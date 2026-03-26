import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import ray

import compare_marl_baseline as cb
from multi_cell_env import MultiCell_5G_SLA_Env


DEFAULT_ROLLOUT_STEPS = 200
DEFAULT_EVAL_SEED = 3026
DEFAULT_EDGE_AGENT = "BS_1"
CENTER_AGENT = "BS_0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze per-step policy behavior for center and one edge agent (IPPO vs MAPPO)."
    )
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED, help="Evaluation seed.")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help="Max rollout steps.",
    )
    parser.add_argument(
        "--edge-agent",
        type=str,
        default=DEFAULT_EDGE_AGENT,
        help="Edge agent id to inspect (e.g., BS_1 ... BS_6).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./results/policy_behavior",
        help="Output prefix for generated files.",
    )
    return parser.parse_args()


def _select_evaluator(algo_key: str) -> Dict:
    evaluators = cb.LEARNED_EVALUATORS.get(algo_key, [])
    if not evaluators:
        raise RuntimeError(f"No evaluator loaded for algo={algo_key}.")

    def ranking_key(item: Dict):
        quality = item.get("quality_score")
        iter_ = item.get("training_iteration")
        total_viol = item.get("center_total_sla_violations")
        return (
            -1e9 if quality is None else float(quality),
            -1e9 if iter_ is None else float(iter_),
            1e9 if total_viol is None else -float(total_viol),
        )

    best = max(evaluators, key=ranking_key)
    return best


def _action_to_ratio(env: MultiCell_5G_SLA_Env, action: np.ndarray) -> np.ndarray:
    return env._action_to_ratios(action).astype(np.float32)


def _safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def run_behavior_rollout(
    algo_key: str,
    evaluator: Dict,
    eval_seed: int,
    rollout_steps: int,
    edge_agent: str,
) -> List[Dict]:
    env = MultiCell_5G_SLA_Env(config=evaluator["env_config"])
    if edge_agent not in env.agents or edge_agent == CENTER_AGENT:
        raise ValueError(f"edge_agent must be one of {env.agents} and not {CENTER_AGENT}. Got {edge_agent!r}")

    obs, reset_infos = env.reset(seed=eval_seed)
    runner, episode, shared_data = cb.build_ippo_episode_context(
        evaluator["algo"],
        obs,
        reset_infos,
    )

    rows: List[Dict] = []
    done = {"__all__": False}
    step = 0

    while not done["__all__"] and step < rollout_steps:
        policy_actions, env_actions, extra_model_outputs = cb.compute_actions_batched(
            evaluator["algo"],
            runner,
            episode,
            shared_data,
        )
        ratio_dict = {
            agent_id: _action_to_ratio(env, action) for agent_id, action in env_actions.items()
        }

        obs, rewards, terminateds, truncateds, infos = env.step(env_actions)
        episode.add_env_step(
            obs,
            policy_actions,
            rewards,
            infos=infos,
            terminateds=terminateds,
            truncateds=truncateds,
            extra_model_outputs=extra_model_outputs,
        )

        center_ratio = ratio_dict.get(CENTER_AGENT, np.zeros(3, dtype=np.float32))
        edge_ratio = ratio_dict.get(edge_agent, np.zeros(3, dtype=np.float32))
        ratio_l1_gap = float(np.sum(np.abs(center_ratio - edge_ratio)))

        for agent_id in (CENTER_AGENT, edge_agent):
            info = infos.get(agent_id, {})
            ratio = ratio_dict.get(agent_id, np.zeros(3, dtype=np.float32))
            tp_slices = np.asarray(info.get("throughput_slices_mbps", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            violations = np.asarray(info.get("violations", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            violations_raw = np.asarray(info.get("violations_raw", np.zeros(3, dtype=np.float32)), dtype=np.float32)

            row = {
                "algo": algo_key,
                "eval_seed": int(eval_seed),
                "train_seed": int(evaluator["train_seed"]),
                "agent": agent_id,
                "step": int(step),
                "ratio_embb": float(ratio[0]),
                "ratio_urllc": float(ratio[1]),
                "ratio_mmtc": float(ratio[2]),
                "embb_tp_mbps": float(tp_slices[0]),
                "urllc_tp_mbps": float(tp_slices[1]),
                "mmtc_tp_mbps": float(tp_slices[2]),
                "total_tp_mbps": _safe_float(info.get("throughput", 0.0)),
                "urllc_delay_ms": _safe_float(info.get("est_urllc_delay", 0.0)) * 1000.0,
                "embb_shortfall_mbps": float(max(0.0, env.sla_props["embb_gbr"] - tp_slices[0])),
                "viol_embb": float(violations[0]),
                "viol_urllc": float(violations[1]),
                "viol_mmtc": float(violations[2]),
                "viol_raw_embb": float(violations_raw[0]),
                "viol_raw_urllc": float(violations_raw[1]),
                "reward": _safe_float(rewards.get(agent_id, 0.0)),
                "reward_local_component": _safe_float(info.get("reward_local_component", 0.0)),
                "reward_neighbor_component": _safe_float(info.get("reward_neighbor_component", 0.0)),
                "reward_dividend_component": _safe_float(info.get("reward_dividend_component", 0.0)),
                "neighbor_penalty_signal": _safe_float(info.get("neighbor_penalty_signal", 0.0)),
                "neighbor_dividend": _safe_float(info.get("neighbor_dividend", 0.0)),
                "center_edge_ratio_l1_gap": ratio_l1_gap,
            }
            rows.append(row)

        done = terminateds
        step += 1

    return rows


def _mean_for(rows: List[Dict], algo: str, agent: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["algo"] == algo and r["agent"] == agent]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _corr_for(rows: List[Dict], algo: str, key_x: str, key_y: str) -> float:
    x = [float(r[key_x]) for r in rows if r["algo"] == algo and r["agent"] == CENTER_AGENT]
    y = [float(r[key_y]) for r in rows if r["algo"] == algo and r["agent"] != CENTER_AGENT]
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def summarize_rows(rows: List[Dict], edge_agent: str) -> Dict:
    summary = {"agents": {}, "cross_agent": {}}
    for algo in ("ippo", "mappo"):
        summary["agents"][algo] = {}
        for agent in (CENTER_AGENT, edge_agent):
            entry = {
                "mean_ratio_embb": _mean_for(rows, algo, agent, "ratio_embb"),
                "mean_ratio_urllc": _mean_for(rows, algo, agent, "ratio_urllc"),
                "mean_ratio_mmtc": _mean_for(rows, algo, agent, "ratio_mmtc"),
                "mean_embb_tp_mbps": _mean_for(rows, algo, agent, "embb_tp_mbps"),
                "mean_urllc_delay_ms": _mean_for(rows, algo, agent, "urllc_delay_ms"),
                "mean_embb_shortfall_mbps": _mean_for(rows, algo, agent, "embb_shortfall_mbps"),
                "mean_reward": _mean_for(rows, algo, agent, "reward"),
                "mean_reward_neighbor_component": _mean_for(rows, algo, agent, "reward_neighbor_component"),
                "mean_reward_dividend_component": _mean_for(rows, algo, agent, "reward_dividend_component"),
                "mean_neighbor_penalty_signal": _mean_for(rows, algo, agent, "neighbor_penalty_signal"),
            }
            summary["agents"][algo][agent] = entry

        summary["cross_agent"][algo] = {
            "mean_center_edge_ratio_l1_gap": float(
                np.mean(
                    [
                        float(r["center_edge_ratio_l1_gap"])
                        for r in rows
                        if r["algo"] == algo and r["agent"] == CENTER_AGENT
                    ]
                )
            ),
            "corr_center_urllc_delay_vs_edge_urllc_ratio": _corr_for(
                rows,
                algo,
                "urllc_delay_ms",
                "ratio_urllc",
            ),
        }

    summary["delta_mappo_minus_ippo"] = {
        "center_mean_embb_tp_mbps": (
            summary["agents"]["mappo"][CENTER_AGENT]["mean_embb_tp_mbps"]
            - summary["agents"]["ippo"][CENTER_AGENT]["mean_embb_tp_mbps"]
        ),
        "center_mean_urllc_delay_ms": (
            summary["agents"]["mappo"][CENTER_AGENT]["mean_urllc_delay_ms"]
            - summary["agents"]["ippo"][CENTER_AGENT]["mean_urllc_delay_ms"]
        ),
        "center_ratio_l1_gap_mean": (
            summary["cross_agent"]["mappo"]["mean_center_edge_ratio_l1_gap"]
            - summary["cross_agent"]["ippo"]["mean_center_edge_ratio_l1_gap"]
        ),
    }
    return summary


def write_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        cb.init_learned_evaluators()

        selected = {}
        for algo in ("ippo", "mappo"):
            evaluator = _select_evaluator(algo)
            selected[algo] = {
                "train_seed": int(evaluator["train_seed"]),
                "training_iteration": int(evaluator["training_iteration"]),
                "quality_score": None if evaluator.get("quality_score") is None else float(evaluator["quality_score"]),
                "checkpoint_path": str(evaluator["checkpoint_path"]),
            }
            print(
                f"Using {algo.upper()} checkpoint: seed={evaluator['train_seed']}, "
                f"iter={evaluator['training_iteration']}, quality={evaluator.get('quality_score')}, "
                f"path={evaluator['checkpoint_path']}"
            )

        rows: List[Dict] = []
        for algo in ("ippo", "mappo"):
            evaluator = _select_evaluator(algo)
            rows.extend(
                run_behavior_rollout(
                    algo_key=algo,
                    evaluator=evaluator,
                    eval_seed=args.eval_seed,
                    rollout_steps=args.rollout_steps,
                    edge_agent=args.edge_agent,
                )
            )

        summary = summarize_rows(rows, edge_agent=args.edge_agent)
        output = {
            "eval_seed": int(args.eval_seed),
            "rollout_steps": int(args.rollout_steps),
            "edge_agent": args.edge_agent,
            "selected_checkpoints": selected,
            "summary": summary,
        }

        csv_path = f"{args.output_prefix}_steps.csv"
        json_path = f"{args.output_prefix}_summary.json"
        write_csv(rows, csv_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print("\n=== Behavior Summary (Center/Edge) ===")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved step-level data: {csv_path}")
        print(f"Saved summary: {json_path}")
    finally:
        cb.stop_learned_evaluators()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

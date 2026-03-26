import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray

import compare_mappo_variants as variants

DEFAULT_SUMMARY_JSON = "./results/mappo_variant_comparison_balanced_summary.json"
DEFAULT_TARGET_KEY = "mappo_current"
DEFAULT_REFERENCE_KEY = "ippo"
DEFAULT_OUTPUT_PREFIX = "./results/mappo_current_violation_diagnostics"

AGENT_ORDER = [f"BS_{idx}" for idx in range(7)]
SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive behavior diagnostics for the system-selected MAPPO checkpoint."
    )
    parser.add_argument("--summary-json", type=str, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--target-key", type=str, default=DEFAULT_TARGET_KEY)
    parser.add_argument("--reference-key", type=str, default=DEFAULT_REFERENCE_KEY)
    parser.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--rollout-steps", type=int, default=variants.base.ROLLOUT_STEPS)
    return parser.parse_args()


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_selection_summary(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selection_summary" not in data:
        raise RuntimeError(f"Missing selection_summary in {path}")
    return data["selection_summary"]


def get_method_by_key(algo_key: str):
    for method in variants.LEARNED_METHODS:
        if method["algo_key"] == algo_key:
            return method
    raise KeyError(f"Unknown algo_key={algo_key}")


def build_evaluator(algo_key: str, selected_item: dict):
    method = get_method_by_key(algo_key)
    env_config = variants.base.LEARNED_ENV_CONFIGS[algo_key]
    checkpoint_path = selected_item["selected_checkpoint"]
    trial_dir = str(Path(checkpoint_path).parent.parent)
    observation_filter = selected_item.get("selected_observation_filter") or variants.base._resolve_trial_observation_filter(trial_dir)
    algo = variants.base.build_learned_eval_algo(observation_filter=observation_filter, env_config=env_config)
    algo.restore(checkpoint_path)
    return {
        "algo_key": algo_key,
        "label": method["label"],
        "algo": algo,
        "env_config": dict(env_config),
        "checkpoint_path": checkpoint_path,
        "training_iteration": int(selected_item.get("selected_iteration", -1)),
        "observation_filter": observation_filter,
    }


def _mean_neighbor_ratios(env, ratio_dict, agent_id):
    neighbor_ids = env.neighbor_map.get(agent_id, [])
    if not neighbor_ids:
        return np.zeros(3, dtype=np.float32)
    arr = np.asarray([ratio_dict[nbr] for nbr in neighbor_ids], dtype=np.float32)
    return np.mean(arr, axis=0).astype(np.float32)


def rollout_diagnostics(evaluator: dict, rollout_steps: int):
    rows = []
    for seed in variants.base.EVAL_SEEDS:
        env = variants.base.MultiCell_5G_SLA_Env(config=evaluator["env_config"])
        obs, reset_infos = env.reset(seed=seed)
        runner, episode, shared_data = variants.base.build_ippo_episode_context(
            evaluator["algo"],
            obs,
            reset_infos,
        )
        done = {"__all__": False}
        step = 0

        while not done["__all__"] and step < rollout_steps:
            pre_arrivals = {agent: env.state[agent][0:3].copy() for agent in env.agents}
            pre_queues = {agent: env.queues[agent].copy() for agent in env.agents}
            pre_se = {agent: env.current_se[agent].copy() for agent in env.agents}
            pre_delay_ms = {agent: float(env.state[agent][12] * 1000.0) for agent in env.agents}
            pre_shortfall = {agent: float(env.state[agent][13]) for agent in env.agents}
            pre_neighbor_ici = {agent: env._get_neighbor_ici_features(agent).copy() for agent in env.agents}

            policy_actions, env_actions, extra_model_outputs = variants.base.compute_actions_batched(
                evaluator["algo"],
                runner,
                episode,
                shared_data,
            )
            ratio_dict = {
                agent_id: env._action_to_ratios(action).astype(np.float32)
                for agent_id, action in env_actions.items()
            }
            mean_neighbor_ratios = {
                agent_id: _mean_neighbor_ratios(env, ratio_dict, agent_id)
                for agent_id in env.agents
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

            for agent_id in env.agents:
                info = infos[agent_id]
                ratio = ratio_dict[agent_id]
                neighbor_ratio = mean_neighbor_ratios[agent_id]
                tp = np.asarray(info.get("throughput_slices_mbps", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                viol = np.asarray(info.get("violations", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                viol_raw = np.asarray(info.get("violations_raw", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                violation_flags = np.asarray(info.get("violation_flags", np.zeros(3, dtype=np.float32)), dtype=np.float32)
                neighbor_ici = pre_neighbor_ici[agent_id]
                rows.append(
                    {
                        "algo_key": evaluator["algo_key"],
                        "algo_label": evaluator["label"],
                        "seed": int(seed),
                        "step": int(step),
                        "quartile": int((4 * step) // max(rollout_steps, 1)),
                        "agent": agent_id,
                        "is_center": int(agent_id == "BS_0"),
                        "arrival_embb": float(pre_arrivals[agent_id][0]),
                        "arrival_urllc": float(pre_arrivals[agent_id][1]),
                        "arrival_mmtc": float(pre_arrivals[agent_id][2]),
                        "queue_embb": float(pre_queues[agent_id][0]),
                        "queue_urllc": float(pre_queues[agent_id][1]),
                        "queue_mmtc": float(pre_queues[agent_id][2]),
                        "se_embb": float(pre_se[agent_id][0]),
                        "se_urllc": float(pre_se[agent_id][1]),
                        "se_mmtc": float(pre_se[agent_id][2]),
                        "prev_delay_ms": float(pre_delay_ms[agent_id]),
                        "prev_embb_shortfall_mbps": float(pre_shortfall[agent_id]),
                        "ratio_embb": float(ratio[0]),
                        "ratio_urllc": float(ratio[1]),
                        "ratio_mmtc": float(ratio[2]),
                        "neighbor_ratio_embb_mean": float(neighbor_ratio[0]),
                        "neighbor_ratio_urllc_mean": float(neighbor_ratio[1]),
                        "neighbor_ratio_mmtc_mean": float(neighbor_ratio[2]),
                        "neighbor_norm_load_embb": float(neighbor_ici[0]),
                        "neighbor_norm_load_urllc": float(neighbor_ici[1]),
                        "est_se_modifier_embb": float(neighbor_ici[2]),
                        "est_se_modifier_urllc": float(neighbor_ici[3]),
                        "embb_tp_mbps": float(tp[0]),
                        "urllc_tp_mbps": float(tp[1]),
                        "mmtc_tp_mbps": float(tp[2]),
                        "system_tp_agent_mbps": float(info.get("throughput", 0.0)),
                        "embb_eval_tp_mbps": float(info.get("embb_eval_tp_mbps", tp[0])),
                        "urllc_delay_ms": float(info.get("est_urllc_delay", 0.0) * 1000.0),
                        "embb_violation": float(viol[0]),
                        "urllc_violation": float(viol[1]),
                        "mmtc_violation": float(viol[2]),
                        "embb_violation_raw": float(viol_raw[0]),
                        "urllc_violation_raw": float(viol_raw[1]),
                        "mmtc_violation_raw": float(viol_raw[2]),
                        "embb_flag": float(violation_flags[0]),
                        "urllc_flag": float(violation_flags[1]),
                        "mmtc_flag": float(violation_flags[2]),
                        "embb_shortfall_mbps": float(max(0.0, env.sla_props["embb_gbr"] - tp[0])),
                        "reward": float(rewards.get(agent_id, 0.0)),
                        "reward_local_component": float(info.get("reward_local_component", 0.0)),
                        "reward_neighbor_component": float(info.get("reward_neighbor_component", 0.0)),
                        "reward_dividend_component": float(info.get("reward_dividend_component", 0.0)),
                        "neighbor_penalty_signal": float(info.get("neighbor_penalty_signal", 0.0)),
                        "neighbor_dividend": float(info.get("neighbor_dividend", 0.0)),
                    }
                )

            done = terminateds
            step += 1
    return rows


def _mean(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([float(r[key]) for r in rows]))


def _p95(rows, key):
    if not rows:
        return float("nan")
    return float(np.percentile([float(r[key]) for r in rows], 95))


def summarize_algo_rows(rows):
    summary = {}
    all_rows = rows
    center_rows = [r for r in rows if r["is_center"] == 1]
    edge_rows = [r for r in rows if r["is_center"] == 0]
    embb_violation_rows = [r for r in rows if r["embb_flag"] > 0.0]
    embb_violation_center = [r for r in center_rows if r["embb_flag"] > 0.0]
    embb_violation_edge = [r for r in edge_rows if r["embb_flag"] > 0.0]

    summary["overall"] = {
        "system_sla_success": {
            "embb": 1.0 - _mean(all_rows, "embb_flag"),
            "urllc": 1.0 - _mean(all_rows, "urllc_flag"),
            "mmtc": 1.0 - _mean(all_rows, "mmtc_flag"),
        },
        "center_sla_success_embb": 1.0 - _mean(center_rows, "embb_flag"),
        "edge_sla_success_embb": 1.0 - _mean(edge_rows, "embb_flag"),
        "system_total_sla_violations": (
            _mean(all_rows, "embb_flag") + _mean(all_rows, "urllc_flag") + _mean(all_rows, "mmtc_flag")
        ),
        "mean_embb_ratio_center": _mean(center_rows, "ratio_embb"),
        "mean_embb_ratio_edge": _mean(edge_rows, "ratio_embb"),
        "mean_urllc_ratio_center": _mean(center_rows, "ratio_urllc"),
        "mean_urllc_ratio_edge": _mean(edge_rows, "ratio_urllc"),
        "mean_embb_tp_center_mbps": _mean(center_rows, "embb_tp_mbps"),
        "mean_embb_tp_edge_mbps": _mean(edge_rows, "embb_tp_mbps"),
        "mean_urllc_delay_center_ms": _mean(center_rows, "urllc_delay_ms"),
        "mean_urllc_delay_edge_ms": _mean(edge_rows, "urllc_delay_ms"),
    }

    summary["embb_violation_conditioned"] = {
        "count": len(embb_violation_rows),
        "share_all_steps": len(embb_violation_rows) / max(len(all_rows), 1),
        "share_center_steps": len(embb_violation_center) / max(len(center_rows), 1),
        "share_edge_steps": len(embb_violation_edge) / max(len(edge_rows), 1),
        "mean_urllc_delay_ms": _mean(embb_violation_rows, "urllc_delay_ms"),
        "p95_urllc_delay_ms": _p95(embb_violation_rows, "urllc_delay_ms"),
        "mean_embb_ratio": _mean(embb_violation_rows, "ratio_embb"),
        "mean_urllc_ratio": _mean(embb_violation_rows, "ratio_urllc"),
        "mean_embb_shortfall_mbps": _mean(embb_violation_rows, "embb_shortfall_mbps"),
        "mean_neighbor_ratio_embb": _mean(embb_violation_rows, "neighbor_ratio_embb_mean"),
        "mean_neighbor_ratio_urllc": _mean(embb_violation_rows, "neighbor_ratio_urllc_mean"),
        "mean_est_se_modifier_embb": _mean(embb_violation_rows, "est_se_modifier_embb"),
        "mean_est_se_modifier_urllc": _mean(embb_violation_rows, "est_se_modifier_urllc"),
        "urllc_safety_share_lt_0_1ms": float(np.mean([r["urllc_delay_ms"] < 0.1 for r in embb_violation_rows])) if embb_violation_rows else float("nan"),
        "urllc_safety_share_lt_0_2ms": float(np.mean([r["urllc_delay_ms"] < 0.2 for r in embb_violation_rows])) if embb_violation_rows else float("nan"),
        "urllc_safety_share_lt_0_5ms": float(np.mean([r["urllc_delay_ms"] < 0.5 for r in embb_violation_rows])) if embb_violation_rows else float("nan"),
        "urllc_safety_share_lt_1_0ms": float(np.mean([r["urllc_delay_ms"] < 1.0 for r in embb_violation_rows])) if embb_violation_rows else float("nan"),
    }

    summary["violation_timing_quartiles"] = []
    for quartile in range(4):
        q_rows = [r for r in rows if r["quartile"] == quartile]
        summary["violation_timing_quartiles"].append(
            {
                "quartile": quartile,
                "embb_violation_rate": _mean(q_rows, "embb_flag"),
                "urllc_violation_rate": _mean(q_rows, "urllc_flag"),
                "mmtc_violation_rate": _mean(q_rows, "mmtc_flag"),
                "mean_embb_ratio": _mean(q_rows, "ratio_embb"),
                "mean_urllc_ratio": _mean(q_rows, "ratio_urllc"),
            }
        )

    summary["per_cell_embb"] = {}
    for agent in AGENT_ORDER:
        agent_rows = [r for r in rows if r["agent"] == agent]
        summary["per_cell_embb"][agent] = {
            "sla_success": 1.0 - _mean(agent_rows, "embb_flag"),
            "mean_shortfall_mbps": _mean(agent_rows, "embb_shortfall_mbps"),
            "mean_embb_tp_mbps": _mean(agent_rows, "embb_tp_mbps"),
            "mean_embb_ratio": _mean(agent_rows, "ratio_embb"),
            "mean_neighbor_ratio_embb": _mean(agent_rows, "neighbor_ratio_embb_mean"),
            "mean_est_se_modifier_embb": _mean(agent_rows, "est_se_modifier_embb"),
        }

    summary["hardest_cells_by_embb_sla"] = sorted(
        [
            {
                "agent": agent,
                **summary["per_cell_embb"][agent],
            }
            for agent in AGENT_ORDER
        ],
        key=lambda item: item["sla_success"],
    )
    return summary


def compare_summaries(target_summary, reference_summary):
    return {
        "system_embb_sla_delta": (
            target_summary["overall"]["system_sla_success"]["embb"]
            - reference_summary["overall"]["system_sla_success"]["embb"]
        ),
        "center_embb_sla_delta": (
            target_summary["overall"]["center_sla_success_embb"]
            - reference_summary["overall"]["center_sla_success_embb"]
        ),
        "edge_embb_sla_delta": (
            target_summary["overall"]["edge_sla_success_embb"]
            - reference_summary["overall"]["edge_sla_success_embb"]
        ),
        "system_tp_proxy_delta_mbps": (
            target_summary["overall"]["mean_embb_tp_center_mbps"] + target_summary["overall"]["mean_embb_tp_edge_mbps"]
            - reference_summary["overall"]["mean_embb_tp_center_mbps"] - reference_summary["overall"]["mean_embb_tp_edge_mbps"]
        ),
        "embb_violation_conditioned_urllc_delay_delta_ms": (
            target_summary["embb_violation_conditioned"]["mean_urllc_delay_ms"]
            - reference_summary["embb_violation_conditioned"]["mean_urllc_delay_ms"]
        ),
        "embb_violation_conditioned_embb_ratio_delta": (
            target_summary["embb_violation_conditioned"]["mean_embb_ratio"]
            - reference_summary["embb_violation_conditioned"]["mean_embb_ratio"]
        ),
    }


def write_csv(rows, path):
    ensure_parent(path)
    if not rows:
        raise RuntimeError("No rows to write.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_violation_rate_plot(path, rows_by_algo):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for algo_key, rows in rows_by_algo.items():
        label = rows[0]["algo_label"] if rows else algo_key
        max_step = int(max(r["step"] for r in rows)) + 1
        embb_rates = []
        center_rates = []
        for step in range(max_step):
            step_rows = [r for r in rows if r["step"] == step]
            center_step_rows = [r for r in step_rows if r["is_center"] == 1]
            embb_rates.append(np.mean([r["embb_flag"] for r in step_rows]))
            center_rates.append(np.mean([r["embb_flag"] for r in center_step_rows]))
        axes[0].plot(embb_rates, label=label)
        axes[1].plot(center_rates, label=label)
    axes[0].set_title("System eMBB Violation Rate Over Time")
    axes[0].set_ylabel("Violation Rate")
    axes[1].set_title("BS_0 eMBB Violation Rate Over Time")
    axes[1].set_ylabel("Violation Rate")
    axes[1].set_xlabel("Step")
    axes[0].grid(True)
    axes[1].grid(True)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_per_cell_sla_plot(path, rows_by_algo):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(AGENT_ORDER))
    width = 0.35
    algo_items = list(rows_by_algo.items())
    offsets = np.linspace(-width / 2, width / 2, num=len(algo_items))
    for offset, (algo_key, rows) in zip(offsets, algo_items):
        label = rows[0]["algo_label"] if rows else algo_key
        vals = []
        for agent in AGENT_ORDER:
            agent_rows = [r for r in rows if r["agent"] == agent]
            vals.append(1.0 - np.mean([r["embb_flag"] for r in agent_rows]))
        ax.bar(x + offset, vals, width=max(width / len(algo_items), 0.18), label=label)
    ax.set_xticks(x, AGENT_ORDER)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("eMBB SLA Success")
    ax.set_title("Per-cell eMBB SLA Success")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_quartile_plot(path, rows_by_algo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    quartiles = np.arange(4)
    width = 0.35
    algo_items = list(rows_by_algo.items())
    offsets = np.linspace(-width / 2, width / 2, num=len(algo_items))
    for offset, (algo_key, rows) in zip(offsets, algo_items):
        label = rows[0]["algo_label"] if rows else algo_key
        embb_rates = []
        urllc_rates = []
        for q in quartiles:
            q_rows = [r for r in rows if r["quartile"] == q]
            embb_rates.append(np.mean([r["embb_flag"] for r in q_rows]))
            urllc_rates.append(np.mean([r["urllc_flag"] for r in q_rows]))
        bar_w = max(width / len(algo_items), 0.18)
        axes[0].bar(quartiles + offset, embb_rates, width=bar_w, label=label)
        axes[1].bar(quartiles + offset, urllc_rates, width=bar_w, label=label)
    axes[0].set_title("eMBB Violation Rate by Episode Quartile")
    axes[1].set_title("URLLC Violation Rate by Episode Quartile")
    for ax in axes:
        ax.set_xticks(quartiles, ["Q1", "Q2", "Q3", "Q4"])
        ax.grid(True, axis="y")
        ax.set_ylabel("Violation Rate")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_shortfall_condition_plot(path, rows_by_algo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = []
    center_embb = []
    center_urllc = []
    edge_embb = []
    edge_urllc = []
    for algo_key, rows in rows_by_algo.items():
        label = rows[0]["algo_label"] if rows else algo_key
        labels.append(label)
        center_short = [r for r in rows if r["is_center"] == 1 and r["embb_flag"] > 0.0]
        edge_short = [r for r in rows if r["is_center"] == 0 and r["embb_flag"] > 0.0]
        center_embb.append(_mean(center_short, "ratio_embb"))
        center_urllc.append(_mean(center_short, "ratio_urllc"))
        edge_embb.append(_mean(edge_short, "ratio_embb"))
        edge_urllc.append(_mean(edge_short, "ratio_urllc"))
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, center_embb, width=w, label="Center eMBB ratio")
    axes[0].bar(x + w / 2, center_urllc, width=w, label="Center URLLC ratio")
    axes[1].bar(x - w / 2, edge_embb, width=w, label="Edge eMBB ratio")
    axes[1].bar(x + w / 2, edge_urllc, width=w, label="Edge URLLC ratio")
    axes[0].set_title("BS_0 Ratios on eMBB Violation Steps")
    axes[1].set_title("Edge Ratios on eMBB Violation Steps")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(True, axis="y")
        ax.set_ylabel("Mean Ratio")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_parent(args.output_prefix)
    variants.configure_base_module()
    selection_summary = load_selection_summary(args.summary_json)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    evaluators = {}
    rows_by_algo = {}
    summaries = {}
    try:
        for algo_key in [args.target_key, args.reference_key]:
            selected_item = selection_summary.get(algo_key)
            if selected_item is None:
                raise RuntimeError(f"Missing selection summary for {algo_key}")
            evaluators[algo_key] = build_evaluator(algo_key, selected_item)
            print(
                f"Using {algo_key}: iter={evaluators[algo_key]['training_iteration']}, "
                f"obs_filter={evaluators[algo_key]['observation_filter']}, "
                f"checkpoint={evaluators[algo_key]['checkpoint_path']}"
            )
            rows_by_algo[algo_key] = rollout_diagnostics(evaluators[algo_key], args.rollout_steps)
            summaries[algo_key] = summarize_algo_rows(rows_by_algo[algo_key])

        comparison = compare_summaries(summaries[args.target_key], summaries[args.reference_key])

        all_rows = rows_by_algo[args.target_key] + rows_by_algo[args.reference_key]
        csv_path = f"{args.output_prefix}_steps.csv"
        json_path = f"{args.output_prefix}_summary.json"
        plot_violation_path = f"{args.output_prefix}_embb_violation_over_time.png"
        plot_cell_path = f"{args.output_prefix}_per_cell_embb_sla.png"
        plot_quartile_path = f"{args.output_prefix}_violation_timing_quartiles.png"
        plot_ratio_path = f"{args.output_prefix}_shortfall_conditioned_ratios.png"

        write_csv(all_rows, csv_path)
        save_violation_rate_plot(plot_violation_path, rows_by_algo)
        save_per_cell_sla_plot(plot_cell_path, rows_by_algo)
        save_quartile_plot(plot_quartile_path, rows_by_algo)
        save_shortfall_condition_plot(plot_ratio_path, rows_by_algo)

        output = {
            "target_key": args.target_key,
            "reference_key": args.reference_key,
            "selected_checkpoints": {
                algo_key: {
                    "checkpoint_path": evaluators[algo_key]["checkpoint_path"],
                    "training_iteration": evaluators[algo_key]["training_iteration"],
                    "observation_filter": evaluators[algo_key]["observation_filter"],
                }
                for algo_key in [args.target_key, args.reference_key]
            },
            "summaries": summaries,
            "comparison": comparison,
            "artifacts": {
                "csv": csv_path,
                "plots": [
                    plot_violation_path,
                    plot_cell_path,
                    plot_quartile_path,
                    plot_ratio_path,
                ],
            },
        }
        Path(json_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n=== Comparison Summary ===")
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        print(f"Saved step-level csv: {csv_path}")
        print(f"Saved summary json: {json_path}")
        print(f"Saved plots: {plot_violation_path}, {plot_cell_path}, {plot_quartile_path}, {plot_ratio_path}")
    finally:
        for evaluator in evaluators.values():
            evaluator["algo"].stop()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

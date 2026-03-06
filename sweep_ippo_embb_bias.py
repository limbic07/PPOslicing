import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import ray

from compare_marl_baseline import (
    EVAL_SEEDS,
    ENV_CONFIG,
    ROLLOUT_STEPS,
    build_ippo_episode_context,
    compute_actions_batched,
    init_ippo_evaluators,
    ratios_to_action,
    stop_ippo_evaluators,
    validate_seed_split,
)
from multi_cell_env import MultiCell_5G_SLA_Env

CENTER_AGENT_ID = "BS_0"
SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
SLICE_COLORS = {
    "eMBB": "#1f77b4",
    "URLLC": "#ff7f0e",
    "mMTC": "#2ca02c",
}
BIAS_EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop bias sweep for IPPO: maximize eMBB SLA under SLA-oriented constraints."
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=ROLLOUT_STEPS,
        help="Maximum rollout steps per evaluation run.",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional evaluation seeds. Defaults to compare_marl_baseline.py EVAL_SEEDS.",
    )
    parser.add_argument(
        "--bias-deltas",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
        help="Absolute ratio deltas shifted from URLLC to eMBB for every agent at every step.",
    )
    parser.add_argument(
        "--urllc-sla-floor",
        type=float,
        default=None,
        help="Optional explicit URLLC system SLA floor. Defaults to the delta=0 baseline.",
    )
    parser.add_argument(
        "--mmtc-sla-floor",
        type=float,
        default=None,
        help="Optional explicit mMTC system SLA floor. Defaults to the delta=0 baseline.",
    )
    parser.add_argument(
        "--urllc-delay-p99-cap-ms",
        type=float,
        default=2.0,
        help="System-wide URLLC p99 delay cap used for feasible-delta selection.",
    )
    parser.add_argument(
        "--selection-tol",
        type=float,
        default=1e-4,
        help="Tolerance when checking non-inferior SLA constraints against baseline.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./results/ippo_embb_bias_sweep",
        help="Output prefix for figure and JSON summary.",
    )
    return parser.parse_args()


def summarize_scalar(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def apply_embb_bias(
    env: MultiCell_5G_SLA_Env,
    env_actions: dict[str, np.ndarray],
    bias_delta: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    biased_actions = {}
    executed_ratios = {}

    for agent_id, action in env_actions.items():
        base_ratios = env._action_to_ratios(action).astype(np.float32)
        if bias_delta <= 0.0:
            biased_action = np.asarray(action, dtype=np.float32).copy()
            executed_ratio = base_ratios
        else:
            shift = min(float(bias_delta), max(float(base_ratios[1]) - BIAS_EPS, 0.0))
            target_ratios = base_ratios.copy()
            target_ratios[0] += shift
            target_ratios[1] -= shift
            target_ratios = np.clip(target_ratios, BIAS_EPS, None)
            target_ratios = (target_ratios / np.sum(target_ratios)).astype(np.float32)
            biased_action = ratios_to_action(target_ratios, env.action_softmax_temperature)
            executed_ratio = env._action_to_ratios(biased_action).astype(np.float32)

        biased_actions[agent_id] = biased_action.astype(np.float32)
        executed_ratios[agent_id] = executed_ratio

    return biased_actions, executed_ratios


def run_biased_rollout(evaluator: dict, eval_seed: int, bias_delta: float, rollout_steps: int) -> dict:
    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    obs, reset_infos = env.reset(seed=eval_seed)
    runner, episode, shared_data = build_ippo_episode_context(evaluator["algo"], obs, reset_infos)

    system_sla_ok = np.zeros(3, dtype=np.float64)
    bs0_sla_ok = np.zeros(3, dtype=np.float64)
    system_steps = 0
    bs0_steps = 0

    center_delay_ms = []
    all_agent_delay_ms = []
    system_throughput_mbps = []
    system_embb_shortfall_mbps = []
    center_embb_shortfall_mbps = []
    center_embb_eval_tp_mbps = []
    executed_ratio_mean = []
    center_ratio = []

    done = {"__all__": False}
    step = 0
    embb_gbr = float(env.sla_props["embb_gbr"])

    while not done["__all__"] and step < rollout_steps:
        policy_actions, env_actions, extra_model_outputs = compute_actions_batched(
            evaluator["algo"],
            runner,
            episode,
            shared_data,
        )
        biased_actions, executed_ratios = apply_embb_bias(env, env_actions, bias_delta)

        obs, rewards, terminateds, truncateds, infos = env.step(biased_actions)
        episode.add_env_step(
            obs,
            policy_actions,
            rewards,
            infos=infos,
            terminateds=terminateds,
            truncateds=truncateds,
            extra_model_outputs=extra_model_outputs,
        )

        step_system_tp = 0.0
        step_embb_shortfalls = []
        for agent_id in env.agents:
            if agent_id not in infos:
                continue

            info = infos[agent_id]
            violations = np.asarray(info["violations"], dtype=np.float64)
            system_sla_ok += (violations <= 0.0).astype(np.float64)
            system_steps += 1

            delay_ms = float(info["est_urllc_delay"] * 1000.0)
            embb_eval_tp_mbps = float(info["embb_eval_tp_mbps"])
            embb_shortfall_mbps = max(0.0, embb_gbr - embb_eval_tp_mbps)
            all_agent_delay_ms.append(delay_ms)
            step_embb_shortfalls.append(embb_shortfall_mbps)
            step_system_tp += float(info["throughput"])

            if agent_id == CENTER_AGENT_ID:
                bs0_sla_ok += (violations <= 0.0).astype(np.float64)
                bs0_steps += 1
                center_delay_ms.append(delay_ms)
                center_embb_shortfall_mbps.append(embb_shortfall_mbps)
                center_embb_eval_tp_mbps.append(embb_eval_tp_mbps)
                center_ratio.append(executed_ratios[agent_id])

        system_throughput_mbps.append(step_system_tp)
        system_embb_shortfall_mbps.append(float(np.mean(step_embb_shortfalls)))
        executed_ratio_mean.append(
            np.asarray([executed_ratios[agent_id] for agent_id in env.agents], dtype=np.float32).mean(axis=0)
        )

        done = terminateds
        step += 1

    return {
        "bias_delta": float(bias_delta),
        "train_seed": int(evaluator["train_seed"]),
        "eval_seed": int(eval_seed),
        "steps": int(step),
        "checkpoint_path": str(evaluator["checkpoint_path"]),
        "quality_score": None if evaluator.get("quality_score") is None else float(evaluator["quality_score"]),
        "sla_sys_success_rate": system_sla_ok / max(float(system_steps), 1.0),
        "sla_bs0_success_rate": bs0_sla_ok / max(float(bs0_steps), 1.0),
        "center_delay_ms": np.asarray(center_delay_ms, dtype=np.float32),
        "all_agent_delay_ms": np.asarray(all_agent_delay_ms, dtype=np.float32),
        "system_throughput_mbps": np.asarray(system_throughput_mbps, dtype=np.float32),
        "system_embb_shortfall_mbps": np.asarray(system_embb_shortfall_mbps, dtype=np.float32),
        "center_embb_shortfall_mbps": np.asarray(center_embb_shortfall_mbps, dtype=np.float32),
        "center_embb_eval_tp_mbps": np.asarray(center_embb_eval_tp_mbps, dtype=np.float32),
        "executed_ratio_mean": np.asarray(executed_ratio_mean, dtype=np.float32),
        "center_ratio": np.asarray(center_ratio, dtype=np.float32),
    }


def aggregate_delta_runs(run_list: list[dict]) -> dict:
    sla_sys = np.asarray([run["sla_sys_success_rate"] for run in run_list], dtype=np.float64)
    sla_bs0 = np.asarray([run["sla_bs0_success_rate"] for run in run_list], dtype=np.float64)
    system_tp_scalar = np.asarray([np.mean(run["system_throughput_mbps"]) for run in run_list], dtype=np.float64)
    system_shortfall_scalar = np.asarray([np.mean(run["system_embb_shortfall_mbps"]) for run in run_list], dtype=np.float64)
    center_shortfall_scalar = np.asarray([np.mean(run["center_embb_shortfall_mbps"]) for run in run_list], dtype=np.float64)
    center_eval_tp_scalar = np.asarray([np.mean(run["center_embb_eval_tp_mbps"]) for run in run_list], dtype=np.float64)
    center_delay_scalar = np.asarray([np.mean(run["center_delay_ms"]) for run in run_list], dtype=np.float64)

    all_agent_delay_ms = np.concatenate([run["all_agent_delay_ms"] for run in run_list], axis=0).astype(np.float32)
    center_delay_ms = np.concatenate([run["center_delay_ms"] for run in run_list], axis=0).astype(np.float32)
    system_tp_series = np.concatenate([run["system_throughput_mbps"] for run in run_list], axis=0).astype(np.float32)
    system_shortfall_series = np.concatenate(
        [run["system_embb_shortfall_mbps"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    center_shortfall_series = np.concatenate(
        [run["center_embb_shortfall_mbps"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    executed_ratio_mean = np.concatenate([run["executed_ratio_mean"] for run in run_list], axis=0).astype(np.float32)
    center_ratio = np.concatenate([run["center_ratio"] for run in run_list], axis=0).astype(np.float32)

    return {
        "num_runs": len(run_list),
        "sla_sys_mean": np.mean(sla_sys, axis=0),
        "sla_sys_std": np.std(sla_sys, axis=0),
        "sla_bs0_mean": np.mean(sla_bs0, axis=0),
        "sla_bs0_std": np.std(sla_bs0, axis=0),
        "system_tp_mbps_mean": float(np.mean(system_tp_scalar)),
        "system_tp_mbps_std": float(np.std(system_tp_scalar)),
        "system_embb_shortfall_mbps_mean": float(np.mean(system_shortfall_scalar)),
        "system_embb_shortfall_mbps_std": float(np.std(system_shortfall_scalar)),
        "center_embb_shortfall_mbps_mean": float(np.mean(center_shortfall_scalar)),
        "center_embb_shortfall_mbps_std": float(np.std(center_shortfall_scalar)),
        "center_embb_eval_tp_mbps_mean": float(np.mean(center_eval_tp_scalar)),
        "center_embb_eval_tp_mbps_std": float(np.std(center_eval_tp_scalar)),
        "center_delay_ms_mean": float(np.mean(center_delay_scalar)),
        "center_delay_ms_std": float(np.std(center_delay_scalar)),
        "system_delay_summary": summarize_scalar(all_agent_delay_ms),
        "center_delay_summary": summarize_scalar(center_delay_ms),
        "system_tp_summary": summarize_scalar(system_tp_series),
        "system_embb_shortfall_summary": summarize_scalar(system_shortfall_series),
        "center_embb_shortfall_summary": summarize_scalar(center_shortfall_series),
        "executed_ratio_mean": {
            slice_name: float(np.mean(executed_ratio_mean[:, idx])) for idx, slice_name in enumerate(SLICE_NAMES)
        },
        "center_ratio_mean": {
            slice_name: float(np.mean(center_ratio[:, idx])) for idx, slice_name in enumerate(SLICE_NAMES)
        },
        "run_metadata": [
            {
                "train_seed": int(run["train_seed"]),
                "eval_seed": int(run["eval_seed"]),
                "steps": int(run["steps"]),
                "checkpoint_path": str(run["checkpoint_path"]),
                "quality_score": run["quality_score"],
            }
            for run in run_list
        ],
    }


def choose_recommended_delta(
    aggregated_by_delta: dict[float, dict],
    urllc_sla_floor: float | None,
    mmtc_sla_floor: float | None,
    urllc_delay_p99_cap_ms: float,
    selection_tol: float,
) -> dict:
    ordered_deltas = sorted(aggregated_by_delta)
    baseline_delta = min(ordered_deltas, key=lambda item: abs(item))
    baseline = aggregated_by_delta[baseline_delta]

    effective_urllc_floor = (
        float(urllc_sla_floor) if urllc_sla_floor is not None else float(baseline["sla_sys_mean"][1])
    )
    effective_mmtc_floor = (
        float(mmtc_sla_floor) if mmtc_sla_floor is not None else float(baseline["sla_sys_mean"][2])
    )

    feasible = []
    for delta in ordered_deltas:
        stats = aggregated_by_delta[delta]
        urllc_sla = float(stats["sla_sys_mean"][1])
        mmtc_sla = float(stats["sla_sys_mean"][2])
        urllc_p99 = float(stats["system_delay_summary"]["p99"])
        is_feasible = (
            urllc_sla >= effective_urllc_floor - selection_tol
            and mmtc_sla >= effective_mmtc_floor - selection_tol
            and urllc_p99 <= urllc_delay_p99_cap_ms + 1e-9
        )
        if is_feasible:
            feasible.append(delta)

    if not feasible:
        recommended_delta = baseline_delta
    else:
        recommended_delta = max(
            feasible,
            key=lambda delta: (
                float(aggregated_by_delta[delta]["sla_sys_mean"][0]),
                -float(aggregated_by_delta[delta]["system_embb_shortfall_mbps_mean"]),
                float(aggregated_by_delta[delta]["system_tp_mbps_mean"]),
                -delta,
            ),
        )

    return {
        "baseline_delta": float(baseline_delta),
        "recommended_delta": float(recommended_delta),
        "feasible_deltas": [float(delta) for delta in feasible],
        "constraint_definition": {
            "urllc_sla_floor": float(effective_urllc_floor),
            "mmtc_sla_floor": float(effective_mmtc_floor),
            "urllc_delay_p99_cap_ms": float(urllc_delay_p99_cap_ms),
            "selection_tol": float(selection_tol),
        },
    }


def plot_bias_sweep(aggregated_by_delta: dict[float, dict], selection: dict, output_path: str):
    deltas = np.asarray(sorted(aggregated_by_delta), dtype=np.float64)
    embb_sla = np.asarray([aggregated_by_delta[delta]["sla_sys_mean"][0] for delta in deltas], dtype=np.float64)
    urllc_sla = np.asarray([aggregated_by_delta[delta]["sla_sys_mean"][1] for delta in deltas], dtype=np.float64)
    mmtc_sla = np.asarray([aggregated_by_delta[delta]["sla_sys_mean"][2] for delta in deltas], dtype=np.float64)
    sys_tp = np.asarray([aggregated_by_delta[delta]["system_tp_mbps_mean"] for delta in deltas], dtype=np.float64)
    sys_shortfall = np.asarray(
        [aggregated_by_delta[delta]["system_embb_shortfall_mbps_mean"] for delta in deltas],
        dtype=np.float64,
    )
    delay_p95 = np.asarray(
        [aggregated_by_delta[delta]["system_delay_summary"]["p95"] for delta in deltas],
        dtype=np.float64,
    )
    delay_p99 = np.asarray(
        [aggregated_by_delta[delta]["system_delay_summary"]["p99"] for delta in deltas],
        dtype=np.float64,
    )
    delay_max = np.asarray(
        [aggregated_by_delta[delta]["system_delay_summary"]["max"] for delta in deltas],
        dtype=np.float64,
    )
    embb_ratio = np.asarray(
        [aggregated_by_delta[delta]["executed_ratio_mean"]["eMBB"] for delta in deltas],
        dtype=np.float64,
    )
    urllc_ratio = np.asarray(
        [aggregated_by_delta[delta]["executed_ratio_mean"]["URLLC"] for delta in deltas],
        dtype=np.float64,
    )
    recommended_delta = float(selection["recommended_delta"])
    feasible_set = set(selection["feasible_deltas"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(deltas, embb_sla, marker="o", linewidth=2.0, color=SLICE_COLORS["eMBB"], label="eMBB SLA")
    ax1.plot(deltas, urllc_sla, marker="o", linewidth=2.0, color=SLICE_COLORS["URLLC"], label="URLLC SLA")
    ax1.plot(deltas, mmtc_sla, marker="o", linewidth=2.0, color=SLICE_COLORS["mMTC"], label="mMTC SLA")
    for delta in deltas:
        if float(delta) in feasible_set:
            ax1.axvspan(delta - 0.001, delta + 0.001, color="#dddddd", alpha=0.25)
    ax1.axvline(recommended_delta, color="#444444", linestyle="--", linewidth=1.5, label="Recommended")
    ax1.set_title("System SLA Success vs eMBB Bias")
    ax1.set_xlabel("Bias delta (URLLC -> eMBB)")
    ax1.set_ylabel("System SLA success")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(deltas, sys_tp, marker="o", linewidth=2.0, color="#4c72b0", label="System throughput")
    ax2.axvline(recommended_delta, color="#444444", linestyle="--", linewidth=1.5)
    ax2.set_title("Throughput and eMBB Shortfall")
    ax2.set_xlabel("Bias delta (URLLC -> eMBB)")
    ax2.set_ylabel("System throughput (Mbps)")
    ax2.grid(True, alpha=0.25)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        deltas,
        sys_shortfall,
        marker="s",
        linewidth=2.0,
        color="#c44e52",
        label="System eMBB shortfall",
    )
    ax2_twin.set_ylabel("System eMBB shortfall (Mbps)")

    ax3.plot(deltas, delay_p95, marker="o", linewidth=2.0, color="#55a868", label="System URLLC delay p95")
    ax3.plot(deltas, delay_p99, marker="o", linewidth=2.0, color="#8172b2", label="System URLLC delay p99")
    ax3.plot(deltas, delay_max, marker="o", linewidth=2.0, color="#dd8452", label="System URLLC delay max")
    ax3.axhline(
        selection["constraint_definition"]["urllc_delay_p99_cap_ms"],
        color="#444444",
        linestyle="--",
        linewidth=1.5,
        label="p99 cap",
    )
    ax3.axvline(recommended_delta, color="#444444", linestyle="--", linewidth=1.5)
    ax3.set_title("System URLLC Delay Tail")
    ax3.set_xlabel("Bias delta (URLLC -> eMBB)")
    ax3.set_ylabel("Delay (ms)")
    ax3.grid(True, alpha=0.25)
    ax3.legend()

    ax4.plot(deltas, embb_ratio, marker="o", linewidth=2.0, color=SLICE_COLORS["eMBB"], label="Executed eMBB ratio")
    ax4.plot(
        deltas,
        urllc_ratio,
        marker="o",
        linewidth=2.0,
        color=SLICE_COLORS["URLLC"],
        label="Executed URLLC ratio",
    )
    ax4.axvline(recommended_delta, color="#444444", linestyle="--", linewidth=1.5)
    ax4.set_title("Mean Executed Ratio After Bias")
    ax4.set_xlabel("Bias delta (URLLC -> eMBB)")
    ax4.set_ylabel("Mean ratio")
    ax4.grid(True, alpha=0.25)
    ax4.legend()

    fig.suptitle("IPPO Closed-Loop eMBB Bias Sweep", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_json_summary(aggregated_by_delta: dict[float, dict], selection: dict) -> dict:
    return {
        "objective": "Maximize system eMBB SLA success subject to SLA-oriented constraints.",
        "selection": selection,
        "deltas": {
            f"{delta:.3f}": {
                "num_runs": int(stats["num_runs"]),
                "sla_sys_mean": [float(item) for item in stats["sla_sys_mean"]],
                "sla_sys_std": [float(item) for item in stats["sla_sys_std"]],
                "sla_bs0_mean": [float(item) for item in stats["sla_bs0_mean"]],
                "sla_bs0_std": [float(item) for item in stats["sla_bs0_std"]],
                "system_tp_mbps_mean": float(stats["system_tp_mbps_mean"]),
                "system_tp_mbps_std": float(stats["system_tp_mbps_std"]),
                "system_embb_shortfall_mbps_mean": float(stats["system_embb_shortfall_mbps_mean"]),
                "system_embb_shortfall_mbps_std": float(stats["system_embb_shortfall_mbps_std"]),
                "center_embb_shortfall_mbps_mean": float(stats["center_embb_shortfall_mbps_mean"]),
                "center_embb_shortfall_mbps_std": float(stats["center_embb_shortfall_mbps_std"]),
                "center_embb_eval_tp_mbps_mean": float(stats["center_embb_eval_tp_mbps_mean"]),
                "center_embb_eval_tp_mbps_std": float(stats["center_embb_eval_tp_mbps_std"]),
                "center_delay_ms_mean": float(stats["center_delay_ms_mean"]),
                "center_delay_ms_std": float(stats["center_delay_ms_std"]),
                "system_delay_summary": stats["system_delay_summary"],
                "center_delay_summary": stats["center_delay_summary"],
                "system_tp_summary": stats["system_tp_summary"],
                "system_embb_shortfall_summary": stats["system_embb_shortfall_summary"],
                "center_embb_shortfall_summary": stats["center_embb_shortfall_summary"],
                "executed_ratio_mean": stats["executed_ratio_mean"],
                "center_ratio_mean": stats["center_ratio_mean"],
                "run_metadata": stats["run_metadata"],
            }
            for delta, stats in sorted(aggregated_by_delta.items())
        },
    }


def print_summary(aggregated_by_delta: dict[float, dict], selection: dict):
    print("=== IPPO eMBB Bias Sweep Summary ===")
    print(
        "Selection objective: maximize system eMBB SLA success subject to "
        f"URLLC SLA >= {selection['constraint_definition']['urllc_sla_floor']:.4f}, "
        f"mMTC SLA >= {selection['constraint_definition']['mmtc_sla_floor']:.4f}, "
        f"system URLLC p99 <= {selection['constraint_definition']['urllc_delay_p99_cap_ms']:.3f} ms"
    )
    for delta, stats in sorted(aggregated_by_delta.items()):
        print(
            f"delta={delta:.3f}: "
            f"SLA_sys[eMBB/URLLC/mMTC]="
            f"{stats['sla_sys_mean'][0]*100:.1f}%/"
            f"{stats['sla_sys_mean'][1]*100:.1f}%/"
            f"{stats['sla_sys_mean'][2]*100:.1f}%, "
            f"SysTP={stats['system_tp_mbps_mean']:.2f} Mbps, "
            f"SysShortfall={stats['system_embb_shortfall_mbps_mean']:.2f} Mbps, "
            f"URLLC p99={stats['system_delay_summary']['p99']:.3f} ms"
        )
    print(
        f"Baseline delta={selection['baseline_delta']:.3f} | "
        f"Feasible deltas={selection['feasible_deltas']} | "
        f"Recommended delta={selection['recommended_delta']:.3f}"
    )


def main():
    args = parse_args()
    eval_seeds = list(EVAL_SEEDS if args.eval_seeds is None else args.eval_seeds)
    bias_deltas = sorted({float(delta) for delta in args.bias_deltas if delta >= 0.0})
    if not bias_deltas:
        raise ValueError("bias_deltas must contain at least one non-negative delta.")

    validate_seed_split()
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    evaluators = init_ippo_evaluators()
    runs_by_delta = defaultdict(list)
    try:
        for delta in bias_deltas:
            for evaluator in evaluators:
                for eval_seed in eval_seeds:
                    print(
                        f"Running biased rollout: delta={delta:.3f}, "
                        f"train_seed={evaluator['train_seed']}, eval_seed={eval_seed}, "
                        f"checkpoint={evaluator['checkpoint_path']}"
                    )
                    run = run_biased_rollout(
                        evaluator=evaluator,
                        eval_seed=eval_seed,
                        bias_delta=delta,
                        rollout_steps=args.rollout_steps,
                    )
                    runs_by_delta[delta].append(run)
    finally:
        stop_ippo_evaluators()
        if ray.is_initialized():
            ray.shutdown()

    aggregated_by_delta = {
        delta: aggregate_delta_runs(run_list) for delta, run_list in sorted(runs_by_delta.items())
    }
    selection = choose_recommended_delta(
        aggregated_by_delta=aggregated_by_delta,
        urllc_sla_floor=args.urllc_sla_floor,
        mmtc_sla_floor=args.mmtc_sla_floor,
        urllc_delay_p99_cap_ms=args.urllc_delay_p99_cap_ms,
        selection_tol=args.selection_tol,
    )

    plot_path = f"{args.output_prefix}.png"
    json_path = f"{args.output_prefix}.json"
    plot_bias_sweep(aggregated_by_delta, selection, plot_path)
    summary = make_json_summary(aggregated_by_delta, selection)
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print_summary(aggregated_by_delta, selection)
    print(f"Saved bias sweep plot to {plot_path}")
    print(f"Saved bias sweep summary to {json_path}")


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray

import compare_marl_baseline as base
from checkpoint_utils import rank_checkpoints_by_metric
from multi_cell_env import MultiCell_5G_SLA_Env

ENV_PROFILE = base.ENV_PROFILE
RESULT_PREFIX = f"./results/mappo_variant_comparison_{ENV_PROFILE}"

LEARNED_METHODS = [
    {
        "algo_key": "ippo",
        "label": "IPPO (Baseline, pure_local)",
        "experiment_env_tag": "balanced_ippo_v1",
        "observation_mode": "pure_local",
        "use_centralized_critic": False,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.0,
        "neighbor_dividend_gamma": 0.0,
        "cooperative_target": "all",
        "neighbor_augmented_include_ici_features": False,
    },
    {
        "algo_key": "mappo_current",
        "label": "MAPPO Current (CTDE v2)",
        "experiment_env_tag": "balanced_mappo_ctde_v2",
        "observation_mode": "neighbor_augmented",
        "use_centralized_critic": True,
        "cooperative_alpha": None,
        "neighbor_liability_beta": None,
        "neighbor_dividend_gamma": None,
        "cooperative_target": "all",
        "neighbor_augmented_include_ici_features": False,
    },
    {
        "algo_key": "mappo_bg",
        "label": "MAPPO beta/gamma",
        "experiment_env_tag": "balanced_mappo_ctde_v2_bg",
        "observation_mode": "neighbor_augmented",
        "use_centralized_critic": True,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.12,
        "neighbor_dividend_gamma": 0.8,
        "cooperative_target": "all",
        "neighbor_augmented_include_ici_features": False,
    },
    {
        "algo_key": "mappo_coop_embb",
        "label": "MAPPO eMBB-only Coop",
        "experiment_env_tag": "balanced_mappo_ctde_v2_coop_embb",
        "observation_mode": "neighbor_augmented",
        "use_centralized_critic": True,
        "cooperative_alpha": None,
        "neighbor_liability_beta": None,
        "neighbor_dividend_gamma": None,
        "cooperative_target": "embb_only",
        "neighbor_augmented_include_ici_features": False,
    },
    {
        "algo_key": "mappo_coop_embb_ici",
        "label": "MAPPO eMBB Coop + ICI",
        "experiment_env_tag": "balanced_mappo_ctde_v2_coop_embb_ici",
        "observation_mode": "neighbor_augmented",
        "use_centralized_critic": True,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.12,
        "neighbor_dividend_gamma": 0.8,
        "cooperative_target": "embb_only",
        "neighbor_augmented_include_ici_features": True,
    },
]

HEURISTIC_ALGOS = [
    ("static", "Static SLA Split"),
    ("priority", "Priority Heuristic"),
    ("max_weight", "Max-Weight (Throughput-Oriented Heuristic)"),
    ("pf", "Proportional Fair"),
]

LEARNED_COLOR_MAP = {
    "ippo": "#9467bd",
    "mappo_current": "#8c564b",
    "mappo_bg": "#e377c2",
    "mappo_coop_embb": "#17becf",
    "mappo_coop_embb_ici": "#bcbd22",
}


def configure_base_module():
    base.LEARNED_METHODS = LEARNED_METHODS
    base.LEARNED_METHOD_BY_KEY = {item["algo_key"]: item for item in LEARNED_METHODS}
    base.ALGORITHMS = HEURISTIC_ALGOS + [(item["algo_key"], item["label"]) for item in LEARNED_METHODS]
    base.LEARNED_ENV_CONFIGS = {item["algo_key"]: base._build_learned_env_config(item) for item in LEARNED_METHODS}
    base.LEARNED_EXPERIMENT_DIRS = {
        item["algo_key"]: [
            f"./ray_results/MAPPO_5G_Slicing_{item['experiment_env_tag']}_seed{seed}"
            for seed in base.TRAIN_SEEDS
        ]
        for item in LEARNED_METHODS
    }
    base.LEARNED_EVALUATORS = {item["algo_key"]: [] for item in LEARNED_METHODS}
    base.PLOT_COLORS.update(LEARNED_COLOR_MAP)


def plot_bar(ax, labels, means, stds, colors, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4.0, color=colors)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")


def save_grouped_sla_plot(path, comparison_algos, aggregated, key, title):
    fig, ax = plt.subplots(figsize=(13, 6))
    x_idx = np.arange(len(comparison_algos))
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width], dtype=np.float64)
    slice_names = ["eMBB", "URLLC", "mMTC"]
    slice_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx_slice, slice_name in enumerate(slice_names):
        vals = np.array([aggregated[k][f"{key}_mean"][idx_slice] for k, _ in comparison_algos], dtype=np.float64)
        stds = np.array([aggregated[k][f"{key}_std"][idx_slice] for k, _ in comparison_algos], dtype=np.float64)
        ax.bar(
            x_idx + offsets[idx_slice],
            vals,
            yerr=stds,
            capsize=3.0,
            width=bar_width,
            color=slice_colors[idx_slice],
            alpha=0.88,
            label=slice_name,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title(title)
    ax.set_xticks(x_idx, [label for _, label in comparison_algos], rotation=20)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary_json(path, comparison_algos, aggregated, selection_summary):
    summary = {"selection_summary": selection_summary, "aggregated": {}}
    for algo_key, algo_label in comparison_algos:
        stats = aggregated[algo_key]
        summary["aggregated"][algo_key] = {
            "label": algo_label,
            "num_runs": stats["num_runs"],
            "jfi_mean": stats["fairness_mean"],
            "jfi_std": stats["fairness_std"],
            "system_tp_mean_mbps": float(np.mean(stats["system_tp_mean"])),
            "system_tp_std_mbps": float(np.std(stats["system_tp_mean"])),
            "delay_mean_ms": float(np.mean(stats["delay_mean"])),
            "delay_std_ms": float(np.std(stats["delay_mean"])),
            "final_cum_reward_mean": float(stats["cum_reward_mean"][-1]),
            "final_cum_reward_std": float(stats["cum_reward_std"][-1]),
            "sla_sys_mean": stats["sla_sys_mean"].tolist(),
            "sla_sys_std": stats["sla_sys_std"].tolist(),
            "sla_bs0_mean": stats["sla_bs0_mean"].tolist(),
            "sla_bs0_std": stats["sla_bs0_std"].tolist(),
            "penalty_total_scalar": stats["penalty_total_scalar"],
            "penalty_embb_scalar": stats["penalty_embb_scalar"],
            "penalty_urllc_scalar": stats["penalty_urllc_scalar"],
            "penalty_mmtc_scalar": stats["penalty_mmtc_scalar"],
            "inference_total_ms_mean": stats["inference_total_ms_mean"],
            "inference_total_ms_std": stats["inference_total_ms_std"],
        }
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_eval_stats(aggregated_algo_stats):
    return {
        "system_total_sla_violations": float(np.sum(1.0 - aggregated_algo_stats["sla_sys_mean"])),
        "bs0_total_sla_violations": float(np.sum(1.0 - aggregated_algo_stats["sla_bs0_mean"])),
        "system_tp_mean_mbps": float(np.mean(aggregated_algo_stats["system_tp_mean"])),
        "delay_mean_ms": float(np.mean(aggregated_algo_stats["delay_mean"])),
        "final_cum_reward": float(aggregated_algo_stats["cum_reward_mean"][-1]),
        "sla_sys_mean": aggregated_algo_stats["sla_sys_mean"].tolist(),
        "sla_bs0_mean": aggregated_algo_stats["sla_bs0_mean"].tolist(),
    }


def candidate_eval_key(summary, training_iteration):
    return (
        summary["system_total_sla_violations"],
        -summary["system_tp_mean_mbps"],
        -summary["sla_sys_mean"][0],
        -summary["sla_bs0_mean"][0],
        -summary["final_cum_reward"],
        -training_iteration,
    )


def evaluate_learned_checkpoint(method, candidate):
    algo_key = method["algo_key"]
    env_config = base.LEARNED_ENV_CONFIGS[algo_key]
    observation_filter = base._resolve_trial_observation_filter(candidate["trial_dir"])
    algo = base.build_learned_eval_algo(observation_filter=observation_filter, env_config=env_config)
    algo.restore(candidate["checkpoint_path"])

    evaluator = {
        "algo_key": algo_key,
        "label": method["label"],
        "algo": algo,
        "env_config": dict(env_config),
        "train_seed": candidate.get("train_seed", base.TRAIN_SEEDS[0]),
        "checkpoint_path": candidate["checkpoint_path"],
        "training_iteration": candidate["training_iteration"],
        "observation_filter": observation_filter,
        "center_total_sla_violations": candidate.get("center_total_sla_violations"),
        "system_total_sla_violations": candidate.get("system_total_sla_violations"),
        "system_throughput_mbps": candidate.get("system_throughput_mbps"),
        "episode_return_mean": candidate.get("episode_return_mean"),
        "quality_score": candidate.get("quality_score"),
    }

    runs = []
    for seed in base.EVAL_SEEDS:
        env = MultiCell_5G_SLA_Env(config=evaluator["env_config"])
        runs.append(base.run_evaluation(env, algo_key, seed, learned_evaluator=evaluator))

    algo.stop()
    aggregated_algo = base.aggregate_results({algo_key: runs})[algo_key]
    summary = summarize_eval_stats(aggregated_algo)
    return evaluator, runs, summary


def sweep_learned_method(method):
    algo_key = method["algo_key"]
    candidates = rank_checkpoints_by_metric(
        base.LEARNED_EXPERIMENT_DIRS[algo_key],
        min_training_iteration=base.MIN_EVAL_CHECKPOINT_ITER,
        fallback_to_any=False,
    )
    if not candidates:
        raise RuntimeError(f"No checkpoints found for {method['label']}")

    best = None
    candidate_records = []
    print(f"\n--- Sweep {method['label']} ({len(candidates)} candidates) ---")
    for candidate in candidates:
        evaluator, runs, summary = evaluate_learned_checkpoint(method, candidate)
        record = {
            "checkpoint_path": candidate["checkpoint_path"],
            "training_iteration": candidate["training_iteration"],
            "observation_filter": evaluator["observation_filter"],
            "center_total_sla_violations": candidate.get("center_total_sla_violations"),
            "system_total_sla_violations_log": candidate.get("system_total_sla_violations"),
            "system_throughput_mbps_log": candidate.get("system_throughput_mbps"),
            "quality_score": candidate.get("quality_score"),
            **summary,
        }
        candidate_records.append(record)
        print(
            f"iter={candidate['training_iteration']}, checkpoint={candidate['checkpoint_path']}, "
            f"eval_total_viol={summary['system_total_sla_violations']:.4f}, "
            f"eval_sys_tp={summary['system_tp_mean_mbps']:.2f} Mbps, "
            f"eval_sla_sys={summary['sla_sys_mean'][0]*100:.1f}%/"
            f"{summary['sla_sys_mean'][1]*100:.1f}%/{summary['sla_sys_mean'][2]*100:.1f}%"
        )
        if best is None or candidate_eval_key(summary, candidate["training_iteration"]) < best["rank_key"]:
            best = {
                "method": method,
                "evaluator": evaluator,
                "runs": runs,
                "summary": summary,
                "rank_key": candidate_eval_key(summary, candidate["training_iteration"]),
                "candidate_records": candidate_records,
            }

    assert best is not None
    best["candidate_records"] = candidate_records
    return best


def run_variant_comparison():
    configure_base_module()
    base.validate_seed_split()
    os.makedirs("./results", exist_ok=True)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    results_by_algo = {key: [] for key, _ in base.ALGORITHMS}
    comparison_algos = list(base.ALGORITHMS)
    selection_summary = {"selection_mode": "eval_sweep_system_total_sla"}

    for seed in base.EVAL_SEEDS:
        print(f"\n=== Heuristic evaluation seed={seed} ===")
        for algo_key, algo_label in HEURISTIC_ALGOS:
            env = MultiCell_5G_SLA_Env(config=base.HEURISTIC_ENV_CONFIG)
            run = base.run_evaluation(env, algo_key, seed)
            results_by_algo[algo_key].append(run)
            print(
                f"[{algo_label}] JFI={run['jfi_embb']:.4f}, "
                f"sys_tp_mean={np.mean(run['system_throughput_mbps']):.2f} Mbps, "
                f"mean_delay_ms={np.mean(run['urllc_delay_ms']):.3f}, "
                f"cum_reward={run['cum_reward'][-1]:.3f}"
            )

    for method in LEARNED_METHODS:
        best = sweep_learned_method(method)
        algo_key = method["algo_key"]
        results_by_algo[algo_key] = best["runs"]
        selection_summary[algo_key] = {
            "label": method["label"],
            "selected_checkpoint": best["evaluator"]["checkpoint_path"],
            "selected_iteration": best["evaluator"]["training_iteration"],
            "selected_observation_filter": best["evaluator"]["observation_filter"],
            "selected_eval_summary": best["summary"],
            "candidate_records": best["candidate_records"],
        }
        print(
            f"Selected {method['label']}: iter={best['evaluator']['training_iteration']}, "
            f"checkpoint={best['evaluator']['checkpoint_path']}, "
            f"eval_total_viol={best['summary']['system_total_sla_violations']:.4f}, "
            f"eval_sys_tp={best['summary']['system_tp_mean_mbps']:.2f} Mbps"
        )

    aggregated = base.aggregate_results(results_by_algo)

    print("\n=== Variant Summary (mean ± std over recorded runs) ===")
    for algo_key, algo_label in comparison_algos:
        stats = aggregated[algo_key]
        print(
            f"{algo_label:>28}: runs={stats['num_runs']}, "
            f"JFI={stats['fairness_mean']:.4f} ± {stats['fairness_std']:.4f}, "
            f"SysTP={np.mean(stats['system_tp_mean']):.2f} Mbps, "
            f"Delay={np.mean(stats['delay_mean']):.3f} ms, "
            f"FinalCumReward={stats['cum_reward_mean'][-1]:.3f}"
        )
        print(
            " " * 30
            + "SLA_sys[eMBB/URLLC/mMTC]="
            + f"{stats['sla_sys_mean'][0]*100:.1f}%/"
            + f"{stats['sla_sys_mean'][1]*100:.1f}%/"
            + f"{stats['sla_sys_mean'][2]*100:.1f}%"
        )
        print(
            " " * 30
            + "SLA_bs0[eMBB/URLLC/mMTC]="
            + f"{stats['sla_bs0_mean'][0]*100:.1f}%/"
            + f"{stats['sla_bs0_mean'][1]*100:.1f}%/"
            + f"{stats['sla_bs0_mean'][2]*100:.1f}%"
        )

    labels = [label for _, label in comparison_algos]
    colors = [base.PLOT_COLORS[key] for key, _ in comparison_algos]

    fig1, ax1 = plt.subplots(figsize=(13, 6))
    plot_bar(
        ax1,
        labels,
        [float(np.mean(aggregated[key]["system_tp_mean"])) for key, _ in comparison_algos],
        [float(np.std(aggregated[key]["system_tp_mean"])) for key, _ in comparison_algos],
        colors,
        "System Throughput Comparison",
        "Mbps",
    )
    fig1.tight_layout()
    throughput_path = f"{RESULT_PREFIX}_system_throughput.png"
    fig1.savefig(throughput_path)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    plot_bar(
        ax2,
        labels,
        [float(aggregated[key]["cum_reward_mean"][-1]) for key, _ in comparison_algos],
        [float(aggregated[key]["cum_reward_std"][-1]) for key, _ in comparison_algos],
        colors,
        "Final Cumulative Reward Comparison (BS_0)",
        "Reward",
    )
    fig2.tight_layout()
    reward_path = f"{RESULT_PREFIX}_final_cum_reward.png"
    fig2.savefig(reward_path)
    plt.close(fig2)

    sla_sys_path = f"{RESULT_PREFIX}_sla_system.png"
    save_grouped_sla_plot(sla_sys_path, comparison_algos, aggregated, "sla_sys", "System-Level SLA Success")

    sla_bs0_path = f"{RESULT_PREFIX}_sla_bs0.png"
    save_grouped_sla_plot(sla_bs0_path, comparison_algos, aggregated, "sla_bs0", "BS_0 SLA Success")

    summary_json_path = f"{RESULT_PREFIX}_summary.json"
    save_summary_json(summary_json_path, comparison_algos, aggregated, selection_summary)

    print("Saved variant comparison outputs:")
    print(f"  - {throughput_path}")
    print(f"  - {reward_path}")
    print(f"  - {sla_sys_path}")
    print(f"  - {sla_bs0_path}")
    print(f"  - {summary_json_path}")

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    run_variant_comparison()

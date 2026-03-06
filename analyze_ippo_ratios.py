import argparse
import json
import os

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
    stop_ippo_evaluators,
    validate_seed_split,
)
from multi_cell_env import MultiCell_5G_SLA_Env

SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
SLICE_COLORS = {
    "eMBB": "#1f77b4",
    "URLLC": "#ff7f0e",
    "mMTC": "#2ca02c",
}
CENTER_AGENT_ID = "BS_0"
AGENT_IDS = tuple(f"BS_{idx}" for idx in range(7))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze how IPPO allocates slice ratios over time and across cells."
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
        help="Optional evaluation seed list. Defaults to compare_marl_baseline.py EVAL_SEEDS.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./results/ippo_ratio_analysis",
        help="Output prefix for generated figure and summary JSON.",
    )
    return parser.parse_args()


def action_to_ratios(env: MultiCell_5G_SLA_Env, action: np.ndarray) -> np.ndarray:
    return env._action_to_ratios(action).astype(np.float32)


def _stack_timeseries(run_list, key: str, rollout_steps: int) -> np.ndarray:
    data = np.full((len(run_list), rollout_steps, 3), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = np.asarray(run[key], dtype=np.float32)
        valid_len = min(len(values), rollout_steps)
        data[idx, :valid_len, :] = values[:valid_len]
    return data


def _stack_agent_timeseries(run_list, key: str, rollout_steps: int) -> np.ndarray:
    data = np.full((len(run_list), rollout_steps, len(AGENT_IDS), 3), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = np.asarray(run[key], dtype=np.float32)
        valid_len = min(values.shape[0], rollout_steps)
        data[idx, :valid_len, :, :] = values[:valid_len, :, :]
    return data


def _stack_scalar_timeseries(run_list, key: str, rollout_steps: int, width: int) -> np.ndarray:
    data = np.full((len(run_list), rollout_steps, width), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = np.asarray(run[key], dtype=np.float32)
        valid_len = min(values.shape[0], rollout_steps)
        data[idx, :valid_len, :] = values[:valid_len, :]
    return data


def _dominant_share(records: np.ndarray) -> dict:
    max_values = np.max(records, axis=1, keepdims=True)
    dominant_mask = np.isclose(records, max_values, rtol=1e-6, atol=1e-6)
    dominant_weights = dominant_mask.astype(np.float32)
    dominant_weights /= np.sum(dominant_weights, axis=1, keepdims=True)
    mean_share = np.mean(dominant_weights, axis=0)
    return {
        slice_name: float(mean_share[slice_idx])
        for slice_idx, slice_name in enumerate(SLICE_NAMES)
    }


def _mean_ratio(records: np.ndarray) -> dict:
    mean_values = np.mean(records, axis=0)
    return {
        slice_name: float(mean_values[slice_idx])
        for slice_idx, slice_name in enumerate(SLICE_NAMES)
    }


def run_ratio_rollout(evaluator: dict, eval_seed: int, rollout_steps: int) -> dict:
    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    obs, reset_infos = env.reset(seed=eval_seed)
    runner, episode, shared_data = build_ippo_episode_context(
        evaluator["algo"], obs, reset_infos
    )

    system_ratio_history = []
    center_ratio_history = []
    edge_ratio_history = []
    agent_ratio_history = []
    edge_to_center_abs_gap_history = []
    agent_to_center_l1_history = []
    all_ratio_records = []
    center_ratio_records = []
    edge_ratio_records = []

    done = {"__all__": False}
    step = 0

    while not done["__all__"] and step < rollout_steps:
        policy_actions, env_actions, extra_model_outputs = compute_actions_batched(
            evaluator["algo"],
            runner,
            episode,
            shared_data,
        )

        ratio_dict = {
            agent_id: action_to_ratios(env, agent_action)
            for agent_id, agent_action in env_actions.items()
        }
        ratio_values = np.asarray(list(ratio_dict.values()), dtype=np.float32)
        edge_values = np.asarray(
            [
                ratio_dict[agent_id]
                for agent_id in env.agents
                if agent_id != CENTER_AGENT_ID
            ],
            dtype=np.float32,
        )

        system_ratio_history.append(np.mean(ratio_values, axis=0))
        center_ratio_history.append(ratio_dict[CENTER_AGENT_ID])
        edge_ratio_history.append(np.mean(edge_values, axis=0))
        ordered_agent_ratios = np.asarray([ratio_dict[agent_id] for agent_id in AGENT_IDS], dtype=np.float32)
        center_ratio = ratio_dict[CENTER_AGENT_ID]
        edge_to_center_abs_gap_history.append(np.mean(np.abs(edge_values - center_ratio), axis=0))
        agent_to_center_l1_history.append(
            np.asarray(
                [
                    np.sum(np.abs(ratio_dict[agent_id] - center_ratio))
                    for agent_id in AGENT_IDS
                ],
                dtype=np.float32,
            )
        )
        agent_ratio_history.append(ordered_agent_ratios)
        all_ratio_records.append(ratio_values)
        center_ratio_records.append(ratio_dict[CENTER_AGENT_ID][None, :])
        edge_ratio_records.append(edge_values)

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

        done = terminateds
        step += 1

    all_records = np.concatenate(all_ratio_records, axis=0).astype(np.float32)
    center_records = np.concatenate(center_ratio_records, axis=0).astype(np.float32)
    edge_records = np.concatenate(edge_ratio_records, axis=0).astype(np.float32)

    return {
        "train_seed": int(evaluator["train_seed"]),
        "eval_seed": int(eval_seed),
        "steps": int(step),
        "checkpoint_path": str(evaluator["checkpoint_path"]),
        "quality_score": (
            None
            if evaluator.get("quality_score") is None
            else float(evaluator["quality_score"])
        ),
        "system_ratio_history": np.asarray(system_ratio_history, dtype=np.float32),
        "center_ratio_history": np.asarray(center_ratio_history, dtype=np.float32),
        "edge_ratio_history": np.asarray(edge_ratio_history, dtype=np.float32),
        "agent_ratio_history": np.asarray(agent_ratio_history, dtype=np.float32),
        "edge_to_center_abs_gap_history": np.asarray(edge_to_center_abs_gap_history, dtype=np.float32),
        "agent_to_center_l1_history": np.asarray(agent_to_center_l1_history, dtype=np.float32),
        "all_ratio_records": all_records,
        "center_ratio_records": center_records,
        "edge_ratio_records": edge_records,
    }


def aggregate_ratio_runs(run_list: list[dict], rollout_steps: int) -> dict:
    system_hist = _stack_timeseries(run_list, "system_ratio_history", rollout_steps)
    center_hist = _stack_timeseries(run_list, "center_ratio_history", rollout_steps)
    edge_hist = _stack_timeseries(run_list, "edge_ratio_history", rollout_steps)
    agent_hist = _stack_agent_timeseries(run_list, "agent_ratio_history", rollout_steps)
    edge_gap_hist = _stack_timeseries(run_list, "edge_to_center_abs_gap_history", rollout_steps)
    center_l1_hist = _stack_scalar_timeseries(
        run_list,
        "agent_to_center_l1_history",
        rollout_steps,
        width=len(AGENT_IDS),
    )

    all_records = np.concatenate(
        [run["all_ratio_records"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    center_records = np.concatenate(
        [run["center_ratio_records"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    edge_records = np.concatenate(
        [run["edge_ratio_records"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    agent_records = np.concatenate(
        [run["agent_ratio_history"] for run in run_list],
        axis=0,
    ).astype(np.float32)
    center_l1_records = np.concatenate(
        [run["agent_to_center_l1_history"] for run in run_list],
        axis=0,
    ).astype(np.float32)

    return {
        "system_ratio_mean": np.nanmean(system_hist, axis=0),
        "system_ratio_std": np.nanstd(system_hist, axis=0),
        "center_ratio_mean": np.nanmean(center_hist, axis=0),
        "center_ratio_std": np.nanstd(center_hist, axis=0),
        "edge_ratio_mean": np.nanmean(edge_hist, axis=0),
        "edge_ratio_std": np.nanstd(edge_hist, axis=0),
        "agent_ratio_mean": np.nanmean(agent_hist, axis=0),
        "agent_ratio_std": np.nanstd(agent_hist, axis=0),
        "edge_to_center_abs_gap_mean": np.nanmean(edge_gap_hist, axis=0),
        "edge_to_center_abs_gap_std": np.nanstd(edge_gap_hist, axis=0),
        "agent_to_center_l1_mean_ts": np.nanmean(center_l1_hist, axis=0),
        "agent_to_center_l1_std_ts": np.nanstd(center_l1_hist, axis=0),
        "overall_mean_ratio": _mean_ratio(all_records),
        "center_mean_ratio": _mean_ratio(center_records),
        "edge_mean_ratio": _mean_ratio(edge_records),
        "agent_mean_ratio": np.nanmean(agent_records, axis=0),
        "agent_to_center_l1_mean": np.nanmean(center_l1_records, axis=0),
        "overall_dominant_share": _dominant_share(all_records),
        "center_dominant_share": _dominant_share(center_records),
        "edge_dominant_share": _dominant_share(edge_records),
    }


def plot_ratio_analysis(aggregated: dict, output_path: str):
    x_axis = np.arange(aggregated["center_ratio_mean"].shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_bs0 = axes[0, 0]
    center_mean = aggregated["center_ratio_mean"]
    stack_values = [center_mean[:, slice_idx] for slice_idx in range(len(SLICE_NAMES))]
    ax_bs0.stackplot(
        x_axis,
        *stack_values,
        labels=SLICE_NAMES,
        colors=[SLICE_COLORS[slice_name] for slice_name in SLICE_NAMES],
        alpha=0.8,
    )
    ax_bs0.set_title("BS_0 Dynamic Slice Allocation")
    ax_bs0.set_xlabel("Time Step (TTI)")
    ax_bs0.set_ylabel("Allocation Ratio")
    ax_bs0.set_ylim(0.0, 1.0)
    ax_bs0.grid(True)
    ax_bs0.legend()

    ax_gap = axes[0, 1]
    gap_mean = aggregated["edge_to_center_abs_gap_mean"]
    gap_std = aggregated["edge_to_center_abs_gap_std"]
    for slice_idx, slice_name in enumerate(SLICE_NAMES):
        color = SLICE_COLORS[slice_name]
        ax_gap.plot(
            x_axis,
            gap_mean[:, slice_idx],
            color=color,
            linewidth=2.0,
            label=slice_name,
        )
        ax_gap.fill_between(
            x_axis,
            gap_mean[:, slice_idx] - gap_std[:, slice_idx],
            gap_mean[:, slice_idx] + gap_std[:, slice_idx],
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
    ax_gap.set_title("Mean |Edge - BS_0| Ratio Gap")
    ax_gap.set_xlabel("Time Step (TTI)")
    ax_gap.set_ylabel("Absolute Ratio Gap")
    ax_gap.set_ylim(0.0, 1.0)
    ax_gap.grid(True)
    ax_gap.legend()

    ax_heatmap = axes[1, 0]
    heatmap = aggregated["agent_mean_ratio"].astype(np.float32)
    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax_heatmap.set_title("Per-Agent Mean Slice Ratio")
    ax_heatmap.set_xticks(np.arange(len(SLICE_NAMES)), SLICE_NAMES)
    ax_heatmap.set_yticks(np.arange(len(AGENT_IDS)), AGENT_IDS)
    for agent_idx in range(len(AGENT_IDS)):
        for slice_idx in range(len(SLICE_NAMES)):
            ax_heatmap.text(
                slice_idx,
                agent_idx,
                f"{heatmap[agent_idx, slice_idx]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    ax_l1 = axes[1, 1]
    agent_l1 = aggregated["agent_to_center_l1_mean"]
    bars = ax_l1.bar(AGENT_IDS, agent_l1, color="#8c564b")
    ax_l1.set_title("Mean L1 Distance to BS_0 Ratio")
    ax_l1.set_ylabel("L1 Distance")
    ax_l1.grid(True, axis="y")
    ax_l1.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, agent_l1):
        ax_l1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_bs0_stackplot(aggregated: dict, output_path: str, num_runs: int):
    x_axis = np.arange(aggregated["center_ratio_mean"].shape[0])
    center_mean = aggregated["center_ratio_mean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        x_axis,
        *[center_mean[:, slice_idx] for slice_idx in range(len(SLICE_NAMES))],
        labels=SLICE_NAMES,
        colors=[SLICE_COLORS[slice_name] for slice_name in SLICE_NAMES],
        alpha=0.8,
    )
    title = "Dynamic Bandwidth Allocation (BS_0)"
    if num_runs > 1:
        title += " [Mean Over Runs]"
    ax.set_title(title)
    ax.set_xlabel("Time Step (TTI)")
    ax.set_ylabel("Bandwidth Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def build_summary(aggregated: dict, evaluators: list[dict], run_list: list[dict], eval_seeds: list[int], rollout_steps: int) -> dict:
    return {
        "rollout_steps": int(rollout_steps),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "num_ippo_checkpoints": len(evaluators),
        "num_runs": len(run_list),
        "evaluators": [
            {
                "train_seed": int(item["train_seed"]),
                "checkpoint_path": str(item["checkpoint_path"]),
                "training_iteration": int(item["training_iteration"]),
                "observation_filter": str(item["observation_filter"]),
                "quality_score": (
                    None
                    if item.get("quality_score") is None
                    else float(item["quality_score"])
                ),
            }
            for item in evaluators
        ],
        "overall_mean_ratio": aggregated["overall_mean_ratio"],
        "center_mean_ratio": aggregated["center_mean_ratio"],
        "edge_mean_ratio": aggregated["edge_mean_ratio"],
        "agent_mean_ratio": {
            agent_id: {
                slice_name: float(aggregated["agent_mean_ratio"][agent_idx, slice_idx])
                for slice_idx, slice_name in enumerate(SLICE_NAMES)
            }
            for agent_idx, agent_id in enumerate(AGENT_IDS)
        },
        "agent_to_center_l1_mean": {
            agent_id: float(aggregated["agent_to_center_l1_mean"][agent_idx])
            for agent_idx, agent_id in enumerate(AGENT_IDS)
        },
        "edge_to_center_abs_gap_mean": {
            slice_name: float(aggregated["edge_to_center_abs_gap_mean"][:, slice_idx].mean())
            for slice_idx, slice_name in enumerate(SLICE_NAMES)
        },
        "overall_dominant_share": aggregated["overall_dominant_share"],
        "center_dominant_share": aggregated["center_dominant_share"],
        "edge_dominant_share": aggregated["edge_dominant_share"],
        "per_run": [
            {
                "train_seed": int(run["train_seed"]),
                "eval_seed": int(run["eval_seed"]),
                "steps": int(run["steps"]),
                "quality_score": run["quality_score"],
                "system_mean_ratio": _mean_ratio(run["all_ratio_records"]),
                "center_mean_ratio": _mean_ratio(run["center_ratio_records"]),
                "edge_mean_ratio": _mean_ratio(run["edge_ratio_records"]),
            }
            for run in run_list
        ],
    }


def print_summary(summary: dict):
    print("\n=== IPPO Ratio Allocation Summary ===")
    print(
        f"Checkpoints={summary['num_ippo_checkpoints']}, "
        f"runs={summary['num_runs']}, "
        f"eval_seeds={summary['eval_seeds']}"
    )

    for scope_key, scope_label in (
        ("overall_mean_ratio", "System mean ratio"),
        ("center_mean_ratio", "Center BS_0 mean ratio"),
        ("edge_mean_ratio", "Edge mean ratio"),
    ):
        ratios = summary[scope_key]
        print(
            f"{scope_label}: "
            f"eMBB={ratios['eMBB']:.3f}, "
            f"URLLC={ratios['URLLC']:.3f}, "
            f"mMTC={ratios['mMTC']:.3f}"
        )

    for scope_key, scope_label in (
        ("overall_dominant_share", "System dominant-slice share"),
        ("center_dominant_share", "Center dominant-slice share"),
        ("edge_dominant_share", "Edge dominant-slice share"),
    ):
        dominant = summary[scope_key]
        print(
            f"{scope_label}: "
            f"eMBB={dominant['eMBB']*100:.1f}%, "
            f"URLLC={dominant['URLLC']*100:.1f}%, "
            f"mMTC={dominant['mMTC']*100:.1f}%"
        )

    gap = summary["edge_to_center_abs_gap_mean"]
    print(
        "Mean |Edge - BS_0| ratio gap: "
        f"eMBB={gap['eMBB']:.4f}, "
        f"URLLC={gap['URLLC']:.4f}, "
        f"mMTC={gap['mMTC']:.4f}"
    )
    print("Per-agent mean L1 distance to BS_0 ratio:")
    for agent_id in AGENT_IDS:
        print(f"  {agent_id}: {summary['agent_to_center_l1_mean'][agent_id]:.4f}")


def main():
    args = parse_args()
    validate_seed_split()
    eval_seeds = args.eval_seeds if args.eval_seeds is not None else list(EVAL_SEEDS)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    try:
        print("Initializing IPPO checkpoints using current compare_marl_baseline selection logic...")
        evaluators = init_ippo_evaluators()
        run_list = []
        for evaluator in evaluators:
            for eval_seed in eval_seeds:
                print(
                    f"Running ratio analysis: train_seed={evaluator['train_seed']}, "
                    f"eval_seed={eval_seed}, iter={evaluator['training_iteration']}, "
                    f"quality={evaluator.get('quality_score')}"
                )
                run_list.append(
                    run_ratio_rollout(
                        evaluator=evaluator,
                        eval_seed=eval_seed,
                        rollout_steps=args.rollout_steps,
                    )
                )

        aggregated = aggregate_ratio_runs(run_list, rollout_steps=args.rollout_steps)
        summary = build_summary(
            aggregated=aggregated,
            evaluators=evaluators,
            run_list=run_list,
            eval_seeds=eval_seeds,
            rollout_steps=args.rollout_steps,
        )

        figure_path = f"{args.output_prefix}.png"
        stackplot_path = f"{args.output_prefix}_bs0_stack.png"
        summary_path = f"{args.output_prefix}.json"
        plot_ratio_analysis(aggregated, output_path=figure_path)
        plot_bs0_stackplot(aggregated, output_path=stackplot_path, num_runs=len(run_list))
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)

        print_summary(summary)
        print(f"Saved ratio plot to {figure_path}")
        print(f"Saved BS_0 stackplot to {stackplot_path}")
        print(f"Saved ratio summary to {summary_path}")
    finally:
        stop_ippo_evaluators()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

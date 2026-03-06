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

CENTER_AGENT_ID = "BS_0"
SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
SLICE_COLORS = {
    "eMBB": "#1f77b4",
    "URLLC": "#ff7f0e",
    "mMTC": "#2ca02c",
}
SLACK_BIN_EDGES_MS = np.asarray([-np.inf, 0.0, 0.5, 1.0, 1.5, 1.8, 1.95, np.inf], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze URLLC slack versus eMBB shortfall under the selected IPPO checkpoints."
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
        "--candidate-shifts",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.10],
        help="Absolute ratio shifts from URLLC to eMBB for immediate one-step counterfactual checks.",
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=0.20,
        help="Maximum absolute ratio shift searched for the per-step one-step safe shift.",
    )
    parser.add_argument(
        "--shift-grid-step",
        type=float,
        default=0.005,
        help="Grid step for per-step max safe shift search.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./results/ippo_slack_tradeoff_analysis",
        help="Output prefix for the generated figure and JSON summary.",
    )
    return parser.parse_args()


def action_to_ratios(env: MultiCell_5G_SLA_Env, action: np.ndarray) -> np.ndarray:
    return env._action_to_ratios(action).astype(np.float32)


def snapshot_env_state(env: MultiCell_5G_SLA_Env) -> dict:
    return {
        "queues": {agent: env.queues[agent].copy() for agent in env.agents},
        "current_se": {agent: env.current_se[agent].copy() for agent in env.agents},
        "arrival_rates_mbps": {
            agent: np.asarray(env.state[agent][0:3], dtype=np.float32).copy() for agent in env.agents
        },
        "embb_tp_history": {
            agent: np.asarray(list(env.embb_tp_history[agent]), dtype=np.float32).copy() for agent in env.agents
        },
    }


def simulate_one_step(env: MultiCell_5G_SLA_Env, snapshot: dict, ratios_dict: dict[str, np.ndarray]) -> dict:
    se_modifiers = env._calculate_interference_and_sinr(ratios_dict)
    agent_metrics = {}
    system_tp_mbps = 0.0

    for agent in env.agents:
        ratios = np.asarray(ratios_dict[agent], dtype=np.float32)
        bw_allocated = ratios * env.total_bandwidth
        effective_se = snapshot["current_se"][agent] * se_modifiers[agent]
        service_rate_mbps = (bw_allocated * effective_se) / 1e6
        service_capacity_mb = service_rate_mbps * env.duration_tti

        arrivals_mb = snapshot["arrival_rates_mbps"][agent] * env.duration_tti
        queue_after_arrivals = snapshot["queues"][agent] + arrivals_mb
        served_mb = np.minimum(service_capacity_mb, queue_after_arrivals)
        queue_after_service = queue_after_arrivals - served_mb
        achieved_throughput_mbps = served_mb / env.duration_tti

        embb_history = snapshot["embb_tp_history"][agent].astype(np.float32).tolist()
        if len(embb_history) >= env.embb_sla_window_tti:
            embb_history = embb_history[-(env.embb_sla_window_tti - 1) :]
        embb_history.append(float(achieved_throughput_mbps[0]))
        embb_eval_tp_mbps = float(np.mean(embb_history))
        embb_shortfall_mbps = max(0.0, env.sla_props["embb_gbr"] - embb_eval_tp_mbps)

        safe_service_rate_mbps = max(float(service_rate_mbps[1]), 0.1)
        est_urllc_delay_s = float(queue_after_service[1] / safe_service_rate_mbps)
        violations_raw = np.zeros(3, dtype=np.float32)
        violations_raw[0] = embb_shortfall_mbps / max(env.sla_props["embb_gbr"], 1e-6)
        delay_excess_s = max(0.0, est_urllc_delay_s - env.sla_props["urllc_max_delay"])
        violations_raw[1] = delay_excess_s / max(env.sla_props["urllc_max_delay"], 1e-6)
        queue_excess = max(0.0, float(queue_after_service[2]) - env.sla_props["mmtc_max_queue"])
        violations_raw[2] = queue_excess / max(env.sla_props["mmtc_max_queue"], 1e-6)

        throughput_total_mbps = float(np.sum(achieved_throughput_mbps))
        system_tp_mbps += throughput_total_mbps
        agent_metrics[agent] = {
            "ratios": ratios.copy(),
            "throughput_slices_mbps": achieved_throughput_mbps.astype(np.float32),
            "throughput_mbps": throughput_total_mbps,
            "queue_after_service": queue_after_service.astype(np.float32),
            "embb_eval_tp_mbps": embb_eval_tp_mbps,
            "embb_shortfall_mbps": embb_shortfall_mbps,
            "est_urllc_delay_ms": est_urllc_delay_s * 1000.0,
            "violations_raw": violations_raw,
        }

    return {
        "agents": agent_metrics,
        "system_throughput_mbps": float(system_tp_mbps),
    }


def shift_center_ratio_to_embb(ratios_dict: dict[str, np.ndarray], shift_ratio: float) -> dict[str, np.ndarray]:
    shifted = {agent: ratios.copy() for agent, ratios in ratios_dict.items()}
    center_ratio = shifted[CENTER_AGENT_ID].copy()
    shift_ratio = float(max(shift_ratio, 0.0))
    shift_ratio = min(shift_ratio, float(center_ratio[1]))
    center_ratio[0] += shift_ratio
    center_ratio[1] -= shift_ratio
    shifted[CENTER_AGENT_ID] = center_ratio.astype(np.float32)
    return shifted


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


def stack_scalar_history(run_list: list[dict], key: str, rollout_steps: int) -> np.ndarray:
    data = np.full((len(run_list), rollout_steps), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = np.asarray(run[key], dtype=np.float32)
        valid_len = min(values.shape[0], rollout_steps)
        data[idx, :valid_len] = values[:valid_len]
    return data


def stack_vector_history(run_list: list[dict], key: str, rollout_steps: int, width: int) -> np.ndarray:
    data = np.full((len(run_list), rollout_steps, width), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = np.asarray(run[key], dtype=np.float32)
        valid_len = min(values.shape[0], rollout_steps)
        data[idx, :valid_len, :] = values[:valid_len, :]
    return data


def compute_slack_bin_summary(records: dict, candidate_shifts: list[float]) -> list[dict]:
    slack_ms = np.asarray(records["center_slack_ms"], dtype=np.float64)
    embb_shortfall = np.asarray(records["center_embb_shortfall_mbps"], dtype=np.float64)
    center_ratios = np.asarray(records["center_ratios"], dtype=np.float64)
    max_safe_shift = np.asarray(records["max_safe_shift_ratio"], dtype=np.float64)
    safe_center_gain = np.asarray(records["max_safe_center_embb_gain_mbps"], dtype=np.float64)
    safe_system_gain = np.asarray(records["max_safe_system_tp_gain_mbps"], dtype=np.float64)
    candidate_safe = np.asarray(records["candidate_safe_flags"], dtype=np.float64)

    summaries = []
    for left_edge, right_edge in zip(SLACK_BIN_EDGES_MS[:-1], SLACK_BIN_EDGES_MS[1:]):
        mask = (slack_ms >= left_edge) & (slack_ms < right_edge)
        count = int(np.sum(mask))
        label_left = "-inf" if not np.isfinite(left_edge) else f"{left_edge:.2f}"
        label_right = "inf" if not np.isfinite(right_edge) else f"{right_edge:.2f}"
        summary = {
            "label": f"[{label_left}, {label_right}) ms",
            "count": count,
            "fraction": float(count / max(slack_ms.size, 1)),
        }
        if count > 0:
            ratio_mean = np.mean(center_ratios[mask], axis=0)
            summary["mean_embb_shortfall_mbps"] = float(np.mean(embb_shortfall[mask]))
            summary["mean_max_safe_shift_ratio"] = float(np.mean(max_safe_shift[mask]))
            summary["mean_safe_center_embb_gain_mbps"] = float(np.mean(safe_center_gain[mask]))
            summary["mean_safe_system_tp_gain_mbps"] = float(np.mean(safe_system_gain[mask]))
            summary["mean_center_ratio"] = {
                slice_name: float(ratio_mean[idx]) for idx, slice_name in enumerate(SLICE_NAMES)
            }
            summary["candidate_safe_rate"] = {
                f"{shift:.3f}": float(np.mean(candidate_safe[mask, shift_idx]))
                for shift_idx, shift in enumerate(candidate_shifts)
            }
        else:
            summary["mean_embb_shortfall_mbps"] = None
            summary["mean_max_safe_shift_ratio"] = None
            summary["mean_safe_center_embb_gain_mbps"] = None
            summary["mean_safe_system_tp_gain_mbps"] = None
            summary["mean_center_ratio"] = {slice_name: None for slice_name in SLICE_NAMES}
            summary["candidate_safe_rate"] = {f"{shift:.3f}": None for shift in candidate_shifts}
        summaries.append(summary)
    return summaries


def run_tradeoff_rollout(
    evaluator: dict,
    eval_seed: int,
    rollout_steps: int,
    candidate_shifts: list[float],
    max_shift: float,
    shift_grid_step: float,
) -> dict:
    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    obs, reset_infos = env.reset(seed=eval_seed)
    runner, episode, shared_data = build_ippo_episode_context(evaluator["algo"], obs, reset_infos)

    center_ratio_history = []
    center_delay_ms_history = []
    center_slack_ms_history = []
    center_embb_shortfall_mbps_history = []
    center_embb_eval_tp_mbps_history = []
    center_embb_tp_mbps_history = []
    system_tp_mbps_history = []
    max_safe_shift_ratio_history = []
    max_safe_center_embb_gain_mbps_history = []
    max_safe_system_tp_gain_mbps_history = []
    candidate_safe_flags_history = []
    candidate_center_embb_gain_history = []
    candidate_system_tp_gain_history = []
    validation_center_delay_error_ms = []
    validation_center_embb_shortfall_error_mbps = []
    validation_system_tp_error_mbps = []

    done = {"__all__": False}
    step = 0
    urllc_sla_ms = float(env.sla_props["urllc_max_delay"] * 1000.0)

    while not done["__all__"] and step < rollout_steps:
        snapshot = snapshot_env_state(env)
        policy_actions, env_actions, extra_model_outputs = compute_actions_batched(
            evaluator["algo"],
            runner,
            episode,
            shared_data,
        )
        ratio_dict = {
            agent_id: action_to_ratios(env, agent_action) for agent_id, agent_action in env_actions.items()
        }

        baseline_prediction = simulate_one_step(env, snapshot, ratio_dict)
        center_ratio = ratio_dict[CENTER_AGENT_ID].copy()
        center_base = baseline_prediction["agents"][CENTER_AGENT_ID]

        candidate_safe_flags = []
        candidate_center_embb_gains = []
        candidate_system_tp_gains = []
        for shift_ratio in candidate_shifts:
            if shift_ratio > float(center_ratio[1]) + 1e-9:
                candidate_safe_flags.append(0.0)
                candidate_center_embb_gains.append(np.nan)
                candidate_system_tp_gains.append(np.nan)
                continue

            shifted_ratios = shift_center_ratio_to_embb(ratio_dict, shift_ratio)
            shifted_prediction = simulate_one_step(env, snapshot, shifted_ratios)
            shifted_center = shifted_prediction["agents"][CENTER_AGENT_ID]
            safe = float(shifted_center["est_urllc_delay_ms"] <= urllc_sla_ms + 1e-9)
            candidate_safe_flags.append(safe)
            candidate_center_embb_gains.append(
                shifted_center["throughput_slices_mbps"][0] - center_base["throughput_slices_mbps"][0]
            )
            candidate_system_tp_gains.append(
                shifted_prediction["system_throughput_mbps"] - baseline_prediction["system_throughput_mbps"]
            )

        max_safe_shift = 0.0
        max_safe_center_embb_gain = 0.0
        max_safe_system_tp_gain = 0.0
        max_feasible_shift = min(float(center_ratio[1]), max_shift)
        search_grid = np.arange(shift_grid_step, max_feasible_shift + 0.5 * shift_grid_step, shift_grid_step)
        for shift_ratio in search_grid:
            shifted_ratios = shift_center_ratio_to_embb(ratio_dict, shift_ratio)
            shifted_prediction = simulate_one_step(env, snapshot, shifted_ratios)
            shifted_center = shifted_prediction["agents"][CENTER_AGENT_ID]
            if shifted_center["est_urllc_delay_ms"] > urllc_sla_ms + 1e-9:
                break
            max_safe_shift = float(shift_ratio)
            max_safe_center_embb_gain = float(
                shifted_center["throughput_slices_mbps"][0] - center_base["throughput_slices_mbps"][0]
            )
            max_safe_system_tp_gain = float(
                shifted_prediction["system_throughput_mbps"] - baseline_prediction["system_throughput_mbps"]
            )

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

        center_info = infos[CENTER_AGENT_ID]
        actual_center_delay_ms = float(center_info["est_urllc_delay"] * 1000.0)
        actual_center_embb_shortfall_mbps = max(
            0.0,
            float(env.sla_props["embb_gbr"] - float(center_info["embb_eval_tp_mbps"])),
        )
        actual_system_tp_mbps = float(sum(float(infos[agent]["throughput"]) for agent in env.agents if agent in infos))

        center_ratio_history.append(center_ratio)
        center_delay_ms_history.append(actual_center_delay_ms)
        center_slack_ms_history.append(urllc_sla_ms - actual_center_delay_ms)
        center_embb_shortfall_mbps_history.append(actual_center_embb_shortfall_mbps)
        center_embb_eval_tp_mbps_history.append(float(center_info["embb_eval_tp_mbps"]))
        center_embb_tp_mbps_history.append(float(center_info["throughput_slices_mbps"][0]))
        system_tp_mbps_history.append(actual_system_tp_mbps)
        max_safe_shift_ratio_history.append(max_safe_shift)
        max_safe_center_embb_gain_mbps_history.append(max_safe_center_embb_gain)
        max_safe_system_tp_gain_mbps_history.append(max_safe_system_tp_gain)
        candidate_safe_flags_history.append(np.asarray(candidate_safe_flags, dtype=np.float32))
        candidate_center_embb_gain_history.append(np.asarray(candidate_center_embb_gains, dtype=np.float32))
        candidate_system_tp_gain_history.append(np.asarray(candidate_system_tp_gains, dtype=np.float32))

        validation_center_delay_error_ms.append(
            abs(center_base["est_urllc_delay_ms"] - actual_center_delay_ms)
        )
        validation_center_embb_shortfall_error_mbps.append(
            abs(center_base["embb_shortfall_mbps"] - actual_center_embb_shortfall_mbps)
        )
        validation_system_tp_error_mbps.append(
            abs(baseline_prediction["system_throughput_mbps"] - actual_system_tp_mbps)
        )

        done = terminateds
        step += 1

    return {
        "train_seed": int(evaluator["train_seed"]),
        "eval_seed": int(eval_seed),
        "steps": int(step),
        "checkpoint_path": str(evaluator["checkpoint_path"]),
        "quality_score": None if evaluator.get("quality_score") is None else float(evaluator["quality_score"]),
        "center_ratios": np.asarray(center_ratio_history, dtype=np.float32),
        "center_delay_ms": np.asarray(center_delay_ms_history, dtype=np.float32),
        "center_slack_ms": np.asarray(center_slack_ms_history, dtype=np.float32),
        "center_embb_shortfall_mbps": np.asarray(center_embb_shortfall_mbps_history, dtype=np.float32),
        "center_embb_eval_tp_mbps": np.asarray(center_embb_eval_tp_mbps_history, dtype=np.float32),
        "center_embb_tp_mbps": np.asarray(center_embb_tp_mbps_history, dtype=np.float32),
        "system_tp_mbps": np.asarray(system_tp_mbps_history, dtype=np.float32),
        "max_safe_shift_ratio": np.asarray(max_safe_shift_ratio_history, dtype=np.float32),
        "max_safe_center_embb_gain_mbps": np.asarray(max_safe_center_embb_gain_mbps_history, dtype=np.float32),
        "max_safe_system_tp_gain_mbps": np.asarray(max_safe_system_tp_gain_mbps_history, dtype=np.float32),
        "candidate_safe_flags": np.asarray(candidate_safe_flags_history, dtype=np.float32),
        "candidate_center_embb_gain_mbps": np.asarray(candidate_center_embb_gain_history, dtype=np.float32),
        "candidate_system_tp_gain_mbps": np.asarray(candidate_system_tp_gain_history, dtype=np.float32),
        "validation_center_delay_error_ms": np.asarray(validation_center_delay_error_ms, dtype=np.float32),
        "validation_center_embb_shortfall_error_mbps": np.asarray(
            validation_center_embb_shortfall_error_mbps,
            dtype=np.float32,
        ),
        "validation_system_tp_error_mbps": np.asarray(validation_system_tp_error_mbps, dtype=np.float32),
    }


def aggregate_tradeoff_runs(run_list: list[dict], rollout_steps: int, candidate_shifts: list[float]) -> dict:
    center_ratio_hist = stack_vector_history(run_list, "center_ratios", rollout_steps, width=3)
    center_delay_hist = stack_scalar_history(run_list, "center_delay_ms", rollout_steps)
    center_slack_hist = stack_scalar_history(run_list, "center_slack_ms", rollout_steps)
    center_shortfall_hist = stack_scalar_history(run_list, "center_embb_shortfall_mbps", rollout_steps)
    center_embb_tp_hist = stack_scalar_history(run_list, "center_embb_tp_mbps", rollout_steps)
    system_tp_hist = stack_scalar_history(run_list, "system_tp_mbps", rollout_steps)
    safe_shift_hist = stack_scalar_history(run_list, "max_safe_shift_ratio", rollout_steps)
    safe_center_gain_hist = stack_scalar_history(run_list, "max_safe_center_embb_gain_mbps", rollout_steps)
    safe_system_gain_hist = stack_scalar_history(run_list, "max_safe_system_tp_gain_mbps", rollout_steps)
    candidate_safe_hist = stack_vector_history(run_list, "candidate_safe_flags", rollout_steps, width=len(candidate_shifts))

    all_records = {
        "center_ratios": np.concatenate([run["center_ratios"] for run in run_list], axis=0).astype(np.float32),
        "center_delay_ms": np.concatenate([run["center_delay_ms"] for run in run_list], axis=0).astype(np.float32),
        "center_slack_ms": np.concatenate([run["center_slack_ms"] for run in run_list], axis=0).astype(np.float32),
        "center_embb_shortfall_mbps": np.concatenate(
            [run["center_embb_shortfall_mbps"] for run in run_list], axis=0
        ).astype(np.float32),
        "center_embb_tp_mbps": np.concatenate([run["center_embb_tp_mbps"] for run in run_list], axis=0).astype(
            np.float32
        ),
        "system_tp_mbps": np.concatenate([run["system_tp_mbps"] for run in run_list], axis=0).astype(np.float32),
        "max_safe_shift_ratio": np.concatenate([run["max_safe_shift_ratio"] for run in run_list], axis=0).astype(
            np.float32
        ),
        "max_safe_center_embb_gain_mbps": np.concatenate(
            [run["max_safe_center_embb_gain_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "max_safe_system_tp_gain_mbps": np.concatenate(
            [run["max_safe_system_tp_gain_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "candidate_safe_flags": np.concatenate([run["candidate_safe_flags"] for run in run_list], axis=0).astype(
            np.float32
        ),
        "candidate_center_embb_gain_mbps": np.concatenate(
            [run["candidate_center_embb_gain_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "candidate_system_tp_gain_mbps": np.concatenate(
            [run["candidate_system_tp_gain_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "validation_center_delay_error_ms": np.concatenate(
            [run["validation_center_delay_error_ms"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "validation_center_embb_shortfall_error_mbps": np.concatenate(
            [run["validation_center_embb_shortfall_error_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
        "validation_system_tp_error_mbps": np.concatenate(
            [run["validation_system_tp_error_mbps"] for run in run_list],
            axis=0,
        ).astype(np.float32),
    }

    candidate_safe_rates = np.mean(all_records["candidate_safe_flags"], axis=0)
    candidate_center_gain_mean = np.nanmean(all_records["candidate_center_embb_gain_mbps"], axis=0)
    candidate_system_gain_mean = np.nanmean(all_records["candidate_system_tp_gain_mbps"], axis=0)
    embb_shortfall = all_records["center_embb_shortfall_mbps"]
    slack_ms = all_records["center_slack_ms"]
    positive_shortfall_mask = embb_shortfall > 1e-6
    positive_shortfall_and_safe_mask = positive_shortfall_mask & (slack_ms > 0.0)

    slack_bin_summary = compute_slack_bin_summary(all_records, candidate_shifts)

    return {
        "timeseries": {
            "center_ratio_mean": np.nanmean(center_ratio_hist, axis=0),
            "center_ratio_std": np.nanstd(center_ratio_hist, axis=0),
            "center_delay_mean": np.nanmean(center_delay_hist, axis=0),
            "center_delay_std": np.nanstd(center_delay_hist, axis=0),
            "center_slack_mean": np.nanmean(center_slack_hist, axis=0),
            "center_slack_std": np.nanstd(center_slack_hist, axis=0),
            "center_embb_shortfall_mean": np.nanmean(center_shortfall_hist, axis=0),
            "center_embb_shortfall_std": np.nanstd(center_shortfall_hist, axis=0),
            "center_embb_tp_mean": np.nanmean(center_embb_tp_hist, axis=0),
            "center_embb_tp_std": np.nanstd(center_embb_tp_hist, axis=0),
            "system_tp_mean": np.nanmean(system_tp_hist, axis=0),
            "system_tp_std": np.nanstd(system_tp_hist, axis=0),
            "max_safe_shift_mean": np.nanmean(safe_shift_hist, axis=0),
            "max_safe_shift_std": np.nanstd(safe_shift_hist, axis=0),
            "max_safe_center_embb_gain_mean": np.nanmean(safe_center_gain_hist, axis=0),
            "max_safe_center_embb_gain_std": np.nanstd(safe_center_gain_hist, axis=0),
            "candidate_safe_mean": np.nanmean(candidate_safe_hist, axis=0),
        },
        "all_records": all_records,
        "summary": {
            "num_runs": len(run_list),
            "total_steps": int(all_records["center_delay_ms"].size),
            "candidate_shifts": [float(shift) for shift in candidate_shifts],
            "center_delay_ms": summarize_scalar(all_records["center_delay_ms"]),
            "center_slack_ms": summarize_scalar(all_records["center_slack_ms"]),
            "center_embb_shortfall_mbps": summarize_scalar(all_records["center_embb_shortfall_mbps"]),
            "center_embb_tp_mbps": summarize_scalar(all_records["center_embb_tp_mbps"]),
            "system_tp_mbps": summarize_scalar(all_records["system_tp_mbps"]),
            "max_safe_shift_ratio": summarize_scalar(all_records["max_safe_shift_ratio"]),
            "max_safe_center_embb_gain_mbps": summarize_scalar(all_records["max_safe_center_embb_gain_mbps"]),
            "max_safe_system_tp_gain_mbps": summarize_scalar(all_records["max_safe_system_tp_gain_mbps"]),
            "candidate_safe_rate": {
                f"{shift:.3f}": float(candidate_safe_rates[idx]) for idx, shift in enumerate(candidate_shifts)
            },
            "candidate_center_embb_gain_mbps_mean": {
                f"{shift:.3f}": float(candidate_center_gain_mean[idx]) for idx, shift in enumerate(candidate_shifts)
            },
            "candidate_system_tp_gain_mbps_mean": {
                f"{shift:.3f}": float(candidate_system_gain_mean[idx]) for idx, shift in enumerate(candidate_shifts)
            },
            "fraction_positive_shortfall": float(np.mean(positive_shortfall_mask.astype(np.float64))),
            "fraction_positive_shortfall_with_positive_slack": float(
                np.mean(positive_shortfall_and_safe_mask.astype(np.float64))
            ),
            "mean_max_safe_shift_given_positive_shortfall": float(
                np.mean(all_records["max_safe_shift_ratio"][positive_shortfall_mask])
            )
            if np.any(positive_shortfall_mask)
            else 0.0,
            "validation_center_delay_error_ms": summarize_scalar(all_records["validation_center_delay_error_ms"]),
            "validation_center_embb_shortfall_error_mbps": summarize_scalar(
                all_records["validation_center_embb_shortfall_error_mbps"]
            ),
            "validation_system_tp_error_mbps": summarize_scalar(all_records["validation_system_tp_error_mbps"]),
            "slack_bin_summary": slack_bin_summary,
        },
    }


def plot_tradeoff_analysis(aggregated: dict, output_path: str):
    ts = aggregated["timeseries"]
    records = aggregated["all_records"]
    summary = aggregated["summary"]
    candidate_shifts = np.asarray(summary["candidate_shifts"], dtype=np.float64)
    steps = np.arange(ts["center_delay_mean"].shape[0], dtype=np.int32)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    delay_mean = ts["center_delay_mean"]
    delay_std = ts["center_delay_std"]
    ax1.plot(steps, delay_mean, color=SLICE_COLORS["URLLC"], linewidth=2.0, label="BS_0 URLLC delay")
    ax1.fill_between(
        steps,
        delay_mean - delay_std,
        delay_mean + delay_std,
        color=SLICE_COLORS["URLLC"],
        alpha=0.2,
    )
    ax1.axhline(2.0, color="#444444", linestyle="--", linewidth=1.5, label="2 ms SLA")
    ax1.set_title("BS_0 URLLC Delay")
    ax1.set_xlabel("TTI")
    ax1.set_ylabel("Delay (ms)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    scatter = ax2.scatter(
        records["center_slack_ms"],
        records["center_embb_shortfall_mbps"],
        c=records["center_ratios"][:, 1],
        cmap="viridis",
        s=18,
        alpha=0.55,
        edgecolors="none",
    )
    ax2.axvline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    ax2.set_title("Slack vs eMBB Shortfall")
    ax2.set_xlabel("BS_0 URLLC slack (ms)")
    ax2.set_ylabel("BS_0 eMBB shortfall (Mbps)")
    ax2.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax2)
    colorbar.set_label("BS_0 URLLC ratio")

    safe_rates = np.asarray([summary["candidate_safe_rate"][f"{shift:.3f}"] for shift in candidate_shifts])
    candidate_center_gain = np.asarray(
        [summary["candidate_center_embb_gain_mbps_mean"][f"{shift:.3f}"] for shift in candidate_shifts]
    )
    ax3.bar(candidate_shifts, safe_rates, width=0.008, color="#4c72b0", alpha=0.85, label="Safe rate")
    ax3.set_ylim(0.0, 1.05)
    ax3.set_title("Immediate Safe Shift Acceptance")
    ax3.set_xlabel("Shift URLLC -> eMBB (ratio)")
    ax3.set_ylabel("Safe fraction")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        candidate_shifts,
        candidate_center_gain,
        color="#dd8452",
        marker="o",
        linewidth=1.8,
        label="Mean BS_0 eMBB gain",
    )
    ax3_twin.set_ylabel("Mean immediate eMBB gain (Mbps)")

    ax4.hist(records["max_safe_shift_ratio"], bins=20, color="#55a868", alpha=0.85, edgecolor="white")
    ax4.set_title("Distribution of Max Safe Shift")
    ax4.set_xlabel("Max safe shift ratio")
    ax4.set_ylabel("Count")
    ax4.grid(True, axis="y", alpha=0.25)

    scatter2 = ax5.scatter(
        records["center_embb_shortfall_mbps"],
        records["max_safe_shift_ratio"],
        c=records["center_delay_ms"],
        cmap="magma_r",
        s=18,
        alpha=0.55,
        edgecolors="none",
    )
    ax5.set_title("Shortfall vs Max Safe Shift")
    ax5.set_xlabel("BS_0 eMBB shortfall (Mbps)")
    ax5.set_ylabel("Max safe shift ratio")
    ax5.grid(True, alpha=0.25)
    colorbar2 = fig.colorbar(scatter2, ax=ax5)
    colorbar2.set_label("BS_0 URLLC delay (ms)")

    slack_bin_summary = summary["slack_bin_summary"]
    labels = [item["label"] for item in slack_bin_summary]
    shortfall_by_bin = np.asarray(
        [np.nan if item["mean_embb_shortfall_mbps"] is None else item["mean_embb_shortfall_mbps"] for item in slack_bin_summary],
        dtype=np.float64,
    )
    safe_shift_by_bin = np.asarray(
        [np.nan if item["mean_max_safe_shift_ratio"] is None else item["mean_max_safe_shift_ratio"] for item in slack_bin_summary],
        dtype=np.float64,
    )
    x_pos = np.arange(len(labels), dtype=np.int32)
    ax6.bar(x_pos, shortfall_by_bin, color="#c44e52", alpha=0.8, label="Mean eMBB shortfall")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels, rotation=25, ha="right")
    ax6.set_title("Slack-Binned Shortfall and Safe Shift")
    ax6.set_xlabel("BS_0 URLLC slack bin")
    ax6.set_ylabel("Mean eMBB shortfall (Mbps)")
    ax6.grid(True, axis="y", alpha=0.25)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(x_pos, safe_shift_by_bin, color="#8172b2", marker="o", linewidth=1.8, label="Mean max safe shift")
    ax6_twin.set_ylabel("Mean max safe shift ratio")

    fig.suptitle("IPPO URLLC Slack vs eMBB Tradeoff (Immediate One-Step Counterfactual)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def to_serializable_summary(aggregated: dict) -> dict:
    summary = aggregated["summary"]
    return {
        "analysis_scope": "Immediate one-step counterfactual around the selected IPPO rollout actions.",
        "num_runs": int(summary["num_runs"]),
        "total_steps": int(summary["total_steps"]),
        "candidate_shifts": [float(item) for item in summary["candidate_shifts"]],
        "center_delay_ms": summary["center_delay_ms"],
        "center_slack_ms": summary["center_slack_ms"],
        "center_embb_shortfall_mbps": summary["center_embb_shortfall_mbps"],
        "center_embb_tp_mbps": summary["center_embb_tp_mbps"],
        "system_tp_mbps": summary["system_tp_mbps"],
        "max_safe_shift_ratio": summary["max_safe_shift_ratio"],
        "max_safe_center_embb_gain_mbps": summary["max_safe_center_embb_gain_mbps"],
        "max_safe_system_tp_gain_mbps": summary["max_safe_system_tp_gain_mbps"],
        "candidate_safe_rate": summary["candidate_safe_rate"],
        "candidate_center_embb_gain_mbps_mean": summary["candidate_center_embb_gain_mbps_mean"],
        "candidate_system_tp_gain_mbps_mean": summary["candidate_system_tp_gain_mbps_mean"],
        "fraction_positive_shortfall": float(summary["fraction_positive_shortfall"]),
        "fraction_positive_shortfall_with_positive_slack": float(
            summary["fraction_positive_shortfall_with_positive_slack"]
        ),
        "mean_max_safe_shift_given_positive_shortfall": float(summary["mean_max_safe_shift_given_positive_shortfall"]),
        "validation_center_delay_error_ms": summary["validation_center_delay_error_ms"],
        "validation_center_embb_shortfall_error_mbps": summary["validation_center_embb_shortfall_error_mbps"],
        "validation_system_tp_error_mbps": summary["validation_system_tp_error_mbps"],
        "slack_bin_summary": summary["slack_bin_summary"],
    }


def print_summary(summary: dict):
    print("=== IPPO Slack / Tradeoff Summary ===")
    delay = summary["center_delay_ms"]
    slack = summary["center_slack_ms"]
    shortfall = summary["center_embb_shortfall_mbps"]
    safe_shift = summary["max_safe_shift_ratio"]
    embb_gain = summary["max_safe_center_embb_gain_mbps"]
    system_gain = summary["max_safe_system_tp_gain_mbps"]
    print(
        f"BS_0 delay mean/p95/p99/max = {delay['mean']:.4f}/{delay['p95']:.4f}/"
        f"{delay['p99']:.4f}/{delay['max']:.4f} ms"
    )
    print(
        f"BS_0 slack mean/p05/p50 = {slack['mean']:.4f}/{slack['p05']:.4f}/{slack['p50']:.4f} ms"
    )
    print(
        f"BS_0 eMBB shortfall mean/p95/max = {shortfall['mean']:.4f}/{shortfall['p95']:.4f}/"
        f"{shortfall['max']:.4f} Mbps"
    )
    print(
        f"Max safe shift mean/p50/p95/max = {safe_shift['mean']:.4f}/{safe_shift['p50']:.4f}/"
        f"{safe_shift['p95']:.4f}/{safe_shift['max']:.4f}"
    )
    print(
        f"Immediate BS_0 eMBB gain at max safe shift mean/p95 = {embb_gain['mean']:.4f}/"
        f"{embb_gain['p95']:.4f} Mbps"
    )
    print(
        f"Immediate system TP gain at max safe shift mean/p95 = {system_gain['mean']:.4f}/"
        f"{system_gain['p95']:.4f} Mbps"
    )
    print(
        f"Positive eMBB shortfall fraction = {summary['fraction_positive_shortfall'] * 100:.1f}% | "
        f"Positive shortfall with positive slack = {summary['fraction_positive_shortfall_with_positive_slack'] * 100:.1f}%"
    )
    print("Candidate safe rates:")
    for shift_key, safe_rate in summary["candidate_safe_rate"].items():
        embb_gain_mean = summary["candidate_center_embb_gain_mbps_mean"][shift_key]
        system_gain_mean = summary["candidate_system_tp_gain_mbps_mean"][shift_key]
        print(
            f"  shift={shift_key}: safe_rate={safe_rate * 100:.1f}%, "
            f"mean_embb_gain={embb_gain_mean:.4f} Mbps, mean_system_gain={system_gain_mean:.4f} Mbps"
        )
    print(
        f"Simulation check max abs error: delay={summary['validation_center_delay_error_ms']['max']:.6f} ms, "
        f"shortfall={summary['validation_center_embb_shortfall_error_mbps']['max']:.6f} Mbps, "
        f"system_tp={summary['validation_system_tp_error_mbps']['max']:.6f} Mbps"
    )


def main():
    args = parse_args()
    eval_seeds = list(EVAL_SEEDS if args.eval_seeds is None else args.eval_seeds)
    candidate_shifts = sorted({float(max(0.0, shift)) for shift in args.candidate_shifts if shift > 0.0})
    if not candidate_shifts:
        raise ValueError("candidate_shifts must contain at least one positive value.")
    if args.shift_grid_step <= 0.0:
        raise ValueError("shift_grid_step must be positive.")
    if args.max_shift <= 0.0:
        raise ValueError("max_shift must be positive.")

    validate_seed_split()
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    evaluators = init_ippo_evaluators()
    run_list = []
    try:
        for evaluator in evaluators:
            for eval_seed in eval_seeds:
                print(
                    f"Running slack/tradeoff rollout for train_seed={evaluator['train_seed']} "
                    f"eval_seed={eval_seed} checkpoint={evaluator['checkpoint_path']}"
                )
                run_list.append(
                    run_tradeoff_rollout(
                        evaluator,
                        eval_seed=eval_seed,
                        rollout_steps=args.rollout_steps,
                        candidate_shifts=candidate_shifts,
                        max_shift=float(args.max_shift),
                        shift_grid_step=float(args.shift_grid_step),
                    )
                )
    finally:
        stop_ippo_evaluators()
        if ray.is_initialized():
            ray.shutdown()

    aggregated = aggregate_tradeoff_runs(run_list, args.rollout_steps, candidate_shifts)
    plot_path = f"{args.output_prefix}.png"
    json_path = f"{args.output_prefix}.json"
    plot_tradeoff_analysis(aggregated, plot_path)
    summary = to_serializable_summary(aggregated)
    summary["run_metadata"] = [
        {
            "train_seed": int(run["train_seed"]),
            "eval_seed": int(run["eval_seed"]),
            "steps": int(run["steps"]),
            "checkpoint_path": str(run["checkpoint_path"]),
            "quality_score": None if run.get("quality_score") is None else float(run["quality_score"]),
        }
        for run in run_list
    ]
    with open(json_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved slack/tradeoff plot to {plot_path}")
    print(f"Saved slack/tradeoff summary to {json_path}")


if __name__ == "__main__":
    main()

import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns
from ray.tune.registry import register_env

from checkpoint_utils import rank_checkpoints_by_metric
from ippo_rl_module import (
    DEFAULT_INITIAL_ACTION_LOG_STD,
    DEFAULT_INITIAL_SLICE_RATIOS,
    build_initialized_rl_module_spec,
)
from multi_cell_env import MultiCell_5G_SLA_Env

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

EPS = 1e-5
PF_MA_ALPHA = 0.9
ROLLOUT_STEPS = 200
TRAIN_SEEDS = [2026, 2027, 2028]
EVAL_SEEDS = [3026, 3027, 3028, 3029]
ENV_PROFILE = "balanced"
OBSERVATION_MODE = "pure_local"
EXPERIMENT_ENV_TAG = "balanced_ippo_v1"
MIN_EVAL_CHECKPOINT_ITER = 50
IPPO_EXPERIMENT_DIRS = [f"./ray_results/MAPPO_5G_Slicing_{EXPERIMENT_ENV_TAG}_seed{seed}" for seed in TRAIN_SEEDS]

ALGORITHMS = [
    ("static", "Static SLA Split"),
    ("priority", "Priority Heuristic"),
    ("max_weight", "Max-Weight (Throughput-Oriented Heuristic)"),
    ("pf", "Proportional Fair"),
    ("ippo", "IPPO (pure_local)"),
]

PLOT_COLORS = {
    "static": "#1f77b4",
    "priority": "#ff7f0e",
    "max_weight": "#2ca02c",
    "pf": "#d62728",
    "ippo": "#9467bd",
}

ENV_CONFIG = {
    "env_profile": ENV_PROFILE,
    "observation_mode": OBSERVATION_MODE,
    "action_softmax_temperature": 3.0,
    "penalty_weight": 0.7,
    "w_embb": 1.0,
    "w_urllc": 0.30,
    "w_mmtc": 0.7,
    "urllc_warning_ratio": 0.90,
    "urllc_tail_ratio": 0.92,
    "urllc_softplus_slope": 10.0,
    "urllc_warning_gain": 0.20,
    "urllc_tail_quad_gain": 2.0,
    "urllc_hard_violation_gain": 1.75,
    "urllc_overflow_gain": 2.2,
    "urllc_exp_coeff": 1.6,
    "urllc_penalty_cap_factor": 10.0,
    "embb_penalty_quad_gain": 2.5,
    "embb_penalty_cubic_gain": 1.2,
    "embb_penalty_cap_factor": 16.0,
    "mmtc_penalty_cap_factor": 5.0,
    "embb_violation_cap": 2.0,
    "urllc_violation_cap": 5.0,
    "mmtc_violation_cap": 2.0,
    "embb_gbr": 220.0,
    "urllc_burst_start_prob": 0.06,
    "urllc_burst_end_prob": 0.35,
    "urllc_burst_mean_mbps": 100.0,
    "urllc_burst_std_mbps": 15.0,
    "binary_reward_throughput_scale": 80.0,
    "binary_penalty_embb": 4.0,
    "binary_penalty_urllc": 6.0,
    "binary_penalty_mmtc": 4.0,
    "binary_urllc_yellow_start_ratio": 0.5,
    "binary_urllc_yellow_penalty": 6.0,
    "tail_reward_throughput_weight": 1.0 / 300.0,
    "center_reward_scale": 1.0,
    "reward_clip_abs": 0.0,
}

DEFAULT_OBSERVATION_FILTER = "MeanStdFilter"
IPPO_EVALUATORS = []


def validate_seed_split():
    overlap = sorted(set(TRAIN_SEEDS).intersection(EVAL_SEEDS))
    if overlap:
        raise ValueError(
            "TRAIN_SEEDS and EVAL_SEEDS must be disjoint for unbiased evaluation. "
            f"Overlapping seeds: {overlap}"
        )


def main_comparison_algorithms():
    return list(ALGORITHMS)


def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def build_ippo_episode_context(algo, obs_dict, infos):
    runner = algo.env_runner_group.local_env_runner
    runner._cached_to_module = None
    episode = runner._new_episode()
    shared_data = {"agent_to_module_mapping_fn": runner.config.policy_mapping_fn}
    episode.add_env_reset(observations=obs_dict, infos=infos)
    return runner, episode, shared_data


def compute_actions_batched(algo, runner, episode, shared_data):
    """Run RLlib new-stack connector pipeline to preserve observation processing."""
    to_module = runner._env_to_module(
        rl_module=runner.module,
        episodes=[episode],
        explore=False,
        shared_data=shared_data,
    )
    to_env = runner.module.forward_inference(to_module)
    to_env = runner._module_to_env(
        rl_module=runner.module,
        data=to_env,
        episodes=[episode],
        explore=False,
        shared_data=shared_data,
    )

    actions = to_env.pop(Columns.ACTIONS)
    actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, actions)
    extra_model_outputs = defaultdict(dict)
    for col, ma_dict_list in to_env.items():
        ma_dict = ma_dict_list[0]
        for agent_id, value in ma_dict.items():
            extra_model_outputs[agent_id][col] = value

    return actions[0], actions_for_env[0], dict(extra_model_outputs)


def ratios_to_action(ratios: np.ndarray, temperature: float) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=np.float32)
    ratios = np.clip(ratios, 1e-8, None)
    ratio_sum = float(np.sum(ratios))
    if ratio_sum <= 0.0:
        ratios = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    else:
        ratios = ratios / ratio_sum

    logits = np.log(ratios)
    logits = logits - np.mean(logits)
    action = logits / max(float(temperature), 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def compute_jain_fairness(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    numerator = float(np.sum(x) ** 2)
    denominator = float(n * np.sum(x**2))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _load_trial_params(trial_dir: str) -> dict:
    trial_path = Path(trial_dir)
    params_json = trial_path / "params.json"
    if params_json.exists():
        with params_json.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        if isinstance(data, dict):
            return data

    params_pkl = trial_path / "params.pkl"
    if params_pkl.exists():
        with params_pkl.open("rb") as file_obj:
            data = pickle.load(file_obj)
        if isinstance(data, dict):
            return data

    return {}


def _resolve_trial_observation_filter(trial_dir: str | None) -> str:
    if not trial_dir:
        return DEFAULT_OBSERVATION_FILTER

    params = _load_trial_params(trial_dir)
    observation_filter = params.get("observation_filter")
    if observation_filter is None:
        return DEFAULT_OBSERVATION_FILTER
    return str(observation_filter)


def build_ippo_eval_algo(observation_filter: str):
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=ENV_CONFIG)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=EVAL_SEEDS[0])
        .rl_module(
            rl_module_spec=build_initialized_rl_module_spec(
                ENV_CONFIG["action_softmax_temperature"],
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu",
                initial_action_ratios=DEFAULT_INITIAL_SLICE_RATIOS,
                initial_action_log_std=DEFAULT_INITIAL_ACTION_LOG_STD,
            )
        )
        .multi_agent(
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(observation_filter=observation_filter, num_env_runners=0)
        .learners(num_learners=0)
    )
    return config.build()


def stop_ippo_evaluators():
    global IPPO_EVALUATORS

    for evaluator in IPPO_EVALUATORS:
        evaluator["algo"].stop()
    IPPO_EVALUATORS = []


def init_ippo_evaluators():
    global IPPO_EVALUATORS

    stop_ippo_evaluators()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    evaluators = []
    restore_errors = []

    for train_seed, experiment_dir in zip(TRAIN_SEEDS, IPPO_EXPERIMENT_DIRS):
        ranked_checkpoints = rank_checkpoints_by_metric(
            experiment_dir,
            min_training_iteration=MIN_EVAL_CHECKPOINT_ITER,
            fallback_to_any=False,
        )
        if not ranked_checkpoints:
            restore_errors.append(
                f"train_seed={train_seed}, experiment_dir={experiment_dir} -> "
                f"no ranked checkpoint found with iter>={MIN_EVAL_CHECKPOINT_ITER}"
            )
            continue

        restored = False
        for item in ranked_checkpoints:
            checkpoint_path = item["checkpoint_path"]
            score = item.get("episode_return_mean")
            iteration = item.get("training_iteration")
            urllc_violation = item.get("center_urllc_violations")
            embb_violation = item.get("center_embb_violations")
            mmtc_violation = item.get("center_mmtc_violations")
            total_sla_violation = item.get("center_total_sla_violations")
            urllc_delay_ms = item.get("center_urllc_delay_ms")
            quality_score = item.get("quality_score")
            base_tp = item.get("center_reward_base_tp")
            system_throughput_mbps = item.get("system_throughput_mbps")
            trial_dir = item.get("trial_dir")
            observation_filter = _resolve_trial_observation_filter(trial_dir)
            algo = build_ippo_eval_algo(observation_filter=observation_filter)

            print(
                f"Trying checkpoint: {checkpoint_path} "
                f"(train_seed={train_seed}, iter={iteration}, obs_filter={observation_filter}, "
                f"total_viol={total_sla_violation}, embb_viol={embb_violation}, "
                f"urllc_viol={urllc_violation}, mmtc_viol={mmtc_violation}, "
                f"urllc_delay_ms={urllc_delay_ms}, system_tp={system_throughput_mbps}, base_tp={base_tp}, "
                f"quality={quality_score}, episode_return_mean={score})"
            )
            try:
                algo.restore(checkpoint_path)
                evaluator = {
                    "algo": algo,
                    "train_seed": train_seed,
                    "checkpoint_path": checkpoint_path,
                    "training_iteration": iteration,
                    "observation_filter": observation_filter,
                    "center_urllc_violations": urllc_violation,
                    "center_embb_violations": embb_violation,
                    "center_mmtc_violations": mmtc_violation,
                    "center_total_sla_violations": total_sla_violation,
                    "center_urllc_delay_ms": urllc_delay_ms,
                    "center_reward_base_tp": base_tp,
                    "system_throughput_mbps": system_throughput_mbps,
                    "episode_return_mean": score,
                    "quality_score": quality_score,
                }
                evaluators.append(evaluator)
                print(
                    f"Loaded IPPO checkpoint: {checkpoint_path} "
                    f"(train_seed={train_seed}, obs_filter={observation_filter}, "
                    f"total_viol={total_sla_violation}, system_tp={system_throughput_mbps}, "
                    f"quality={quality_score})"
                )
                restored = True
                break
            except Exception as exc:  # noqa: PERF203
                algo.stop()
                restore_errors.append(
                    f"train_seed={train_seed}, checkpoint={checkpoint_path}, "
                    f"obs_filter={observation_filter} -> {exc}"
                )

        if not restored:
            print(f"Warning: no restorable checkpoint found for train_seed={train_seed}.")

    if not evaluators:
        error_preview = "\n".join(restore_errors[:5])
        raise RuntimeError(
            "No compatible per-seed IPPO checkpoint could be restored.\n"
            f"Sample restore errors:\n{error_preview}"
        )

    IPPO_EVALUATORS = evaluators
    return IPPO_EVALUATORS


def run_evaluation(env, algo_name, seed, ippo_evaluator=None):
    if algo_name == "ippo" and ippo_evaluator is None:
        raise RuntimeError("IPPO evaluator is not initialized. Call init_ippo_evaluators() first.")

    obs, reset_infos = env.reset(seed=seed)
    done = {"__all__": False}
    ippo_runner = None
    ippo_episode = None
    ippo_shared_data = None

    if algo_name == "ippo":
        ippo_runner, ippo_episode, ippo_shared_data = build_ippo_episode_context(
            ippo_evaluator["algo"], obs, reset_infos
        )

    center_reward = []
    center_reward_base_tp = []
    center_urllc_delay_ms = []
    center_embb_shortfall = []
    center_throughput_mbps = []
    system_throughput_mbps = []
    center_penalty_total = []
    center_penalty_embb = []
    center_penalty_urllc = []
    center_penalty_mmtc = []
    center_penalty_raw_embb = []
    center_penalty_raw_urllc = []
    center_penalty_raw_mmtc = []
    inference_overhead_total_ms = []
    inference_overhead_per_agent_ms = []

    embb_cumulative_throughput = {agent: 0.0 for agent in env.agents}

    pf_avg_throughput = {
        agent: np.full(3, 1.0, dtype=np.float32) for agent in env.agents
    }

    # SLA success counters: slice order [eMBB, URLLC, mMTC].
    sla_ok_sys = np.zeros(3, dtype=np.float64)
    sla_total_sys = 0
    sla_ok_bs0 = np.zeros(3, dtype=np.float64)
    sla_total_bs0 = 0

    steps = 0
    while not done["__all__"] and steps < ROLLOUT_STEPS:
        actions = {}

        if algo_name == "static":
            # Industrial-leaning static split: keep a small fixed mMTC slice.
            static_ratios = np.array([0.45, 0.45, 0.10], dtype=np.float32)
            for agent in env.agents:
                actions[agent] = ratios_to_action(static_ratios, env.action_softmax_temperature)

        elif algo_name == "priority":
            urllc_priority_ratios = np.array([0.08333333, 0.8333333, 0.08333333], dtype=np.float32)
            embb_priority_ratios = np.array([0.72, 0.20, 0.08], dtype=np.float32)
            for agent in env.agents:
                if env.queues[agent][1] > 0.05:
                    actions[agent] = ratios_to_action(urllc_priority_ratios, env.action_softmax_temperature)
                else:
                    actions[agent] = ratios_to_action(embb_priority_ratios, env.action_softmax_temperature)

        elif algo_name == "max_weight":
            # Throughput-oriented heuristic: queue backlog weighted by instant SE.
            for agent in env.agents:
                queue_vec = np.maximum(env.queues[agent], 0.0)
                se_vec = np.maximum(env.current_se[agent], 0.0)
                weights = queue_vec * se_vec
                if float(np.sum(weights)) <= EPS:
                    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                ratios = weights / np.sum(weights)
                actions[agent] = ratios_to_action(ratios, env.action_softmax_temperature)

        elif algo_name == "pf":
            for agent in env.agents:
                # Estimated rate if the slice gets full bandwidth.
                rate_est_mbps = (env.total_bandwidth * np.maximum(env.current_se[agent], 0.0)) / 1e6
                weights = rate_est_mbps / (pf_avg_throughput[agent] + EPS)
                if float(np.sum(weights)) <= EPS:
                    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                ratios = weights / np.sum(weights)
                actions[agent] = ratios_to_action(ratios, env.action_softmax_temperature)

        elif algo_name == "ippo":
            start = time.perf_counter()
            policy_actions, actions, extra_model_outputs = compute_actions_batched(
                ippo_evaluator["algo"],
                ippo_runner,
                ippo_episode,
                ippo_shared_data,
            )
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            inference_overhead_total_ms.append(elapsed_ms)
            inference_overhead_per_agent_ms.append(elapsed_ms / max(len(obs), 1))
        else:
            extra_model_outputs = None

        if algo_name not in {"static", "priority", "max_weight", "pf", "ippo"}:
            raise ValueError(f"Unknown algorithm name: {algo_name}")

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        if algo_name == "ippo":
            ippo_episode.add_env_step(
                obs,
                policy_actions,
                rewards,
                infos=infos,
                terminateds=terminateds,
                truncateds=truncateds,
                extra_model_outputs=extra_model_outputs,
            )

        step_system_tp_mbps = 0.0
        for agent in env.agents:
            if agent not in infos:
                continue
            step_system_tp_mbps += float(infos[agent]["throughput"])

            violations = np.asarray(infos[agent]["violations"], dtype=np.float64)
            sla_ok_sys += (violations <= 0.0).astype(np.float64)
            sla_total_sys += 1
            if agent == "BS_0":
                sla_ok_bs0 += (violations <= 0.0).astype(np.float64)
                sla_total_bs0 += 1

            embb_tp_mbps = float(infos[agent]["throughput_slices_mbps"][0])
            embb_cumulative_throughput[agent] += embb_tp_mbps

            if algo_name == "pf":
                inst_tp = np.asarray(infos[agent]["throughput_slices_mbps"], dtype=np.float32)
                pf_avg_throughput[agent] = (
                    PF_MA_ALPHA * pf_avg_throughput[agent] + (1.0 - PF_MA_ALPHA) * inst_tp
                )

        center_info = infos["BS_0"]
        penalties_clipped = np.asarray(center_info.get("penalties", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        penalties_raw = np.asarray(center_info.get("penalty_raw", penalties_clipped), dtype=np.float32)

        if env.reward_mode == "binary_sla_reward":
            throughput_reward_weight = 1.0 / max(env.binary_reward_throughput_scale, 1e-6)
        elif env.reward_mode == "simple_local_sla":
            throughput_reward_weight = env.simple_reward_throughput_weight
        else:
            throughput_reward_weight = env.tail_reward_throughput_weight
        reward_base_tp = float(
            center_info.get("reward_base_tp", center_info["throughput"] * throughput_reward_weight)
        )
        penalty_embb = float(center_info.get("penalty_embb", penalties_clipped[0]))
        penalty_urllc = float(center_info.get("penalty_urllc", penalties_clipped[1]))
        penalty_mmtc = float(center_info.get("penalty_mmtc", penalties_clipped[2]))
        penalty_total = float(center_info.get("penalty_total", penalty_embb + penalty_urllc + penalty_mmtc))

        center_reward.append(float(rewards["BS_0"]))
        center_reward_base_tp.append(reward_base_tp)
        center_urllc_delay_ms.append(float(center_info["est_urllc_delay"] * 1000.0))
        center_embb_shortfall.append(float(env.state["BS_0"][13]))
        center_throughput_mbps.append(float(center_info["throughput"]))
        system_throughput_mbps.append(step_system_tp_mbps)
        center_penalty_total.append(penalty_total)
        center_penalty_embb.append(penalty_embb)
        center_penalty_urllc.append(penalty_urllc)
        center_penalty_mmtc.append(penalty_mmtc)
        center_penalty_raw_embb.append(float(center_info.get("penalty_raw_embb", penalties_raw[0])))
        center_penalty_raw_urllc.append(float(center_info.get("penalty_raw_urllc", penalties_raw[1])))
        center_penalty_raw_mmtc.append(float(center_info.get("penalty_raw_mmtc", penalties_raw[2])))

        done = terminateds
        steps += 1

    embb_vec = np.array([embb_cumulative_throughput[agent] for agent in env.agents], dtype=np.float64)
    jfi_embb = compute_jain_fairness(embb_vec)
    sla_sys_success = sla_ok_sys / max(float(sla_total_sys), 1.0)
    sla_bs0_success = sla_ok_bs0 / max(float(sla_total_bs0), 1.0)

    result = {
        "reward": np.asarray(center_reward, dtype=np.float32),
        "reward_base_tp": np.asarray(center_reward_base_tp, dtype=np.float32),
        "cum_reward": np.cumsum(np.asarray(center_reward, dtype=np.float32)),
        "urllc_delay_ms": np.asarray(center_urllc_delay_ms, dtype=np.float32),
        "embb_shortfall": np.asarray(center_embb_shortfall, dtype=np.float32),
        "center_throughput_mbps": np.asarray(center_throughput_mbps, dtype=np.float32),
        "system_throughput_mbps": np.asarray(system_throughput_mbps, dtype=np.float32),
        "penalty_total": np.asarray(center_penalty_total, dtype=np.float32),
        "penalty_embb": np.asarray(center_penalty_embb, dtype=np.float32),
        "penalty_urllc": np.asarray(center_penalty_urllc, dtype=np.float32),
        "penalty_mmtc": np.asarray(center_penalty_mmtc, dtype=np.float32),
        "penalty_raw_embb": np.asarray(center_penalty_raw_embb, dtype=np.float32),
        "penalty_raw_urllc": np.asarray(center_penalty_raw_urllc, dtype=np.float32),
        "penalty_raw_mmtc": np.asarray(center_penalty_raw_mmtc, dtype=np.float32),
        "sla_sys_success_rate": np.asarray(sla_sys_success, dtype=np.float64),
        "sla_bs0_success_rate": np.asarray(sla_bs0_success, dtype=np.float64),
        "jfi_embb": float(jfi_embb),
        "mean_inference_total_ms": (
            float(np.mean(inference_overhead_total_ms)) if inference_overhead_total_ms else np.nan
        ),
        "p95_inference_total_ms": (
            float(np.percentile(inference_overhead_total_ms, 95)) if inference_overhead_total_ms else np.nan
        ),
        "mean_inference_per_agent_ms": (
            float(np.mean(inference_overhead_per_agent_ms)) if inference_overhead_per_agent_ms else np.nan
        ),
    }
    if algo_name == "ippo":
        result["train_seed"] = int(ippo_evaluator["train_seed"])
        result["checkpoint_path"] = str(ippo_evaluator["checkpoint_path"])
    return result


def stack_metric(run_list, metric_key):
    data = np.full((len(run_list), ROLLOUT_STEPS), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = run[metric_key]
        valid_len = min(len(values), ROLLOUT_STEPS)
        data[idx, :valid_len] = values[:valid_len]
    return data


def aggregate_results(results_by_algo):
    aggregated = {}
    for algo_key, run_list in results_by_algo.items():
        reward_base_tp = stack_metric(run_list, "reward_base_tp")
        delay = stack_metric(run_list, "urllc_delay_ms")
        cum_reward = stack_metric(run_list, "cum_reward")
        shortfall = stack_metric(run_list, "embb_shortfall")
        center_tp = stack_metric(run_list, "center_throughput_mbps")
        system_tp = stack_metric(run_list, "system_throughput_mbps")
        penalty_total = stack_metric(run_list, "penalty_total")
        penalty_embb = stack_metric(run_list, "penalty_embb")
        penalty_urllc = stack_metric(run_list, "penalty_urllc")
        penalty_mmtc = stack_metric(run_list, "penalty_mmtc")
        penalty_raw_embb = stack_metric(run_list, "penalty_raw_embb")
        penalty_raw_urllc = stack_metric(run_list, "penalty_raw_urllc")
        penalty_raw_mmtc = stack_metric(run_list, "penalty_raw_mmtc")

        fairness = np.asarray([run["jfi_embb"] for run in run_list], dtype=np.float64)
        inf_total_ms = np.asarray([run["mean_inference_total_ms"] for run in run_list], dtype=np.float64)
        inf_total_ms_finite = inf_total_ms[np.isfinite(inf_total_ms)]
        inf_p95_total_ms = np.asarray([run["p95_inference_total_ms"] for run in run_list], dtype=np.float64)
        inf_p95_total_ms_finite = inf_p95_total_ms[np.isfinite(inf_p95_total_ms)]
        inf_per_agent_ms = np.asarray([run["mean_inference_per_agent_ms"] for run in run_list], dtype=np.float64)
        inf_per_agent_ms_finite = inf_per_agent_ms[np.isfinite(inf_per_agent_ms)]
        sla_sys = np.asarray([run["sla_sys_success_rate"] for run in run_list], dtype=np.float64)
        sla_bs0 = np.asarray([run["sla_bs0_success_rate"] for run in run_list], dtype=np.float64)

        penalty_embb_scalar = float(np.nanmean(penalty_embb))
        penalty_urllc_scalar = float(np.nanmean(penalty_urllc))
        penalty_mmtc_scalar = float(np.nanmean(penalty_mmtc))
        penalty_total_scalar = float(np.nanmean(penalty_total))
        penalty_share_denom = max(penalty_embb_scalar + penalty_urllc_scalar + penalty_mmtc_scalar, 1e-8)

        aggregated[algo_key] = {
            "num_runs": len(run_list),
            "reward_base_tp_mean": np.nanmean(reward_base_tp, axis=0),
            "reward_base_tp_std": np.nanstd(reward_base_tp, axis=0),
            "delay_mean": np.nanmean(delay, axis=0),
            "delay_std": np.nanstd(delay, axis=0),
            "cum_reward_mean": np.nanmean(cum_reward, axis=0),
            "cum_reward_std": np.nanstd(cum_reward, axis=0),
            "shortfall_mean": np.nanmean(shortfall, axis=0),
            "shortfall_std": np.nanstd(shortfall, axis=0),
            "center_tp_mean": np.nanmean(center_tp, axis=0),
            "center_tp_std": np.nanstd(center_tp, axis=0),
            "system_tp_mean": np.nanmean(system_tp, axis=0),
            "system_tp_std": np.nanstd(system_tp, axis=0),
            "penalty_total_mean": np.nanmean(penalty_total, axis=0),
            "penalty_total_std": np.nanstd(penalty_total, axis=0),
            "penalty_embb_mean": np.nanmean(penalty_embb, axis=0),
            "penalty_embb_std": np.nanstd(penalty_embb, axis=0),
            "penalty_urllc_mean": np.nanmean(penalty_urllc, axis=0),
            "penalty_urllc_std": np.nanstd(penalty_urllc, axis=0),
            "penalty_mmtc_mean": np.nanmean(penalty_mmtc, axis=0),
            "penalty_mmtc_std": np.nanstd(penalty_mmtc, axis=0),
            "penalty_raw_embb_mean": np.nanmean(penalty_raw_embb, axis=0),
            "penalty_raw_urllc_mean": np.nanmean(penalty_raw_urllc, axis=0),
            "penalty_raw_mmtc_mean": np.nanmean(penalty_raw_mmtc, axis=0),
            "penalty_embb_scalar": penalty_embb_scalar,
            "penalty_urllc_scalar": penalty_urllc_scalar,
            "penalty_mmtc_scalar": penalty_mmtc_scalar,
            "penalty_total_scalar": penalty_total_scalar,
            "penalty_share_embb": penalty_embb_scalar / penalty_share_denom,
            "penalty_share_urllc": penalty_urllc_scalar / penalty_share_denom,
            "penalty_share_mmtc": penalty_mmtc_scalar / penalty_share_denom,
            "fairness_mean": float(np.nanmean(fairness)),
            "fairness_std": float(np.nanstd(fairness)),
            "inference_total_ms_mean": (
                float(np.mean(inf_total_ms_finite)) if inf_total_ms_finite.size > 0 else np.nan
            ),
            "inference_total_ms_std": (
                float(np.std(inf_total_ms_finite)) if inf_total_ms_finite.size > 0 else np.nan
            ),
            "inference_p95_total_ms_mean": (
                float(np.mean(inf_p95_total_ms_finite)) if inf_p95_total_ms_finite.size > 0 else np.nan
            ),
            "inference_p95_total_ms_std": (
                float(np.std(inf_p95_total_ms_finite)) if inf_p95_total_ms_finite.size > 0 else np.nan
            ),
            "inference_per_agent_ms_mean": (
                float(np.mean(inf_per_agent_ms_finite)) if inf_per_agent_ms_finite.size > 0 else np.nan
            ),
            "inference_per_agent_ms_std": (
                float(np.std(inf_per_agent_ms_finite)) if inf_per_agent_ms_finite.size > 0 else np.nan
            ),
            "sla_sys_mean": np.nanmean(sla_sys, axis=0),
            "sla_sys_std": np.nanstd(sla_sys, axis=0),
            "sla_bs0_mean": np.nanmean(sla_bs0, axis=0),
            "sla_bs0_std": np.nanstd(sla_bs0, axis=0),
        }

    return aggregated


def plot_with_band(ax, x_axis, mean, std, label, color):
    ax.plot(x_axis, mean, label=label, linewidth=2.0, color=color)
    ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.18, linewidth=0.0)


def run_baselines():
    validate_seed_split()
    print(f"Initializing IPPO checkpoints by training seed (min_iter={MIN_EVAL_CHECKPOINT_ITER})...")
    ippo_evaluators = init_ippo_evaluators()
    print("Loaded IPPO evaluators:")
    for evaluator in ippo_evaluators:
        print(
            f"  train_seed={evaluator['train_seed']}, "
            f"iter={evaluator['training_iteration']}, "
            f"obs_filter={evaluator['observation_filter']}, "
            f"quality={evaluator['quality_score']}, "
            f"checkpoint={evaluator['checkpoint_path']}"
        )

    results_by_algo = {key: [] for key, _ in ALGORITHMS}

    for seed in EVAL_SEEDS:
        print(f"\n=== Evaluation seed={seed} ===")
        for algo_key, algo_label in ALGORITHMS:
            if algo_key != "ippo":
                env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
                run = run_evaluation(env, algo_key, seed)
                results_by_algo[algo_key].append(run)

                print(
                    f"[{algo_label}] "
                    f"JFI={run['jfi_embb']:.4f}, "
                    f"sys_tp_mean={np.mean(run['system_throughput_mbps']):.2f} Mbps, "
                    f"mean_delay_ms={np.mean(run['urllc_delay_ms']):.3f}, "
                    f"cum_reward={run['cum_reward'][-1]:.3f}"
                )
                continue

            for evaluator in ippo_evaluators:
                env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
                run = run_evaluation(env, algo_key, seed, ippo_evaluator=evaluator)
                results_by_algo[algo_key].append(run)

                print(
                    f"[{algo_label}] "
                    f"train_seed={evaluator['train_seed']}, "
                    f"JFI={run['jfi_embb']:.4f}, "
                    f"sys_tp_mean={np.mean(run['system_throughput_mbps']):.2f} Mbps, "
                    f"mean_delay_ms={np.mean(run['urllc_delay_ms']):.3f}, "
                    f"cum_reward={run['cum_reward'][-1]:.3f}"
                )
                print(
                    f"[{algo_label}] "
                    f"train_seed={evaluator['train_seed']}, "
                    f"inference_total={run['mean_inference_total_ms']:.4f} ms, "
                    f"inference_per_agent={run['mean_inference_per_agent_ms']:.4f} ms, "
                    f"inference_p95_total={run['p95_inference_total_ms']:.4f} ms"
                )

    aggregated = aggregate_results(results_by_algo)

    print("\n=== Summary (mean ± std over recorded runs) ===")
    print(
        f"Heuristic baselines use {len(EVAL_SEEDS)} evaluation seeds each; "
        f"IPPO uses {len(ippo_evaluators)} training-seed checkpoints x {len(EVAL_SEEDS)} "
        f"evaluation seeds = {len(results_by_algo['ippo'])} runs."
    )
    for algo_key, algo_label in main_comparison_algorithms():
        stats = aggregated[algo_key]
        fairness_text = f"JFI={stats['fairness_mean']:.4f} ± {stats['fairness_std']:.4f}"
        sys_tp_text = f"SysTP={np.mean(stats['system_tp_mean']):.2f} Mbps"
        delay_text = f"Delay={np.mean(stats['delay_mean']):.3f} ms"
        reward_text = f"FinalCumReward={stats['cum_reward_mean'][-1]:.3f}"
        sla_sys_text = (
            "SLA_sys[eMBB/URLLC/mMTC]="
            f"{stats['sla_sys_mean'][0]*100:.1f}%/"
            f"{stats['sla_sys_mean'][1]*100:.1f}%/"
            f"{stats['sla_sys_mean'][2]*100:.1f}%"
        )
        penalty_text = (
            "Penalty[total/eMBB/URLLC/mMTC]="
            f"{stats['penalty_total_scalar']:.3f}/"
            f"{stats['penalty_embb_scalar']:.3f}/"
            f"{stats['penalty_urllc_scalar']:.3f}/"
            f"{stats['penalty_mmtc_scalar']:.3f}"
        )
        penalty_share_text = (
            "PenaltyShare[eMBB/URLLC/mMTC]="
            f"{stats['penalty_share_embb']*100:.1f}%/"
            f"{stats['penalty_share_urllc']*100:.1f}%/"
            f"{stats['penalty_share_mmtc']*100:.1f}%"
        )
        reward_base_text = (
            "BaseTPReward="
            f"{np.mean(stats['reward_base_tp_mean']):.3f}"
        )
        print(
            f"{algo_label:>20}: runs={stats['num_runs']}, "
            f"{fairness_text}, {sys_tp_text}, {delay_text}, {reward_text}"
        )
        print(f"{' ':>22}{sla_sys_text}")
        print(f"{' ':>22}{reward_base_text}, {penalty_text}")
        print(f"{' ':>22}{penalty_share_text}")
        if algo_key == "ippo":
            print(
                " " * 22
                + "Inference total="
                + f"{stats['inference_total_ms_mean']:.4f} ± {stats['inference_total_ms_std']:.4f} ms, "
                + "per-agent="
                + f"{stats['inference_per_agent_ms_mean']:.4f} ± {stats['inference_per_agent_ms_std']:.4f} ms, "
                + "p95(total)="
                + f"{stats['inference_p95_total_ms_mean']:.4f} ± {stats['inference_p95_total_ms_std']:.4f} ms "
                + "(TTI=0.5ms)"
            )

    x = np.arange(ROLLOUT_STEPS)
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    ax1 = axes[0, 0]
    for algo_key, algo_label in main_comparison_algorithms():
        stats = aggregated[algo_key]
        plot_with_band(
            ax1,
            x,
            stats["delay_mean"],
            stats["delay_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax1.axhline(y=2.0, color="black", linestyle="--", linewidth=1.5, label="SLA Deadline (2ms)")
    ax1.set_title("URLLC Delay (mean ± std)")
    ax1.set_xlabel("Time Step (TTI)")
    ax1.set_ylabel("Delay (ms)")
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[0, 1]
    for algo_key, algo_label in main_comparison_algorithms():
        stats = aggregated[algo_key]
        plot_with_band(
            ax2,
            x,
            stats["cum_reward_mean"],
            stats["cum_reward_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax2.set_title("Cumulative Reward (mean ± std, BS_0)")
    ax2.set_xlabel("Time Step (TTI)")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True)
    ax2.legend()

    ax3 = axes[1, 0]
    for algo_key, algo_label in main_comparison_algorithms():
        stats = aggregated[algo_key]
        plot_with_band(
            ax3,
            x,
            stats["shortfall_mean"],
            stats["shortfall_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax3.set_title("eMBB Shortfall (mean ± std, BS_0)")
    ax3.set_xlabel("Time Step (TTI)")
    ax3.set_ylabel("Shortfall (Mbps)")
    ax3.grid(True)
    ax3.legend()

    ax4 = axes[1, 1]
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        plot_with_band(
            ax4,
            x,
            stats["system_tp_mean"],
            stats["system_tp_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax4.set_title("System Throughput (mean ± std, 7 cells)")
    ax4.set_xlabel("Time Step (TTI)")
    ax4.set_ylabel("Throughput (Mbps)")
    ax4.grid(True)
    ax4.legend()

    ax5 = axes[2, 0]
    comparison_algos = main_comparison_algorithms()
    labels = [label for _, label in comparison_algos]
    fairness_means = [aggregated[key]["fairness_mean"] for key, _ in comparison_algos]
    fairness_stds = [aggregated[key]["fairness_std"] for key, _ in comparison_algos]
    bars = ax5.bar(labels, fairness_means, yerr=fairness_stds, capsize=4.0, color=[PLOT_COLORS[k] for k, _ in comparison_algos])
    ax5.set_title("Jain Fairness Index (eMBB Throughput)")
    ax5.set_ylabel("JFI")
    ax5.set_ylim(0.0, 1.05)
    ax5.grid(True, axis="y")
    ax5.tick_params(axis="x", rotation=20)

    ippo_stats = aggregated["ippo"]
    ax5.text(
        0.02,
        0.92,
        (
            "IPPO inference overhead:\n"
            f"total={ippo_stats['inference_total_ms_mean']:.4f} ms\n"
            f"per-agent={ippo_stats['inference_per_agent_ms_mean']:.4f} ms\n"
            f"p95(total)={ippo_stats['inference_p95_total_ms_mean']:.4f} ms"
        ),
        transform=ax5.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    for bar, score in zip(bars, fairness_means):
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax6 = axes[2, 1]
    x_idx = np.arange(len(comparison_algos))
    width = 0.25
    embb_sla = np.array([aggregated[key]["sla_sys_mean"][0] for key, _ in comparison_algos], dtype=np.float64)
    urllc_sla = np.array([aggregated[key]["sla_sys_mean"][1] for key, _ in comparison_algos], dtype=np.float64)
    mmtc_sla = np.array([aggregated[key]["sla_sys_mean"][2] for key, _ in comparison_algos], dtype=np.float64)

    ax6.bar(x_idx - width, embb_sla, width=width, label="eMBB SLA success", color="#1f77b4")
    ax6.bar(x_idx, urllc_sla, width=width, label="URLLC SLA success", color="#ff7f0e")
    ax6.bar(x_idx + width, mmtc_sla, width=width, label="mMTC SLA success", color="#2ca02c")
    ax6.set_xticks(x_idx, labels)
    ax6.set_ylim(0.0, 1.05)
    ax6.set_ylabel("Success Rate")
    ax6.set_title("System SLA Success Rate (mean over seeds)")
    ax6.tick_params(axis="x", rotation=20)
    ax6.grid(True, axis="y")
    ax6.legend()

    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    output_path = f"./results/marl_baseline_comparison_{ENV_PROFILE}_multiseed.png"
    plt.savefig(output_path)
    print(f"Saved comparison plots to {output_path}")

    stop_ippo_evaluators()
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    run_baselines()

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
    CENTRALIZED_CRITIC_GLOBAL_DIM,
    DEFAULT_INITIAL_ACTION_LOG_STD,
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
MIN_EVAL_CHECKPOINT_ITER = 50
LEARNED_SELECTION_MODE = "eval_sweep_system_total_sla"
BASE_PROFILE_OVERRIDES = MultiCell_5G_SLA_Env._get_env_profile_overrides(ENV_PROFILE)

LEARNED_METHODS = [
    {
        "algo_key": "ippo",
        "label": "IPPO (Baseline, pure_local)",
        "experiment_env_tag": "balanced_ippo_v5",
        "observation_mode": "pure_local",
        "use_centralized_critic": False,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.0,
        "neighbor_dividend_gamma": 0.0,
        "cooperative_target": "all",
        "neighbor_augmented_include_ici_features": False,
    },
    {
        "algo_key": "mappo",
        "label": "MAPPO (Proposed, CTDE)",
        "experiment_env_tag": "balanced_mappo_ctde_v5",
        "observation_mode": "neighbor_augmented",
        "use_centralized_critic": True,
        "cooperative_alpha": None,
        "neighbor_liability_beta": None,
        "neighbor_dividend_gamma": None,
        "cooperative_target": "all",
        "neighbor_augmented_include_ici_features": False,
    },
]
LEARNED_METHOD_BY_KEY = {item["algo_key"]: item for item in LEARNED_METHODS}

ALGORITHMS = [
    ("static", "Static SLA Split"),
    ("priority", "Priority Heuristic"),
    ("max_weight", "Max-Weight (Throughput-Oriented Heuristic)"),
    ("pf", "Proportional Fair"),
    ("ippo", LEARNED_METHOD_BY_KEY["ippo"]["label"]),
    ("mappo", LEARNED_METHOD_BY_KEY["mappo"]["label"]),
]

PLOT_COLORS = {
    "static": "#1f77b4",
    "priority": "#ff7f0e",
    "max_weight": "#2ca02c",
    "pf": "#d62728",
    "ippo": "#9467bd",
    "mappo": "#8c564b",
}

DELAY_PLOT_SCALE_EXCLUDE_ALGOS = {"max_weight"}
DELAY_PLOT_SCALE_PERCENTILE = 98.0
DELAY_PLOT_SCALE_MARGIN = 1.20
DELAY_PLOT_MIN_UPPER_MS = 2.5

BASE_ENV_CONFIG = {
    "env_profile": ENV_PROFILE,
    "action_softmax_temperature": float(BASE_PROFILE_OVERRIDES.get("action_softmax_temperature", 1.0)),
}

HEURISTIC_ENV_CONFIG = {
    **BASE_ENV_CONFIG,
    "observation_mode": "pure_local",
    "use_centralized_critic": False,
}

def _build_learned_env_config(method):
    env_cfg = {
        **BASE_ENV_CONFIG,
        "observation_mode": method["observation_mode"],
        "use_centralized_critic": method["use_centralized_critic"],
        "centralized_critic_global_dim": CENTRALIZED_CRITIC_GLOBAL_DIM,
    }
    if method["cooperative_alpha"] is not None:
        env_cfg["cooperative_alpha"] = method["cooperative_alpha"]
    if method["neighbor_liability_beta"] is not None:
        env_cfg["neighbor_liability_beta"] = method["neighbor_liability_beta"]
    if method["neighbor_dividend_gamma"] is not None:
        env_cfg["neighbor_dividend_gamma"] = method["neighbor_dividend_gamma"]
    env_cfg["cooperative_target"] = method.get("cooperative_target", "all")
    env_cfg["neighbor_augmented_include_ici_features"] = bool(
        method.get("neighbor_augmented_include_ici_features", False)
    )
    return env_cfg


LEARNED_ENV_CONFIGS = {method["algo_key"]: _build_learned_env_config(method) for method in LEARNED_METHODS}

LEARNED_EXPERIMENT_DIRS = {
    method["algo_key"]: [
        f"./ray_results/MAPPO_5G_Slicing_{method['experiment_env_tag']}_seed{seed}"
        for seed in TRAIN_SEEDS
    ]
    for method in LEARNED_METHODS
}

DEFAULT_OBSERVATION_FILTER = "MeanStdFilter"
LEARNED_EVALUATORS = {item["algo_key"]: [] for item in LEARNED_METHODS}


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


def build_learned_eval_algo(observation_filter: str, env_config: dict):
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=EVAL_SEEDS[0])
        .rl_module(
            rl_module_spec=build_initialized_rl_module_spec(
                env_config["action_softmax_temperature"],
                fcnet_hiddens=[256, 256],
                fcnet_activation="tanh",
                initial_action_log_std=DEFAULT_INITIAL_ACTION_LOG_STD,
                observation_mode=env_config["observation_mode"],
                include_ici_features=bool(env_config.get("neighbor_augmented_include_ici_features", False)),
                use_centralized_critic=bool(env_config.get("use_centralized_critic", False)),
                critic_global_dim=int(env_config.get("centralized_critic_global_dim", CENTRALIZED_CRITIC_GLOBAL_DIM)),
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


def evaluate_learned_checkpoint_candidate(method, candidate):
    algo_key = method["algo_key"]
    env_config = LEARNED_ENV_CONFIGS[algo_key]
    observation_filter = _resolve_trial_observation_filter(candidate["trial_dir"])
    algo = build_learned_eval_algo(observation_filter=observation_filter, env_config=env_config)
    algo.restore(candidate["checkpoint_path"])

    evaluator = {
        "algo_key": algo_key,
        "label": method["label"],
        "algo": algo,
        "env_config": dict(env_config),
        "train_seed": candidate.get("train_seed", TRAIN_SEEDS[0]),
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
    for seed in EVAL_SEEDS:
        env = MultiCell_5G_SLA_Env(config=evaluator["env_config"])
        runs.append(run_evaluation(env, algo_key, seed, learned_evaluator=evaluator))

    aggregated_algo = aggregate_results({algo_key: runs})[algo_key]
    summary = summarize_eval_stats(aggregated_algo)
    return evaluator, runs, summary


def stop_learned_evaluators():
    global LEARNED_EVALUATORS

    for evaluators in LEARNED_EVALUATORS.values():
        for evaluator in evaluators:
            evaluator["algo"].stop()
    LEARNED_EVALUATORS = {item["algo_key"]: [] for item in LEARNED_METHODS}


def init_learned_evaluators():
    global LEARNED_EVALUATORS

    stop_learned_evaluators()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    evaluators_by_algo = {item["algo_key"]: [] for item in LEARNED_METHODS}
    restore_errors = []

    for method in LEARNED_METHODS:
        algo_key = method["algo_key"]
        experiment_dirs = LEARNED_EXPERIMENT_DIRS[algo_key]
        env_config = LEARNED_ENV_CONFIGS[algo_key]
        method_label = method["label"]
        ranked_checkpoints = rank_checkpoints_by_metric(
            experiment_dirs,
            min_training_iteration=MIN_EVAL_CHECKPOINT_ITER,
            fallback_to_any=False,
        )
        if not ranked_checkpoints:
            restore_errors.append(
                f"{method_label}, experiment_dirs={experiment_dirs} -> "
                f"no ranked checkpoint found with iter>={MIN_EVAL_CHECKPOINT_ITER}"
            )
            continue

        best = None
        print(f"\n--- Eval sweep {method_label} ({len(ranked_checkpoints)} candidates) ---")
        for item in ranked_checkpoints:
            checkpoint_path = item["checkpoint_path"]
            score = item.get("episode_return_mean")
            iteration = item.get("training_iteration")
            total_sla_violation = item.get("center_total_sla_violations")
            quality_score = item.get("quality_score")
            system_throughput_mbps = item.get("system_throughput_mbps")
            train_seed = item.get("train_seed")
            print(
                f"Trying {method_label} checkpoint: {checkpoint_path} "
                f"(train_seed={train_seed}, iter={iteration}, "
                f"log_center_total_viol={total_sla_violation}, log_system_tp={system_throughput_mbps}, "
                f"quality={quality_score}, episode_return_mean={score})"
            )
            try:
                evaluator, _, summary = evaluate_learned_checkpoint_candidate(method, item)
            except Exception as exc:  # noqa: PERF203
                restore_errors.append(
                    f"{method_label}, checkpoint={checkpoint_path}, train_seed={train_seed} -> {exc}"
                )
                continue

            print(
                f"  eval_total_viol={summary['system_total_sla_violations']:.4f}, "
                f"eval_bs0_total_viol={summary['bs0_total_sla_violations']:.4f}, "
                f"eval_sys_tp={summary['system_tp_mean_mbps']:.2f} Mbps, "
                f"eval_sla_sys[eMBB]={summary['sla_sys_mean'][0]*100:.2f}%, "
                f"eval_sla_bs0[eMBB]={summary['sla_bs0_mean'][0]*100:.2f}%"
            )

            rank_key = candidate_eval_key(summary, iteration)
            if best is None or rank_key < best["rank_key"]:
                if best is not None:
                    best["evaluator"]["algo"].stop()
                best = {
                    "evaluator": evaluator,
                    "summary": summary,
                    "rank_key": rank_key,
                }
            else:
                evaluator["algo"].stop()

        if best is None:
            print(f"Warning: no restorable checkpoint found for {method_label}.")
            continue

        best_evaluator = best["evaluator"]
        best_summary = best["summary"]
        evaluators_by_algo[algo_key].append(best_evaluator)
        print(
            f"Loaded {method_label} checkpoint via {LEARNED_SELECTION_MODE}: "
            f"{best_evaluator['checkpoint_path']} "
            f"(train_seed={best_evaluator['train_seed']}, iter={best_evaluator['training_iteration']}, "
            f"obs_filter={best_evaluator['observation_filter']}, "
            f"eval_total_viol={best_summary['system_total_sla_violations']:.4f}, "
            f"eval_sys_tp={best_summary['system_tp_mean_mbps']:.2f} Mbps, "
            f"eval_sla_sys[eMBB]={best_summary['sla_sys_mean'][0]*100:.2f}%, "
            f"eval_sla_bs0[eMBB]={best_summary['sla_bs0_mean'][0]*100:.2f}%)"
        )

    missing_methods = [m["label"] for m in LEARNED_METHODS if not evaluators_by_algo[m["algo_key"]]]
    if missing_methods:
        error_preview = "\n".join(restore_errors[:8])
        raise RuntimeError(
            "Missing learned evaluators for: "
            + ", ".join(missing_methods)
            + "\nSample restore errors:\n"
            + error_preview
        )

    LEARNED_EVALUATORS = evaluators_by_algo
    return LEARNED_EVALUATORS


def run_evaluation(env, algo_name, seed, learned_evaluator=None):
    if algo_name in LEARNED_METHOD_BY_KEY and learned_evaluator is None:
        raise RuntimeError(f"{algo_name} evaluator is not initialized. Call init_learned_evaluators() first.")

    obs, reset_infos = env.reset(seed=seed)
    done = {"__all__": False}
    learned_runner = None
    learned_episode = None
    learned_shared_data = None

    if algo_name in LEARNED_METHOD_BY_KEY:
        learned_runner, learned_episode, learned_shared_data = build_ippo_episode_context(
            learned_evaluator["algo"], obs, reset_infos
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
            # Archive-style single-agent heuristic, ported to the current
            # multi-cell benchmark using the same effective Softmax ratios.
            # The legacy heuristic switched on obs[4] (URLLC queue feature),
            # so we intentionally read the pure_local observation instead of
            # the raw env queue directly to preserve the old decision rule.
            normal_ratios = np.array([0.6621907, 0.29754147, 0.04026786], dtype=np.float32)
            emergency_ratios = np.array([0.4435102, 0.4901546, 0.0663352], dtype=np.float32)
            for agent in env.agents:
                agent_obs = obs[agent]
                urllc_queue_feature = float(agent_obs[4])
                if urllc_queue_feature > 0.005:
                    actions[agent] = ratios_to_action(emergency_ratios, env.action_softmax_temperature)
                else:
                    actions[agent] = ratios_to_action(normal_ratios, env.action_softmax_temperature)

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

        elif algo_name in LEARNED_METHOD_BY_KEY:
            start = time.perf_counter()
            policy_actions, actions, extra_model_outputs = compute_actions_batched(
                learned_evaluator["algo"],
                learned_runner,
                learned_episode,
                learned_shared_data,
            )
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            inference_overhead_total_ms.append(elapsed_ms)
            inference_overhead_per_agent_ms.append(elapsed_ms / max(len(obs), 1))
        else:
            extra_model_outputs = None

        if algo_name not in {"static", "priority", "max_weight", "pf", *LEARNED_METHOD_BY_KEY.keys()}:
            raise ValueError(f"Unknown algorithm name: {algo_name}")

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        if algo_name in LEARNED_METHOD_BY_KEY:
            learned_episode.add_env_step(
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
        elif env.reward_mode == "archive_local_sla":
            throughput_reward_weight = 1.0 / 100.0
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
    if algo_name in LEARNED_METHOD_BY_KEY:
        result["train_seed"] = int(learned_evaluator["train_seed"])
        result["checkpoint_path"] = str(learned_evaluator["checkpoint_path"])
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
    print(
        f"Initializing learned checkpoints via {LEARNED_SELECTION_MODE} "
        f"(min_iter={MIN_EVAL_CHECKPOINT_ITER})..."
    )
    learned_evaluators_by_algo = init_learned_evaluators()
    for method in LEARNED_METHODS:
        algo_key = method["algo_key"]
        method_label = method["label"]
        print(f"Loaded {method_label} evaluators:")
        for evaluator in learned_evaluators_by_algo[algo_key]:
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
            if algo_key not in LEARNED_METHOD_BY_KEY:
                env = MultiCell_5G_SLA_Env(config=HEURISTIC_ENV_CONFIG)
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

            for evaluator in learned_evaluators_by_algo[algo_key]:
                env = MultiCell_5G_SLA_Env(config=evaluator["env_config"])
                run = run_evaluation(env, algo_key, seed, learned_evaluator=evaluator)
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
    learned_run_text = "; ".join(
        [
            f"{m['label']} uses {len(learned_evaluators_by_algo[m['algo_key']])} training-seed checkpoints "
            f"x {len(EVAL_SEEDS)} evaluation seeds = {len(results_by_algo[m['algo_key']])} runs"
            for m in LEARNED_METHODS
        ]
    )
    print(f"Heuristic baselines use {len(EVAL_SEEDS)} evaluation seeds each; {learned_run_text}.")
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
        sla_bs0_text = (
            "SLA_bs0[eMBB/URLLC/mMTC]="
            f"{stats['sla_bs0_mean'][0]*100:.1f}%/"
            f"{stats['sla_bs0_mean'][1]*100:.1f}%/"
            f"{stats['sla_bs0_mean'][2]*100:.1f}%"
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
        print(f"{' ':>22}{sla_bs0_text}")
        print(f"{' ':>22}{reward_base_text}, {penalty_text}")
        print(f"{' ':>22}{penalty_share_text}")
        if algo_key in LEARNED_METHOD_BY_KEY:
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
    comparison_algos = main_comparison_algorithms()
    labels = [label for _, label in comparison_algos]
    x_idx = np.arange(len(comparison_algos))
    os.makedirs("./results", exist_ok=True)

    # Figure 1: URLLC delay timeline.
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for algo_key, algo_label in comparison_algos:
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
    delay_scale_keys = [k for k, _ in comparison_algos if k not in DELAY_PLOT_SCALE_EXCLUDE_ALGOS]
    if not delay_scale_keys:
        delay_scale_keys = [k for k, _ in comparison_algos]
    delay_scale_values = np.concatenate(
        [np.asarray(aggregated[k]["delay_mean"], dtype=np.float64).ravel() for k in delay_scale_keys]
    )
    delay_scale_q = float(np.nanpercentile(delay_scale_values, DELAY_PLOT_SCALE_PERCENTILE))
    delay_ylim_upper = max(
        DELAY_PLOT_MIN_UPPER_MS,
        2.05,
        delay_scale_q * DELAY_PLOT_SCALE_MARGIN,
    )
    ax1.set_ylim(0.0, delay_ylim_upper)
    clipped_algos = [
        algo_label
        for algo_key, algo_label in comparison_algos
        if float(np.nanmax(np.asarray(aggregated[algo_key]["delay_mean"], dtype=np.float64))) > delay_ylim_upper
    ]
    if clipped_algos:
        clipped_text = ", ".join(clipped_algos)
        ax1.text(
            0.99,
            0.02,
            f"Clipped above {delay_ylim_upper:.2f} ms: {clipped_text}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="dimgray",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    delay_path = f"./results/marl_baseline_delay_{ENV_PROFILE}_multiseed.png"
    fig1.savefig(delay_path)
    plt.close(fig1)

    # Figure 2: Cumulative reward timeline (exclude MAPPO line by request).
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for algo_key, algo_label in comparison_algos:
        if algo_key == "mappo":
            continue
        stats = aggregated[algo_key]
        plot_with_band(
            ax2,
            x,
            stats["cum_reward_mean"],
            stats["cum_reward_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax2.set_title("Cumulative Reward (mean ± std, BS_0, MAPPO Hidden)")
    ax2.set_xlabel("Time Step (TTI)")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    reward_path = f"./results/marl_baseline_cum_reward_{ENV_PROFILE}_multiseed.png"
    fig2.savefig(reward_path)
    plt.close(fig2)

    # Figure 3: Average system throughput bar chart.
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sys_tp_means = np.array([np.mean(aggregated[key]["system_tp_mean"]) for key, _ in comparison_algos], dtype=np.float64)
    sys_tp_stds = np.array([np.std(aggregated[key]["system_tp_mean"]) for key, _ in comparison_algos], dtype=np.float64)
    ax3.bar(
        x_idx,
        sys_tp_means,
        yerr=sys_tp_stds,
        capsize=4.0,
        color=[PLOT_COLORS[key] for key, _ in comparison_algos],
    )
    ax3.set_xticks(x_idx, labels, rotation=20)
    ax3.set_title("Average System Throughput (7 cells)")
    ax3.set_ylabel("Mbps")
    ax3.grid(True, axis="y")
    fig3.tight_layout()
    throughput_path = f"./results/marl_baseline_system_throughput_{ENV_PROFILE}_multiseed.png"
    fig3.savefig(throughput_path)
    plt.close(fig3)

    # Figure 4: Fairness + shortfall table.
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.axis("off")
    table_rows = []
    for algo_key, algo_label in comparison_algos:
        stats = aggregated[algo_key]
        table_rows.append(
            [
                algo_label,
                f"{stats['fairness_mean']:.4f}",
                f"{np.mean(stats['shortfall_mean']):.2f}",
            ]
        )
    table = ax4.table(
        cellText=table_rows,
        colLabels=["Algorithm", "Jain Fairness (eMBB)", "Mean eMBB Shortfall (Mbps)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax4.set_title("Fairness + eMBB Shortfall Summary", pad=12)
    fig4.tight_layout()
    table_path = f"./results/marl_baseline_fairness_shortfall_{ENV_PROFILE}_multiseed.png"
    fig4.savefig(table_path)
    plt.close(fig4)

    # Figure 5: SLA success grouped bars for all algorithms (system-level).
    fig5, ax5_sys = plt.subplots(1, 1, figsize=(12, 6))
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width], dtype=np.float64)
    slice_names = ["eMBB", "URLLC", "mMTC"]
    slice_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx_slice, slice_name in enumerate(slice_names):
        vals_sys = np.array([aggregated[k]["sla_sys_mean"][idx_slice] for k, _ in comparison_algos], dtype=np.float64)
        std_sys = np.array([aggregated[k]["sla_sys_std"][idx_slice] for k, _ in comparison_algos], dtype=np.float64)
        x_pos = x_idx + offsets[idx_slice]
        ax5_sys.bar(
            x_pos,
            vals_sys,
            yerr=std_sys,
            capsize=3.0,
            width=bar_width,
            color=slice_colors[idx_slice],
            alpha=0.85,
            label=slice_name,
        )

    ax5_sys.set_ylim(0.0, 1.05)
    ax5_sys.set_ylabel("Success Rate")
    ax5_sys.set_title("SLA Success Rate by Algorithm (System-Level)")
    ax5_sys.grid(True, axis="y")
    ax5_sys.set_xticks(x_idx, labels, rotation=20)
    ax5_sys.legend(loc="lower right")
    fig5.tight_layout()
    sla_system_path = f"./results/marl_baseline_sla_success_system_{ENV_PROFILE}_multiseed.png"
    fig5.savefig(sla_system_path)
    plt.close(fig5)

    # Figure 6: BS_0 SLA success, IPPO vs MAPPO only.
    bs0_pair_algos = [(k, v) for k, v in comparison_algos if k in ("ippo", "mappo")]
    if len(bs0_pair_algos) != 2:
        raise RuntimeError("Expected both IPPO and MAPPO in comparison algorithms for BS_0 SLA figure.")
    bs0_labels = [label for _, label in bs0_pair_algos]
    bs0_x = np.arange(len(bs0_pair_algos))

    fig6, ax6 = plt.subplots(1, 1, figsize=(9, 6))
    for idx_slice, slice_name in enumerate(slice_names):
        vals_bs0 = np.array([aggregated[k]["sla_bs0_mean"][idx_slice] for k, _ in bs0_pair_algos], dtype=np.float64)
        std_bs0 = np.array([aggregated[k]["sla_bs0_std"][idx_slice] for k, _ in bs0_pair_algos], dtype=np.float64)
        x_pos = bs0_x + offsets[idx_slice]
        ax6.bar(
            x_pos,
            vals_bs0,
            yerr=std_bs0,
            capsize=3.0,
            width=bar_width,
            color=slice_colors[idx_slice],
            alpha=0.88,
            label=slice_name,
        )
    ax6.set_ylim(0.0, 1.05)
    ax6.set_ylabel("Success Rate")
    ax6.set_title("BS_0 SLA Success: IPPO vs MAPPO")
    ax6.set_xticks(bs0_x, bs0_labels, rotation=10)
    ax6.grid(True, axis="y")
    ax6.legend(loc="lower right")
    fig6.tight_layout()
    bs0_sla_pair_path = f"./results/marl_baseline_bs0_ippo_mappo_sla_{ENV_PROFILE}_multiseed.png"
    fig6.savefig(bs0_sla_pair_path)
    plt.close(fig6)

    print("Saved comparison plots:")
    print(f"  - {delay_path}")
    print(f"  - {reward_path}")
    print(f"  - {throughput_path}")
    print(f"  - {table_path}")
    print(f"  - {sla_system_path}")
    print(f"  - {bs0_sla_pair_path}")

    stop_learned_evaluators()
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    run_baselines()

import argparse
import logging
import math
import os
import warnings

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

from ippo_rl_module import (
    DEFAULT_INITIAL_ACTION_LOG_STD,
    DEFAULT_INITIAL_SLICE_RATIOS,
    build_initialized_rl_module_spec,
    ratios_to_raw_action_means,
)
from multi_cell_env import MultiCell_5G_SLA_Env

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_memory_usage_threshold"] = "1.0"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

logging.getLogger("ray").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

TRAIN_SEEDS = [2026, 2027, 2028]
FULL_DEFAULT_SEEDS = [2026]
TRAIN_ITERS = 200
QUICK_TRAIN_SEEDS = [2026]
QUICK_TRAIN_ITERS = 60
DEFAULT_ENV_PROFILE = "balanced"
DEFAULT_OBSERVATION_MODE = "pure_local"
EXPERIMENT_PREFIX = "MAPPO_5G_Slicing"
EXPERIMENT_ENV_TAGS = {
    ("harsh", "pure_local"): "harsh_ippo_v1",
    ("balanced", "pure_local"): "balanced_ippo_v1",
    ("harsh", "neighbor_augmented"): "harsh_neighbor_v4",
    ("balanced", "neighbor_augmented"): "balanced_neighbor_v6",
}
ENV_CONFIG = {
    "env_profile": DEFAULT_ENV_PROFILE,
    "observation_mode": DEFAULT_OBSERVATION_MODE,
    "action_softmax_temperature": 3.0,
    # P1 tail-risk shaping: lighter blanket pressure, stronger punishment near/after URLLC deadline.
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

# PPO stability knobs for continuous-control MARL.
PPO_LR_START = 5e-5
PPO_LR_END = 5e-6
PPO_ENTROPY_START = 0.01
PPO_ENTROPY_END = 5e-4
PPO_TRAIN_BATCH = 2400
PPO_MINI_BATCH = 512
PPO_NUM_SGD_ITER = 6
PPO_GRAD_CLIP = 0.5
QUICK_PPO_TRAIN_BATCH = 1200
QUICK_PPO_MINI_BATCH = 256
QUICK_PPO_NUM_SGD_ITER = 3
QUICK_ROLLOUT_FRAGMENT_LENGTH = 100
INITIAL_ACTION_RATIOS = DEFAULT_INITIAL_SLICE_RATIOS.tolist()
INITIAL_ACTION_LOG_STD = DEFAULT_INITIAL_ACTION_LOG_STD


# Register environment
def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)

class SLACallbacks(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        module = algorithm.get_module("center_policy")
        model_cfg = module.config.model_config_dict or {}
        init_ratios = model_cfg.get("initial_action_ratios", INITIAL_ACTION_RATIOS)
        init_log_std = float(model_cfg.get("initial_action_log_std", INITIAL_ACTION_LOG_STD))
        target_means = ratios_to_raw_action_means(
            init_ratios,
            model_cfg.get("action_softmax_temperature", ENV_CONFIG["action_softmax_temperature"]),
        )
        print(
            "Initialized PPO actor prior: "
            f"target_ratios={list(init_ratios)}, "
            f"raw_action_means={np.round(target_means, 6).tolist()}, "
            f"initial_log_std={init_log_std}"
        )

    def on_episode_step(self, **kwargs):
        episode = kwargs.get("episode")
        if episode is None:
            return

        # New stack (env_runner) path.
        info = None
        infos = None
        if hasattr(episode, "get_infos"):
            infos = episode.get_infos(-1)
            if isinstance(infos, dict):
                info = infos.get("BS_0")
        # Old stack fallback.
        elif hasattr(episode, "last_info_for"):
            info = episode.last_info_for("BS_0")

        if not info:
            return

        metrics_logger = kwargs.get("metrics_logger")
        has_episode_custom_metrics = hasattr(episode, "custom_metrics")

        def log_custom_metric(metric_name: str, metric_value: float):
            metric_value = float(metric_value)
            if metrics_logger is not None and hasattr(metrics_logger, "log_value"):
                metrics_logger.log_value(("custom_metrics", metric_name), metric_value)
            if has_episode_custom_metrics:
                episode.custom_metrics[metric_name] = metric_value

        if "est_urllc_delay" in info:
            log_custom_metric("center_urllc_delay_ms", info["est_urllc_delay"] * 1000.0)
        if "violations" in info:
            log_custom_metric("center_urllc_violations", info["violations"][1])
            log_custom_metric("center_embb_violations", info["violations"][0])
            log_custom_metric("center_mmtc_violations", info["violations"][2])
            log_custom_metric("center_total_sla_violations", float(np.sum(np.asarray(info["violations"]))))
            log_custom_metric("center_embb_sla_ok", float(info["violations"][0] <= 0.0))
            log_custom_metric("center_urllc_sla_ok", float(info["violations"][1] <= 0.0))
            log_custom_metric("center_mmtc_sla_ok", float(info["violations"][2] <= 0.0))
        if "violations_raw" in info:
            log_custom_metric("center_urllc_violations_raw", info["violations_raw"][1])
            log_custom_metric("center_embb_violations_raw", info["violations_raw"][0])
            log_custom_metric("center_mmtc_violations_raw", info["violations_raw"][2])

        for metric_name in (
            "reward_base_tp",
            "reward_sla_bonus",
            "local_reward",
            "local_reward_unclipped",
            "role_reward_scale",
            "penalty_total",
            "penalty_embb",
            "penalty_urllc",
            "penalty_mmtc",
            "penalty_raw_embb",
            "penalty_raw_urllc",
            "penalty_raw_mmtc",
        ):
            if metric_name in info:
                log_custom_metric(f"center_{metric_name}", info[metric_name])

        if (
            "penalty_total" in info
            and "penalty_embb" in info
            and "penalty_urllc" in info
            and "penalty_mmtc" in info
        ):
            penalty_total = max(float(info["penalty_total"]), 1e-8)
            log_custom_metric("center_penalty_share_embb", float(info["penalty_embb"]) / penalty_total)
            log_custom_metric("center_penalty_share_urllc", float(info["penalty_urllc"]) / penalty_total)
            log_custom_metric("center_penalty_share_mmtc", float(info["penalty_mmtc"]) / penalty_total)

        if isinstance(infos, dict):
            system_throughput_mbps = 0.0
            has_system_tp = False
            for agent_info in infos.values():
                if isinstance(agent_info, dict) and "throughput" in agent_info:
                    system_throughput_mbps += float(agent_info["throughput"])
                    has_system_tp = True
            if has_system_tp:
                log_custom_metric("system_throughput_mbps", system_throughput_mbps)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def resolve_hw_profile(hw_profile: str, mode: str):
    cpu_total = os.cpu_count() or 8
    if mode == "quick":
        # Fast iteration profile for early-stage debugging/tuning.
        num_env_runners = max(2, min(4, cpu_total - 8))
        num_envs_per_env_runner = 1
        rollout_fragment_length = QUICK_ROLLOUT_FRAGMENT_LENGTH
        train_batch_floor = QUICK_PPO_TRAIN_BATCH
        mini_batch_size_per_learner = QUICK_PPO_MINI_BATCH
        num_sgd_iter = QUICK_PPO_NUM_SGD_ITER
        observation_filter = "MeanStdFilter"
        return {
            "cpu_total": cpu_total,
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_env_runner,
            "rollout_fragment_length": rollout_fragment_length,
            "train_batch_floor": train_batch_floor,
            "mini_batch_size_per_learner": mini_batch_size_per_learner,
            "num_sgd_iter": num_sgd_iter,
            "observation_filter": observation_filter,
        }

    # RLlib new env_runner stack currently does not support:
    # multi-agent + num_envs_per_env_runner > 1.
    # We therefore scale with more env_runners and fragment length.
    if hw_profile == "maxperf":
        num_env_runners = max(2, min(12, cpu_total - 2))
        num_envs_per_env_runner = 1
        # Keep fragment geometry compatible with maxperf train-batch targets.
        rollout_fragment_length = 400
        train_batch_floor = 4800
        mini_batch_size_per_learner = 1024
        num_sgd_iter = 8
        observation_filter = "MeanStdFilter"
    else:
        num_env_runners = max(2, min(6, cpu_total - 6))
        num_envs_per_env_runner = 1
        rollout_fragment_length = 200
        train_batch_floor = PPO_TRAIN_BATCH
        mini_batch_size_per_learner = PPO_MINI_BATCH
        num_sgd_iter = PPO_NUM_SGD_ITER
        observation_filter = "MeanStdFilter"

    return {
        "cpu_total": cpu_total,
        "num_env_runners": num_env_runners,
        "num_envs_per_env_runner": num_envs_per_env_runner,
        "rollout_fragment_length": rollout_fragment_length,
        "train_batch_floor": train_batch_floor,
        "mini_batch_size_per_learner": mini_batch_size_per_learner,
        "num_sgd_iter": num_sgd_iter,
        "observation_filter": observation_filter,
    }


def _resolve_train_batch_per_learner(hw_cfg):
    num_env_runners = int(hw_cfg["num_env_runners"])
    num_envs_per_env_runner = int(hw_cfg["num_envs_per_env_runner"])
    rollout_fragment_length = hw_cfg["rollout_fragment_length"]
    train_batch_floor = int(hw_cfg["train_batch_floor"])

    if not isinstance(rollout_fragment_length, int) or rollout_fragment_length <= 0:
        return train_batch_floor

    fragment_bundle = num_env_runners * num_envs_per_env_runner * rollout_fragment_length
    if fragment_bundle <= 0:
        return train_batch_floor

    target = max(train_batch_floor, fragment_bundle)
    nearest_multiple = max(1, round(target / fragment_bundle)) * fragment_bundle
    nearest_gap_ratio = abs(nearest_multiple - target) / max(float(target), 1.0)

    # Align to rollout geometry. Prefer nearest if within RLlib's 10% tolerance.
    if nearest_gap_ratio <= 0.10:
        return int(nearest_multiple)

    return int(math.ceil(target / fragment_bundle) * fragment_bundle)


def build_config(seed, num_learners, num_gpus_per_learner, hw_cfg, env_config):
    train_batch_per_learner = _resolve_train_batch_per_learner(hw_cfg)

    return (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=seed)
        .callbacks(SLACallbacks)
        .rl_module(
            rl_module_spec=build_initialized_rl_module_spec(
                env_config["action_softmax_temperature"],
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu",
                initial_action_ratios=INITIAL_ACTION_RATIOS,
                initial_action_log_std=INITIAL_ACTION_LOG_STD,
            )
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            lr=PPO_LR_START,
            lr_schedule=[[0, PPO_LR_START], [500000, PPO_LR_END]],
            vf_loss_coeff=0.5,
            clip_param=0.1,
            entropy_coeff=[[0, PPO_ENTROPY_START], [500000, PPO_ENTROPY_END]],
            train_batch_size_per_learner=train_batch_per_learner,
            mini_batch_size_per_learner=hw_cfg["mini_batch_size_per_learner"],
            num_sgd_iter=hw_cfg["num_sgd_iter"],
            grad_clip=PPO_GRAD_CLIP,
        )
        .env_runners(
            observation_filter=hw_cfg["observation_filter"],
            num_env_runners=hw_cfg["num_env_runners"],
            num_envs_per_env_runner=hw_cfg["num_envs_per_env_runner"],
            rollout_fragment_length=hw_cfg["rollout_fragment_length"],
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .multi_agent(
            # Heterogeneous topology: center and edge use two shared policies.
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
    )


def get_experiment_env_tag(env_profile: str, observation_mode: str) -> str:
    key = (env_profile, observation_mode)
    if key not in EXPERIMENT_ENV_TAGS:
        raise ValueError(
            "Unsupported experiment tagging combination: "
            f"env_profile={env_profile!r}, observation_mode={observation_mode!r}"
        )
    return EXPERIMENT_ENV_TAGS[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MARL with quick/full profiles. "
        "Use quick during early debugging, full for final reporting."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick: 1 seed + fewer iterations; full: default 1 seed + full iterations.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional manual seed list. Overrides --mode seed preset.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Optional manual iteration count. Overrides --mode preset.",
    )
    parser.add_argument(
        "--hw-profile",
        choices=["balanced", "maxperf"],
        default="balanced",
        help=(
            "Hardware utilization preset. "
            "Ignored when --mode quick (quick mode uses fast-iteration defaults)."
        ),
    )
    parser.add_argument(
        "--env-profile",
        choices=["harsh", "balanced"],
        default=DEFAULT_ENV_PROFILE,
        help="Environment profile. balanced targets near-feasible 3-SLA training.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=["pure_local", "neighbor_augmented"],
        default=DEFAULT_OBSERVATION_MODE,
        help="Observation mode. pure_local is the fully local IPPO baseline.",
    )
    return parser.parse_args()


def resolve_train_plan(args):
    if args.seeds is not None and len(args.seeds) > 0:
        # Keep input order but remove duplicates.
        seeds = list(dict.fromkeys(args.seeds))
    elif args.mode == "full":
        seeds = FULL_DEFAULT_SEEDS
    else:
        seeds = QUICK_TRAIN_SEEDS

    if args.iters is not None and args.iters > 0:
        train_iters = args.iters
    elif args.mode == "full":
        train_iters = TRAIN_ITERS
    else:
        train_iters = QUICK_TRAIN_ITERS

    return seeds, train_iters


def main():
    args = parse_args()
    seeds, train_iters = resolve_train_plan(args)
    hw_cfg = resolve_hw_profile(args.hw_profile, args.mode)

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    use_cuda = torch.cuda.is_available()
    num_learners = 1
    num_gpus_per_learner = 1 if use_cuda else 0

    print(
        "Training plan: "
        f"mode={args.mode}, seeds={seeds}, train_iters={train_iters}, "
        f"use_cuda={use_cuda}, num_gpus_per_learner={num_gpus_per_learner}, "
        f"hw_profile={args.hw_profile}, cpu_total={hw_cfg['cpu_total']}, "
        f"num_env_runners={hw_cfg['num_env_runners']}, "
        f"num_envs_per_env_runner={hw_cfg['num_envs_per_env_runner']}, "
        f"rollout_fragment_length={hw_cfg['rollout_fragment_length']}, "
        f"train_batch_floor={hw_cfg['train_batch_floor']}, "
        f"mini_batch={hw_cfg['mini_batch_size_per_learner']}, "
        f"num_sgd_iter={hw_cfg['num_sgd_iter']}, "
        f"observation_filter={hw_cfg['observation_filter']}"
    )

    # Use Tune for multi-seed training orchestration
    env_config = dict(ENV_CONFIG)
    env_config["env_profile"] = args.env_profile
    env_config["observation_mode"] = args.observation_mode
    experiment_env_tag = get_experiment_env_tag(args.env_profile, args.observation_mode)
    for seed in seeds:
        config = build_config(seed, num_learners, num_gpus_per_learner, hw_cfg, env_config)
        experiment_name = f"{EXPERIMENT_PREFIX}_{experiment_env_tag}_seed{seed}"
        print(
            "Starting MARL training "
            f"(seed={seed}, new_api_stack=True, num_learners={num_learners}, "
            f"num_gpus_per_learner={num_gpus_per_learner}, env_profile={args.env_profile}, "
            f"observation_mode={args.observation_mode}, "
            f"experiment_env_tag={experiment_env_tag}, "
            f"name={experiment_name})..."
        )
        tune.run(
            "PPO",
            name=experiment_name,
            stop={"training_iteration": train_iters},
            config=config.to_dict(),
            checkpoint_freq=50,
            checkpoint_at_end=True,
            storage_path=os.path.abspath("./ray_results"),
            reuse_actors=True,
        )
    
    print("Training completed. Check ./ray_results for logs and checkpoints.")
    ray.shutdown()
    
if __name__ == "__main__":
    main()

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
    CENTRALIZED_CRITIC_GLOBAL_DIM,
    DEFAULT_INITIAL_ACTION_LOG_STD,
    build_initialized_rl_module_spec,
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
DEFAULT_MAPPO_COOPERATIVE_ALPHA = 0.5
DEFAULT_NEIGHBOR_LIABILITY_BETA = 0.5
DEFAULT_NEIGHBOR_DIVIDEND_GAMMA = 1.0
EXPERIMENT_PREFIX = "MAPPO_5G_Slicing"
EXPERIMENT_ENV_TAGS = {
    ("harsh", "pure_local"): "harsh_ippo_v2",
    ("balanced", "pure_local"): "balanced_ippo_v5",
    ("harsh", "neighbor_augmented"): "harsh_neighbor_v5",
    ("balanced", "neighbor_augmented"): "balanced_neighbor_v7",
}
EXPERIMENT_CTDE_ENV_TAGS = {
    ("harsh", "neighbor_augmented"): "harsh_mappo_ctde_v3",
    ("balanced", "neighbor_augmented"): "balanced_mappo_ctde_v5",
}
ENV_CONFIG = {
    "env_profile": DEFAULT_ENV_PROFILE,
    "observation_mode": DEFAULT_OBSERVATION_MODE,
    "use_centralized_critic": False,
    "centralized_critic_global_dim": CENTRALIZED_CRITIC_GLOBAL_DIM,
    "action_softmax_temperature": 1.0,
}

# PPO stability knobs for continuous-control MARL.
PPO_LR_START = 1e-5
PPO_LR_END = 1e-6
PPO_ENTROPY_START = 0.01
PPO_ENTROPY_END = 5e-4
PPO_TRAIN_BATCH = 2400
PPO_MINI_BATCH = 512
PPO_NUM_SGD_ITER = 2
PPO_GRAD_CLIP = 0.1
QUICK_PPO_TRAIN_BATCH = 1200
QUICK_PPO_MINI_BATCH = 256
QUICK_PPO_NUM_SGD_ITER = 2
QUICK_ROLLOUT_FRAGMENT_LENGTH = 100
INITIAL_ACTION_LOG_STD = DEFAULT_INITIAL_ACTION_LOG_STD


# Register environment
def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)

class SLACallbacks(DefaultCallbacks):
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
            "local_reward_for_coop",
            "local_penalty_total",
            "local_net_gain",
            "neighbor_only_penalty_total",
            "neighbor_only_penalty_mean",
            "neighbor_count",
            "neighborhood_penalty_total",
            "neighborhood_penalty_mean",
            "neighborhood_size",
            "cooperative_alpha",
            "cooperative_beta",
            "reward_final",
            "reward_local_component",
            "reward_neighbor_component",
            "neighbor_coop_term",
            "neighbor_coop_weight_sum",
            "neighbor_coop_weighted_risk_sum",
            "neighbor_coop_risk_mean",
            "neighbor_penalty_signal",
            "neighbor_prev_penalty_signal",
            "neighbor_penalty_improve",
            "neighbor_throughput_signal",
            "neighbor_prev_throughput_signal",
            "neighbor_throughput_improve",
            "neighbor_dividend_gate",
            "neighbor_dividend_raw",
            "neighbor_dividend",
            "reward_dividend_component",
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
            system_violations = []
            for agent_info in infos.values():
                if not isinstance(agent_info, dict):
                    continue
                if "throughput" in agent_info:
                    system_throughput_mbps += float(agent_info["throughput"])
                    has_system_tp = True
                if "violations" in agent_info:
                    system_violations.append(np.asarray(agent_info["violations"], dtype=np.float32))
            if has_system_tp:
                log_custom_metric("system_throughput_mbps", system_throughput_mbps)
            if system_violations:
                system_violations = np.asarray(system_violations, dtype=np.float32)
                system_slice_success = np.mean(system_violations <= 0.0, axis=0)
                system_slice_violation = np.mean(system_violations, axis=0)
                log_custom_metric("system_embb_sla_ok", system_slice_success[0])
                log_custom_metric("system_urllc_sla_ok", system_slice_success[1])
                log_custom_metric("system_mmtc_sla_ok", system_slice_success[2])
                log_custom_metric("system_embb_violations", system_slice_violation[0])
                log_custom_metric("system_urllc_violations", system_slice_violation[1])
                log_custom_metric("system_mmtc_violations", system_slice_violation[2])
                log_custom_metric(
                    "system_total_sla_violations",
                    float(np.sum(1.0 - system_slice_success)),
                )


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
                fcnet_activation="tanh",
                initial_action_log_std=INITIAL_ACTION_LOG_STD,
                observation_mode=env_config["observation_mode"],
                include_ici_features=bool(env_config.get("neighbor_augmented_include_ici_features", False)),
                use_centralized_critic=bool(env_config.get("use_centralized_critic", False)),
                critic_global_dim=int(env_config.get("centralized_critic_global_dim", CENTRALIZED_CRITIC_GLOBAL_DIM)),
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


def get_experiment_env_tag(env_profile: str, observation_mode: str, use_centralized_critic: bool) -> str:
    key = (env_profile, observation_mode)
    if use_centralized_critic:
        if key not in EXPERIMENT_CTDE_ENV_TAGS:
            raise ValueError(
                "CTDE requires neighbor_augmented observation mode with a supported env profile. "
                f"Got env_profile={env_profile!r}, observation_mode={observation_mode!r}"
            )
        return EXPERIMENT_CTDE_ENV_TAGS[key]

    if key not in EXPERIMENT_ENV_TAGS:
        raise ValueError(
            "Unsupported experiment tagging combination: "
            f"env_profile={env_profile!r}, observation_mode={observation_mode!r}"
        )
    return EXPERIMENT_ENV_TAGS[key]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MARL with streamlined presets. "
        "Use quick for debugging and full for formal runs."
    )
    parser.add_argument(
        "--algo",
        choices=["ippo", "mappo"],
        default="mappo",
        help=(
            "Algorithm preset. "
            "ippo -> pure_local + local reward; "
            "mappo -> neighbor_augmented + centralized critic + cooperative reward."
        ),
    )
    # Backward-compatible alias for older commands.
    parser.add_argument("--algo-mode", dest="algo", choices=["ippo", "mappo"], help=argparse.SUPPRESS)
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
        "--mappo-variant",
        choices=["current", "beta_gamma", "coop_embb", "coop_embb_ici"],
        default="current",
        help=(
            "MAPPO experiment preset. "
            "current: current baseline; "
            "beta_gamma: alpha=1,beta=0.12,gamma=0.8; "
            "coop_embb: cooperation only tracks neighbor eMBB; "
            "coop_embb_ici: eMBB-only cooperation + beta/gamma + direct ICI features."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override cooperative alpha for MAPPO only. IPPO ignores this value.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override neighbor joint-liability beta for MAPPO only. IPPO always uses beta=0.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override neighbor dividend gamma for MAPPO only. IPPO always uses gamma=0.",
    )
    parser.add_argument(
        "--coop-target",
        choices=["all", "embb_only"],
        default=None,
        help="Override cooperative target for MAPPO only. all keeps current behavior.",
    )
    parser.add_argument(
        "--neighbor-ici-features",
        action="store_true",
        help="Enable direct same-slice ICI features in neighbor_augmented observation.",
    )
    # Backward-compatible aliases for older commands.
    parser.add_argument(
        "--mappo-cooperative-alpha",
        dest="alpha",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--neighbor-liability-beta",
        dest="beta",
        type=float,
        help=argparse.SUPPRESS,
    )
    # Deprecated knobs kept only for compatibility and ignored by new preset logic.
    parser.add_argument("--observation-mode", choices=["pure_local", "neighbor_augmented"], help=argparse.SUPPRESS)
    parser.add_argument("--use-centralized-critic", action="store_true", help=argparse.SUPPRESS)
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


def resolve_algorithm_mode(args):
    if args.observation_mode is not None or bool(args.use_centralized_critic):
        print(
            "Deprecated flags --observation-mode/--use-centralized-critic are ignored. "
            "Topology is now derived from --algo preset."
        )

    if args.algo == "mappo":
        variant_defaults = {
            "current": {
                "alpha": None,
                "beta": None,
                "gamma": None,
                "coop_target": "all",
                "include_ici_features": False,
                "tag_suffix": "",
            },
            "beta_gamma": {
                "alpha": 1.0,
                "beta": 0.12,
                "gamma": 0.8,
                "coop_target": "all",
                "include_ici_features": False,
                "tag_suffix": "_bg",
            },
            "coop_embb": {
                "alpha": None,
                "beta": None,
                "gamma": None,
                "coop_target": "embb_only",
                "include_ici_features": False,
                "tag_suffix": "_coop_embb",
            },
            "coop_embb_ici": {
                "alpha": 1.0,
                "beta": 0.12,
                "gamma": 0.8,
                "coop_target": "embb_only",
                "include_ici_features": True,
                "tag_suffix": "_coop_embb_ici",
            },
        }
        preset = variant_defaults[args.mappo_variant]
        cooperative_alpha = preset["alpha"] if args.alpha is None else float(args.alpha)
        neighbor_liability_beta = preset["beta"] if args.beta is None else float(args.beta)
        neighbor_dividend_gamma = preset["gamma"] if args.gamma is None else float(args.gamma)
        cooperative_target = preset["coop_target"] if args.coop_target is None else str(args.coop_target)
        include_ici_features = bool(preset["include_ici_features"] or args.neighbor_ici_features)
        if cooperative_alpha is not None and not 0.0 <= cooperative_alpha <= 1.0:
            raise ValueError(f"--alpha must be in [0, 1]. Got {cooperative_alpha}.")
        if neighbor_liability_beta is not None and neighbor_liability_beta < 0.0:
            raise ValueError(f"--beta must be >= 0. Got {neighbor_liability_beta}.")
        if neighbor_dividend_gamma is not None and neighbor_dividend_gamma < 0.0:
            raise ValueError(f"--gamma must be >= 0. Got {neighbor_dividend_gamma}.")
        return {
            "observation_mode": "neighbor_augmented",
            "use_centralized_critic": True,
            "cooperative_alpha": cooperative_alpha,
            "neighbor_liability_beta": neighbor_liability_beta,
            "neighbor_dividend_gamma": neighbor_dividend_gamma,
            "cooperative_target": cooperative_target,
            "include_ici_features": include_ici_features,
            "tag_suffix": preset["tag_suffix"],
        }

    if args.alpha is not None or args.beta is not None or args.gamma is not None:
        print("algo=ippo ignores --alpha/--beta/--gamma; enforced alpha=1.0, beta=0.0, gamma=0.0.")
    return {
        "observation_mode": "pure_local",
        "use_centralized_critic": False,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.0,
        "neighbor_dividend_gamma": 0.0,
        "cooperative_target": "all",
        "include_ici_features": False,
        "tag_suffix": "",
    }


def main():
    args = parse_args()
    seeds, train_iters = resolve_train_plan(args)
    resolved_mode = resolve_algorithm_mode(args)
    resolved_observation_mode = resolved_mode["observation_mode"]
    resolved_use_centralized_critic = resolved_mode["use_centralized_critic"]
    resolved_cooperative_alpha = resolved_mode["cooperative_alpha"]
    resolved_neighbor_liability_beta = resolved_mode["neighbor_liability_beta"]
    resolved_neighbor_dividend_gamma = resolved_mode["neighbor_dividend_gamma"]
    resolved_cooperative_target = resolved_mode["cooperative_target"]
    resolved_include_ici_features = resolved_mode["include_ici_features"]
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
        f"observation_filter={hw_cfg['observation_filter']}, "
        f"algo={args.algo}, "
        f"mappo_variant={args.mappo_variant if args.algo == 'mappo' else 'n/a'}, "
        f"observation_mode={resolved_observation_mode}, "
        f"neighbor_augmented_include_ici_features={resolved_include_ici_features}, "
        f"use_centralized_critic={resolved_use_centralized_critic}, "
        f"cooperative_alpha={resolved_cooperative_alpha if resolved_cooperative_alpha is not None else 'profile'}, "
        f"neighbor_liability_beta={resolved_neighbor_liability_beta if resolved_neighbor_liability_beta is not None else 'profile'}, "
        f"neighbor_dividend_gamma={resolved_neighbor_dividend_gamma if resolved_neighbor_dividend_gamma is not None else 'profile'}, "
        f"cooperative_target={resolved_cooperative_target}"
    )

    # Use Tune for multi-seed training orchestration
    env_config = dict(ENV_CONFIG)
    env_config["env_profile"] = args.env_profile
    profile_overrides = MultiCell_5G_SLA_Env._get_env_profile_overrides(args.env_profile)
    env_config["action_softmax_temperature"] = float(
        profile_overrides.get("action_softmax_temperature", env_config["action_softmax_temperature"])
    )
    env_config["observation_mode"] = resolved_observation_mode
    env_config["neighbor_augmented_include_ici_features"] = resolved_include_ici_features
    env_config["use_centralized_critic"] = resolved_use_centralized_critic
    env_config["cooperative_target"] = resolved_cooperative_target
    if resolved_cooperative_alpha is not None:
        env_config["cooperative_alpha"] = float(resolved_cooperative_alpha)
    if resolved_neighbor_liability_beta is not None:
        env_config["neighbor_liability_beta"] = float(resolved_neighbor_liability_beta)
    if resolved_neighbor_dividend_gamma is not None:
        env_config["neighbor_dividend_gamma"] = float(resolved_neighbor_dividend_gamma)
    experiment_env_tag = get_experiment_env_tag(
        args.env_profile,
        resolved_observation_mode,
        resolved_use_centralized_critic,
    ) + resolved_mode["tag_suffix"]
    for seed in seeds:
        config = build_config(seed, num_learners, num_gpus_per_learner, hw_cfg, env_config)
        experiment_name = f"{EXPERIMENT_PREFIX}_{experiment_env_tag}_seed{seed}"
        print(
            "Starting MARL training "
            f"(seed={seed}, new_api_stack=True, num_learners={num_learners}, "
            f"num_gpus_per_learner={num_gpus_per_learner}, env_profile={args.env_profile}, "
            f"algo={args.algo}, "
            f"mappo_variant={args.mappo_variant if args.algo == 'mappo' else 'n/a'}, "
            f"observation_mode={resolved_observation_mode}, "
            f"neighbor_augmented_include_ici_features={resolved_include_ici_features}, "
            f"use_centralized_critic={resolved_use_centralized_critic}, "
            f"cooperative_alpha={resolved_cooperative_alpha if resolved_cooperative_alpha is not None else 'profile'}, "
            f"neighbor_liability_beta={resolved_neighbor_liability_beta if resolved_neighbor_liability_beta is not None else 'profile'}, "
            f"neighbor_dividend_gamma={resolved_neighbor_dividend_gamma if resolved_neighbor_dividend_gamma is not None else 'profile'}, "
            f"cooperative_target={resolved_cooperative_target}, "
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

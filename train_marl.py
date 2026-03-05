import argparse
import logging
import os
import warnings

import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

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
EXPERIMENT_PREFIX = "MAPPO_5G_Slicing_seed"
ENV_CONFIG = {
    # Rebalanced penalties: avoid over-killing URLLC and strengthen eMBB shortfall pain.
    "penalty_weight": 0.7,
    "urllc_warning_ratio": 0.65,
    "urllc_softplus_slope": 12.0,
    "urllc_warning_gain": 1.0,
    "urllc_overflow_gain": 6.0,
    "urllc_exp_coeff": 2.5,
    "urllc_penalty_cap_factor": 20.0,
    "embb_penalty_quad_gain": 1.2,
    "embb_penalty_cap_factor": 10.0,
    # Symmetric ICI settings.
    "ici_gain": 0.65,
    "se_modifier_floor": 0.3,
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
        if metrics_logger is not None and hasattr(metrics_logger, "log_value"):
            if "est_urllc_delay" in info:
                metrics_logger.log_value(
                    ("custom_metrics", "center_urllc_delay_ms"),
                    float(info["est_urllc_delay"] * 1000.0),
                )
            if "violations" in info:
                metrics_logger.log_value(
                    ("custom_metrics", "center_urllc_violations"),
                    float(info["violations"][1]),
                )
                metrics_logger.log_value(
                    ("custom_metrics", "center_embb_violations"),
                    float(info["violations"][0]),
                )

        if hasattr(episode, "custom_metrics"):
            if "est_urllc_delay" in info:
                episode.custom_metrics["center_urllc_delay_ms"] = info["est_urllc_delay"] * 1000.0
            if "violations" in info:
                episode.custom_metrics["center_urllc_violations"] = info["violations"][1]
                episode.custom_metrics["center_embb_violations"] = info["violations"][0]


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
        observation_filter = "NoFilter"
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
        rollout_fragment_length = 300
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


def build_config(seed, num_learners, num_gpus_per_learner, hw_cfg):
    auto_train_batch = (
        hw_cfg["num_env_runners"]
        * hw_cfg["num_envs_per_env_runner"]
        * hw_cfg["rollout_fragment_length"]
    )
    train_batch_per_learner = max(hw_cfg["train_batch_floor"], auto_train_batch)

    return (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=ENV_CONFIG)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=seed)
        .callbacks(SLACallbacks)
        .rl_module(model_config_dict={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
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
    for seed in seeds:
        config = build_config(seed, num_learners, num_gpus_per_learner, hw_cfg)
        experiment_name = f"{EXPERIMENT_PREFIX}{seed}"
        print(
            "Starting MARL training "
            f"(seed={seed}, new_api_stack=True, num_learners={num_learners}, "
            f"num_gpus_per_learner={num_gpus_per_learner}, name={experiment_name})..."
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

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
TRAIN_ITERS = 200
EXPERIMENT_PREFIX = "MAPPO_5G_Slicing_seed"
ENV_CONFIG = {
    # Stronger URLLC soft-cliff.
    "penalty_weight": 0.7,
    "urllc_warning_ratio": 0.5,
    "urllc_softplus_slope": 16.0,
    "urllc_warning_gain": 1.5,
    "urllc_overflow_gain": 10.0,
    "urllc_exp_coeff": 4.0,
    "urllc_penalty_cap_factor": 40.0,
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


def build_config(seed, num_learners, num_gpus_per_learner):
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
            train_batch_size_per_learner=PPO_TRAIN_BATCH,
            mini_batch_size_per_learner=PPO_MINI_BATCH,
            num_sgd_iter=PPO_NUM_SGD_ITER,
            grad_clip=PPO_GRAD_CLIP,
        )
        .env_runners(
            observation_filter="MeanStdFilter",
            num_env_runners=6,
            num_envs_per_env_runner=1,
            rollout_fragment_length=200,
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


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    use_cuda = torch.cuda.is_available()
    num_learners = 1
    num_gpus_per_learner = 1 if use_cuda else 0

    # Use Tune for multi-seed training orchestration
    for seed in TRAIN_SEEDS:
        config = build_config(seed, num_learners, num_gpus_per_learner)
        experiment_name = f"{EXPERIMENT_PREFIX}{seed}"
        print(
            "Starting MARL training "
            f"(seed={seed}, new_api_stack=True, num_learners={num_learners}, "
            f"num_gpus_per_learner={num_gpus_per_learner}, name={experiment_name})..."
        )
        tune.run(
            "PPO",
            name=experiment_name,
            stop={"training_iteration": TRAIN_ITERS},
            config=config.to_dict(),
            checkpoint_freq=50,
            checkpoint_at_end=True,
            storage_path=os.path.abspath("./ray_results"),
        )
    
    print("Training completed. Check ./ray_results for logs and checkpoints.")
    ray.shutdown()
    
if __name__ == "__main__":
    main()

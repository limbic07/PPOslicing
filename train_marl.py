import logging
import sys
import os
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
logging.getLogger('ray').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['RAY_memory_usage_threshold'] = '1.0'
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env
from multi_cell_env import MultiCell_5G_SLA_Env

# Register environment
def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)

register_env("MultiCell_5G_SLA_Env", env_creator)


from ray.rllib.algorithms.callbacks import DefaultCallbacks

class SLACallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Retrieve the latest info dictionary from the last step
        info = episode.last_info_for("BS_0")
        if info:
            # We add custom metrics which Ray will automatically log to Tensorboard
            # 'custom_metrics' prefix is handled by Ray Tune
            if "est_urllc_delay" in info:
                episode.custom_metrics["center_urllc_delay_ms"] = info["est_urllc_delay"] * 1000.0
            if "violations" in info:
                episode.custom_metrics["center_urllc_violations"] = info["violations"][1]
                episode.custom_metrics["center_embb_violations"] = info["violations"][0]


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Configuration for Multi-Agent PPO (MAPPO) with parameter sharing
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env")
        # Ensure we have the right neural net structure for the 14-dim input
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .learners(
            num_learners=1,
            num_gpus_per_learner=1
        )
        .callbacks(SLACallbacks)
        .training(
            gamma=0.99,
            lr=5e-5,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
            num_sgd_iter=10,
        )
        
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1,
            rollout_fragment_length=200,
        )
        .multi_agent(
            # Parameter Sharing Setup
            # All 7 base stations share the same policy (Neural Network)
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .resources(num_gpus=1) # Enable GPU if available
    )

    # Use Tune for training orchestration
    print("Starting MARL training...")
    results = tune.run(
        "PPO",
        name="MAPPO_5G_Slicing",
        stop={"training_iteration": 30},  # Train for 500 iterations
        config=config.to_dict(),
        checkpoint_freq=1,                # Checkpoint every 50 iterations
        checkpoint_at_end=True,            # Save model at the end
        storage_path=os.path.abspath("./ray_results"),
    )
    
    print("Training completed. Check ./ray_results for logs and checkpoints.")
    
if __name__ == "__main__":
    main()

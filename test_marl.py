import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.core import Columns
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from checkpoint_utils import rank_checkpoints_by_metric
from multi_cell_env import MultiCell_5G_SLA_Env

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_memory_usage_threshold"] = "1.0"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

logging.getLogger("ray").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

EVAL_SEED = 2026
TRAIN_SEEDS = [2026, 2027, 2028]
EXPERIMENT_DIRS = [f"./ray_results/MAPPO_5G_Slicing_seed{seed}" for seed in TRAIN_SEEDS]
ENV_CONFIG = {
    "penalty_weight": 0.7,
    "urllc_warning_ratio": 0.65,
    "urllc_softplus_slope": 12.0,
    "urllc_warning_gain": 1.0,
    "urllc_overflow_gain": 6.0,
    "urllc_exp_coeff": 2.5,
    "urllc_penalty_cap_factor": 20.0,
    "embb_penalty_quad_gain": 1.2,
    "embb_penalty_cap_factor": 10.0,
    "ici_gain": 0.65,
    "se_modifier_floor": 0.3,
}

def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)

register_env("MultiCell_5G_SLA_Env", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def compute_action_new_stack(algo, obs: np.ndarray, policy_id: str) -> np.ndarray:
    module = algo.get_module(policy_id)
    obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        module_out = module.forward_inference({Columns.OBS: obs_batch})
        dist_cls = module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS]).to_deterministic()
        action = action_dist.sample()[0].cpu().numpy().astype(np.float32)

    return np.clip(action, -1.0, 1.0)

def run_test():
    ray.init(ignore_reinit_error=True)
    
    # Needs to match training config
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=ENV_CONFIG)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=EVAL_SEED)
        .rl_module(model_config_dict={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
        .multi_agent(
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(observation_filter="MeanStdFilter", num_env_runners=0)
        .learners(num_learners=0)
    )
    
    algo = config.build()
    ranked_checkpoints = rank_checkpoints_by_metric(EXPERIMENT_DIRS)
    if not ranked_checkpoints:
        raise FileNotFoundError(
            "No ranked checkpoints found in MAPPO seed experiment dirs. "
            "Please run train_marl.py first."
        )

    restore_errors = []
    restored_checkpoint = None
    for item in ranked_checkpoints:
        checkpoint_path = item["checkpoint_path"]
        score = item.get("episode_return_mean")
        iteration = item.get("training_iteration")
        urllc_violation = item.get("center_urllc_violations")
        urllc_delay_ms = item.get("center_urllc_delay_ms")
        print(
            f"Trying checkpoint: {checkpoint_path} "
            f"(iter={iteration}, urllc_viol={urllc_violation}, "
            f"urllc_delay_ms={urllc_delay_ms}, episode_return_mean={score})"
        )
        try:
            algo.restore(checkpoint_path)
            restored_checkpoint = checkpoint_path
            print(
                f"Loaded best available checkpoint: {checkpoint_path} "
                f"(iter={iteration}, urllc_viol={urllc_violation}, "
                f"urllc_delay_ms={urllc_delay_ms}, episode_return_mean={score})"
            )
            break
        except Exception as exc:  # noqa: PERF203
            restore_errors.append(f"{checkpoint_path} -> {exc}")

    if restored_checkpoint is None:
        error_preview = "\n".join(restore_errors[:3])
        raise RuntimeError(
            "Failed to restore any ranked checkpoint.\n"
            f"Sample restore errors:\n{error_preview}"
        )

    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    obs, _ = env.reset(seed=EVAL_SEED)
    
    rewards_history = {agent: [] for agent in env.agents}
    throughput_history = {agent: [] for agent in env.agents}
    urllc_delay_history = {agent: [] for agent in env.agents}
    
    done = {"__all__": False}
    step = 0
    
    print("Running evaluation...")
    while not done["__all__"] and step < 200:
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            actions[agent_id] = compute_action_new_stack(algo, agent_obs, policy_id=policy_id)

            
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        for agent in env.agents:
            if agent in rewards:
                rewards_history[agent].append(rewards[agent])
                throughput_history[agent].append(infos[agent]["throughput"])
                urllc_delay_history[agent].append(infos[agent]["est_urllc_delay"] * 1000) # ms
                
        done = terminateds
        step += 1

    print("Plotting results...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Center Cell (BS_0) vs Avg Edge Cell Throughput
    plt.subplot(2, 2, 1)
    plt.plot(throughput_history["BS_0"], label="Center Cell (BS_0) Throughput")
    avg_edge_throughput = np.mean([throughput_history[f"BS_{i}"] for i in range(1, 7)], axis=0)
    plt.plot(avg_edge_throughput, label="Avg Edge Cell Throughput")
    plt.title("Throughput Comparison (Center vs Edge)")
    plt.xlabel("Step")
    plt.ylabel("Throughput (Mbps)")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Center Cell URLLC Delay
    plt.subplot(2, 2, 2)
    plt.plot(urllc_delay_history["BS_0"], label="BS_0 URLLC Delay", color='red')
    plt.axhline(y=2.0, color='black', linestyle='--', label='2ms Deadline')
    plt.title("Center Cell URLLC Delay")
    plt.xlabel("Step")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Cumulative Rewards
    plt.subplot(2, 2, 3)
    plt.plot(np.cumsum(rewards_history["BS_0"]), label="BS_0 Reward")
    avg_edge_reward = np.mean([np.cumsum(rewards_history[f"BS_{i}"]) for i in range(1, 7)], axis=0)
    plt.plot(avg_edge_reward, label="Avg Edge Reward")
    plt.title("Cumulative Cooperative Rewards")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("./results/marl_evaluation.png")
    print("Results saved to ./results/marl_evaluation.png")
    algo.stop()
    ray.shutdown()
    
if __name__ == "__main__":
    run_test()

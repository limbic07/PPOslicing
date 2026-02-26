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
from multi_cell_env import MultiCell_5G_SLA_Env
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import numpy as np

def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)

register_env("MultiCell_5G_SLA_Env", env_creator)

def run_test():
    ray.init(ignore_reinit_error=True)
    
    # Needs to match training config
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env")
        .framework("torch")\
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)\
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
    )
    
    # In a real scenario we'd restore a checkpoint. Since we're just testing the script, we'll run an untrained model.
    # algo = config.build()
    # algo.restore("path/to/checkpoint")
    algo = config.build()

    env = MultiCell_5G_SLA_Env()
    obs, info = env.reset()
    
    rewards_history = {agent: [] for agent in env.agents}
    throughput_history = {agent: [] for agent in env.agents}
    urllc_delay_history = {agent: [] for agent in env.agents}
    
    done = {"__all__": False}
    step = 0
    
    print("Running evaluation...")
    while not done["__all__"] and step < 200:
        actions = {}
        for agent_id, agent_obs in obs.items():
            # Get action from shared policy
            # RLlib 2.34 standard inference path
            action_out = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False
            )
            actions[agent_id] = action_out

            
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
    
    plt.tight_layout()
    plt.savefig("./results/marl_evaluation.png")
    print("Results saved to ./results/marl_evaluation.png")
    
if __name__ == "__main__":
    run_test()

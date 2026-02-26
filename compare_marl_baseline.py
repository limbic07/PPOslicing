import ray
from ray.rllib.algorithms.ppo import PPOConfig
from multi_cell_env import MultiCell_5G_SLA_Env
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import numpy as np

def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)

register_env("MultiCell_5G_SLA_Env", env_creator)

def run_baselines():
    env = MultiCell_5G_SLA_Env()
    
    # We will test two baselines and the PPO model
    # Baseline 1: Static Allocation (Proportional Fair / Equal Split) -> [0.33, 0.33, 0.34]
    # Baseline 2: Heuristic / Priority Based -> Always gives more to URLLC when queue is > 0, else eMBB gets it.

    # 1. Run Static Baseline
    obs, _ = env.reset(seed=42)
    static_rewards = []
    static_urllc_delay = []
    static_embb_shortfall = []
    
    for _ in range(200):
        actions = {agent: np.array([0.0, 0.0, 0.0]) for agent in env.agents} # mapped to [0.33, 0.33, 0.34]
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        static_rewards.append(rewards["BS_0"])
        static_urllc_delay.append(infos["BS_0"]["est_urllc_delay"] * 1000)
        static_embb_shortfall.append(env.state["BS_0"][13])
        if terminateds["__all__"]: break

    # 2. Run Priority Heuristic Baseline
    obs, _ = env.reset(seed=42) # Use same seed for fair comparison
    heuristic_rewards = []
    heuristic_urllc_delay = []
    heuristic_embb_shortfall = []
    
    for _ in range(200):
        actions = {}
        for agent in env.agents:
            # If URLLC queue is accumulating (meaning a burst is happening)
            if env.queues[agent][1] > 0.05:  # Reacts immediately to ANY queue buildup
                actions[agent] = np.array([-0.8, 1.0, -0.8]) # Max out URLLC, starve others
            else:
                actions[agent] = np.array([0.8, -0.5, -0.8]) # Prioritize eMBB normally
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        heuristic_rewards.append(rewards["BS_0"])
        heuristic_urllc_delay.append(infos["BS_0"]["est_urllc_delay"] * 1000)
        heuristic_embb_shortfall.append(env.state["BS_0"][13])
        if terminateds["__all__"]: break

    # 3. Run Trained PPO Model
    ray.init(ignore_reinit_error=True)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import logging
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    logging.getLogger('ray').setLevel(logging.ERROR)
    
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env")
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
    )
    
    algo = config.build()
    
    # Try to find the latest checkpoint in ray_results
    import glob
    checkpoints = glob.glob('./ray_results/MAPPO_5G_Slicing/*/checkpoint_*')
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading MAPPO model from: {latest_checkpoint}")
        try:
            algo.restore(latest_checkpoint)
        except Exception as e:
            print(f"Could not load checkpoint due to API mismatch (common with Ray new/old stack changes). Falling back to untrained/partially trained initialization for graphing. Error: {e}")
    else:
        print("No checkpoint found in ./ray_results/. Using initialized model.")

    obs, _ = env.reset(seed=42)
    ppo_rewards = []
    ppo_urllc_delay = []
    ppo_embb_shortfall = []
    
    for _ in range(200):
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = algo.compute_single_action(
                agent_obs,
                policy_id="shared_policy",
                explore=False
            )
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        ppo_rewards.append(rewards["BS_0"])
        ppo_urllc_delay.append(infos["BS_0"]["est_urllc_delay"] * 1000)
        ppo_embb_shortfall.append(env.state["BS_0"][13])
        if terminateds["__all__"]: break

    # --- Plotting ---
    print("Generating baseline comparison plots...")
    plt.figure(figsize=(18, 12))
    
    # 1. URLLC Delay Comparison
    plt.subplot(2, 2, 1)
    plt.plot(static_urllc_delay, label='Static Allocation', alpha=0.7)
    plt.plot(heuristic_urllc_delay, label='Priority Heuristic', alpha=0.7)
    plt.plot(ppo_urllc_delay, label='MAPPO (Proposed)', linewidth=2.5, color='green')
    plt.axhline(y=2.0, color='r', linestyle='--', label='SLA Deadline (2ms)')
    plt.title('URLLC Delay over Time (BS_0)')
    plt.xlabel('Time Step (TTI)')
    plt.ylabel('Delay (ms)')
    plt.legend()
    plt.grid(True)
    
    # 2. Cumulative Reward Comparison
    plt.subplot(2, 2, 2)
    plt.plot(np.cumsum(static_rewards), label='Static Allocation')
    plt.plot(np.cumsum(heuristic_rewards), label='Priority Heuristic')
    plt.plot(np.cumsum(ppo_rewards), label='MAPPO (Proposed)', linewidth=2.5, color='green')
    plt.title('Cumulative System Reward (BS_0)')
    plt.xlabel('Time Step (TTI)')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    
    # 3. eMBB Throughput Shortfall
    plt.subplot(2, 2, 3)
    plt.plot(static_embb_shortfall, label='Static Allocation')
    plt.plot(heuristic_embb_shortfall, label='Priority Heuristic')
    plt.plot(ppo_embb_shortfall, label='MAPPO (Proposed)', linewidth=2.5, color='green')
    plt.title('eMBB SLA Shortfall (Lower is better)')
    plt.xlabel('Time Step (TTI)')
    plt.ylabel('Shortfall (Mbps)')
    plt.legend()
    plt.grid(True)
    
    # 4. Boxplot for Delay Violations (Quantitative)
    plt.subplot(2, 2, 4)
    data = [
        [d for d in static_urllc_delay if d > 2.0],
        [d for d in heuristic_urllc_delay if d > 2.0],
        [d for d in ppo_urllc_delay if d > 2.0]
    ]
    plt.boxplot(data, labels=['Static', 'Heuristic', 'MAPPO'])
    plt.title('Magnitude of URLLC SLA Violations (>2ms)')
    plt.ylabel('Delay Spike (ms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/marl_baseline_comparison_harsh.png')
    print("Saved comparison plots to ./results/marl_baseline_comparison_harsh.png")

if __name__ == "__main__":
    run_baselines()

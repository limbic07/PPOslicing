import os

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.core import Columns
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from checkpoint_utils import rank_checkpoints_by_metric
from multi_cell_env import MultiCell_5G_SLA_Env

EVAL_SEED = 2026
ROLLOUT_STEPS = 200
TRAIN_SEEDS = [2026, 2027, 2028]
MAPPO_EXPERIMENT_DIRS = [f"./ray_results/MAPPO_5G_Slicing_seed{seed}" for seed in TRAIN_SEEDS]
ENV_CONFIG = {
    "penalty_weight": 0.7,
    "urllc_warning_ratio": 0.5,
    "urllc_softplus_slope": 16.0,
    "urllc_warning_gain": 1.5,
    "urllc_overflow_gain": 10.0,
    "urllc_exp_coeff": 4.0,
    "urllc_penalty_cap_factor": 40.0,
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

def run_baselines():
    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    
    # We will test two baselines and the PPO model
    # Baseline 1: Static Allocation (Proportional Fair / Equal Split) -> [0.33, 0.33, 0.34]
    # Baseline 2: Heuristic / Priority Based -> Always gives more to URLLC when queue is > 0, else eMBB gets it.

    # 1. Run Static Baseline
    obs, _ = env.reset(seed=EVAL_SEED)
    static_rewards = []
    static_urllc_delay = []
    static_embb_shortfall = []
    
    for _ in range(ROLLOUT_STEPS):
        actions = {agent: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent in env.agents} # mapped to [0.33, 0.33, 0.34]
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        static_rewards.append(rewards["BS_0"])
        static_urllc_delay.append(infos["BS_0"]["est_urllc_delay"] * 1000)
        static_embb_shortfall.append(env.state["BS_0"][13])
        if terminateds["__all__"]: break

    # 2. Run Priority Heuristic Baseline
    obs, _ = env.reset(seed=EVAL_SEED) # Use same seed for fair comparison
    heuristic_rewards = []
    heuristic_urllc_delay = []
    heuristic_embb_shortfall = []
    
    for _ in range(ROLLOUT_STEPS):
        actions = {}
        for agent in env.agents:
            # If URLLC queue is accumulating (meaning a burst is happening)
            if env.queues[agent][1] > 0.05:  # Reacts immediately to ANY queue buildup
                actions[agent] = np.array([-0.8, 1.0, -0.8], dtype=np.float32) # Max out URLLC, starve others
            else:
                actions[agent] = np.array([0.8, -0.5, -0.8], dtype=np.float32) # Prioritize eMBB normally
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        heuristic_rewards.append(rewards["BS_0"])
        heuristic_urllc_delay.append(infos["BS_0"]["est_urllc_delay"] * 1000)
        heuristic_embb_shortfall.append(env.state["BS_0"][13])
        if terminateds["__all__"]: break

    # 3. Run Trained PPO Model
    ray.init(ignore_reinit_error=True)
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
    
    # Must load a trained checkpoint; fail fast to avoid invalid comparisons.
    ranked_checkpoints = rank_checkpoints_by_metric(MAPPO_EXPERIMENT_DIRS)
    if not ranked_checkpoints:
        raise FileNotFoundError(
            "No ranked checkpoint found under MAPPO seed experiment dirs. "
            "Please run train_marl.py first."
        )

    restore_errors = []
    restored_checkpoint = None
    for item in ranked_checkpoints:
        checkpoint_dir = item["checkpoint_path"]
        score = item.get("episode_return_mean")
        iteration = item.get("training_iteration")
        urllc_violation = item.get("center_urllc_violations")
        urllc_delay_ms = item.get("center_urllc_delay_ms")
        print(
            f"Trying checkpoint: {checkpoint_dir} "
            f"(iter={iteration}, urllc_viol={urllc_violation}, "
            f"urllc_delay_ms={urllc_delay_ms}, episode_return_mean={score})"
        )
        try:
            algo.restore(checkpoint_dir)
            restored_checkpoint = checkpoint_dir
            print(
                f"Loaded best available checkpoint: {checkpoint_dir} "
                f"(iter={iteration}, urllc_viol={urllc_violation}, "
                f"urllc_delay_ms={urllc_delay_ms}, episode_return_mean={score})"
            )
            break
        except Exception as exc:  # noqa: PERF203
            restore_errors.append(f"{checkpoint_dir} -> {exc}")

    if restored_checkpoint is None:
        error_preview = "\n".join(restore_errors[:3])
        raise RuntimeError(
            "No compatible checkpoint could be restored with the new env_runners API stack. "
            "Please retrain using the migrated train_marl.py.\n"
            f"Sample restore errors:\n{error_preview}"
        )

    print(f"Loaded MAPPO model from: {restored_checkpoint}")

    obs, _ = env.reset(seed=EVAL_SEED)
    ppo_rewards = []
    ppo_urllc_delay = []
    ppo_embb_shortfall = []
    
    for _ in range(ROLLOUT_STEPS):
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            actions[agent_id] = compute_action_new_stack(algo, agent_obs, policy_id=policy_id)
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
    plt.boxplot(data, tick_labels=['Static', 'Heuristic', 'MAPPO'])
    plt.title('Magnitude of URLLC SLA Violations (>2ms)')
    plt.ylabel('Delay Spike (ms)')
    plt.grid(True)
    
    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    plt.savefig('./results/marl_baseline_comparison_harsh.png')
    print("Saved comparison plots to ./results/marl_baseline_comparison_harsh.png")

    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    run_baselines()

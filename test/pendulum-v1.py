import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import torch


class FiveGSlicingEnv(gym.Env):
    def __init__(self, num_slices=3, total_rbs=100):
        super(FiveGSlicingEnv, self).__init__()

        self.num_slices = num_slices
        self.total_rbs = total_rbs

        # Action: scaling factors for RR allocation [s1, s2, s3]
        self.action_space = spaces.Box(
            low=0.5, high=2.0, shape=(num_slices,), dtype=np.float32
        )

        # State: demand + utilization + channel quality + RR allocation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4 * num_slices,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 1000

    def _get_rr_allocation(self):
        """Round Robin baseline - equal distribution"""
        return np.ones(self.num_slices) / self.num_slices

    def _calculate_kpis(self, allocation):
        """Calculate network KPIs based on resource allocation"""
        # Simplified KPI model - replace with actual network calculations
        throughput = allocation * self.state[0:self.num_slices] * self.state[2 * self.num_slices:3 * self.num_slices]
        delay = np.maximum(0.1, 1.0 - allocation * 0.8)  # Inverse relationship
        return throughput, delay

    def step(self, action):
        # Get RR baseline
        rr_allocation = self._get_rr_allocation()

        # Apply agent's scaling factors
        scaled_allocation = rr_allocation * action
        final_allocation = scaled_allocation / np.sum(scaled_allocation)

        # Calculate network performance
        throughput, delay = self._calculate_kpis(final_allocation)

        # Reward function components
        total_throughput = np.sum(throughput)
        sla_violations = np.sum(delay > 0.5)  # URLLC delay threshold

        # Combined reward
        reward = np.log(total_throughput + 1) - 10 * sla_violations

        # Update state (simulate network dynamics)
        self._update_state()

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.state, reward, done, {
            'throughput': throughput,
            'delay': delay,
            'allocation': final_allocation
        }

    def _update_state(self):
        # Simulate changing network conditions
        demand = np.random.rand(self.num_slices)
        utilization = self.state[self.num_slices:2 * self.num_slices] if hasattr(self, 'state') else np.zeros(
            self.num_slices)
        channel_quality = np.random.rand(self.num_slices) * 0.5 + 0.5  # 0.5-1.0
        rr_alloc = self._get_rr_allocation()

        self.state = np.concatenate([demand, utilization, channel_quality, rr_alloc])

    def reset(self):
        self.current_step = 0
        self._update_state()
        return self.state

    def render(self, mode='human'):
        print(f"Step {self.current_step}: State={self.state}")


# Training setup
def train_agent():
    env = FiveGSlicingEnv()

    # PPO configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_5g_tensorboard/"
    )

    # Start training
    print("Starting PPO training for 5G slicing...")
    model.learn(total_timesteps=100000)
    model.save("ppo_5g_slicing")

    return model, env


# Evaluation
def evaluate_model(model, env, episodes=10):
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        for step in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
        print(f"Final Allocation: {info['allocation']}")


if __name__ == "__main__":
    # Train and evaluate
    trained_model, environment = train_agent()
    evaluate_model(trained_model, environment)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

# PPO implementation for CartPole environment
class ActorCritic(nn.Module):
    """Neural network for actor-critic architecture"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers for feature extraction
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        # Actor network - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic network - outputs state value
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared = self.shared_layer(state)
        action_probs = self.actor(shared)
        state_value = self.critic(shared)
        return action_probs, state_value

# Experience replay buffer
class Memory:
    """Stores trajectory data for training"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """Clear memory after each update"""
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

# Proximal Policy Optimization agent
class PPO:
    """PPO algorithm implementation"""
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        # PPO hyperparameters
        self.gamma = gamma      # discount factor
        self.eps_clip = eps_clip  # clipping parameter
        self.K_epochs = K_epochs  # update epochs

    def select_action(self, state, memory):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(device)
        action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        # Store experience in memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def update(self, memory):
        """Update policy using PPO algorithm"""
        # Convert memory to tensors
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # normalize rewards

        # Multiple epochs of optimization
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()

            # PPO objective function
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach().squeeze()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()  # actor loss
            loss_critic = self.MseLoss(state_values.squeeze(), rewards)  # critic loss
            loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy.mean()  # total loss

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Hyperparameters
lr = 0.002
gamma = 0.99
eps_clip = 0.2
K_epochs = 4
max_episodes = 1000
max_timesteps = 300

# Initialize PPO agent and memory
ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)
memory = Memory()

# Training loop
episode_rewards = []

for episode in range(1, max_episodes + 1):
    state, _ = env.reset()
    total_reward = 0

    for t in range(max_timesteps):
        action = ppo.select_action(state, memory)
        state, reward, done, _, _ = env.step(action)

        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        total_reward += reward

        if done:
            break

    # Update policy and clear memory
    ppo.update(memory)
    memory.clear()
    episode_rewards.append(total_reward)

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

# Simple data visualization
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, alpha=0.7, linewidth=0.8)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('PPO Training Performance on CartPole')
plt.grid(True, alpha=0.3)
plt.show()
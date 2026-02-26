import numpy as np
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

class MultiCell_5G_SLA_Env(MultiAgentEnv):
    """
    7-Cell Hexagonal 5G Slicing Environment with SLA Constraints & Cooperative Reward.
    7小区六边形蜂窝 5G 切片环境，带 SLA 约束和合作式奖励。
    """

    def __init__(self, config=None):
        super().__init__()

        # --- Multi-Cell Topology ---
        # 0 is the center macro cell (BS_0)
        # 1-6 are the 6 surrounding neighbor cells (BS_1 to BS_6)
        self.num_cells = 7
        self.agents = [f"BS_{i}" for i in range(self.num_cells)]
        self._agent_ids = set(self.agents)

        # --- System Constants ---
        self.total_bandwidth = 100e6  # 100 MHz
        self.duration_tti = 0.5e-3  # 0.5 ms
        self.noise_power = 1e-12  # Background thermal noise (Watts) - W

        # Transmission Power of Macro BS
        self.bs_tx_power = 40.0  # 40 Watts (46 dBm typical for macro)

        # Path Loss Exponent and Distances
        # simplified geometric model: r = distance to center cell
        self.cell_radius_m = 500.0  # 500 meters inter-site distance
        self.path_loss_exp = 3.5  # Typical urban macro path loss exponent

        # 物理信道演进相关参数
        self.mean_se = np.array([4.5, 2.5, 1.5], dtype=np.float32)  # Base Spectral Efficiency
        self.rho_se = 0.9

        # --- SLA Parameters ---
        self.sla_props = {
            "embb_gbr": 180.0,
            "urllc_max_delay": 0.002,
            "mmtc_max_queue": 1.0,
        }

        # --- Spaces ---
        # Action space: [eMBB_Ratio, URLLC_Ratio, mMTC_Ratio] in [-1, 1]
        self.action_space = spaces.Box(low=np.float32(-1.0), high=np.float32(1.0), shape=(3,), dtype=np.float32)

        # Observation space: 14 dimensions per cell (same as single-cell)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        # State storage: Dictionary tracking states for all agents
        self.state = {agent: np.zeros(14, dtype=np.float32) for agent in self.agents}
        self.queues = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        self.current_se = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        
        self.max_steps = 200
        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}

        obs = {}
        for agent in self.agents:
            self.queues[agent] = np.zeros(3, dtype=np.float32)
            self.burst_state[agent] = False
            self.current_se[agent] = (self.mean_se + np.random.normal(0, 0.2, size=3)).astype(np.float32)
            self._update_agent_state(agent)
            
            # Initial dummy previous actions
            self.state[agent][9:12] = [0.33, 0.33, 0.34]
            
            scaled_obs = np.zeros(14, dtype=np.float32)
            # Demands: typically 150-350, 10-150, 10 -> scale down
            scaled_obs[0:3] = self.state[agent][0:3] / np.array([400.0, 200.0, 20.0], dtype=np.float32)
            # Queues: typically 0 to 20+ -> scale down
            scaled_obs[3:6] = self.state[agent][3:6] / np.array([20.0, 10.0, 2.0], dtype=np.float32)
            # SE: typically 2.0 - 6.0 -> scale down
            scaled_obs[6:9] = self.state[agent][6:9] / 6.0
            # Prev actions: already [0, 1]
            scaled_obs[9:12] = self.state[agent][9:12]
            # Delay: 0 to 0.010 -> scale down (10ms)
            scaled_obs[12] = self.state[agent][12] / 0.010
            # Shortfall: 0 to 120 -> scale down
            scaled_obs[13] = self.state[agent][13] / 120.0
            
            obs[agent] = scaled_obs

        return obs, {}

    def _update_agent_state(self, agent):
        
        # eMBB: Heavy background traffic, wants 350 Mbps
        arr_embb = np.clip(np.random.normal(250, 40), 180, 350).astype(np.float32)

        # URLLC: Markov Burst State (Sustained bursts to kill static allocation)
        if self.burst_state[agent]:
            if np.random.rand() > 0.8: # 20% chance to end burst (avg 5 TTIs)
                self.burst_state[agent] = False
        else:
            if np.random.rand() > 0.95: # 5% chance to start burst
                self.burst_state[agent] = True
                
        if self.burst_state[agent]:
            arr_urllc = np.float32(np.random.normal(160, 20)) # Massive 250Mbps burst
        else:
            arr_urllc = np.float32(np.random.normal(10, 2))


        # 3. mMTC: Sensor Data
        arr_mmtc = np.float32(np.random.normal(10, 1))

        # 4. 物理信道 (Spectral Efficiency)
        noise = np.array([
            np.random.normal(0, 0.4),
            np.random.normal(0, 0.2),
            np.random.normal(0, 0.15)
        ])
        
        self.current_se[agent] = (self.rho_se * self.current_se[agent]) + \
                                 ((1 - self.rho_se) * self.mean_se) + \
                                 (np.sqrt(1 - self.rho_se**2) * noise)
                          
        self.current_se[agent] = np.clip(self.current_se[agent], [2.0, 1.0, 0.5], [6.0, 4.0, 2.5])

        # 更新状态数组
        self.state[agent][0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[agent][6:9] = self.current_se[agent]
        # queue sizes
        self.state[agent][3:6] = self.queues[agent]

    def _calculate_interference_and_sinr(self, ratios_dict):
        """
        计算同频干扰 (ICI) 与 SINR。
        简单模型：中心基站 BS_0 受到其余 6 个基站的干扰。
        其他边缘基站的干扰暂不计算以简化模型。
        """
        # 计算 BS_0 在各个切片上的接收功率 (假设 UE 距离基站 d = 100m)
        d_ue = 100.0
        # 路径损耗简化模型： P_rx = P_tx * (d_ue)^(-path_loss_exp)
        # 此处我们只关注相对 SINR 变化，可以直接建模 interference penalty
        
        se_modifiers = {agent: np.ones(3, dtype=np.float32) for agent in self.agents}
        
        # 对于中心基站 BS_0：
        center_agent = "BS_0"
        if center_agent in ratios_dict:
            # 取邻居基站的分配比例
            neighbor_ratios = []
            for i in range(1, 7):
                agent_name = f"BS_{i}"
                if agent_name in ratios_dict:
                    neighbor_ratios.append(ratios_dict[agent_name])
            
            if neighbor_ratios:
                neighbor_ratios_np = np.array(neighbor_ratios) # shape: (6, 3)
                # 如果所有邻居基站都在同一个切片上分配了大量带宽，同频碰撞概率极高
                # 简单计算：邻居分配的平均比例作为干扰因子
                interference_factor = np.mean(neighbor_ratios_np, axis=0) # shape: (3,)
                
                # 干扰因子越大，中心基站的 SE 下降越多
                # 最多下降 50%
                se_modifiers[center_agent] = 1.0 - (interference_factor * 0.5)

        return se_modifiers

    def step(self, action_dict):
        """
        Execute simultaneous actions for all BS agents.
        """
        self.current_step += 1

        # Parse actions
        ratios_dict = {}
        for agent, action in action_dict.items():
            action_positive = (action + 1.0) / 2.0
            weights = np.clip(action_positive, 0.01, 1.0)
            ratios = weights / np.sum(weights)
            ratios_dict[agent] = ratios
            self.state[agent][9:12] = ratios

        # --- Physics Layer & Inter-Cell Interference (ICI) Calculation ---
        se_modifiers = self._calculate_interference_and_sinr(ratios_dict)

        obs = {}
        rewards = {}
        agent_rewards = {}
        infos = {}

        for agent in self.agents:
            if agent not in ratios_dict:
                continue
                
            ratios = ratios_dict[agent]
            bw_allocated = ratios * self.total_bandwidth
            
            # Apply interference modifier to SE
            effective_se = self.current_se[agent] * se_modifiers[agent]
            service_rate_mbps = (bw_allocated * effective_se) / 1e6
            service_capacity_mb = service_rate_mbps * self.duration_tti

            # Queue Evolution
            arrivals_mb = self.state[agent][0:3] * self.duration_tti
            self.queues[agent] += arrivals_mb
            served_mb = np.minimum(service_capacity_mb, self.queues[agent])
            self.queues[agent] -= served_mb
            
            achieved_throughput_mbps = served_mb / self.duration_tti

            # SLA Violations
            violations = np.zeros(3)
            
            embb_shortfall = max(0.0, self.sla_props['embb_gbr'] - achieved_throughput_mbps[0])
            violations[0] = embb_shortfall / self.sla_props['embb_gbr']

            safe_service_rate = max(service_rate_mbps[1], 0.1)
            est_delay = self.queues[agent][1] / safe_service_rate
            delay_excess = max(0.0, est_delay - self.sla_props['urllc_max_delay'])
            violations[1] = delay_excess / self.sla_props['urllc_max_delay']

            queue_excess = max(0.0, self.queues[agent][2] - self.sla_props['mmtc_max_queue'])
            violations[2] = queue_excess / self.sla_props['mmtc_max_queue']
            
            if self.queues[agent][2] > self.sla_props['mmtc_max_queue']:
                self.queues[agent][2] = self.sla_props['mmtc_max_queue']

            # Reward calculation for this agent
            reward = np.sum(achieved_throughput_mbps) / 300.0  # Normalized to max ~1.0
            penalty_weight = 0.5
            pen_embb = min(violations[0] * penalty_weight, penalty_weight * 5.0)
            pen_urllc = min(violations[1] * (penalty_weight * 3.0), penalty_weight * 10.0)
            pen_mmtc = min(violations[2] * penalty_weight, penalty_weight * 5.0)
            reward -= (pen_embb + pen_urllc + pen_mmtc)

            agent_rewards[agent] = reward

            # Update State
            self._update_agent_state(agent)
            self.state[agent][12] = est_delay
            self.state[agent][13] = embb_shortfall
            
            
            scaled_obs = np.zeros(14, dtype=np.float32)
            # Demands: typically 150-350, 10-150, 10 -> scale down
            scaled_obs[0:3] = self.state[agent][0:3] / np.array([400.0, 200.0, 20.0], dtype=np.float32)
            # Queues: typically 0 to 20+ -> scale down
            scaled_obs[3:6] = self.state[agent][3:6] / np.array([20.0, 10.0, 2.0], dtype=np.float32)
            # SE: typically 2.0 - 6.0 -> scale down
            scaled_obs[6:9] = self.state[agent][6:9] / 6.0
            # Prev actions: already [0, 1]
            scaled_obs[9:12] = self.state[agent][9:12]
            # Delay: 0 to 0.010 -> scale down (10ms)
            scaled_obs[12] = self.state[agent][12] / 0.010
            # Shortfall: 0 to 120 -> scale down
            scaled_obs[13] = self.state[agent][13] / 120.0
            
            obs[agent] = scaled_obs
            infos[agent] = {
                "queue_sizes": self.queues[agent].copy(),
                "violations": violations,
                "throughput": np.sum(achieved_throughput_mbps),
                "est_urllc_delay": est_delay,
                "local_reward": reward
            }

        # --- Cooperative Reward (MARL) ---
        # Agent reward is local reward + average reward of system
        total_system_reward = sum(agent_rewards.values())
        avg_system_reward = total_system_reward / len(self.agents)
        
        # 合作比例: 0.5 * 本地奖励 + 0.5 * 全局平均奖励
        alpha = 0.5
        for agent in self.agents:
            if agent in agent_rewards:
                rewards[agent] = alpha * agent_rewards[agent] + (1 - alpha) * avg_system_reward

        # Done flags
        terminateds = {"__all__": self.current_step >= self.max_steps}
        truncateds = {"__all__": False}
        
        # Set individual done flags
        for agent in self.agents:
            terminateds[agent] = terminateds["__all__"]
            truncateds[agent] = truncateds["__all__"]

        return obs, rewards, terminateds, truncateds, infos


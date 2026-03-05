import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

class MultiCell_5G_SLA_Env(MultiAgentEnv):
    """
    7-Cell Hexagonal 5G Slicing Environment with SLA Constraints & Cooperative Reward.
    7小区六边形蜂窝 5G 切片环境，带 SLA 约束和合作式奖励。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        # --- Multi-Cell Topology ---
        # 0 is the center macro cell (BS_0)
        # 1-6 are the 6 surrounding neighbor cells (BS_1 to BS_6)
        self.num_cells = 7
        self.agents = [f"BS_{i}" for i in range(self.num_cells)]
        self._agent_ids = set(self.agents)
        self.neighbor_map = {
            "BS_0": ["BS_1", "BS_2", "BS_3", "BS_4", "BS_5", "BS_6"],
            "BS_1": ["BS_0", "BS_2", "BS_6"],
            "BS_2": ["BS_0", "BS_1", "BS_3"],
            "BS_3": ["BS_0", "BS_2", "BS_4"],
            "BS_4": ["BS_0", "BS_3", "BS_5"],
            "BS_5": ["BS_0", "BS_4", "BS_6"],
            "BS_6": ["BS_0", "BS_5", "BS_1"],
        }

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

        # --- Reward shaping params (tunable) ---
        # Continuous URLLC soft-cliff: warning before 2ms and sharp increase after deadline.
        self.penalty_weight = float(self.config.get("penalty_weight", 0.5))
        self.urllc_warning_ratio = float(self.config.get("urllc_warning_ratio", 0.6))
        self.urllc_softplus_slope = float(self.config.get("urllc_softplus_slope", 12.0))
        self.urllc_warning_gain = float(self.config.get("urllc_warning_gain", 0.8))
        self.urllc_overflow_gain = float(self.config.get("urllc_overflow_gain", 3.0))
        self.urllc_exp_coeff = float(self.config.get("urllc_exp_coeff", 3.0))
        self.urllc_penalty_cap_factor = float(self.config.get("urllc_penalty_cap_factor", 20.0))
        self.embb_penalty_quad_gain = float(self.config.get("embb_penalty_quad_gain", 1.2))
        self.embb_penalty_cap_factor = float(self.config.get("embb_penalty_cap_factor", 10.0))
        self.ici_gain = float(self.config.get("ici_gain", 0.6))
        self.se_modifier_floor = float(self.config.get("se_modifier_floor", 0.35))
        self.max_neighbors = 6.0

        # --- Spaces ---
        # Per-agent spaces required by RLlib env_runner + connector v2 stack.
        # Obs dims: 14 local features + 3 neighbor previous-action means.
        self._agent_action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0), shape=(3,), dtype=np.float32
        )
        self._agent_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.action_spaces = {agent: self._agent_action_space for agent in self.agents}
        self.observation_spaces = {agent: self._agent_observation_space for agent in self.agents}
        self.action_space = spaces.Dict(self.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)

        # State storage: Dictionary tracking states for all agents
        self.state = {agent: np.zeros(14, dtype=np.float32) for agent in self.agents}
        self.queues = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        self.current_se = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        
        self.max_steps = 200
        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}
        self.np_random, _ = np_random(None)

    def observation_space_sample(self, agent_ids=None):
        sampled_agent_ids = agent_ids or self.agents
        return {agent_id: self._agent_observation_space.sample() for agent_id in sampled_agent_ids}

    def action_space_sample(self, agent_ids=None):
        sampled_agent_ids = agent_ids or self.agents
        return {agent_id: self._agent_action_space.sample() for agent_id in sampled_agent_ids}

    def observation_space_contains(self, x):
        if not isinstance(x, dict):
            return False
        return all(
            agent_id in self.observation_spaces and self._agent_observation_space.contains(agent_obs)
            for agent_id, agent_obs in x.items()
        )

    def action_space_contains(self, x):
        if not isinstance(x, dict):
            return False
        return all(
            agent_id in self.action_spaces and self._agent_action_space.contains(agent_action)
            for agent_id, agent_action in x.items()
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = np_random(seed)

        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}

        obs = {}
        for agent in self.agents:
            self.queues[agent] = np.zeros(3, dtype=np.float32)
            self.burst_state[agent] = False
            self.current_se[agent] = (
                self.mean_se + self.np_random.normal(0.0, 0.2, size=3)
            ).astype(np.float32)
            self._update_agent_state(agent)
            
            # Initial dummy previous actions
            self.state[agent][9:12] = [0.33, 0.33, 0.34]
            
            obs[agent] = self._build_scaled_obs(agent)

        return obs, {}

    def _update_agent_state(self, agent):
        
        # eMBB: Heavy background traffic, wants 350 Mbps
        arr_embb = np.float32(
            np.clip(self.np_random.normal(250.0, 40.0), 180.0, 350.0)
        )

        # URLLC: Markov Burst State (Sustained bursts to kill static allocation)
        if self.burst_state[agent]:
            if self.np_random.random() > 0.8: # 20% chance to end burst (avg 5 TTIs)
                self.burst_state[agent] = False
        else:
            if self.np_random.random() > 0.95: # 5% chance to start burst
                self.burst_state[agent] = True
                
        if self.burst_state[agent]:
            arr_urllc = np.float32(self.np_random.normal(160.0, 20.0))
        else:
            arr_urllc = np.float32(self.np_random.normal(10.0, 2.0))


        # 3. mMTC: Sensor Data
        arr_mmtc = np.float32(self.np_random.normal(10.0, 1.0))

        # 4. 物理信道 (Spectral Efficiency)
        noise = np.array([
            self.np_random.normal(0.0, 0.4),
            self.np_random.normal(0.0, 0.2),
            self.np_random.normal(0.0, 0.15)
        ], dtype=np.float32)
        
        self.current_se[agent] = (self.rho_se * self.current_se[agent]) + \
                                 ((1 - self.rho_se) * self.mean_se) + \
                                 (np.sqrt(1 - self.rho_se**2) * noise)
                          
        self.current_se[agent] = np.clip(
            self.current_se[agent], [2.0, 1.0, 0.5], [6.0, 4.0, 2.5]
        ).astype(np.float32)

        # 更新状态数组
        self.state[agent][0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[agent][6:9] = self.current_se[agent]
        # queue sizes
        self.state[agent][3:6] = self.queues[agent]

    def _calculate_interference_and_sinr(self, ratios_dict):
        """对 7 小区都计算同频干扰 (ICI)，避免中心/边缘建模不对称。"""
        se_modifiers = {agent: np.ones(3, dtype=np.float32) for agent in self.agents}

        for agent in self.agents:
            if agent not in ratios_dict:
                continue

            neighbor_ratios = [
                ratios_dict[neighbor]
                for neighbor in self.neighbor_map.get(agent, [])
                if neighbor in ratios_dict
            ]
            if not neighbor_ratios:
                continue

            neighbor_ratios_np = np.array(neighbor_ratios, dtype=np.float32)
            # Keep center harder than edge by normalizing to max-neighbor count (6).
            normalized_neighbor_load = np.sum(neighbor_ratios_np, axis=0) / self.max_neighbors
            reduction = self.ici_gain * normalized_neighbor_load
            se_modifiers[agent] = np.clip(1.0 - reduction, self.se_modifier_floor, 1.0).astype(np.float32)

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
            violations = np.zeros(3, dtype=np.float32)
            
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
            reward = np.sum(achieved_throughput_mbps) / 300.0  # Normalized throughput

            # Continuous smooth penalty with soft clipping (no hard step/cliff penalty).
            embb_violation = float(violations[0])
            raw_pen_embb = self.penalty_weight * (
                embb_violation + self.embb_penalty_quad_gain * (embb_violation ** 2)
            )
            raw_pen_urllc = self._urllc_soft_cliff_penalty(est_delay)
            raw_pen_mmtc = violations[2] * self.penalty_weight

            pen_embb = self._soft_clip_penalty(raw_pen_embb, self.penalty_weight * self.embb_penalty_cap_factor)
            pen_urllc = self._soft_clip_penalty(raw_pen_urllc, self.penalty_weight * self.urllc_penalty_cap_factor)
            pen_mmtc = self._soft_clip_penalty(raw_pen_mmtc, self.penalty_weight * 5.0)
            reward -= (pen_embb + pen_urllc + pen_mmtc)
            if not np.isfinite(reward):
                reward = -100.0


            agent_rewards[agent] = reward

            # Update State
            self._update_agent_state(agent)
            self.state[agent][12] = est_delay
            self.state[agent][13] = embb_shortfall
            
            
            obs[agent] = self._build_scaled_obs(agent)
            infos[agent] = {
                "queue_sizes": self.queues[agent].copy(),
                "violations": violations,
                "throughput": np.sum(achieved_throughput_mbps),
                "throughput_slices_mbps": achieved_throughput_mbps.astype(np.float32),
                "est_urllc_delay": est_delay,
                "urllc_delay_ratio": est_delay / self.sla_props["urllc_max_delay"],
                "neighbor_prev_action_mean": self._get_neighbor_prev_action_mean(agent),
                "local_reward": reward,
                "penalties": np.array([pen_embb, pen_urllc, pen_mmtc], dtype=np.float32),
            }

        # --- Cooperative Reward (MARL) ---
        # Agent reward is local reward + average reward of system
        total_system_reward = sum(agent_rewards.values())
        avg_system_reward = total_system_reward / len(self.agents)
        
        # 合作比例: 0.5 * 本地奖励 + 0.5 * 全局平均奖励
        alpha = 0.7
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

    @staticmethod
    def _soft_clip_penalty(penalty_value: float, max_penalty: float) -> float:
        """Smoothly saturate penalty magnitude to avoid reward spikes."""
        if max_penalty <= 0:
            return float(penalty_value)
        return float(max_penalty * np.tanh(penalty_value / max_penalty))

    @staticmethod
    def _softplus(x: float) -> float:
        """Numerically stable softplus for continuous penalty shaping."""
        x = float(x)
        return float(np.log1p(np.exp(-abs(x))) + max(x, 0.0))

    def _urllc_soft_cliff_penalty(self, est_delay: float) -> float:
        """
        Continuous URLLC soft-cliff penalty:
        1) warning term activates before deadline,
        2) overflow term grows exponentially after deadline.
        """
        max_delay = self.sla_props["urllc_max_delay"]
        delay_ratio = float(est_delay / max_delay)
        slope = self.urllc_softplus_slope

        warning_term = self._softplus(slope * (delay_ratio - self.urllc_warning_ratio)) / slope
        overflow_ratio = max(0.0, delay_ratio - 1.0)
        overflow_ratio = min(overflow_ratio, 3.0)
        overflow_term = np.expm1(self.urllc_exp_coeff * overflow_ratio)

        return float(
            self.penalty_weight
            * (
                self.urllc_warning_gain * warning_term
                + self.urllc_overflow_gain * overflow_term
            )
        )

    def _get_neighbor_prev_action_mean(self, agent):
        neighbor_ids = self.neighbor_map.get(agent, [])
        if not neighbor_ids:
            return np.array([0.33, 0.33, 0.34], dtype=np.float32)

        prev_actions = np.array([self.state[neighbor][9:12] for neighbor in neighbor_ids], dtype=np.float32)
        return np.mean(prev_actions, axis=0).astype(np.float32)

    def _build_scaled_obs(self, agent):
        scaled_obs = np.zeros(17, dtype=np.float32)
        scaled_obs[0:3] = self.state[agent][0:3] / np.array([400.0, 200.0, 20.0], dtype=np.float32)
        scaled_obs[3:6] = self.state[agent][3:6] / np.array([20.0, 10.0, 2.0], dtype=np.float32)
        scaled_obs[6:9] = self.state[agent][6:9] / 6.0
        scaled_obs[9:12] = self.state[agent][9:12]
        scaled_obs[12] = self.state[agent][12] / 0.010
        scaled_obs[13] = self.state[agent][13] / 120.0
        scaled_obs[14:17] = self._get_neighbor_prev_action_mean(agent)
        scaled_obs = np.nan_to_num(scaled_obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(scaled_obs, -10.0, 10.0).astype(np.float32)

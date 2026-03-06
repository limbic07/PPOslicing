from collections import deque

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

from ippo_rl_module import DEFAULT_INITIAL_SLICE_RATIOS

class MultiCell_5G_SLA_Env(MultiAgentEnv):
    """
    7-Cell Hexagonal 5G Slicing Environment with SLA Constraints & Cooperative Reward.
    7小区六边形蜂窝 5G 切片环境，带 SLA 约束和合作式奖励。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        self.env_profile = str(self.config.get("env_profile", "harsh")).lower()
        profile_overrides = self._get_env_profile_overrides(self.env_profile)

        def cfg_value(key, default):
            return self.config.get(key, profile_overrides.get(key, default))

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
            "embb_gbr": float(cfg_value("embb_gbr", 180.0)),
            "urllc_max_delay": float(cfg_value("urllc_max_delay", 0.002)),
            "mmtc_max_queue": float(cfg_value("mmtc_max_queue", 1.0)),
        }
        self.embb_sla_window_tti = int(cfg_value("embb_sla_window_tti", 1))
        if self.embb_sla_window_tti <= 0:
            raise ValueError("embb_sla_window_tti must be >= 1")
        self.reward_mode = str(cfg_value("reward_mode", "tail_risk_coop")).lower()
        self.cooperative_alpha = float(cfg_value("cooperative_alpha", 0.7))
        self.action_softmax_temperature = float(cfg_value("action_softmax_temperature", 3.0))
        self.observation_mode = str(cfg_value("observation_mode", "pure_local")).lower()
        if self.action_softmax_temperature <= 0.0:
            raise ValueError("action_softmax_temperature must be > 0")
        if self.observation_mode not in {"pure_local", "neighbor_augmented"}:
            raise ValueError(
                f"Unsupported observation_mode={self.observation_mode!r}. "
                "Expected one of ['neighbor_augmented', 'pure_local']"
            )

        # --- Reward shaping params (tunable) ---
        # Continuous URLLC soft-cliff: warning before 2ms and sharp increase after deadline.
        self.penalty_weight = float(self.config.get("penalty_weight", 0.5))
        self.w_embb = float(self.config.get("w_embb", self.penalty_weight))
        self.w_urllc = float(self.config.get("w_urllc", self.penalty_weight))
        self.w_mmtc = float(self.config.get("w_mmtc", self.penalty_weight))
        self.urllc_warning_ratio = float(self.config.get("urllc_warning_ratio", 0.9))
        self.urllc_tail_ratio = float(self.config.get("urllc_tail_ratio", 0.92))
        self.urllc_softplus_slope = float(self.config.get("urllc_softplus_slope", 12.0))
        self.urllc_warning_gain = float(self.config.get("urllc_warning_gain", 0.2))
        self.urllc_tail_quad_gain = float(self.config.get("urllc_tail_quad_gain", 2.0))
        self.urllc_hard_violation_gain = float(self.config.get("urllc_hard_violation_gain", 1.75))
        self.urllc_overflow_gain = float(self.config.get("urllc_overflow_gain", 3.0))
        self.urllc_exp_coeff = float(self.config.get("urllc_exp_coeff", 1.6))
        self.urllc_penalty_cap_factor = float(self.config.get("urllc_penalty_cap_factor", 10.0))
        self.embb_penalty_quad_gain = float(self.config.get("embb_penalty_quad_gain", 1.2))
        self.embb_penalty_cubic_gain = float(self.config.get("embb_penalty_cubic_gain", 0.0))
        self.embb_penalty_cap_factor = float(self.config.get("embb_penalty_cap_factor", 10.0))
        self.mmtc_penalty_cap_factor = float(self.config.get("mmtc_penalty_cap_factor", 5.0))
        self.embb_violation_cap = float(cfg_value("embb_violation_cap", 2.0))
        self.urllc_violation_cap = float(cfg_value("urllc_violation_cap", 5.0))
        self.mmtc_violation_cap = float(cfg_value("mmtc_violation_cap", 2.0))
        self.ici_gain = float(cfg_value("ici_gain", 0.6))
        self.se_modifier_floor = float(cfg_value("se_modifier_floor", 0.35))
        self.max_neighbors = 6.0
        self.urllc_nominal_mean_mbps = float(cfg_value("urllc_nominal_mean_mbps", 10.0))
        self.urllc_nominal_std_mbps = float(cfg_value("urllc_nominal_std_mbps", 2.0))
        self.urllc_burst_start_prob = float(cfg_value("urllc_burst_start_prob", 0.05))
        self.urllc_burst_end_prob = float(cfg_value("urllc_burst_end_prob", 0.2))
        self.urllc_burst_mean_mbps = float(cfg_value("urllc_burst_mean_mbps", 160.0))
        self.urllc_burst_std_mbps = float(cfg_value("urllc_burst_std_mbps", 20.0))
        self.interference_neighbor_normalization = str(
            cfg_value("interference_neighbor_normalization", "fixed_max")
        ).lower()
        self.tail_reward_throughput_weight = float(cfg_value("tail_reward_throughput_weight", 1.0 / 300.0))
        self.binary_reward_throughput_scale = float(cfg_value("binary_reward_throughput_scale", 100.0))
        self.binary_penalty_embb = float(cfg_value("binary_penalty_embb", 6.0))
        self.binary_penalty_urllc = float(cfg_value("binary_penalty_urllc", 12.0))
        self.binary_penalty_mmtc = float(cfg_value("binary_penalty_mmtc", 6.0))
        self.simple_reward_embb_target_weight = float(cfg_value("simple_reward_embb_target_weight", 1.2))
        self.simple_reward_throughput_weight = float(cfg_value("simple_reward_throughput_weight", 0.2))
        self.simple_reward_bonus_embb = float(cfg_value("simple_reward_bonus_embb", 0.60))
        self.simple_reward_bonus_urllc = float(cfg_value("simple_reward_bonus_urllc", 0.30))
        self.simple_reward_bonus_mmtc = float(cfg_value("simple_reward_bonus_mmtc", 0.10))
        self.simple_reward_penalty_embb_linear = float(cfg_value("simple_reward_penalty_embb_linear", 1.5))
        self.simple_reward_penalty_embb_quad = float(cfg_value("simple_reward_penalty_embb_quad", 1.0))
        self.simple_reward_penalty_urllc_linear = float(cfg_value("simple_reward_penalty_urllc_linear", 1.0))
        self.simple_reward_penalty_urllc_quad = float(cfg_value("simple_reward_penalty_urllc_quad", 0.5))
        self.simple_reward_penalty_mmtc_linear = float(cfg_value("simple_reward_penalty_mmtc_linear", 0.5))
        self.center_reward_scale = float(cfg_value("center_reward_scale", 1.0))
        self.reward_clip_abs = float(cfg_value("reward_clip_abs", 10.0))

        # --- Spaces ---
        # Per-agent spaces required by RLlib env_runner + connector v2 stack.
        # Obs dims depend on observation_mode:
        # pure_local=14, neighbor_augmented=20.
        self._agent_action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0), shape=(3,), dtype=np.float32
        )
        self.obs_dim = 14 if self.observation_mode == "pure_local" else 20
        self._agent_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_spaces = {agent: self._agent_action_space for agent in self.agents}
        self.observation_spaces = {agent: self._agent_observation_space for agent in self.agents}
        self.action_space = spaces.Dict(self.action_spaces)
        self.observation_space = spaces.Dict(self.observation_spaces)

        # State storage: Dictionary tracking states for all agents
        self.state = {agent: np.zeros(14, dtype=np.float32) for agent in self.agents}
        self.queues = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        self.current_se = {agent: np.zeros(3, dtype=np.float32) for agent in self.agents}
        self.embb_tp_history = {
            agent: deque(maxlen=self.embb_sla_window_tti) for agent in self.agents
        }
        
        self.max_steps = 200
        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}
        self.np_random, _ = np_random(None)

    @staticmethod
    def _get_env_profile_overrides(env_profile):
        profiles = {
            "harsh": {
                "embb_sla_window_tti": 1,
                "reward_mode": "tail_risk_coop",
                "cooperative_alpha": 0.7,
                "action_softmax_temperature": 3.0,
                "urllc_burst_start_prob": 0.05,
                "urllc_burst_end_prob": 0.2,
                "urllc_burst_mean_mbps": 160.0,
                "urllc_burst_std_mbps": 20.0,
                "ici_gain": 0.65,
                "se_modifier_floor": 0.3,
                "interference_neighbor_normalization": "fixed_max",
                "tail_reward_throughput_weight": 1.0 / 300.0,
                "embb_violation_cap": 2.0,
                "urllc_violation_cap": 5.0,
                "mmtc_violation_cap": 2.0,
            },
            "balanced": {
                "embb_gbr": 220.0,
                "embb_sla_window_tti": 8,
                "reward_mode": "binary_sla_reward",
                "cooperative_alpha": 1.0,
                "action_softmax_temperature": 3.0,
                "urllc_burst_start_prob": 0.06,
                "urllc_burst_end_prob": 0.35,
                "urllc_burst_mean_mbps": 100.0,
                "urllc_burst_std_mbps": 15.0,
                "ici_gain": 0.50,
                "se_modifier_floor": 0.45,
                "interference_neighbor_normalization": "actual_neighbors",
                "binary_reward_throughput_scale": 100.0,
                "binary_penalty_embb": 6.0,
                "binary_penalty_urllc": 12.0,
                "binary_penalty_mmtc": 6.0,
                "center_reward_scale": 1.0,
                "reward_clip_abs": 0.0,
                "embb_violation_cap": 2.0,
                "urllc_violation_cap": 5.0,
                "mmtc_violation_cap": 2.0,
            },
        }
        if env_profile not in profiles:
            raise ValueError(f"Unsupported env_profile={env_profile!r}. Expected one of {sorted(profiles)}")
        return profiles[env_profile]

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
            self.embb_tp_history[agent].clear()
            self.burst_state[agent] = False
            self.current_se[agent] = (
                self.mean_se + self.np_random.normal(0.0, 0.2, size=3)
            ).astype(np.float32)
            self._update_agent_state(agent)
            
            # Initial previous-ratio prior for the first step.
            self.state[agent][9:12] = DEFAULT_INITIAL_SLICE_RATIOS
            
            obs[agent] = self._build_obs(agent)

        return obs, {}

    def _update_agent_state(self, agent):
        
        # eMBB: Heavy background traffic, wants 350 Mbps
        arr_embb = np.float32(
            np.clip(self.np_random.normal(250.0, 40.0), 180.0, 350.0)
        )

        # URLLC: Markov Burst State (Sustained bursts to kill static allocation)
        if self.burst_state[agent]:
            if self.np_random.random() < self.urllc_burst_end_prob:
                self.burst_state[agent] = False
        else:
            if self.np_random.random() < self.urllc_burst_start_prob:
                self.burst_state[agent] = True
                
        if self.burst_state[agent]:
            arr_urllc = np.float32(
                np.clip(
                    self.np_random.normal(self.urllc_burst_mean_mbps, self.urllc_burst_std_mbps),
                    0.0,
                    None,
                )
            )
        else:
            arr_urllc = np.float32(
                np.clip(
                    self.np_random.normal(self.urllc_nominal_mean_mbps, self.urllc_nominal_std_mbps),
                    0.0,
                    None,
                )
            )


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
            if self.interference_neighbor_normalization == "actual_neighbors":
                normalizer = float(len(neighbor_ratios))
            elif self.interference_neighbor_normalization == "fixed_max":
                normalizer = self.max_neighbors
            else:
                raise ValueError(
                    "Unsupported interference_neighbor_normalization="
                    f"{self.interference_neighbor_normalization!r}"
                )

            normalized_neighbor_load = np.sum(neighbor_ratios_np, axis=0) / max(normalizer, 1.0)
            reduction = self.ici_gain * normalized_neighbor_load
            se_modifiers[agent] = np.clip(1.0 - reduction, self.se_modifier_floor, 1.0).astype(np.float32)

        return se_modifiers

    def _compute_tail_risk_reward(self, throughput_mbps, violations, est_delay):
        embb_violation = float(violations[0])
        raw_pen_embb = self.w_embb * (
            embb_violation
            + self.embb_penalty_quad_gain * (embb_violation ** 2)
            + self.embb_penalty_cubic_gain * (embb_violation ** 3)
        )
        raw_pen_urllc = self._urllc_soft_cliff_penalty(est_delay)
        raw_pen_mmtc = float(violations[2]) * self.w_mmtc

        pen_embb = self._soft_clip_penalty(raw_pen_embb, self.w_embb * self.embb_penalty_cap_factor)
        pen_urllc = self._soft_clip_penalty(raw_pen_urllc, self.w_urllc * self.urllc_penalty_cap_factor)
        pen_mmtc = self._soft_clip_penalty(raw_pen_mmtc, self.w_mmtc * self.mmtc_penalty_cap_factor)
        penalty_total = float(pen_embb + pen_urllc + pen_mmtc)
        throughput_bonus = float(self.tail_reward_throughput_weight * throughput_mbps)
        reward_local = float(throughput_bonus - penalty_total)

        return {
            "local_reward": reward_local,
            "reward_base_tp": throughput_bonus,
            "reward_sla_bonus": 0.0,
            "penalty_total": penalty_total,
            "penalty_raw_embb": float(raw_pen_embb),
            "penalty_raw_urllc": float(raw_pen_urllc),
            "penalty_raw_mmtc": float(raw_pen_mmtc),
            "penalty_embb": float(pen_embb),
            "penalty_urllc": float(pen_urllc),
            "penalty_mmtc": float(pen_mmtc),
        }

    def _compute_binary_sla_reward(self, throughput_mbps, violation_flags):
        violation_flags = np.asarray(violation_flags, dtype=np.float32)
        throughput_bonus = float(throughput_mbps / max(self.binary_reward_throughput_scale, 1e-6))
        pen_embb = float(self.binary_penalty_embb * violation_flags[0])
        pen_urllc = float(self.binary_penalty_urllc * violation_flags[1])
        pen_mmtc = float(self.binary_penalty_mmtc * violation_flags[2])
        penalty_total = float(pen_embb + pen_urllc + pen_mmtc)
        reward_local = float(throughput_bonus - penalty_total)

        return {
            "local_reward": reward_local,
            "reward_base_tp": throughput_bonus,
            "reward_sla_bonus": 0.0,
            "penalty_total": penalty_total,
            "penalty_raw_embb": pen_embb,
            "penalty_raw_urllc": pen_urllc,
            "penalty_raw_mmtc": pen_mmtc,
            "penalty_embb": pen_embb,
            "penalty_urllc": pen_urllc,
            "penalty_mmtc": pen_mmtc,
        }

    def _compute_simple_local_reward(self, throughput_mbps, embb_eval_tp_mbps, violations):
        embb_violation = float(violations[0])
        urllc_violation = float(violations[1])
        mmtc_violation = float(violations[2])

        # IPPO local reward should care about "how close eMBB is to GBR",
        # not just total throughput. This prevents the policy from settling
        # into a low-risk but low-eMBB static allocation.
        embb_target_ratio = float(
            np.clip(embb_eval_tp_mbps / max(self.sla_props["embb_gbr"], 1e-6), 0.0, 1.2)
        )
        total_tp_ratio = float(np.clip(throughput_mbps / 250.0, 0.0, 1.2))
        embb_target_reward = self.simple_reward_embb_target_weight * embb_target_ratio
        throughput_bonus = self.simple_reward_throughput_weight * total_tp_ratio

        sla_bonus = (
            self.simple_reward_bonus_embb * float(embb_violation <= 0.0)
            + self.simple_reward_bonus_urllc * float(urllc_violation <= 0.0)
            + self.simple_reward_bonus_mmtc * float(mmtc_violation <= 0.0)
        )

        pen_embb = (
            self.simple_reward_penalty_embb_linear * embb_violation
            + self.simple_reward_penalty_embb_quad * (embb_violation ** 2)
        )
        pen_urllc = (
            self.simple_reward_penalty_urllc_linear * urllc_violation
            + self.simple_reward_penalty_urllc_quad * (urllc_violation ** 2)
        )
        pen_mmtc = self.simple_reward_penalty_mmtc_linear * mmtc_violation
        penalty_total = float(pen_embb + pen_urllc + pen_mmtc)
        reward_local = float(embb_target_reward + throughput_bonus + sla_bonus - penalty_total)

        return {
            "local_reward": reward_local,
            "reward_base_tp": float(throughput_bonus),
            "reward_sla_bonus": float(embb_target_reward + sla_bonus),
            "penalty_total": penalty_total,
            "penalty_raw_embb": float(pen_embb),
            "penalty_raw_urllc": float(pen_urllc),
            "penalty_raw_mmtc": float(pen_mmtc),
            "penalty_embb": float(pen_embb),
            "penalty_urllc": float(pen_urllc),
            "penalty_mmtc": float(pen_mmtc),
        }

    def _action_to_ratios(self, action):
        logits = np.asarray(action, dtype=np.float32) * self.action_softmax_temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        ratio_sum = float(np.sum(exp_logits))
        if not np.isfinite(ratio_sum) or ratio_sum <= 0.0:
            return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        return (exp_logits / ratio_sum).astype(np.float32)

    def _cap_violations(self, violations_raw):
        caps = np.array(
            [self.embb_violation_cap, self.urllc_violation_cap, self.mmtc_violation_cap],
            dtype=np.float32,
        )
        violations_raw = np.maximum(np.asarray(violations_raw, dtype=np.float32), 0.0)
        return np.minimum(violations_raw, caps).astype(np.float32)

    def _get_role_reward_scale(self, agent):
        if agent == "BS_0":
            return float(self.center_reward_scale)
        return 1.0

    def _clip_reward(self, reward_value):
        if self.reward_clip_abs <= 0.0:
            return float(reward_value)
        return float(np.clip(reward_value, -self.reward_clip_abs, self.reward_clip_abs))

    def step(self, action_dict):
        """
        Execute simultaneous actions for all BS agents.
        """
        self.current_step += 1

        # Parse actions
        ratios_dict = {}
        for agent, action in action_dict.items():
            ratios = self._action_to_ratios(action)
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
            self.embb_tp_history[agent].append(float(achieved_throughput_mbps[0]))
            embb_eval_tp_mbps = float(np.mean(self.embb_tp_history[agent]))

            # SLA Violations
            violations_raw = np.zeros(3, dtype=np.float32)
            
            embb_shortfall = max(0.0, self.sla_props['embb_gbr'] - embb_eval_tp_mbps)
            violations_raw[0] = embb_shortfall / self.sla_props['embb_gbr']

            safe_service_rate = max(service_rate_mbps[1], 0.1)
            est_delay = self.queues[agent][1] / safe_service_rate
            delay_excess = max(0.0, est_delay - self.sla_props['urllc_max_delay'])
            violations_raw[1] = delay_excess / self.sla_props['urllc_max_delay']

            queue_excess = max(0.0, self.queues[agent][2] - self.sla_props['mmtc_max_queue'])
            violations_raw[2] = queue_excess / self.sla_props['mmtc_max_queue']
            violations = self._cap_violations(violations_raw)
            violation_flags = (violations_raw > 0.0).astype(np.float32)
            
            if self.queues[agent][2] > self.sla_props['mmtc_max_queue']:
                self.queues[agent][2] = self.sla_props['mmtc_max_queue']

            
            # Reward calculation for this agent
            throughput_mbps_total = float(np.sum(achieved_throughput_mbps))

            if self.reward_mode == "tail_risk_coop":
                reward_terms = self._compute_tail_risk_reward(throughput_mbps_total, violations, est_delay)
            elif self.reward_mode == "binary_sla_reward":
                reward_terms = self._compute_binary_sla_reward(throughput_mbps_total, violation_flags)
            elif self.reward_mode == "simple_local_sla":
                reward_terms = self._compute_simple_local_reward(
                    throughput_mbps_total,
                    embb_eval_tp_mbps,
                    violations,
                )
            else:
                raise ValueError(f"Unsupported reward_mode={self.reward_mode!r}")

            reward_local_unclipped = float(reward_terms["local_reward"] * self._get_role_reward_scale(agent))
            reward_local = self._clip_reward(reward_local_unclipped)
            if not np.isfinite(reward_local):
                reward_local = -100.0


            agent_rewards[agent] = reward_local

            # Update State
            self._update_agent_state(agent)
            # --- 必须加 min() 保护滤波器不被探索期极值毒害 ---
            self.state[agent][12] = min(float(est_delay), 1.0)
            self.state[agent][13] = min(float(embb_shortfall), float(self.sla_props['embb_gbr']))
            
            
            obs[agent] = self._build_obs(agent)
            infos[agent] = {
                "queue_sizes": self.queues[agent].copy(),
                "violations": violations,
                "violations_raw": violations_raw,
                "violation_flags": violation_flags,
                "throughput": np.sum(achieved_throughput_mbps),
                "throughput_slices_mbps": achieved_throughput_mbps.astype(np.float32),
                "embb_eval_tp_mbps": np.float32(embb_eval_tp_mbps),
                "embb_sla_window_tti": np.int32(self.embb_sla_window_tti),
                "est_urllc_delay": est_delay,
                "urllc_delay_ratio": est_delay / self.sla_props["urllc_max_delay"],
                "neighbor_prev_action_mean": self._get_neighbor_prev_action_mean(agent),
                "neighbor_urgency_features": self._get_neighbor_urgency_features(agent),
                "local_reward": reward_local,
                "local_reward_unclipped": np.float32(reward_local_unclipped),
                "role_reward_scale": np.float32(self._get_role_reward_scale(agent)),
                "reward_base_tp": np.float32(reward_terms["reward_base_tp"]),
                "reward_sla_bonus": np.float32(reward_terms["reward_sla_bonus"]),
                "penalty_total": np.float32(reward_terms["penalty_total"]),
                "penalty_raw_embb": np.float32(reward_terms["penalty_raw_embb"]),
                "penalty_raw_urllc": np.float32(reward_terms["penalty_raw_urllc"]),
                "penalty_raw_mmtc": np.float32(reward_terms["penalty_raw_mmtc"]),
                "penalty_embb": np.float32(reward_terms["penalty_embb"]),
                "penalty_urllc": np.float32(reward_terms["penalty_urllc"]),
                "penalty_mmtc": np.float32(reward_terms["penalty_mmtc"]),
                "penalty_raw": np.array(
                    [
                        reward_terms["penalty_raw_embb"],
                        reward_terms["penalty_raw_urllc"],
                        reward_terms["penalty_raw_mmtc"],
                    ],
                    dtype=np.float32,
                ),
                "penalties": np.array(
                    [
                        reward_terms["penalty_embb"],
                        reward_terms["penalty_urllc"],
                        reward_terms["penalty_mmtc"],
                    ],
                    dtype=np.float32,
                ),
            }

        # --- Cooperative Reward (MARL) ---
        # Agent reward is local reward + average reward of system
        total_system_reward = sum(agent_rewards.values())
        avg_system_reward = total_system_reward / len(self.agents)
        
        # 合作比例: 0.5 * 本地奖励 + 0.5 * 全局平均奖励
        alpha = self.cooperative_alpha
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
        1) warning term activates shortly before deadline,
        2) tail term increases in the last pre-deadline region,
        3) crossing the deadline adds a fixed hit plus super-linear overflow.
        """
        max_delay = self.sla_props["urllc_max_delay"]
        delay_ratio = float(est_delay / max_delay)
        slope = self.urllc_softplus_slope

        warning_term = self._softplus(slope * (delay_ratio - self.urllc_warning_ratio)) / slope
        tail_excess = max(0.0, delay_ratio - self.urllc_tail_ratio)
        overflow_ratio = max(0.0, delay_ratio - 1.0)
        overflow_ratio = min(overflow_ratio, 3.0)
        overflow_term = np.expm1(self.urllc_exp_coeff * overflow_ratio)
        hard_violation_term = 1.0 if overflow_ratio > 0.0 else 0.0

        return float(
            self.w_urllc
            * (
                self.urllc_warning_gain * warning_term
                + self.urllc_tail_quad_gain * (tail_excess ** 2)
                + self.urllc_hard_violation_gain * hard_violation_term
                + self.urllc_overflow_gain * overflow_term
            )
        )

    def _get_neighbor_prev_action_mean(self, agent):
        neighbor_ids = self.neighbor_map.get(agent, [])
        if not neighbor_ids:
            return DEFAULT_INITIAL_SLICE_RATIOS.copy()

        prev_actions = np.array([self.state[neighbor][9:12] for neighbor in neighbor_ids], dtype=np.float32)
        return np.mean(prev_actions, axis=0).astype(np.float32)

    def _get_neighbor_urgency_features(self, agent):
        neighbor_ids = self.neighbor_map.get(agent, [])
        if not neighbor_ids:
            return np.zeros(3, dtype=np.float32)

        neighbor_urllc_queue_mean = float(np.mean([self.queues[neighbor][1] for neighbor in neighbor_ids]))
        neighbor_urllc_delay_ratio_mean = float(
            np.mean([self.state[neighbor][12] / self.sla_props["urllc_max_delay"] for neighbor in neighbor_ids])
        )
        neighbor_embb_shortfall_mean = float(np.mean([self.state[neighbor][13] for neighbor in neighbor_ids]))
        return np.array(
            [neighbor_urllc_queue_mean, neighbor_urllc_delay_ratio_mean, neighbor_embb_shortfall_mean],
            dtype=np.float32,
        )

    def _build_obs(self, agent):
        if self.observation_mode == "pure_local":
            obs = np.zeros(14, dtype=np.float32)
            obs[0:14] = self.state[agent]
        else:
            obs = np.zeros(20, dtype=np.float32)
            obs[0:14] = self.state[agent]
            obs[14:17] = self._get_neighbor_prev_action_mean(agent)
            obs[17:20] = self._get_neighbor_urgency_features(agent)
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

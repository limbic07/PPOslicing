from collections import deque

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

from ippo_rl_module import CENTRALIZED_CRITIC_GLOBAL_DIM, resolve_local_obs_dim

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
        self.neighbor_liability_beta = float(cfg_value("neighbor_liability_beta", 0.5))
        self.action_softmax_temperature = float(cfg_value("action_softmax_temperature", 1.0))
        self.observation_mode = str(cfg_value("observation_mode", "pure_local")).lower()
        self.include_ici_features = bool(cfg_value("neighbor_augmented_include_ici_features", False))
        self.cooperative_target = str(cfg_value("cooperative_target", "all")).lower()
        self.neighbor_urgency_pooling = str(cfg_value("neighbor_urgency_pooling", "max")).lower()
        self.use_centralized_critic = bool(cfg_value("use_centralized_critic", False))
        self.centralized_critic_global_dim = int(
            cfg_value("centralized_critic_global_dim", CENTRALIZED_CRITIC_GLOBAL_DIM)
        )
        self.expected_centralized_critic_global_dim = self.num_cells * 12
        if (
            self.use_centralized_critic
            and self.centralized_critic_global_dim != self.expected_centralized_critic_global_dim
        ):
            raise ValueError(
                "centralized_critic_global_dim must match flattened global feature size: "
                f"{self.expected_centralized_critic_global_dim} (num_cells={self.num_cells}, per_cell=12). "
                f"Got {self.centralized_critic_global_dim}."
            )
        if self.centralized_critic_global_dim < 0:
            raise ValueError("centralized_critic_global_dim must be >= 0")
        if self.action_softmax_temperature <= 0.0:
            raise ValueError("action_softmax_temperature must be > 0")
        if self.neighbor_liability_beta < 0.0:
            raise ValueError("neighbor_liability_beta must be >= 0")
        if self.observation_mode not in {"pure_local", "neighbor_augmented"}:
            raise ValueError(
                f"Unsupported observation_mode={self.observation_mode!r}. "
                "Expected one of ['neighbor_augmented', 'pure_local']"
            )
        if self.neighbor_urgency_pooling not in {"mean", "max"}:
            raise ValueError(
                f"Unsupported neighbor_urgency_pooling={self.neighbor_urgency_pooling!r}. "
                "Expected one of ['mean', 'max']"
            )
        if self.cooperative_target not in {"all", "embb_only"}:
            raise ValueError(
                f"Unsupported cooperative_target={self.cooperative_target!r}. "
                "Expected one of ['all', 'embb_only']"
            )

        # --- Reward shaping params (tunable) ---
        # Continuous URLLC soft-cliff: warning before 2ms and sharp increase after deadline.
        self.penalty_weight = float(cfg_value("penalty_weight", 0.5))
        self.w_embb = float(cfg_value("w_embb", self.penalty_weight))
        self.w_urllc = float(cfg_value("w_urllc", self.penalty_weight))
        self.w_mmtc = float(cfg_value("w_mmtc", self.penalty_weight))
        self.urllc_warning_ratio = float(cfg_value("urllc_warning_ratio", 0.9))
        self.urllc_tail_ratio = float(cfg_value("urllc_tail_ratio", 0.92))
        self.urllc_softplus_slope = float(cfg_value("urllc_softplus_slope", 12.0))
        self.urllc_warning_gain = float(cfg_value("urllc_warning_gain", 0.2))
        self.urllc_tail_quad_gain = float(cfg_value("urllc_tail_quad_gain", 2.0))
        self.urllc_hard_violation_gain = float(cfg_value("urllc_hard_violation_gain", 1.75))
        self.urllc_overflow_gain = float(cfg_value("urllc_overflow_gain", 3.0))
        self.urllc_exp_coeff = float(cfg_value("urllc_exp_coeff", 1.6))
        self.urllc_penalty_cap_factor = float(cfg_value("urllc_penalty_cap_factor", 10.0))
        self.embb_penalty_quad_gain = float(cfg_value("embb_penalty_quad_gain", 1.2))
        self.embb_penalty_cubic_gain = float(cfg_value("embb_penalty_cubic_gain", 0.0))
        self.embb_penalty_cap_factor = float(cfg_value("embb_penalty_cap_factor", 10.0))
        self.mmtc_penalty_cap_factor = float(cfg_value("mmtc_penalty_cap_factor", 5.0))
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
        self.urllc_regional_event_start_prob = float(cfg_value("urllc_regional_event_start_prob", 0.0))
        self.urllc_regional_event_end_prob = float(cfg_value("urllc_regional_event_end_prob", 0.0))
        self.urllc_regional_event_start_boost = float(cfg_value("urllc_regional_event_start_boost", 0.0))
        self.urllc_regional_event_burst_mean_boost_mbps = float(
            cfg_value("urllc_regional_event_burst_mean_boost_mbps", 0.0)
        )
        self.embb_mean_mbps = float(cfg_value("embb_mean_mbps", 250.0))
        self.embb_std_mbps = float(cfg_value("embb_std_mbps", 40.0))
        self.embb_clip_low_mbps = float(cfg_value("embb_clip_low_mbps", 180.0))
        self.embb_clip_high_mbps = float(cfg_value("embb_clip_high_mbps", 350.0))
        self.embb_hotspot_rho = float(cfg_value("embb_hotspot_rho", 0.0))
        self.embb_hotspot_std_mbps = float(cfg_value("embb_hotspot_std_mbps", 0.0))
        self.embb_hotspot_clip_mbps = float(cfg_value("embb_hotspot_clip_mbps", 0.0))
        self.embb_center_bias_mbps = float(cfg_value("embb_center_bias_mbps", 0.0))
        self.embb_edge_bias_mbps = float(cfg_value("embb_edge_bias_mbps", 0.0))
        self.urllc_center_burst_start_bias = float(cfg_value("urllc_center_burst_start_bias", 0.0))
        self.urllc_edge_burst_start_bias = float(cfg_value("urllc_edge_burst_start_bias", 0.0))
        self.interference_neighbor_normalization = str(
            cfg_value("interference_neighbor_normalization", "fixed_max")
        ).lower()
        self.tail_reward_throughput_weight = float(cfg_value("tail_reward_throughput_weight", 1.0 / 300.0))
        self.binary_reward_throughput_scale = float(cfg_value("binary_reward_throughput_scale", 60.0))
        self.binary_penalty_embb = float(cfg_value("binary_penalty_embb", 3.0))
        self.binary_penalty_urllc = float(cfg_value("binary_penalty_urllc", 4.0))
        self.binary_penalty_mmtc = float(cfg_value("binary_penalty_mmtc", 3.0))
        self.binary_urllc_yellow_start_ratio = float(cfg_value("binary_urllc_yellow_start_ratio", 0.5))
        self.binary_urllc_yellow_penalty = float(
            cfg_value("binary_urllc_yellow_penalty", self.binary_penalty_urllc)
        )
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
        self.neighbor_penalty_degree_ref = float(cfg_value("neighbor_penalty_degree_ref", 3.0))
        self.neighbor_dividend_gamma = float(cfg_value("neighbor_dividend_gamma", 1.0))
        self.neighbor_dividend_penalty_weight = float(cfg_value("neighbor_dividend_penalty_weight", 1.0))
        self.neighbor_dividend_throughput_weight = float(cfg_value("neighbor_dividend_throughput_weight", 0.25))
        self.neighbor_dividend_max = float(cfg_value("neighbor_dividend_max", 1.5))
        self.neighbor_dividend_gate_urllc_delay_ratio = float(
            cfg_value("neighbor_dividend_gate_urllc_delay_ratio", 0.85)
        )
        self.neighbor_dividend_gate_embb_shortfall_ratio = float(
            cfg_value("neighbor_dividend_gate_embb_shortfall_ratio", 0.10)
        )
        self.center_reward_scale = float(cfg_value("center_reward_scale", 1.0))
        self.reward_clip_abs = float(cfg_value("reward_clip_abs", 10.0))
        if self.neighbor_penalty_degree_ref <= 0.0:
            raise ValueError("neighbor_penalty_degree_ref must be > 0")
        if self.neighbor_dividend_gamma < 0.0:
            raise ValueError("neighbor_dividend_gamma must be >= 0")
        if self.neighbor_dividend_max < 0.0:
            raise ValueError("neighbor_dividend_max must be >= 0")
        if self.neighbor_dividend_gate_urllc_delay_ratio < 0.0:
            raise ValueError("neighbor_dividend_gate_urllc_delay_ratio must be >= 0")
        if self.neighbor_dividend_gate_embb_shortfall_ratio < 0.0:
            raise ValueError("neighbor_dividend_gate_embb_shortfall_ratio must be >= 0")

        # --- Spaces ---
        # Per-agent spaces required by RLlib env_runner + connector v2 stack.
        # Obs dims depend on observation_mode:
        # pure_local=9, neighbor_augmented=15 or 19 (with ICI features).
        self._agent_action_space = spaces.Box(
            low=np.float32(-1.0), high=np.float32(1.0), shape=(3,), dtype=np.float32
        )
        self.local_obs_dim = resolve_local_obs_dim(
            self.observation_mode,
            include_ici_features=self.include_ici_features,
        )
        critic_context_dim = self.centralized_critic_global_dim if self.use_centralized_critic else 0
        self.obs_dim = self.local_obs_dim + critic_context_dim
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
        self.embb_hotspot_bias_mbps = {agent: np.float32(0.0) for agent in self.agents}
        self.embb_tp_history = {
            agent: deque(maxlen=self.embb_sla_window_tti) for agent in self.agents
        }
        
        self.max_steps = 200
        self.current_step = 0
        self.burst_state = {agent: False for agent in self.agents}
        self.regional_urllc_event_active = False
        self.regional_urllc_event_anchor = None
        self.prev_neighbor_penalty_signal = {agent: 0.0 for agent in self.agents}
        self.prev_neighbor_throughput_signal = {agent: 0.0 for agent in self.agents}
        self.np_random, _ = np_random(None)

    @staticmethod
    def _get_env_profile_overrides(env_profile):
        profiles = {
            "harsh": {
                "embb_sla_window_tti": 1,
                "reward_mode": "tail_risk_coop",
                "cooperative_alpha": 0.7,
                "neighbor_liability_beta": 0.35,
                "neighbor_penalty_degree_ref": 3.0,
                "neighbor_dividend_gamma": 0.0,
                "neighbor_dividend_penalty_weight": 1.0,
                "neighbor_dividend_throughput_weight": 0.25,
                "neighbor_dividend_max": 1.5,
                "neighbor_dividend_gate_urllc_delay_ratio": 0.85,
                "neighbor_dividend_gate_embb_shortfall_ratio": 0.10,
                "neighbor_urgency_pooling": "max",
                "action_softmax_temperature": 1.0,
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
                "embb_gbr": 200.0,
                "embb_sla_window_tti": 20,
                "reward_mode": "archive_local_sla",
                "cooperative_alpha": 0.5,
                "neighbor_liability_beta": 0.35,
                "neighbor_penalty_degree_ref": 3.0,
                "neighbor_dividend_gamma": 1.0,
                "neighbor_dividend_penalty_weight": 1.0,
                "neighbor_dividend_throughput_weight": 0.25,
                "neighbor_dividend_max": 1.5,
                "neighbor_dividend_gate_urllc_delay_ratio": 0.85,
                "neighbor_dividend_gate_embb_shortfall_ratio": 0.10,
                "neighbor_urgency_pooling": "max",
                "action_softmax_temperature": 1.0,
                "urllc_burst_start_prob": 0.06,
                "urllc_burst_end_prob": 0.25,
                "urllc_burst_mean_mbps": 100.0,
                "urllc_burst_std_mbps": 15.0,
                "urllc_regional_event_start_prob": 0.04,
                "urllc_regional_event_end_prob": 0.25,
                "urllc_regional_event_start_boost": 0.18,
                "urllc_regional_event_burst_mean_boost_mbps": 20.0,
                "embb_mean_mbps": 250.0,
                "embb_std_mbps": 40.0,
                "embb_clip_low_mbps": 180.0,
                "embb_clip_high_mbps": 350.0,
                "embb_hotspot_rho": 0.92,
                "embb_hotspot_std_mbps": 24.0,
                "embb_hotspot_clip_mbps": 55.0,
                "embb_center_bias_mbps": 15.0,
                "embb_edge_bias_mbps": 0.0,
                "urllc_center_burst_start_bias": 0.0,
                "urllc_edge_burst_start_bias": 0.01,
                "ici_gain": 0.42,
                "se_modifier_floor": 0.50,
                "interference_neighbor_normalization": "actual_neighbors",
                "binary_reward_throughput_scale": 45.0,
                "binary_penalty_embb": 2.0,
                "binary_penalty_urllc": 3.0,
                "binary_penalty_mmtc": 2.0,
                "binary_urllc_yellow_start_ratio": 0.65,
                "binary_urllc_yellow_penalty": 2.5,
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
        self.regional_urllc_event_active = False
        self.regional_urllc_event_anchor = None
        self.prev_neighbor_penalty_signal = {agent: 0.0 for agent in self.agents}
        self.prev_neighbor_throughput_signal = {agent: 0.0 for agent in self.agents}

        obs = {}
        self._update_regional_urllc_event()
        for agent in self.agents:
            self.state[agent].fill(0.0)
            self.queues[agent] = np.zeros(3, dtype=np.float32)
            self.embb_tp_history[agent].clear()
            self.burst_state[agent] = False
            self.embb_hotspot_bias_mbps[agent] = np.float32(0.0)
            self.current_se[agent] = (
                self.mean_se + self.np_random.normal(0.0, 0.2, size=3)
            ).astype(np.float32)
            self._update_agent_state(agent)
            obs[agent] = self._build_obs(agent)

        return obs, {}

    def _is_center_agent(self, agent):
        return agent == "BS_0"

    def _get_embb_role_bias_mbps(self, agent):
        return self.embb_center_bias_mbps if self._is_center_agent(agent) else self.embb_edge_bias_mbps

    def _get_urllc_burst_start_bias(self, agent):
        return self.urllc_center_burst_start_bias if self._is_center_agent(agent) else self.urllc_edge_burst_start_bias

    def _regional_urllc_affected_agents(self):
        if not self.regional_urllc_event_active or self.regional_urllc_event_anchor is None:
            return set()
        anchor = self.regional_urllc_event_anchor
        return {anchor, *self.neighbor_map.get(anchor, [])}

    def _update_regional_urllc_event(self):
        if self.urllc_regional_event_start_prob <= 0.0:
            self.regional_urllc_event_active = False
            self.regional_urllc_event_anchor = None
            return

        if self.regional_urllc_event_active:
            if self.np_random.random() < self.urllc_regional_event_end_prob:
                self.regional_urllc_event_active = False
                self.regional_urllc_event_anchor = None
        else:
            if self.np_random.random() < self.urllc_regional_event_start_prob:
                self.regional_urllc_event_active = True
                self.regional_urllc_event_anchor = str(self.np_random.choice(self.agents))

    def _update_agent_state(self, agent):
        # 1. eMBB: Heavy background traffic with temporally correlated hotspot bias.
        if self.embb_hotspot_rho > 0.0 and self.embb_hotspot_std_mbps > 0.0 and self.embb_hotspot_clip_mbps > 0.0:
            hotspot_noise = float(self.np_random.normal(0.0, self.embb_hotspot_std_mbps))
            hotspot_bias = (
                self.embb_hotspot_rho * float(self.embb_hotspot_bias_mbps[agent])
                + np.sqrt(max(1.0 - (self.embb_hotspot_rho ** 2), 0.0)) * hotspot_noise
            )
            hotspot_bias = float(
                np.clip(hotspot_bias, -self.embb_hotspot_clip_mbps, self.embb_hotspot_clip_mbps)
            )
            self.embb_hotspot_bias_mbps[agent] = np.float32(hotspot_bias)
        else:
            self.embb_hotspot_bias_mbps[agent] = np.float32(0.0)

        embb_target_mean_mbps = (
            self.embb_mean_mbps
            + float(self.embb_hotspot_bias_mbps[agent])
            + self._get_embb_role_bias_mbps(agent)
        )
        arr_embb = np.float32(
            np.clip(
                self.np_random.normal(embb_target_mean_mbps, self.embb_std_mbps),
                self.embb_clip_low_mbps,
                self.embb_clip_high_mbps,
            )
        )

        # 2. URLLC: Markov Burst State (Sustained bursts to kill static allocation)
        regional_agents = self._regional_urllc_affected_agents()
        regional_event_affects_agent = agent in regional_agents
        burst_start_prob = self.urllc_burst_start_prob + self._get_urllc_burst_start_bias(agent)
        if regional_event_affects_agent:
            burst_start_prob += self.urllc_regional_event_start_boost
        burst_start_prob = float(np.clip(burst_start_prob, 0.0, 1.0))

        if self.burst_state[agent]:
            if self.np_random.random() < self.urllc_burst_end_prob:
                self.burst_state[agent] = False
        else:
            if self.np_random.random() < burst_start_prob:
                self.burst_state[agent] = True

        burst_mean_mbps = self.urllc_burst_mean_mbps
        if regional_event_affects_agent:
            burst_mean_mbps += self.urllc_regional_event_burst_mean_boost_mbps

        if self.burst_state[agent]:
            arr_urllc = np.float32(
                np.clip(
                    self.np_random.normal(burst_mean_mbps, self.urllc_burst_std_mbps),
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

    def _estimate_neighbor_normalized_load(self, agent):
        neighbor_ids = self.neighbor_map.get(agent, [])
        if not neighbor_ids:
            return np.zeros(3, dtype=np.float32)

        prev_actions = np.array([self.state[neighbor][9:12] for neighbor in neighbor_ids], dtype=np.float32)
        if self.interference_neighbor_normalization == "actual_neighbors":
            normalizer = float(len(neighbor_ids))
        elif self.interference_neighbor_normalization == "fixed_max":
            normalizer = self.max_neighbors
        else:
            raise ValueError(
                "Unsupported interference_neighbor_normalization="
                f"{self.interference_neighbor_normalization!r}"
            )
        return (np.sum(prev_actions, axis=0) / max(normalizer, 1.0)).astype(np.float32)

    def _get_neighbor_ici_features(self, agent):
        normalized_neighbor_load = self._estimate_neighbor_normalized_load(agent)
        se_modifiers = np.clip(
            1.0 - (self.ici_gain * normalized_neighbor_load),
            self.se_modifier_floor,
            1.0,
        ).astype(np.float32)
        return np.array(
            [
                normalized_neighbor_load[0],
                normalized_neighbor_load[1],
                se_modifiers[0],
                se_modifiers[1],
            ],
            dtype=np.float32,
        )

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

    def _compute_binary_sla_reward(self, throughput_mbps, violation_flags, est_delay):
        violation_flags = np.asarray(violation_flags, dtype=np.float32)
        throughput_bonus = float(throughput_mbps / max(self.binary_reward_throughput_scale, 1e-6))
        pen_embb = float(self.binary_penalty_embb * violation_flags[0])
        max_delay = max(self.sla_props["urllc_max_delay"], 1e-6)
        urllc_delay_ratio = max(float(est_delay), 0.0) / max_delay
        yellow_start = float(np.clip(self.binary_urllc_yellow_start_ratio, 0.0, 0.999999))
        yellow_progress = float(np.clip((urllc_delay_ratio - yellow_start) / max(1.0 - yellow_start, 1e-6), 0.0, 1.0))
        pen_urllc_yellow = float(self.binary_urllc_yellow_penalty * yellow_progress)
        pen_urllc_red = float(self.binary_penalty_urllc * violation_flags[1])
        pen_urllc = float(pen_urllc_red + pen_urllc_yellow)
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
            "penalty_urllc_yellow": pen_urllc_yellow,
            "penalty_urllc_red": pen_urllc_red,
        }

    def _compute_archive_local_reward(self, achieved_throughput_mbps, est_delay, mmtc_overflow_flag):
        throughput_slices = np.asarray(achieved_throughput_mbps, dtype=np.float32)
        throughput_total = float(np.sum(throughput_slices))

        embb_flag = float(throughput_slices[0] < self.sla_props["embb_gbr"])
        urllc_flag = float(est_delay > self.sla_props["urllc_max_delay"])
        mmtc_flag = float(bool(mmtc_overflow_flag))

        throughput_bonus = float(throughput_total / 100.0)
        pen_embb = 10.0 * embb_flag
        pen_urllc = 20.0 * urllc_flag
        pen_mmtc = 10.0 * mmtc_flag
        penalty_total = float(pen_embb + pen_urllc + pen_mmtc)
        reward_local = float(throughput_bonus - penalty_total)

        return {
            "local_reward": reward_local,
            "reward_base_tp": throughput_bonus,
            "reward_sla_bonus": 0.0,
            "penalty_total": penalty_total,
            "penalty_raw_embb": float(pen_embb),
            "penalty_raw_urllc": float(pen_urllc),
            "penalty_raw_mmtc": float(pen_mmtc),
            "penalty_embb": float(pen_embb),
            "penalty_urllc": float(pen_urllc),
            "penalty_mmtc": float(pen_mmtc),
            "local_archive_violation_flags": np.array([embb_flag, urllc_flag, mmtc_flag], dtype=np.float32),
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
        self._update_regional_urllc_event()

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
                reward_terms = self._compute_binary_sla_reward(throughput_mbps_total, violation_flags, est_delay)
            elif self.reward_mode == "archive_local_sla":
                reward_terms = self._compute_archive_local_reward(
                    achieved_throughput_mbps,
                    est_delay,
                    queue_excess > 0.0,
                )
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
                "embb_hotspot_bias_mbps": np.float32(self.embb_hotspot_bias_mbps[agent]),
                "embb_sla_window_tti": np.int32(self.embb_sla_window_tti),
                "est_urllc_delay": est_delay,
                "urllc_delay_ratio": est_delay / self.sla_props["urllc_max_delay"],
                "regional_urllc_event_active": np.float32(self.regional_urllc_event_active),
                "regional_urllc_event_affects_agent": np.float32(agent in self._regional_urllc_affected_agents()),
                "neighbor_prev_action_mean": self._get_neighbor_prev_action_mean(agent),
                "neighbor_urgency_features": self._get_neighbor_urgency_features(agent),
                "neighbor_ici_features": self._get_neighbor_ici_features(agent),
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
                "penalty_urllc_yellow": np.float32(reward_terms.get("penalty_urllc_yellow", 0.0)),
                "penalty_urllc_red": np.float32(reward_terms.get("penalty_urllc_red", 0.0)),
                "local_archive_violation_flags": np.asarray(
                    reward_terms.get("local_archive_violation_flags", violation_flags),
                    dtype=np.float32,
                ),
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

        # --- One-hop Neighborhood Joint Liability + Dividend Reward (MARL) ---
        # Reward_i = alpha * local_reward_i - beta * P_nbr_i + dividend_i
        # where:
        #   P_nbr_i = sum_{j in N(i)} penalty_j / degree_ref
        #   dividend_i = gate_i * clip(
        #       gamma * (w_p * max(0, P_nbr_prev_i - P_nbr_i) +
        #                w_t * max(0, T_nbr_i - T_nbr_prev_i)),
        #       0, dividend_max
        #   )
        alpha = float(self.cooperative_alpha)
        beta = float(self.neighbor_liability_beta)
        degree_ref = max(float(self.neighbor_penalty_degree_ref), 1e-6)
        throughput_norm = max(float(self.binary_reward_throughput_scale) * degree_ref, 1e-6)
        next_neighbor_penalty_signal = {}
        next_neighbor_throughput_signal = {}

        for agent in self.agents:
            if agent not in agent_rewards or agent not in infos:
                continue

            local_reward_for_coop = float(agent_rewards[agent])
            local_penalty = float(infos[agent].get("penalty_total", 0.0))
            local_net_gain = local_reward_for_coop + local_penalty

            neighbor_only_penalty_total = 0.0
            neighbor_only_throughput_total = 0.0
            external_neighbor_count = 0
            for neighbor in self.neighbor_map.get(agent, []):
                neighbor_info = infos.get(neighbor)
                if neighbor_info is None:
                    continue
                if self.cooperative_target == "embb_only":
                    neighbor_only_penalty_total += float(neighbor_info.get("penalty_embb", 0.0))
                    embb_tp = float(neighbor_info.get("embb_eval_tp_mbps", 0.0))
                    neighbor_only_throughput_total += embb_tp / max(self.sla_props["embb_gbr"], 1e-6)
                else:
                    neighbor_only_penalty_total += float(neighbor_info.get("penalty_total", 0.0))
                    neighbor_only_throughput_total += float(neighbor_info.get("throughput", 0.0))
                external_neighbor_count += 1

            neighbor_only_penalty_mean = neighbor_only_penalty_total / max(external_neighbor_count, 1)
            neighbor_penalty_signal = neighbor_only_penalty_total / degree_ref
            neighbor_throughput_signal = neighbor_only_throughput_total / throughput_norm
            prev_neighbor_penalty_signal = self.prev_neighbor_penalty_signal.get(agent, neighbor_penalty_signal)
            prev_neighbor_throughput_signal = self.prev_neighbor_throughput_signal.get(agent, neighbor_throughput_signal)
            if self.current_step <= 1:
                prev_neighbor_penalty_signal = neighbor_penalty_signal
                prev_neighbor_throughput_signal = neighbor_throughput_signal
            neighbor_penalty_improve = max(0.0, prev_neighbor_penalty_signal - neighbor_penalty_signal)
            neighbor_throughput_improve = max(0.0, neighbor_throughput_signal - prev_neighbor_throughput_signal)
            neighborhood_penalty_total = local_penalty + neighbor_only_penalty_total
            neighborhood_size = 1 + external_neighbor_count
            neighborhood_penalty_mean = neighborhood_penalty_total / max(neighborhood_size, 1)
            embb_shortfall_ratio = float(infos[agent].get("violations_raw", np.zeros(3, dtype=np.float32))[0])
            urllc_delay_ratio = float(infos[agent].get("urllc_delay_ratio", 0.0))
            urllc_gate_term = urllc_delay_ratio / max(self.neighbor_dividend_gate_urllc_delay_ratio, 1e-6)
            embb_gate_term = embb_shortfall_ratio / max(self.neighbor_dividend_gate_embb_shortfall_ratio, 1e-6)
            dividend_gate = float(1.0 / (1.0 + urllc_gate_term + embb_gate_term))
            neighbor_dividend_raw = float(
                self.neighbor_dividend_gamma
                * (
                    self.neighbor_dividend_penalty_weight * neighbor_penalty_improve
                    + self.neighbor_dividend_throughput_weight * neighbor_throughput_improve
                )
            )
            neighbor_dividend = float(
                dividend_gate * np.clip(neighbor_dividend_raw, 0.0, self.neighbor_dividend_max)
            )

            reward_local_component = alpha * local_reward_for_coop
            reward_neighbor_component = -beta * neighbor_penalty_signal
            reward_dividend_component = neighbor_dividend
            reward_final = reward_local_component + reward_neighbor_component + reward_dividend_component
            reward_final = self._clip_reward(reward_final)
            if not np.isfinite(reward_final):
                reward_final = -100.0
            rewards[agent] = reward_final
            next_neighbor_penalty_signal[agent] = neighbor_penalty_signal
            next_neighbor_throughput_signal[agent] = neighbor_throughput_signal

            infos[agent]["local_reward_for_coop"] = np.float32(local_reward_for_coop)
            infos[agent]["local_penalty_total"] = np.float32(local_penalty)
            infos[agent]["local_net_gain"] = np.float32(local_net_gain)
            infos[agent]["neighbor_only_penalty_total"] = np.float32(neighbor_only_penalty_total)
            infos[agent]["neighbor_only_penalty_mean"] = np.float32(neighbor_only_penalty_mean)
            infos[agent]["neighbor_penalty_signal"] = np.float32(neighbor_penalty_signal)
            infos[agent]["neighbor_throughput_signal"] = np.float32(neighbor_throughput_signal)
            infos[agent]["neighbor_prev_penalty_signal"] = np.float32(prev_neighbor_penalty_signal)
            infos[agent]["neighbor_prev_throughput_signal"] = np.float32(prev_neighbor_throughput_signal)
            infos[agent]["neighbor_penalty_improve"] = np.float32(neighbor_penalty_improve)
            infos[agent]["neighbor_throughput_improve"] = np.float32(neighbor_throughput_improve)
            infos[agent]["neighbor_dividend_gate"] = np.float32(dividend_gate)
            infos[agent]["neighbor_dividend_raw"] = np.float32(neighbor_dividend_raw)
            infos[agent]["neighbor_dividend"] = np.float32(neighbor_dividend)
            infos[agent]["neighbor_count"] = np.float32(external_neighbor_count)
            infos[agent]["neighborhood_penalty_total"] = np.float32(neighborhood_penalty_total)
            infos[agent]["neighborhood_penalty_mean"] = np.float32(neighborhood_penalty_mean)
            infos[agent]["neighborhood_size"] = np.float32(neighborhood_size)
            infos[agent]["cooperative_alpha"] = np.float32(alpha)
            infos[agent]["cooperative_beta"] = np.float32(beta)
            infos[agent]["neighbor_penalty_degree_ref"] = np.float32(self.neighbor_penalty_degree_ref)
            infos[agent]["neighbor_dividend_gamma"] = np.float32(self.neighbor_dividend_gamma)
            infos[agent]["neighbor_dividend_penalty_weight"] = np.float32(self.neighbor_dividend_penalty_weight)
            infos[agent]["neighbor_dividend_throughput_weight"] = np.float32(self.neighbor_dividend_throughput_weight)
            infos[agent]["neighbor_dividend_max"] = np.float32(self.neighbor_dividend_max)
            infos[agent]["cooperative_target"] = self.cooperative_target
            infos[agent]["reward_final"] = np.float32(reward_final)
            # Keep legacy metric names for downstream scripts/dashboards.
            infos[agent]["neighbor_coop_term"] = np.float32(-neighbor_penalty_signal)
            infos[agent]["neighbor_coop_weight_sum"] = np.float32(external_neighbor_count)
            infos[agent]["neighbor_coop_weighted_risk_sum"] = np.float32(neighbor_only_penalty_total)
            infos[agent]["neighbor_coop_risk_mean"] = np.float32(neighbor_penalty_signal)
            infos[agent]["reward_local_component"] = np.float32(reward_local_component)
            infos[agent]["reward_neighbor_component"] = np.float32(reward_neighbor_component)
            infos[agent]["reward_dividend_component"] = np.float32(reward_dividend_component)

        self.prev_neighbor_penalty_signal.update(next_neighbor_penalty_signal)
        self.prev_neighbor_throughput_signal.update(next_neighbor_throughput_signal)

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
            return np.zeros(3, dtype=np.float32)

        prev_actions = np.array([self.state[neighbor][9:12] for neighbor in neighbor_ids], dtype=np.float32)
        return np.mean(prev_actions, axis=0).astype(np.float32)

    def _pool_neighbor_values(self, values) -> float:
        values_arr = np.asarray(values, dtype=np.float32)
        if values_arr.size == 0:
            return 0.0
        if self.neighbor_urgency_pooling == "max":
            return float(np.max(values_arr))
        return float(np.mean(values_arr))

    def _get_neighbor_urgency_features(self, agent):
        neighbor_ids = self.neighbor_map.get(agent, [])
        if not neighbor_ids:
            return np.zeros(3, dtype=np.float32)

        neighbor_urllc_queue_stat = self._pool_neighbor_values(
            [self.queues[neighbor][1] for neighbor in neighbor_ids]
        )
        neighbor_urllc_delay_ratio_stat = self._pool_neighbor_values(
            [self.state[neighbor][12] / self.sla_props["urllc_max_delay"] for neighbor in neighbor_ids]
        )
        neighbor_embb_shortfall_stat = self._pool_neighbor_values(
            [self.state[neighbor][13] for neighbor in neighbor_ids]
        )
        return np.array(
            [neighbor_urllc_queue_stat, neighbor_urllc_delay_ratio_stat, neighbor_embb_shortfall_stat],
            dtype=np.float32,
        )

    def _build_obs(self, agent):
        if self.observation_mode == "pure_local":
            local_obs = np.zeros(9, dtype=np.float32)
            local_obs[0:3] = self.state[agent][0:3]
            local_obs[3:6] = self.state[agent][3:6]
            local_obs[6:9] = self.state[agent][6:9]
        else:
            local_obs = np.zeros(self.local_obs_dim, dtype=np.float32)
            local_obs[0:3] = self.state[agent][0:3]
            local_obs[3:6] = self.state[agent][3:6]
            local_obs[6:9] = self.state[agent][6:9]
            local_obs[9:12] = self._get_neighbor_prev_action_mean(agent)
            local_obs[12:15] = self._get_neighbor_urgency_features(agent)
            if self.include_ici_features:
                local_obs[15:19] = self._get_neighbor_ici_features(agent)

        if not self.use_centralized_critic:
            return np.nan_to_num(local_obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

        global_ctx = self._get_centralized_critic_global_features()
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[: self.local_obs_dim] = local_obs
        obs[self.local_obs_dim :] = global_ctx
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _get_centralized_critic_global_features(self):
        # Preserve spatial topology: flatten per-cell features in fixed BS order.
        per_cell_features = []
        for agent in self.agents:
            per_cell_features.extend(
                [
                    self.state[agent][0:3],      # demand/arrivals
                    self.queues[agent],          # current queue sizes
                    self.current_se[agent],      # current SE
                    self.state[agent][9:12],     # previous action ratios
                ]
            )
        features = np.concatenate(per_cell_features, axis=0).astype(np.float32)

        if features.size != self.centralized_critic_global_dim:
            raise ValueError(
                "Flattened centralized critic feature size mismatch. "
                f"expected={self.centralized_critic_global_dim}, got={features.size}"
            )
        return features

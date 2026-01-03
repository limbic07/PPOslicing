import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FiveG_SLA_Env(gym.Env):
    """
    Advanced 5G Slicing Environment (V4 - Safe Barrier Reward).
    Features:
    1. 3GPP Physics: 20MHz Bandwidth, 0.5ms TTI.
    2. Dynamic Load: Randomized load factors during training for robustness.
    3. Reward Shaping: Exponential Barrier Function with numerical clamping.
    """

    def __init__(self):
        super(FiveG_SLA_Env, self).__init__()

        # --- 1. 物理层参数 (Aligned with 3GPP) ---
        self.total_bandwidth = 20e6  # 20 MHz (Resource Constrained)
        self.duration_tti = 0.5e-3  # 0.5 ms (Numerology 1)

        # --- 2. SLA 约束 (Strict Mode) ---
        self.sla_props = {
            'embb_gbr': 40.0,  # GBR 40 Mbps
            'urllc_max_delay': 0.002,  # Latency < 2ms (Strict!)
            'mmtc_max_queue': 5.0
        }

        # 动作空间: [-1, 1] 连续值 (PPO 友好)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 状态空间: 9维 [Traffic(3), Queues(3), SE(3)]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(9,), dtype=np.float32)

        # 内部变量初始化
        self.state = np.zeros(9, dtype=np.float32)
        self.queues = np.zeros(3, dtype=np.float32)
        self.load_factor = 1.0
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queues = np.zeros(3, dtype=np.float32)

        # 🌟 关键：动态负载训练 (Domain Randomization)
        # 每一局开始时，随机设定系统负载倍率 (0.5倍 到 1.6倍)
        # 这迫使 Agent 既学会处理空闲，也学会处理极端拥塞
        self.load_factor = np.random.uniform(0.5, 1.6)

        self._update_state()
        return self.state.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1

        # --- 1. 物理层计算 ---
        # 动作映射 (Softmax)
        action = np.clip(action, -10, 10)
        exp_action = np.exp(action)
        ratios = exp_action / np.sum(exp_action)
        bw_allocated = ratios * self.total_bandwidth

        # 计算传输能力 (Capacity)
        se = self.state[6:9]
        service_rate_mbps = (bw_allocated * se) / 1e6
        service_capacity_mb = service_rate_mbps * self.duration_tti

        # 排队演进 (Queue Dynamics)
        arrivals_mb = self.state[0:3] * self.duration_tti
        self.queues += arrivals_mb
        served_mb = np.minimum(service_capacity_mb, self.queues)
        self.queues -= served_mb
        achieved_throughput_mbps = served_mb / self.duration_tti

        # --- 2. SLA 违约检测 & 线性惩罚项计算 ---
        violations = np.zeros(3)
        penalty = 0.0

        # eMBB: GBR 违约判定
        target_embb = self.sla_props['embb_gbr']
        if self.queues[0] > 0.1:  # 只有有积压时才考核
            if achieved_throughput_mbps[0] < target_embb:
                violations[0] = 1.0
                # 线性惩罚：缺口越大罚越多
                penalty += 0.5 * (target_embb - achieved_throughput_mbps[0])

        # URLLC: 延迟违约
        est_delay = 0.0
        if service_rate_mbps[1] > 1e-6:
            est_delay = self.queues[1] / service_rate_mbps[1]
        else:
            if self.queues[1] > 0: est_delay = 0.01  # 10ms 默认延迟

        if est_delay > self.sla_props['urllc_max_delay']:
            violations[1] = 1.0
            # 线性惩罚：延迟超出的绝对值
            # 权重设高一些，因为 URLLC 是严苛约束
            penalty += 200.0 * (est_delay - self.sla_props['urllc_max_delay'])

        # mMTC: 队列溢出违约
        if self.queues[2] > self.sla_props['mmtc_max_queue']:
            violations[2] = 1.0
            # 线性惩罚：超出队列的部分
            penalty += 10.0 * (self.queues[2] - self.sla_props['mmtc_max_queue'])
            self.queues[2] = self.sla_props['mmtc_max_queue']  # 丢包

        # --- 3. 最终线性奖励函数 ---
        # 基础奖励：吞吐量（单位：100Mbps 对应 1.0 Reward）
        reward_throughput = np.sum(achieved_throughput_mbps) / 100.0

        # 静态违约惩罚 (Fixed Penalty)
        reward_static_violation = -(2.0 * violations[0] + 50.0 * violations[1] + 5.0 * violations[2])

        # 组合最终奖励
        reward = reward_throughput + reward_static_violation - penalty

        # --- 4. 更新状态 ---
        self._update_state()
        self.state[3:6] = self.queues

        info = {
            "queue_sizes": self.queues.copy(),
            "violations": violations,
            "throughput": np.sum(achieved_throughput_mbps),
            "est_delay_urllc": est_delay
        }

        terminated = self.current_step >= self.max_steps
        return self.state.astype(np.float32), float(reward), terminated, False, info

    def _update_state(self):
        """
        生成流量并应用 Load Factor
        """
        # eMBB: 截断高斯分布 (大流量)
        arr_embb = np.clip(np.random.normal(60, 10), 40, 90) * self.load_factor

        # URLLC: 泊松突发
        if np.random.rand() > 0.8:
            arr_urllc = np.random.normal(25, 5) * self.load_factor
        else:
            arr_urllc = np.random.normal(5, 1) * self.load_factor

        # mMTC: 周期性小包
        arr_mmtc = np.random.normal(2, 0.1) * self.load_factor

        # 信道质量 (SE)
        se_embb = np.random.uniform(3.0, 6.0)
        se_urllc = np.random.uniform(1.5, 3.5)
        se_mmtc = np.random.uniform(1.0, 2.5)

        self.state[0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[6:9] = [se_embb, se_urllc, se_mmtc]
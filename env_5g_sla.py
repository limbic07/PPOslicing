import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FiveG_SLA_Env(gym.Env):
    """
    Advanced 5G Slicing Environment with SLA Constraints & Queueing.
    带有 SLA 约束和排队机制的进阶 5G 切片环境。

    SLA Definitions:
    1. eMBB: GBR (Guaranteed Bit Rate) constraint. 必须满足最低速率。
    2. URLLC: Latency constraint (Delay Budget). 延迟不能超过阈值。
    3. mMTC: Packet Loss constraint (Queue overflow). 队列不能溢出。
    """

    def __init__(self):
        super(FiveG_SLA_Env, self).__init__()

        # --- System Constants (系统常数) ---
        self.total_bandwidth = 100e6  # 100 MHz
        self.duration_tti = 1e-3  # 1ms per TTI (Time Slot duration)

        # --- SLA Parameters (SLA 参数配置) ---
        self.sla_props = {
            # eMBB: Minimum 100 Mbps throughput required
            'embb_gbr': 100.0,

            # URLLC: Max 5ms delay allowed.
            # 延迟估算公式: Delay = Queue_Size / Service_Rate
            'urllc_max_delay': 0.005,

            # mMTC: Max queue size (buffer depth) to prevent packet loss
            'mmtc_max_queue': 100.0  # Mb data in buffer
        }

        # --- Action Space (动作空间) ---
        # [eMBB_Ratio, URLLC_Ratio, mMTC_Ratio] (Normalized to sum=1)
        # We use [-1, 1] for PPO stability, mapped to [0, 1] later
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- Observation Space (状态空间) ---
        # 维度增加到 9 维:
        # 1-3. Traffic Arrivals (Instantaneous Demand) [Mbps]
        # 4-6. Queue Backlog (Accumulated Data) [Mb] <-- NEW! (新增队列状态)
        # 7-9. Spectral Efficiency (Channel Quality) [bits/s/Hz]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(9,), dtype=np.float32)

        # Internal State
        self.state = np.zeros(9, dtype=np.float32)

        # Initialize Queues (Data backlog in Megabits)
        # 队列初始值为0
        self.queues = np.zeros(3, dtype=np.float32)

        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queues = np.zeros(3)  # Reset queues
        self._update_state()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # --- 1. Action Mapping (动作映射) ---
        # Map [-1, 1] -> [0, 1] and Normalize
        exp_action = np.exp(action)  # Softmax trick for positive ratios
        ratios = exp_action / np.sum(exp_action)

        bw_allocated = ratios * self.total_bandwidth

        # --- 2. Calculate Capacity (计算当前时刻的服务能力) ---
        # Capacity (Mbps) = BW (Hz) * SE (bits/s/Hz) / 1e6
        # SE is in state indices 6,7,8
        se = self.state[6:9]
        service_rate_mbps = (bw_allocated * se) / 1e6

        # Convert Service Rate to "Data served in 1ms" (Mb per TTI)
        # 这一步能处理多少兆比特数据：速率 * 时间(1ms)
        service_capacity_mb = service_rate_mbps * self.duration_tti

        # --- 3. Queue Evolution (排队演进逻辑 - 核心!) ---
        # New Queue = Old Queue + Arrival - Served
        # Arrival is in state indices 0,1,2 (Mbps) -> convert to Mb per TTI
        arrivals_mb = self.state[0:3] * self.duration_tti

        # Update Queues (Add arrivals)
        self.queues += arrivals_mb

        # Calculate actually served data (cannot serve more than what's in queue)
        # 实际服务量：取Capacity和当前队列的最小值
        served_mb = np.minimum(service_capacity_mb, self.queues)

        # Update Queues (Subtract served)
        self.queues -= served_mb

        # Calculate Real-time Throughput (Mbps) for Reward
        achieved_throughput_mbps = served_mb / self.duration_tti

        # --- 4. SLA Violation Check (SLA 违约检查) ---
        violations = np.zeros(3)

        # (1) eMBB: Check GBR
        # If throughput < 100 Mbps, it's a violation
        if achieved_throughput_mbps[0] < self.sla_props['embb_gbr']:
            violations[0] = 1.0  # Boolean flag or margin

        # (2) URLLC: Check Latency (Little's Law approximation)
        # Estimated Delay = Queue_Size (Mb) / Service_Rate (Mbps)
        # Prevent division by zero
        if service_rate_mbps[1] > 0:
            est_delay = self.queues[1] / service_rate_mbps[1]
        else:
            est_delay = 1.0  # Infinite delay if no bandwidth

        if est_delay > self.sla_props['urllc_max_delay']:
            violations[1] = 1.0  # Violation!

        # (3) mMTC: Check Queue Overflow
        if self.queues[2] > self.sla_props['mmtc_max_queue']:
            violations[2] = 1.0  # Buffer Overflow
            self.queues[2] = self.sla_props['mmtc_max_queue']  # Drop packets

        # --- 5. Reward Function (SLA-Aware) ---
        # Base Reward: Total Throughput (encourages serving data)
        reward = np.sum(achieved_throughput_mbps) / 100.0  # Normalize

        # Penalties: Heavy punishment for SLA violations
        penalty_weight = 10.0

        reward -= penalty_weight * violations[0]  # eMBB penalty
        reward -= (penalty_weight * 2) * violations[1]  # URLLC penalty (Critical!)
        reward -= penalty_weight * violations[2]  # mMTC penalty

        # --- 6. Update State & Finish ---
        self._update_state()

        # Update Observation with new queues
        # State: [Arrivals(3), Queues(3), SE(3)]
        self.state[3:6] = self.queues

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "queue_sizes": self.queues.copy(),
            "violations": violations,
            "est_urllc_delay": est_delay if 'est_delay' in locals() else 0
        }

        return self.state, reward, terminated, truncated, info

    def _update_state(self):
        """
        Generate new traffic arrivals and channel conditions.
        """
        # 1. Traffic Arrivals (Mbps) -> similar to previous code
        arr_embb = np.random.uniform(100, 500)
        arr_urllc = np.random.poisson(lam=20) * 5.0  # Random bursts
        arr_mmtc = np.random.normal(30, 5)

        # 2. Spectral Efficiency (SE)
        se_embb = np.random.uniform(4.0, 8.0)
        se_urllc = np.random.uniform(2.0, 5.0)
        se_mmtc = np.random.uniform(1.0, 3.0)

        # Fill state (Queues will be filled in step function or kept from prev)
        # We only update Arrivals and SE here. Queues are persistent.
        self.state[0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[6:9] = [se_embb, se_urllc, se_mmtc]
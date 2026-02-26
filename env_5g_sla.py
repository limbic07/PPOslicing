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
        # 考虑到后续多小区扩展，设定标准 5G Sub-6GHz 单载波带宽
        self.total_bandwidth = 100e6  # 100 MHz (Typical for C-band)
        self.duration_tti = 0.5e-3  # 0.5ms per TTI (Numerology 1, 30kHz SCS)

        # --- SLA Parameters (SLA 参数配置) ---
        self.sla_props = {
            # eMBB: Minimum 120 Mbps throughput required
            # 提高底线 (100 -> 120): 压缩 Agent 的容错空间，让它不能肆无忌惮地剥削 eMBB
            'embb_gbr': 120.0,

            # URLLC: Max 2ms delay allowed.
            'urllc_max_delay': 0.002,

            # mMTC: Max queue size (buffer depth) to prevent packet loss
            # 修复漏洞：降至 1.0 Mb。若设为 20Mb，200步内根本无法填满，导致约束失效。
            'mmtc_max_queue': 1.0  # Mb data in buffer
        }

        # --- Action Space (动作空间) ---
        # [eMBB_Ratio, URLLC_Ratio, mMTC_Ratio] (Normalized to sum=1)
        # We use [-1, 1] for PPO stability, mapped to [0, 1] later
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- Observation Space (状态空间) ---
        # 维度扩展到 14 维 (强化特征工程):
        # 1-3 (0:3). Traffic Arrivals (Instantaneous Demand) [Mbps]
        # 4-6 (3:6). Queue Backlog (Accumulated Data) [Mb]
        # 7-9 (6:9). Spectral Efficiency (Channel Quality) [bits/s/Hz]
        # 10-12 (9:12). Previous Action (Bandwidth Ratios allocated in the last step) [0~1]
        # 13 (12). URLLC Estimated Delay [ms] (Critical Alert Feature)
        # 14 (13). eMBB GBR Shortfall [Mbps] (Critical Alert Feature)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(14,), dtype=np.float32)

        # Internal State
        self.state = np.zeros(14, dtype=np.float32)
        
        # 辅助状态记忆
        self.prev_ratios = np.zeros(3, dtype=np.float32)
        self.est_delay_urllc = 0.0
        self.shortfall_embb = 0.0

        # Initialize Queues (Data backlog in Megabits)
        # 队列初始值为0
        self.queues = np.zeros(3, dtype=np.float32)

        # --- Channel Model Parameters (AR(1) 过程参数) ---
        # eMBB, URLLC, mMTC 的信道均值
        self.mean_se = np.array([4.5, 2.5, 1.75], dtype=np.float32)
        self.current_se = self.mean_se.copy()
        # 时间相关系数 rho (0~1)。越接近1，信道变化越平滑
        self.rho_se = 0.9

        self.max_steps = 200
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.queues = np.zeros(3, dtype=np.float32)  # Reset queues
        # 重置信道状态为均值附近，防止每次环境重置信道都一模一样
        self.current_se = self.mean_se + np.random.normal(0, 0.2, size=3)
        self._update_state()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # --- 1. Action Mapping (动作映射) ---
        # 弃用容易导致溢出和梯度消失的 np.exp (Softmax)
        # 改用更稳定的线性映射：将 PPO 的 [-1, 1] 映射到 [0, 1]
        action_positive = (action + 1.0) / 2.0
        
        # 设定极小值 0.01 防止除零并保证每个切片最少有一点点微弱控制信道带宽
        weights = np.clip(action_positive, 0.01, 1.0)
        ratios = weights / np.sum(weights)

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

        # --- 4. Continuous SLA Violation Check (平滑违约检查) ---
        # 弃用“非0即1”的悬崖式惩罚，改为连续的偏离度计算
        violations = np.zeros(3)

        # (1) eMBB: Check GBR
        # 计算低于 GBR 的差值，并归一化
        embb_shortfall = max(0.0, self.sla_props['embb_gbr'] - achieved_throughput_mbps[0])
        violations[0] = embb_shortfall / self.sla_props['embb_gbr']

        # (2) URLLC: Check Latency (Little's Law approximation)
        # Prevent division by zero with a small floor value
        safe_service_rate = max(service_rate_mbps[1], 0.1)
        est_delay = self.queues[1] / safe_service_rate
        
        # 计算超出最大延迟的量，并归一化（超出2ms就算1.0）
        delay_excess = max(0.0, est_delay - self.sla_props['urllc_max_delay'])
        violations[1] = delay_excess / self.sla_props['urllc_max_delay']

        # (3) mMTC: Check Queue Overflow
        queue_excess = max(0.0, self.queues[2] - self.sla_props['mmtc_max_queue'])
        violations[2] = queue_excess / self.sla_props['mmtc_max_queue']
        
        # 物理约束：实际队列大小被限制在最大容量内（多出的包被丢弃）
        if self.queues[2] > self.sla_props['mmtc_max_queue']:
            self.queues[2] = self.sla_props['mmtc_max_queue']

        # --- 5. Reward Function (SLA-Aware Smooth Penalty) ---
        # Base Reward: Total Throughput (encourages serving data)
        reward = np.sum(achieved_throughput_mbps) / 100.0  # Normalize

        # Penalties: Continuous and scaled
        penalty_weight = 10.0

        # 对惩罚进行软截断 (Soft Clipping)，防止单个极端的恶劣状态导致梯度爆炸
        # eMBB 最大扣除 50 分，URLLC 最大扣除 100 分
        pen_embb = min(violations[0] * penalty_weight, penalty_weight * 5.0)
        pen_urllc = min(violations[1] * (penalty_weight * 2.0), penalty_weight * 10.0)
        pen_mmtc = min(violations[2] * penalty_weight, penalty_weight * 5.0)

        reward -= (pen_embb + pen_urllc + pen_mmtc)

        # --- 6. Update State & Finish ---
        self._update_state()

        # Update Observation with new queues and auxiliary features
        # State mapping: 
        # [Arrivals(0:3), Queues(3:6), SE(6:9), PrevRatios(9:12), EstDelay(12), Shortfall(13)]
        self.state[3:6] = self.queues
        self.state[9:12] = ratios  # 记录当前动作给下一次状态用
        self.state[12] = est_delay
        self.state[13] = embb_shortfall

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "queue_sizes": self.queues.copy(),
            "violations": violations,
            "throughput": np.sum(achieved_throughput_mbps),
            "est_urllc_delay": est_delay if 'est_delay' in locals() else 0
        }

        return self.state, reward, terminated, truncated, info

    def _update_state(self):

        # --- 1. eMBB: Video Streaming (Truncated Gaussian) ---
        # 100MHz 管道，SE均值4.5，满载约 450Mbps。
        # 设定 eMBB 需求在 150~350 Mbps 之间波动
        arr_embb = np.clip(np.random.normal(250, 40), 150, 350)

        # --- 2. URLLC: Industrial Automation (Poisson Burst) ---
        # 模拟机器臂控制/自动驾驶。平时极低，罕见但剧烈的突发。
        # 突发时可能需要瞬间消耗极大的带宽来保证 2ms 延迟
        # 修改：加大突发烈度 (100 -> 150) 以彻底击穿静态分配的防御
        if np.random.rand() > 0.9:  # 10% 概率突发 (降低频率，符合 URLLC 偶发特性)
            arr_urllc = np.random.normal(150, 30)  # 极强突发状态，需求激增
        else:
            arr_urllc = np.random.normal(10, 2)  # 静默状态

        # --- 3. mMTC: Sensor Data (Constant + Noise) ---
        # 恒定且需求小，但对缓冲区溢出敏感
        arr_mmtc = np.random.normal(10, 1)

        # --- 4. 物理信道 (Spectral Efficiency) ---
        # 弃用纯随机数，改用 AR(1) 过程模拟真实的时间相关衰落信道 (Time-Correlated Fading Channel)
        # 产生新的随机高斯波动 (innovation)
        noise = np.array([
            np.random.normal(0, 0.4),  # eMBB 的波动幅度
            np.random.normal(0, 0.2),  # URLLC 的波动幅度
            np.random.normal(0, 0.15)  # mMTC 的波动幅度
        ])
        
        # AR(1) 演进公式: SE_t = rho * SE_{t-1} + (1 - rho) * Mean + sqrt(1 - rho^2) * Noise
        self.current_se = (self.rho_se * self.current_se) + \
                          ((1 - self.rho_se) * self.mean_se) + \
                          (np.sqrt(1 - self.rho_se**2) * noise)
                          
        # 极值裁剪，保证物理意义 (SE 不能无限大，也不能过小)
        self.current_se = np.clip(self.current_se, [2.0, 1.0, 0.5], [6.0, 4.0, 2.5])

        # 更新状态
        self.state[0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[6:9] = self.current_se
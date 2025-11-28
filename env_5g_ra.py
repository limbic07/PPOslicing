import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FiveGResourceAllocationEnv(gym.Env):
    """
    Custom Environment for 5G/6G Network Slicing Resource Allocation.
    自定义 5G/6G 网络切片资源分配环境。

    Scope (适用范围): Resource Allocation Agent
    Assumptions (核心假设):
    1. Single Cell (单基站)
    2. Equal Power Allocation (等功率分配 - PSD is constant)
    3. Rule-based User Association (基于规则的用户关联 - implicit in slice aggregation)
    4. Perfect CSI (完美信道状态信息)
    5. Block Fading (块衰落信道)
    """

    def __init__(self):
        super(FiveGResourceAllocationEnv, self).__init__()

        # --- System Parameters (系统参数) ---
        self.total_bandwidth = 100e6  # 100 MHz Total Bandwidth
        self.total_power = 40.0  # 40 Watts (approx 46 dBm)
        self.noise_spectral_density = -174  # dBm/Hz

        # Calculate Noise Power in Watts for 1 Hz (计算1Hz带宽内的噪声功率 - 线性值)
        # Noise (dBm) = -174
        # Noise (Watts) = 10^((-174-30)/10)
        self.noise_per_hz = 10 ** ((-174 - 30) / 10)

        # Simulation Steps per Episode (每个回合的步数)
        self.max_steps = 200
        self.current_step = 0

        # --- Action Space (动作空间) ---
        # We need to allocate bandwidth ratios for 3 slices: [eMBB, URLLC, mMTC]
        # PPO outputs a continuous vector. We will normalize it to sum to 1 in the step() function.
        # 动作: 3个连续值，代表三个切片的权重。我们在 step 函数中会用 Softmax 或归一化处理。
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- Observation Space (状态空间) ---
        # What the Agent sees (Agent 能看到什么):
        # 1. Current Demand for eMBB (Mbps)
        # 2. Current Demand for URLLC (Mbps)
        # 3. Current Demand for mMTC (Mbps)
        # 4. Average Spectral Efficiency (SE) of eMBB Users (bits/s/Hz) - indicating Channel Quality
        # 5. Average Spectral Efficiency (SE) of URLLC Users
        # 6. Average Spectral Efficiency (SE) of mMTC Users
        # Total: 6 dimensions (总共6维状态)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

        # Internal state holder
        self.state = np.zeros(6)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        重置环境，开始新回合。
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Generate initial state (traffic and channel quality)
        self._update_state()

        return self.state, {}

    def step(self, action):
        """
        Execute one time step within the environment.
        执行一步动作。
        """
        self.current_step += 1

        # --- 1. Process Action (处理动作) ---
        # Map action from [-1, 1] to [0, 1]
        action_positive = (action + 1) / 2.0

        # Normalize action to sum to 1 (Bandwidth Fractions)
        # 防止除以0，加一个极小值
        weights = np.clip(action_positive, 0.01, 1.0)
        allocation_ratios = weights / np.sum(weights)

        # Calculate allocated bandwidth for each slice (Hz)
        # 计算每个切片实际分到的带宽
        bw_allocated = allocation_ratios * self.total_bandwidth

        # --- 2. Physics & Calculation (物理层计算) ---
        # Get Traffic Demand (Mbps) and Spectral Efficiency (SE) from current state
        # 从状态中获取当前的需求和信道质量
        demands_mbps = self.state[0:3]
        spectral_efficiencies = self.state[3:6]  # bits/s/Hz

        # Calculate Capacity (Shannon Formula Simplified with SE)
        # Capacity = Bandwidth * Spectral Efficiency
        # 假设：等功率分配下，SE 已经包含了 SINR 的影响
        capacity_bps = bw_allocated * spectral_efficiencies
        capacity_mbps = capacity_bps / 1e6

        # --- 3. Service Logic (业务逻辑) ---
        # Calculate served traffic (can't serve more than capacity, nor more than demand)
        # 计算实际服务的流量。对于 eMBB (index 0)，通常是 Full Buffer，有多少传多少。
        # 对于 URLLC/mMTC，如果 Capacity > Demand，则浪费了带宽；如果 Capacity < Demand，则发生丢包/延迟。

        served_mbps = np.minimum(capacity_mbps, demands_mbps)

        # Special case for eMBB (Index 0): Assumption "Full Buffer"
        # eMBB 假设是无限需求，只要给带宽就能跑满 (bounded by a peak value in simulation for stability)
        # 这里为了训练简单，我们保留 minimum 逻辑，但在 _update_state 里让 eMBB 需求很高。

        # --- 4. Reward Calculation (核心：奖励函数设计) ---
        # Goal: Satisfy URLLC/mMTC requirements first, then maximize eMBB throughput.
        # 目标：优先满足 URLLC 和 mMTC 的硬性需求，剩下的资源尽可能让 eMBB 跑得快。

        # Penalty for unmet demand (latency/packet loss proxy)
        # 计算未满足需求的缺口
        unmet_demand = demands_mbps - served_mbps

        # Define weights for reward (可调整的超参数)
        alpha_embb = 0.5
        beta_urllc = 5.0  # High penalty for URLLC (Critical!)
        gamma_mmtc = 2.0

        # Reward Formula:
        # + eMBB Throughput (Positive Reward)
        # - URLLC Violation (Heavy Negative Reward)
        # - mMTC Violation (Moderate Negative Reward)

        reward = (alpha_embb * served_mbps[0]) - \
                 (beta_urllc * unmet_demand[1]) - \
                 (gamma_mmtc * unmet_demand[2])

        # Optional: Normalize reward for training stability (Scaling)
        reward = reward / 10.0

        # --- 5. Next State (状态更新) ---
        self._update_state()

        # --- 6. Termination Check (结束检查) ---
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps:
            terminated = True

        # Info dict for debugging/plotting
        info = {
            "ratios": allocation_ratios,
            "served_throughput": served_mbps,
            "demand": demands_mbps,
            "unmet": unmet_demand
        }

        return self.state, reward, terminated, truncated, info

    def _update_state(self):
        """
        Simulate Traffic Dynamics and Channel Fading.
        模拟流量动态和信道衰落 (Block Fading)。
        """
        # --- A. Traffic Generation (Traffic Models) ---

        # 1. eMBB: Full Buffer / High Throughput (Normal Dist)
        # 模拟：需求一直在 500Mbps 到 1500Mbps 之间波动
        demand_embb = np.random.uniform(500, 1500)

        # 2. URLLC: Poisson Arrival (Bursty)
        # 模拟：泊松分布，突发流量。有时很低，有时突然很高。
        # Lam = 50, Scale factor implies packet size sum
        demand_urllc = np.random.poisson(lam=5) * 10.0  # roughly 20-80 Mbps
        if np.random.rand() > 0.9:  # 10% chance of huge burst
            demand_urllc += 100

            # 3. mMTC: Periodic / Constant Low Data
        # 模拟：大量传感器，总流量相对稳定，有小幅波动
        demand_mmtc = np.random.normal(loc=30, scale=5)  # ~30 Mbps

        # --- B. Channel Variation (Block Fading) ---
        # Instead of calculating path loss for every user, we simulate the 'Average SE' of the slice.
        # 不计算具体路径损耗，直接模拟切片的“平均频谱效率”。

        # eMBB SE: Range 2 - 6 bits/s/Hz (Good channel usually)
        se_embb = np.random.uniform(2.0, 6.0)

        # URLLC SE: Range 1 - 4 bits/s/Hz (Lower because requires high reliability/low MCS)
        se_urllc = np.random.uniform(1.0, 4.0)

        # mMTC SE: Range 0.5 - 2 bits/s/Hz (Cell edge devices, poor coverage)
        se_mmtc = np.random.uniform(0.5, 2.0)

        # Update state vector
        self.state = np.array([
            demand_embb, demand_urllc, demand_mmtc,
            se_embb, se_urllc, se_mmtc
        ], dtype=np.float32)

    def render(self):
        # Optional: Print current status
        print(f"Step: {self.current_step}")
        print(f"Demand: {self.state[0:3]}")
        print(f"SE    : {self.state[3:6]}")


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
        self.state = np.zeros(9)

        # Initialize Queues (Data backlog in Megabits)
        # 队列初始值为0
        self.queues = np.zeros(3)

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
        arr_embb = np.random.uniform(500, 1500)
        arr_urllc = np.random.poisson(lam=20) * 5.0  # Random bursts
        arr_mmtc = np.random.normal(30, 5)

        # 2. Spectral Efficiency (SE)
        se_embb = np.random.uniform(2.0, 6.0)
        se_urllc = np.random.uniform(1.0, 4.0)
        se_mmtc = np.random.uniform(0.5, 2.0)

        # Fill state (Queues will be filled in step function or kept from prev)
        # We only update Arrivals and SE here. Queues are persistent.
        self.state[0:3] = [arr_embb, arr_urllc, arr_mmtc]
        self.state[6:9] = [se_embb, se_urllc, se_mmtc]
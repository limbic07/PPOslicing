# 系统模型与强化学习算法设计改进报告
# System Model and RL Algorithm Design Improvement Report

---

## 🇨🇳 中文版 (Chinese Version)

### 1. 马尔可夫决策过程 (MDP) 核心设计与扩充

#### 1.1 状态空间定义 (从 9D 扩充至 14D)
为了让智能体全面感知物理层的动态演进并做出前瞻性决策，我们将每个基站的状态观测空间 (Observation Space) 从早期的 9 维扩充到了 **14 维特征**：
*   **`[0:2]` 业务请求量 (Traffic Arrivals)**：当前 TTI 内三个切片到达的新数据需求量。我们在环境模型中加入了马尔可夫突发流，URLLC 突发时会有巨大的瞬间增量。
*   **`[3:5]` 队列积压 (Queue Backlogs)**：当前缓存在基站缓冲区中未被服务的数据量，直接反映当前网络拥堵程度。
*   **`[6:8]` 频谱效率 (Spectral Efficiency, SE)**：由 AR(1) 时序模型演进生成的当前信道状态。
*   **`[9:11]` 历史动作 (Previous Actions)**：智能体在上一个时间步对三个切片分配的带宽比例。由于采用前馈神经网络 (MLP) 而非 LSTM，该特征提供了强化学习中必需的时间因果关系 (Temporal Context)，有效防止分配策略发生高频振荡。
*   **`[12]` URLLC 预估延迟 (Estimated Delay)**：核心衍生告警特征，由系统物理引擎前置计算（公式：`当前 URLLC 队列长度 / 当前分配速率`）。
*   **`[13]` eMBB 吞吐量缺口 (eMBB Shortfall)**：记录当前 eMBB 距离 GBR (180 Mbps) 目标的差值，用于防止模型在极度偏袒 URLLC 时彻底饿死 eMBB。

*(注：在送入神经网络前，上述 14 维物理量通过内置滤波器进行了动态的均值方差归一化，消除了量纲差异带来的梯度振荡问题。)*

### 2. 多小区同频干扰 (Inter-Cell Interference, ICI) 建模
在此次迭代中，我们将环境正式从单基站升级为了包含中心站 (BS_0) 和边缘站 (BS_1 ~ BS_6) 的 **7小区系统**。为了在数学上对齐多小区之间的物理耦合，我们在基站环境类 `_calculate_interference_and_sinr` 中引入了基于动作空间的干扰模型：
*   **干扰因子提取**：对于中心基站，系统会实时提取其周围 6 个邻居在同一时刻对相同切片分配的带宽比例（即 `Action Weights` 矩阵）。
*   **频谱效率折损机制**：由于邻居在相同信道上分配大量资源必然导致严重碰撞，我们将邻居分配比例的均值作为干扰因子，按比例削减中心基站的频谱效率：
    $$SE_{effective} = SE_{base} \times (1.0 - \text{Interference\_Factor} \times 0.5)$$
*   **算法博弈意义**：这一环境动力学的修改，彻底改变了强化学习的寻优方向。边缘基站现在不仅要满足本地切片，还必须学会在中心基站面临 URLLC 突发（高干扰敏感度）时主动进行**频谱静默 (Spectrum Blanking)**。


### 3. 多智能体环境构建与 CTDE 架构 (Multi-Agent Environment & CTDE)
为了适配 Ray RLlib 框架并实现真正的并发调度，我们重构了底层环境引擎，使其继承自 `MultiAgentEnv` 接口。
#### 3.1 字典化并行空间 (Dictionary-based Parallel Spaces)
*   **Agent 标识**：系统注册了 7 个独立的 Agent ID (`BS_0` 到 `BS_6`)。
*   **并发执行 (Simultaneous Execution)**：在核心的 `step(action_dict)` 函数中，环境接收一个包含所有基站动作的字典，并在同一个时间步（TTI）内**同步并行计算** 7 个小区的干扰、吞吐量和队列演进。随后，环境打包并返回包含 7 个独立 `obs`, `reward`, `done` 状态的字典，实现了严格的时间同步仿真。

#### 3.2 集中式训练与分布式执行 (CTDE) & 参数共享
*   **策略映射 (Policy Mapping)**：我们将 7 个基站的决策大脑映射到了**同一个共享策略网络 (Shared Policy)** 上。
*   **学术优势**：这不仅极大地提升了多智能体的样本采样效率（每次 step 能收集 7 份经验），且符合 **CTDE (Centralized Training with Decentralized Execution)** 架构。训练时，所有基站的经验被汇总到一个中央 Learner 更新梯度；但在实际执行阶段，每个基站仅依赖其本地获取的 14 维局部观测状态（Local Observation）进行独立推理，无需基站间的超低延迟实时通信，具有极高的工程落地价值。

### 4. 奖励函数 (Reward Function) 重构与合作博弈机制
奖励函数是决定多智能体能否学会上述“退让”策略的关键，我们对 Reward 进行了两次重大的架构调整：

#### 4.1 引入悬崖惩罚 (Cliff Penalty) 保证极低延迟
单纯的线性延迟惩罚在复杂系统中容易导致智能体陷入“端水”策略（即：宁愿让 URLLC 稍微超时，也要去换取 eMBB 巨大的吞吐量正向收益）。
为了杜绝这一现象，我们采用了“一票否决”式的非线性悬崖惩罚：
*   **机制**：一旦 URLLC 预估延迟越过 2ms 的绝对红线，除常规的违约比例扣分外，算法会**立刻追加 -5.0 的巨额基底惩罚**。
*   **效果**：这种惩罚将任何违反延迟红线的动作期望收益瞬间拉成负数，在梯度层面上逼迫网络将“保障 2ms”作为不可违背的第一准则，而非可以用来交易的筹码。

#### 4.2 合作博弈分账权重 (Cooperative Weight $\alpha$)
为了落实 MARL 的协作机制，我们将系统改为完全合作博弈（Cooperative Game）。每个基站的最终反馈奖励由其本地表现和全系统的平均表现加权组成：
$$Reward_{agent} = \alpha \times R_{local} + (1 - \alpha) \times R_{global\_average}$$
目前最优的设置是 **$\alpha = 0.7$**。
*   保留 70% 的本地奖励，是为了给神经网络提供明确的局部梯度，加速收敛，避免陷入多智能体常见的学分分配难题（Credit Assignment Problem）。
*   引入 30% 的全局共享奖励，是驱动边缘基站牺牲自我吞吐量、减少 ICI、成全中心基站突发传输的源动力。

---

## 🇬🇧 English Version

### 1. Core Design and Expansion of Markov Decision Process (MDP)

#### 1.1 State Space Definition (Expansion from 9D to 14D)
To enable the agents to comprehensively perceive the dynamic evolution of the physical layer and make forward-looking decisions, we expanded the observation space for each base station from the initial 9 dimensions to **14 feature dimensions**:
*   **`[0:2]` Traffic Arrivals**: The volume of new data requests arriving in the current TTI for the three slices. We incorporated a Markov burst flow model, resulting in massive instantaneous increments during URLLC bursts.
*   **`[3:5]` Queue Backlogs**: The volume of unserved data currently buffered at the base station, directly reflecting the ongoing network congestion level.
*   **`[6:8]` Spectral Efficiency (SE)**: The current channel state generated through the evolution of an AR(1) time-series model.
*   **`[9:11]` Previous Actions**: The bandwidth allocation ratios determined by the agent in the previous time step. Since we use a Multi-Layer Perceptron (MLP) instead of an LSTM, this feature supplies the necessary temporal context (causality) in RL, effectively preventing high-frequency oscillations in allocation strategies.
*   **`[12]` Estimated URLLC Delay**: A core derivative warning feature pre-calculated by the system's physical engine (Formula: `Current URLLC Queue Length / Current Allocation Rate`).
*   **`[13]` eMBB Shortfall**: Records the gap between the current eMBB throughput and the GBR target (180 Mbps), utilized to prevent the model from completely starving eMBB when heavily favoring URLLC.

*(Note: Before being fed into the neural network, these 14 physical quantities undergo dynamic mean-variance normalization via built-in filters, eliminating gradient oscillation issues caused by dimensional differences.)*

### 2. Multi-Cell Inter-Cell Interference (ICI) Modeling
In this iteration, we officially upgraded the environment from a single base station to a **7-Cell system** encompassing a central macro station (BS_0) and edge stations (BS_1 ~ BS_6). To mathematically align the physical coupling among multiple cells, we introduced an action-space-based interference model within the base station environment class `_calculate_interference_and_sinr`:
*   **Interference Factor Extraction**: For the central base station, the system monitors the bandwidth allocation ratios (i.e., `Action Weights` matrix) of its 6 neighbors for the same slice in real-time.
*   **Spectral Efficiency Degradation Mechanism**: Since neighbors allocating substantial resources on the same channel inevitably cause severe collisions, we utilize the mean of the neighbors' allocation ratios as an interference factor to proportionally discount the central base station's spectral efficiency:
    $$SE_{effective} = SE_{base} \times (1.0 - \text{Interference\_Factor} \times 0.5)$$
*   **Algorithmic Game-Theoretic Significance**: This modification to environmental dynamics fundamentally alters the RL optimization direction. Edge base stations must now not only satisfy local slice demands but also learn to proactively execute **Spectrum Blanking** when the central base station encounters a URLLC burst (high interference sensitivity).


### 3. Multi-Agent Environment Setup and CTDE Architecture
To accommodate the Ray RLlib framework and enable genuine concurrent scheduling, we refactored the underlying environmental engine to inherit from the `MultiAgentEnv` interface.
#### 3.1 Dictionary-Based Parallel Spaces
*   **Agent Identification**: The system registers 7 distinct Agent IDs (`BS_0` through `BS_6`).
*   **Simultaneous Execution**: In the core `step(action_dict)` function, the environment receives a dictionary containing the actions of all base stations and **synchronously computes** the interference, throughput, and queue evolution for all 7 cells within the exact same time step (TTI). Subsequently, the environment packages and returns dictionaries containing 7 independent `obs`, `reward`, and `done` flags, achieving strictly time-synchronized simulation.

#### 3.2 Centralized Training with Decentralized Execution (CTDE) & Parameter Sharing
*   **Policy Mapping**: We mapped the decision-making brains of all 7 base stations onto a **Single Shared Policy Network**.
*   **Academic Advantages**: This not only drastically enhances the multi-agent sample efficiency (gathering 7 pieces of experience per step) but also strictly adheres to the **CTDE (Centralized Training with Decentralized Execution)** paradigm. During training, experiences from all base stations are aggregated into a central Learner for gradient updates. However, during the actual execution phase, each base station infers independently relying solely on its local 14D observation, eliminating the need for ultra-low-latency real-time communication between base stations and ensuring high practical viability for engineering deployment.

### 4. Reward Function Reconstruction and Cooperative Game Mechanism
The reward function is pivotal in determining whether the multi-agent system can master the aforementioned "yielding" strategy. We executed two major architectural adjustments to the Reward:

#### 4.1 Implementing a Cliff Penalty to Ensure Ultra-Low Latency
A purely linear delay penalty in complex systems often leads the agent into an "averaging" strategy (i.e., preferring to let URLLC slightly exceed the deadline to secure massive positive throughput returns from eMBB).
To eradicate this phenomenon, we employed a "veto-style" non-linear cliff penalty:
*   **Mechanism**: The moment the estimated URLLC delay crosses the absolute red line of 2ms, in addition to the conventional proportional deduction, the algorithm **instantly imposes a massive base penalty of -5.0**.
*   **Effect**: This penalty instantaneously drags the expected return of any action violating the delay red line into the negative, forcing the gradients to treat "securing 2ms" as an inviolable primary axiom rather than a tradable commodity.

#### 4.2 Cooperative Game Profit-Sharing Weight ($\alpha$)
To actualize the MARL collaborative mechanism, we transitioned the system into a fully Cooperative Game. The final feedback reward for each base station is a weighted composition of its local performance and the average performance of the entire system:
$$Reward_{agent} = \alpha \times R_{local} + (1 - \alpha) \times R_{global\_average}$$
The currently optimal configuration is **$\alpha = 0.7$**.
*   Retaining 70% of the local reward provides clear local gradients to the neural network, accelerating convergence and evading the Credit Assignment Problem typical in multi-agent systems.
*   Integrating 30% of the globally shared reward constitutes the fundamental driving force motivating edge base stations to sacrifice their own throughput and reduce ICI to accommodate the burst transmissions of the central base station.

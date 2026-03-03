# Recent Progress Report on 5G Network Slicing RL Optimization
# 5G 网络切片强化学习优化近期进展汇报

---

## 🇬🇧 English Version (For Advisor)

### Overview
Recently, we have substantially upgraded the single-cell PPO resource allocation model to align with realistic 5G NR physical parameters and resolved the gradient oscillation issues. Concurrently, we successfully transitioned the architecture to a Multi-Agent Reinforcement Learning (MARL) framework, simulating a 7-cell network with physical Inter-Cell Interference (ICI) and cooperative game mechanisms, which has now achieved stable convergence.

### 1. Refinement of the Single-Cell Environment
*   **Realistic Physical Parameters**: The system bandwidth has been scaled up to 100 MHz to strictly follow the Sub-6GHz C-Band standard, ensuring the simulation aligns with practical 5G deployments.
*   **State Space Expansion (9D $\rightarrow$ 14D)**: To provide the neural network with temporal context and preemptive warning capabilities, the observation space was expanded. The new 14D state now includes:
    *   `[0:2]` Traffic Arrivals (incorporating Markov burst flows).
    *   `[3:5]` Queue Backlogs.
    *   `[6:8]` AR(1) Fading Spectral Efficiency (SE).
    *   `[9:11]` **Previous Action Ratios**: Provides causality for the MLP to prevent high-frequency allocation oscillations.
    *   `[12]` **Estimated URLLC Delay**: Pre-calculated warning signal (`Queue Length / Current Service Rate`).
    *   `[13]` **eMBB Shortfall**: Gap to the 180 Mbps GBR target.
    *(Note: All states are strictly normalized using `MeanStdFilter` to $\mathcal{N}(0,1)$ before entering the network, which fundamentally solved the non-convergence issue.)*
*   **Reward Function Reform (Cliff Penalty)**:
    Previously, using a linear penalty for URLLC delay violations caused the agent to adopt a compromised "averaging" strategy (sacrificing latency slightly to gain massive throughput rewards from eMBB).
    We introduced a non-linear **Cliff Penalty**: If the estimated URLLC delay crosses the 2ms threshold, an instantaneous heavy base penalty ($-5.0$) is applied on top of the linear penalty. This effectively acts as a veto, forcing the agent to prioritize URLLC bursts at all costs. Preliminary single-cell tests show this drastically tightened the latency bounds (figures will be attached below).

### 2. Expansion to Multi-Cell MARL (Core Focus)
We successfully migrated the environment to a 7-Cell Hexagonal topology (1 center macro cell, 6 edge cells), implemented using Ray RLlib's `MultiAgentEnv` with the CTDE (Centralized Training with Decentralized Execution) paradigm via Parameter Sharing.

*   **Inter-Cell Interference (ICI) Modeling**:
    To physically couple the cells, we introduced an action-space-based ICI penalty. At each TTI, if the 6 edge cells allocate a high proportion of their bandwidth to the same slice (e.g., URLLC) simultaneously, it creates a high probability of PRB collisions. We model this by extracting the mean Action Weights of neighbors to discount the center cell's Spectral Efficiency:
    $$SE_{effective} = SE_{base} \times (1.0 - \text{Mean\_Neighbor\_Action\_Weight} \times 0.5)$$
*   **Cooperative Game Reward Design**:
    To mitigate this ICI, we shifted from a purely competitive formulation to a cooperative game. The reward for each base station is now a weighted sum of its local SLA performance and the global average performance:
    $$Reward_i = \alpha \times R_{local\_i} + (1 - \alpha) \times R_{global\_average}$$
    We empirically set **$\alpha = 0.7$**. The 30% global weight provides the necessary mathematical motivation for edge cells to learn **Spectrum Blanking**—proactively yielding their own bandwidth to reduce interference when the center cell experiences a severe URLLC burst.

### 3. Current Status and Next Steps
The MARL MAPPO model is now fully functional and successfully converging under the multi-cell interference setup. It has learned to execute the cooperative blanking strategy. We are currently preparing comprehensive baseline comparisons (e.g., vs. Static Allocation and Priority Heuristics) to rigorously benchmark these multi-agent performance gains.

*(Insert comparison plots here)*

---

## 🇨🇳 中文版 (For Personal Reference)

### 概述 (Overview)
近期，我们将单基站 PPO 资源分配模型大幅升级，使其对齐了真实的 5G NR 物理参数，并彻底解决了之前存在的梯度振荡不收敛问题。同时，我们成功将架构跨越到了多智能体强化学习 (MARL) 框架，构建了包含物理同频干扰 (ICI) 和合作博弈机制的 7 小区网络，目前该大模型已能稳定收敛。

### 1. 单基站环境的深度完善
*   **真实物理参数**：系统总带宽提升至 100 MHz，严格对标 Sub-6GHz C-Band 标准，使仿真结果不再是玩具模型，而是贴近实际部署。
*   **状态空间扩充 (9D $\rightarrow$ 14D)**：为了让神经网络拥有“时间记忆”和“前瞻预警”能力，我们将状态扩充到了 14 维：
    *   `[0:2]` 业务请求量 (引入了马尔可夫突发流)。
    *   `[3:5]` 队列积压。
    *   `[6:8]` AR(1) 时序衰落信道效率 (SE)。
    *   `[9:11]` **上一时刻的动作分配比例**：为不含 LSTM 的网络提供时间因果关系，防止分配策略出现高频抽搐。
    *   `[12]` **URLLC 预估延迟**：前置计算的预警信号 (`队列长度 / 当前服务速率`)。
    *   `[13]` **eMBB 吞吐量缺口**：距离 180 Mbps GBR 目标的差值。
    *（注：送入网络前，所有状态都经过了严格的 `MeanStdFilter` 动态正态归一化，这是解决模型此前无法收敛的绝对核心操作。）*
*   **奖励函数重构 (悬崖惩罚 Cliff Penalty)**：
    以前：用纯线性的惩罚计算 URLLC 超时。这导致模型变成“端水大师”（宁愿让 URLLC 超时一点点，也要去换取 eMBB 极其庞大的吞吐量正向得分）。
    现在：引入了非线性的**悬崖惩罚**。只要预估延迟跨过 2ms 红线，除了常规扣分，**瞬间砸下 -5.0 的巨额基底惩罚**。这在数学上等同于“一票否决”，逼迫模型不惜一切代价优先处理 URLLC 突发。单基站初步测试显示延迟上限被死死压住了（图表附后）。

### 2. 多小区 MARL 扩展 (核心重点)
我们将环境升级为了 7小区六边形蜂窝网络（1 个中心宏站，6 个边缘站）。使用 Ray RLlib 的 `MultiAgentEnv` 接口，并通过“参数共享”实现了 CTDE (集中式训练，分布式执行) 架构。

*   **同频干扰 (ICI) 建模**：
    为了在物理层面上耦合各个基站，我们引入了基于动作空间的 ICI 惩罚。在同一个 TTI 内，如果周围 6 个边缘基站都把大量带宽分配给了某一切片（比如都在处理 URLLC），就会产生极高的 PRB 物理碰撞概率。
    我们在代码中提取邻居“动作分配比例”的均值，作为干扰乘数，直接打折中心基站的频谱效率：
    $$SE_{effective} = SE_{base} \times (1.0 - \text{邻居动作比例均值} \times 0.5)$$
*   **合作博弈奖励设计 (Cooperative Game)**：
    为了应对上述干扰，我们将系统从竞争博弈改为了合作博弈。现在每个基站的得分，是它自己的 SLA 表现与全网平均表现的加权和：
    $$Reward_i = \alpha \times R_{local\_i} + (1 - \alpha) \times R_{global\_average}$$
    我们通过实验将 $\alpha$ 定为 `0.7`。这 30% 的全局绑定权重，在数学上赋予了边缘基站“大局观”——当中心基站爆发 URLLC 流量而面临严重干扰时，边缘基站有动力主动进行**频谱静默 (Spectrum Blanking，即主动让出带宽)**，牺牲自己的局部小分来换取全局大分。

### 3. 当前状态与下一步
目前，搭载上述机制的 MAPPO 多智能体模型不仅成功跑通，而且已经能够在强干扰下稳定收敛，学会了边缘退让策略。我们正在着手准备详细的 Baseline 对比（如对比静态分配和启发式绝对优先级算法），以严谨的图表来证明多智能体协作带来的巨大增益。

*(此处预留手动贴图位置)*


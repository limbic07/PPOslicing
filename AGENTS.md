# 🤖 5G网络切片PPO算法优化 - Agent 开发指南 (AGENTS.md)

欢迎！这是为自动化编码 Agent（如 Cursor, Copilot, 以及其他基于 LLM 的编程助手）准备的代码库指南。
本项目是一个毕业设计项目（Graduation Design），核心目标是利用 **PPO (Proximal Policy Optimization) 强化学习算法** 解决 **5G/6G 网络切片资源分配 (Resource Allocation)** 和 **SLA 保障** 问题。

请在阅读或修改代码之前仔细阅读以下规范。

---

## 🚀 1. 运行与测试命令 (Build, Lint & Test Commands)

本项目是基于 Python 的强化学习算法实现，不需要传统的编译构建过程。

### 1.1 环境配置 (Environment Setup)
如果需要重新安装依赖，请确保以下核心库被正确安装：
```bash
pip install gymnasium stable-baselines3[extra] numpy matplotlib
```

### 1.2 运行训练与断点续训 (Run & Resume Training)
启动 PPO 模型的正式训练（含 TensorBoard 监控和模型保存）：
```bash
python train_formal.py
```
> **💡 断点续训支持**: `train_formal.py` 支持自动断点续训。训练中途按下 `Ctrl+C` 会自动安全保存 `latest_model.zip` 和 `latest_vec_normalize.pkl`。再次运行命令，模型会自动识别断点并继续追加训练步数，且 TensorBoard 曲线会无缝拼接。
> **注意**: 训练环境对 `VecNormalize` 有强依赖，必须连同环境统计数据一同保存和加载。

### 1.3 运行评估与测试 (Run Evaluation & Testing)
验证训练好的模型，并绘制 SLA 和带宽分配仿真图：
```bash
python test_formal.py
python analyze_urllc.py
python compare_baseline.py
```
> **注意**: 运行测试时会从 `models_formal/` 加载模型。所有的图表和测试结果都会自动统一保存到 `./results/` 文件夹中。

### 1.4 代码格式化与 Lint (Formatting & Linting)
推荐使用 `flake8` 和 `black` 进行代码检查：
```bash
black . 
flake8 . --max-line-length=120
```

---

## 📝 2. 代码风格与 RL 规范 (Code Style & RL Conventions)

为了保证代码的可读性和统一性，请遵守以下规则：

### 2.1 语言与沟通 (Language & Communication)
- **对话语言**: Agent 在解释代码和与用户沟通时**必须使用中文**。
- **专业术语**: 涉及强化学习、通信等专业术语时应**使用英文补充说明**（例如：动作空间 Action Space，频谱效率 Spectral Efficiency，吞吐量 Throughput 等）。

### 2.2 强化学习空间定义 (RL Space Definitions)
- **动作空间 (Action Space)**: 
  - 使用 `spaces.Box(-1.0, 1.0)` 以利于 PPO 输出。
  - **切勿使用 Softmax**。在 `step()` 中必须使用**线性映射和极值裁剪**（`np.clip((action+1)/2, 0.01, 1.0)`），以防除零或饿死某一确定的切片（Starvation）。
- **状态空间 (Observation Space)**: 当前采用 **14 维特征工程**：
  - `[0:3]` 需求量 (Demand/Arrivals)
  - `[3:6]` 队列堆积 (Queue Backlog)
  - `[6:9]` AR(1) 时序演进信道效率 (Spectral Efficiency)
  - `[9:12]` 上一步动作 (Previous Ratios) - 用于提供 MDP 中的时间因果记忆。
  - `[12]` URLLC 预估延迟 (Estimated Delay) - 核心告警特征。
  - `[13]` eMBB GBR 吞吐量缺口 (Shortfall) - 防止挤压 eMBB。
- **奖励函数 (Reward Function)**: 
  - **禁止使用阶跃/悬崖函数** (Step functions)。
  - 必须使用**连续平滑惩罚 (Continuous Smooth Penalty)**，并引入 **软截断 (Soft Clipping)** 防止偶发极值导致的梯度爆炸 (Gradient Explosion)。

### 2.3 命名约定 (Naming Conventions)
- **单位后缀 (极其重要)**: 涉及物理量的变量，**强烈建议**在变量名末尾加上单位后缀，以防单位换算错误：
  - `_mbps` (兆比特每秒), `_hz` (赫兹), `_mb` (兆比特), `_tti` (传输时间间隔)

---

## 🛠️ 3. 通信与 5G 物理参数上下文 (5G Physical Context)

该项目严格对标 5G NR 标准，适用于后续的多小区 (Multi-Cell) 扩展。

- **系统宏观参数**: 
  - 带宽 (Bandwidth): `100 MHz` (Sub-6GHz C-Band 标准)
  - TTI 时隙: `0.5 ms` (Numerology 1, 30kHz SCS)
  - 信道模型: 使用 **AR(1) 马尔可夫高斯过程** (Time-Correlated Fading Channel)，禁止使用纯白噪声。
- **三种网络切片及健康负载设计 (Traffic & SLA)**:
  1. **eMBB**: 背景常态负载。需求正态分布 (`150-350 Mbps`)。受 **GBR (Guaranteed Bit Rate) 100Mbps** 约束。
  2. **URLLC**: 系统刺客。极强突发性，平时 `10 Mbps`，突发时高达 `100 Mbps`。受 **严格 2ms 延迟 (Max Delay)** 约束。
  3. **mMTC**: 背景噪音。恒定小流 `10 Mbps`。受 **1.0 Mb 队列溢出 (Queue Overflow)** 约束。

> **动态博弈提示**: 该参数下系统平均容量约为 300 Mbps。平时平稳运行，但一旦 URLLC 突发（总需求飙升至 360 Mbps 导致过载），Agent 必须学会主动剥削 eMBB 并在 2ms 内排空 URLLC 队列，这体现了切片资源分配的核心冲突。

### 🔧 4. Agent 行为准则 (Agent Directives)
- 在修改系统动力学公式后，Agent **必须**通过 `test_formal.py` 确认未破坏物理逻辑。
- 排查 Bug (如 NaN loss, 性能暴跌) 时，优先打印 `step()` 内部关键变量（reward 和 状态）的 Value Range 以排查溢出。

### 👤 5. Agent 角色与交互原则 (Persona & Interaction Principles)
- **角色设定 (Persona)**: 你是一名精通**强化学习算法 (RL Algorithms)** 和 **5G/6G 通信原理 (5G/6G Principles)** 的高级研发工程师。
- **主动提问与需求确认 (Proactive Questioning & Clarification)**: 在开展工作时，你必须**多向用户提问**以明确具体需求，探明边界条件和业务目标。
- **高置信度执行 (High Confidence Execution)**: 只有在你确认对用户的意图和需求有 **95% 以上的把握** 时，才能开始编写或修改代码。在存在歧义时，宁可多问，不可盲动。
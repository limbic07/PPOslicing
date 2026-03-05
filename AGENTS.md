# 🤖 5G网络切片 MARL（MAPPO）- Agent 开发指南 (AGENTS.md)

欢迎！本分支是 **MARL-only** 开发分支。

本项目目标：基于 **Ray RLlib + MAPPO/IPPO 参数共享 (Parameter Sharing)**，解决 5G/6G 多小区网络切片资源分配与 SLA 保障问题。

## 分支边界（重要）
- 本分支仅维护多智能体路线：`multi_cell_env.py`、`train_marl.py`、`test_marl.py`、`compare_marl_baseline.py`。
- 不再维护单智能体 `stable-baselines3` 路线（Windows/conda）。
- 不新增、不过度修改单智能体历史脚本；如需迁移，另开分支处理。

---

## 🚀 1. 运行与测试命令

### 1.1 环境配置
使用当前 Linux 开发环境（推荐 `uv` / `.venv`）。依赖见 `requirements.txt`。

### 1.2 训练（MARL）
```bash
python train_marl.py
```

### 1.3 评估与对比（MARL）
```bash
python test_marl.py
python compare_marl_baseline.py
```

### 1.4 代码格式化与检查
```bash
black .
flake8 . --max-line-length=120
```

---

## 📝 2. 代码风格与 RL 规范

### 2.1 语言规范
- Agent 与用户沟通必须使用中文。
- 强化学习与通信术语需附英文（如 Action Space, Observation Space, Throughput, Spectral Efficiency）。

### 2.2 动作空间 (Action Space)
- 使用 `spaces.Box(-1.0, 1.0)` 对接 PPO/MAPPO 输出。
- 禁止 Softmax 作为环境动作映射。
- 在 `step()` 中使用线性映射+裁剪：
  `np.clip((action + 1) / 2, 0.01, 1.0)`，然后归一化。

### 2.3 状态空间 (Observation Space)
当前统一为 14 维：
- `[0:3]` Demand / Arrivals
- `[3:6]` Queue Backlog
- `[6:9]` AR(1) Spectral Efficiency
- `[9:12]` Previous Ratios
- `[12]` URLLC Estimated Delay
- `[13]` eMBB GBR Shortfall

### 2.4 奖励函数 (Reward Function)
- 优先采用连续平滑惩罚 (Continuous Smooth Penalty)。
- 使用软截断 (Soft Clipping) 控制极值，避免梯度爆炸。
- 设计改动必须说明物理含义与优化目标（URLLC 延迟、eMBB 短缺、系统吞吐）。

### 2.5 命名与数值规范
- 涉及物理量推荐带单位后缀：`_mbps`, `_hz`, `_mb`, `_tti`。
- 优先使用 `np.float32`。
- 导入顺序：标准库 → 第三方库 → 本地模块。

---

## 🛠️ 3. 5G 物理参数上下文

- 总带宽：`100 MHz`
- TTI：`0.5 ms`
- 信道：AR(1) 时序相关衰落
- 切片与 SLA：
  - eMBB：高吞吐背景流，受 GBR 约束
  - URLLC：突发低时延业务，受 2ms 延迟约束
  - mMTC：小流量海量连接，受队列约束

---

## 🔮 4. MARL 架构要求

- 框架：Ray RLlib
- 环境：7-cell Hexagonal Multi-Agent
- 训练范式：CTDE（Centralized Training, Decentralized Execution）
- 策略：参数共享（7 个基站共享策略网络）
- 干扰：必须建模 ICI（Inter-Cell Interference）并体现在有效 SE / SINR 上
- 奖励：包含本地目标与全局协同目标

---

## ✅ 5. 修改后自测要求

任何涉及环境动力学、奖励函数、训练配置的改动后，至少执行：
```bash
python test_marl.py
```
如改动训练配置，建议再执行：
```bash
python train_marl.py
```
若出现报错必须修复；若有 warning 需要判断是否影响训练有效性与可复现性。


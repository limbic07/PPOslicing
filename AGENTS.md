# 🤖 5G网络切片 MARL（MAPPO/IPPO）开发指南（AGENTS.md）

欢迎！本分支是 **MARL-only** 开发分支。

本项目目标：基于 **Ray RLlib（new API stack）** 的多智能体训练，解决 5G/6G 多小区网络切片资源分配与 SLA 保障问题。

## 0. 分支边界（重要）
- 本分支仅维护多智能体路线：`multi_cell_env.py`、`train_marl.py`、`test_marl.py`、`compare_marl_baseline.py`、`smoke_test_trained_model.py`、`checkpoint_utils.py`、`ippo_rl_module.py`。
- 单智能体旧方案归档在 `archive/`，不作为本分支主线维护对象。

---

## 1. 运行与测试命令

### 1.1 环境
- 使用当前 Linux 开发环境（`uv` / `.venv`）。
- 依赖以 `requirements.txt` 为准。

### 1.2 训练（MARL）
- 默认直接运行：
```bash
python train_marl.py
```
- 当前默认等价于：
```bash
python train_marl.py --mode full --hw-profile balanced --env-profile balanced --observation-mode pure_local
```
- 默认训练计划：`full + balanced(hw) + balanced(env) + pure_local(IPPO) + seed=[2026] + 200 iterations`。
- 常用切换：
```bash
python train_marl.py --mode quick
python train_marl.py --mode full --env-profile harsh
python train_marl.py --mode full --observation-mode pure_local
python train_marl.py --mode full --observation-mode neighbor_augmented
python train_marl.py --mode full --seeds 2026 2027 2028
python train_marl.py --mode full --hw-profile maxperf
python train_marl.py --mode full --iters 120
```

### 1.3 评估与对比（MARL）
```bash
python test_marl.py
python compare_marl_baseline.py
python smoke_test_trained_model.py
```

### 1.4 代码格式化与检查
```bash
black .
flake8 . --max-line-length=120
```

---

## 2. 代码风格与沟通规范

### 2.1 沟通语言
- Agent 与用户沟通使用中文。
- 强化学习与通信术语建议附英文，例如：
  - 动作空间（Action Space）
  - 观测空间（Observation Space）
  - 吞吐（Throughput）
  - 频谱效率（Spectral Efficiency, SE）

### 2.2 沟通风格与技术挑战
- Agent 应作为严格导师（Strict Mentor）与用户协作，不应一味附和。
- 当用户的技术判断、实验设计、指标解释或工程假设存在错误、混淆或不严谨之处时，Agent 必须直接指出，并说明原因。
- 如有必要，Agent 可以明确反驳用户的错误前提，但应保持专业、克制、以技术事实为依据。
- 在分析方案、代码、实验结果时，Agent 应主动提出问题，推动用户澄清假设、边界条件、指标口径和目标。
- 如果用户的说法可能导致错误结论、无效实验或不可靠对比，Agent 必须及时提醒，不能为了迎合用户而弱化问题。

### 2.3 命名与数值
- 物理量推荐带单位后缀：`_mbps`, `_ms`, `_hz`, `_tti`。
- 优先使用 `np.float32`（分析统计可用 `float64`）。
- 导入顺序：标准库 → 第三方库 → 本地模块。

---

## 3. 环境与算法现状（必须与代码一致）

### 3.0 环境档位（仅两档）
- `harsh`：原始高压环境，用于压力测试。
- `balanced`：面向“三 SLA 接近 100%”目标的可行环境。
- 当前训练/测试/对比脚本默认使用 `balanced`。

### 3.1 环境与物理参数
- 环境：`MultiCell_5G_SLA_Env`（7-cell hexagonal）。
- 总带宽：`100 MHz`。
- TTI：`0.5 ms`。
- 信道：AR(1) 时序相关衰落。
- 干扰：全小区对称建模 ICI（Inter-Cell Interference），并作用于有效 SE。
- `balanced` 环境下，ICI 对边缘小区按“实际邻居数”归一化，不再被固定 `6` 邻居分母稀释。

### 3.2 动作空间（Action Space）
- 每个 agent 动作：`spaces.Box(-1.0, 1.0, shape=(3,))`。
- 环境 `step()` 中映射规则：
  - 温度缩放 `Softmax`：`ratios = softmax(action * temperature)`
  - 当前默认 `action_softmax_temperature = 3.0`
  - 不再使用旧的线性变换 + 裁剪映射。

### 3.3 观测空间（Observation Space）
- 当前支持两种模式：
  - `pure_local`：**14 维**，纯本地 IPPO 基线，不含任何邻居信息
  - `neighbor_augmented`：**20 维**，用于后续协同增强实验
- `pure_local` 14 维：
  - `[0:3]` Demand/Arrivals
  - `[3:6]` Queue Backlog
  - `[6:9]` Spectral Efficiency
  - `[9:12]` Previous Ratios
  - `[12]` URLLC Estimated Delay
  - `[13]` eMBB GBR Shortfall
- `neighbor_augmented` 20 维：
  - `[0:14]` 纯本地 14 维
  - `[14:17]` 邻居上一时刻动作均值（neighbor previous-action mean）
  - `[17:20]` 邻居紧急度统计（URLLC queue mean / URLLC delay-ratio mean / eMBB shortfall mean）
- 环境直接输出原始物理量（raw physical values），不再在环境内部做手工除法缩放。
- 观测归一化统一交给 RLlib `MeanStdFilter`。

### 3.4 策略共享（Parameter Sharing）
- 当前不是“全 7 小区单策略”。
- 现为 **双共享策略**：
  - `center_policy`：`BS_0`
  - `edge_policy`：`BS_1 ~ BS_6`
- `policy_mapping_fn` 必须与训练/测试/评估保持一致。
- 当前纯本地 IPPO 主线使用显式 actor 初始先验：
  - 目标首步分配偏好：`[0.4, 0.4, 0.2]`
  - 通过 `Softmax^{-1}` 映射为初始 raw action mean
  - 初始 `log_std = -1.5`
  - 逻辑统一封装在 `ippo_rl_module.py`

### 3.5 当前主线定位
- 当前基础路线是 **纯本地 IPPO（pure_local）**：
  - 无全局状态（global state）
  - 无邻居观测（neighbor information）
  - `cooperative_alpha = 1.0`，纯本地 reward
- `neighbor_augmented` 仅作为后续从 IPPO 走向更强协同方法的过渡模式，不作为当前默认主线。

### 3.6 奖励函数（Reward Function）
- `harsh` 仍保留旧的连续惩罚/tail-risk 逻辑，用于压力测试。
- `balanced` 主线已切回单智能体风格的极简二值 reward：
  - `reward = total_throughput_mbps / 100 - 10*eMBB_flag - 20*URLLC_flag - 10*mMTC_flag`
  - `violations` 继续保留连续比例用于分析与绘图
  - reward 只看布尔违约标志（`violation_flags`）
- 当前 ENV_CONFIG（train/test/compare 同步）为：
  - `penalty_weight=0.7`
  - `w_embb=1.0, w_urllc=0.30, w_mmtc=0.7`
  - `urllc_warning_ratio=0.90, urllc_tail_ratio=0.92, urllc_softplus_slope=10.0`
  - `urllc_warning_gain=0.20, urllc_tail_quad_gain=2.0, urllc_hard_violation_gain=1.75`
  - `urllc_overflow_gain=2.2, urllc_exp_coeff=1.6, urllc_penalty_cap_factor=10.0`
  - `embb_penalty_quad_gain=2.5, embb_penalty_cubic_gain=1.2, embb_penalty_cap_factor=16.0`
  - `mmtc_penalty_cap_factor=5.0`
  - `embb_violation_cap=2.0, urllc_violation_cap=5.0, mmtc_violation_cap=2.0`
- `balanced` 环境下，训练奖励改为 binary SLA reward：
  - `cooperative_alpha = 1.0`
  - `binary_reward_throughput_scale = 80.0`
  - `binary_penalty_embb/urllc/mmtc = 4 / 6 / 4`
  - `binary_urllc_yellow_start_ratio = 0.5`（1ms 黄灯起点，2ms 红线）
  - `binary_urllc_yellow_penalty = 6.0`（黄灯区随延迟线性增加）
  - `center_reward_scale = 1.0`
  - `reward_clip_abs = 0.0`

### 3.7 奖励可观测化（Penalty Observability）
- 环境 `info` 已输出：
  - `reward_base_tp`, `local_reward`, `penalty_total`
  - `local_reward_unclipped`, `role_reward_scale`
  - `violations`（封顶后的训练/评估用违规比例）
  - `violations_raw`（原始物理违规比例，仅用于分析）
  - `violation_flags`（reward 实际使用的布尔违约标志）
  - `penalty_{embb,urllc,mmtc}`（裁剪后）
  - `penalty_raw_{embb,urllc,mmtc}`（裁剪前）
- 训练 callback 已记录中心小区上述指标和 penalty share 到 TensorBoard custom metrics。
- 训练 callback 额外记录 `center_{embb,urllc,mmtc}_sla_ok`，用于后续 checkpoint 综合选模。

### 3.8 `balanced` 环境定义
- `balanced` 当前设置：
  - `eMBB GBR = 220 Mbps`
  - `URLLC max delay = 2 ms`
  - `mMTC max queue = 1.0`
- 环境动力学与评估口径：
  - `URLLC burst` 增强：`start_prob=0.06`, `end_prob=0.35`, `burst_mean=100 Mbps`, `burst_std=15 Mbps`
  - `ICI` 设为中等偏强：`ici_gain=0.50`, `se_modifier_floor=0.45`
  - `ICI` 归一化：`actual_neighbors`
  - `eMBB` 从逐 TTI 硬判改为 `8 TTI` 滑动平均吞吐评估
  - 动作映射：`Softmax(temperature=3.0)`
- 该设置已通过本地可行性检查：
  - `Priority Heuristic` 不再能达到系统级 `eMBB/URLLC/mMTC = 100%/100%/100%`

---

## 4. 训练配置约定（train_marl.py）

### 4.1 API 栈
- 使用 RLlib new API stack：
  - `enable_rl_module_and_learner=True`
  - `enable_env_runner_and_connector_v2=True`

### 4.2 模式与硬件档位
- `--mode quick`：快速迭代调参（更小 batch / 更短 fragment / 更少迭代）。
- `--mode full`：正式训练。
- `--hw-profile balanced|maxperf`：full 模式下控制 env_runners 与 batch 规模。

### 4.3 默认与可复现性
- 默认：`full + balanced(hw) + balanced(env) + pure_local + seed=[2026]`。
- 标准多种子列表保留：`[2026, 2027, 2028]`，可通过 `--seeds` 手工切换。
- 训练结果目录按环境版本隔离：`ray_results/MAPPO_5G_Slicing_{experiment_env_tag}_seed{seed}`。
- 当前目录标签：
  - `harsh + pure_local -> harsh_ippo_v1`
  - `balanced + pure_local -> balanced_ippo_v1`
  - `harsh + neighbor_augmented -> harsh_neighbor_v4`
  - `balanced + neighbor_augmented -> balanced_neighbor_v6`
- `quick` / `full` 训练都使用 `MeanStdFilter`，不再使用 `NoFilter`。

---

## 5. 评估框架约定（compare_marl_baseline.py）

### 5.1 基线集合
- `Static SLA Split`
- `Priority Heuristic`
- `Max-Weight`
- `Proportional Fair`
- `Max-Throughput (Upper Bound)`
- `IPPO (pure_local)`

### 5.2 Seed 划分
- 训练 seed：`[2026, 2027, 2028]`（用于 checkpoint 池）。
- 评估 seed：`[3026, 3027, 3028, 3029]`（与训练 seed 强制不重合）。

### 5.3 指标口径
- 时序曲线（含均值±方差带）：
  - URLLC Delay
  - Cumulative Reward（BS_0）
  - eMBB Shortfall（BS_0）
  - System Throughput（7 cells）
- 汇总指标：
  - Jain Fairness Index（eMBB 累计吞吐）
  - SLA success rate（system + BS_0）
  - Penalty 绝对值与占比（含 raw/clipped 分解）
  - IPPO 推理开销：total/per-agent/p95（ms）

### 5.4 Checkpoint 选择
- 使用 `checkpoint_utils.rank_checkpoints_by_metric()`：
  - `compare_marl_baseline.py` 当前使用 `min_training_iteration=50`，禁止早期 checkpoint 混入正式对比
  - 当前主排序规则：先最小化 `center_total_sla_violations = embb_viol + urllc_viol + mmtc_viol`
  - 次排序规则：最大化 `system_throughput_mbps`（若历史日志无该指标，则回退到 `center_reward_base_tp` 作为吞吐 proxy）
  - `quality_score` 仅作为次级辅助摘要分，不再作为首要排序依据
  - 设计原则：先保“三 SLA 总体违约最少”，再比较吞吐，避免高吞吐掩盖 SLA 失效

---

## 6. 修改后最小自测要求

- 修改 `multi_cell_env.py` / `train_marl.py` / 评估主逻辑后，至少执行：
```bash
python smoke_test_trained_model.py
```
- 涉及训练配置或 policy mapping 变更，建议追加：
```bash
python train_marl.py --mode quick
python test_marl.py
python compare_marl_baseline.py
```
- 若报错必须修复；warning 需判断是否影响有效性或复现性。

---

## 7. 交付总结（强制）

- 每次发生代码修改后，回复中必须提供完整总结，至少包含：
  1. 修改文件清单（逐个文件说明）。
  2. 每个文件核心改动点（新增/删除/重构）。
  3. 行为变化与预期影响（训练、评估、指标、兼容性）。
  4. 执行过的验证命令与结果（通过/失败、关键输出）。
  5. 未完成项、风险项与建议下一步。
- 若本次无代码改动，也必须明确写出“本次无文件修改”。

---

## 8. 当前分支注意事项

- `ray_results/` 与 `results/` 被 `.gitignore` 忽略，默认不纳入版本管理。
- 评估脚本依赖已训练 checkpoint；若切换 `env_profile`，需要使用对应 profile 重新训练得到的 checkpoint。
- 如需论文级实验，请固定配置后再切回多 seed 全量运行（避免开发阶段过早拉满实验成本）。

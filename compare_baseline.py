import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_5g_sla import FiveG_SLA_Env

# --- 配置路径 (确保是你最新训练的模型) ---
model_path = "./models_formal/best_model.zip"
stats_path = "./models_formal/vec_normalize.pkl"

# 测试步数 (跑久一点，数据更稳)
N_TEST_STEPS = 2000


# ==========================================
# 1. 定义 Baseline Agents (针对拥塞场景优化)
# ==========================================

class StaticAgent:
    """
    静态分配 (针对 20MHz 拥塞场景优化):
    资源极其有限。
    策略：eMBB 60%, URLLC 35%, mMTC 5%
    Action (Softmax前): [0.5, 0.0, -2.0]
    """

    def predict(self, obs, deterministic=True):
        return np.array([[0.5, 0.0, -2.0]]), None


class HeuristicAgent:
    """
    启发式代理 (V6 - 保守防御版):
    痛点分析：之前的版本平时给 eMBB 太多(>80%)，导致 URLLC 潜伏期延迟积累。
    修正策略：
    1. 平时模式：向 Static 靠拢，给 URLLC 留足 30% (Static 是 35%)。
    2. 紧急模式：比 Static 更强，给 URLLC 50% (Static 只有 35%)。
    """

    def predict(self, obs, deterministic=True):
        urllc_queue = obs[0][4]

        # 阈值极低：只要有一点点积压，马上切换，防止延迟累积
        if urllc_queue > 0.005:
            # --- 紧急模式 (Emergency) ---
            # 目标：eMBB 45%, URLLC 50%, mMTC 5%
            # Static 只有 35% 给 URLLC，我们给 50%，清空队列速度更快！
            # eMBB 45% * 80Mbps = 36Mbps。
            # 虽然稍微低于 40Mbps GBR，但因为只持续很短时间清队列，
            # 平均下来 eMBB 速率可能还是达标的。这是为了救 URLLC 必须做的牺牲。
            # Action: [-0.1, 0.0, -2.0]
            # e^-0.1≈0.9, e^0=1.0 -> URLLC 占比 > eMBB
            return np.array([[-0.1, 0.0, -2.0]]), None
        else:
            # --- 平时模式 (Normal) ---
            # 目标：eMBB 65%, URLLC 30%, mMTC 5%
            # 之前的版本给 eMBB 85%，太激进了。
            # 现在只比 Static (60%) 多拿 5% 的好处。
            # Action: [0.8, 0.0, -2.0]
            # e^0.8≈2.22, e^0=1.0 -> eMBB 占比 ≈ 68%
            return np.array([[0.8, 0.0, -2.0]]), None


# ==========================================
# 2. 评估函数 (基于物理指标)
# ==========================================

def evaluate_agent(agent_name, agent, env, steps, is_ppo=False):
    print(f"正在评估: {agent_name} ...")

    # 强制重置环境和种子
    env.seed(2025)
    obs = env.reset()

    total_throughput = 0.0
    total_violations = 0

    for _ in range(steps):
        # 预测动作
        action, _ = agent.predict(obs, deterministic=True)

        # 执行
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]  # VecEnv 包了一层

        # --- 核心：提取真实物理指标 ---
        # 我们不看 Reward，只看物理量，这样对比才公平

        # 1. 获取真实环境
        real_env = env.envs[0]

        # 2. 计算 SLA 违约 (只要有任意一个切片违约，该时刻就算违约)
        # info['violations'] 是 [1, 0, 0] 这种
        if np.sum(info['violations']) > 0:
            total_violations += 1

        # 3. 计算实际总吞吐量 (Mbps)
        # PPO 环境里把 throughput 放到了 info 吗？如果没有，我们就手动算
        # 手动计算方法：
        # eMBB Throughput (Mbps) = Served_Mb / 0.0005s (TTI)
        # 简单起见，利用 real_env 内部状态
        # (需要你在 env_5g_sla.py 的 info 里加了 'throughput'，如果没有，下面这行会报错)
        # 如果你没加，请用下面被注释掉的代码手动算：

        if 'throughput' in info:
            total_throughput += info['throughput']
        else:
            # 手动补救计算 (Backup calculation)
            # 反推 Action 比例
            raw_action = action[0]
            exp_a = np.exp(raw_action)
            ratios = exp_a / np.sum(exp_a)
            # 容量
            se = real_env.state[6:9]
            cap_mbps = (ratios * 20e6 * se) / 1e6
            # 需求
            demands = real_env.state[0:3]
            # 实际流量 (min(capacity, demand + queue))
            # 这里简化：近似等于 capacity (在拥塞时) 或 demand (在空闲时)
            # 为准确起见，建议你在 env 文件里加 info['throughput']
            # 这里暂且假设 env 里有，或者用 0 代替 (仅看违约率)
            total_throughput += 0

    avg_throughput = total_throughput / steps
    violation_rate = (total_violations / steps) * 100.0

    print(f"  -> [{agent_name}] Throughput: {avg_throughput:.2f} Mbps | Violation: {violation_rate:.2f}%")
    return avg_throughput, violation_rate


# ==========================================
# 3. 主程序
# ==========================================

def run_comparison():
    # --- A. 准备 PPO 环境 (带归一化) ---
    # 必须加载 vec_normalize.pkl，否则 PPO 也就是个瞎子
    env_ppo = DummyVecEnv([lambda: FiveG_SLA_Env()])
    try:
        env_ppo = VecNormalize.load(stats_path, env_ppo)
        env_ppo.training = False  # 测试模式
        env_ppo.norm_reward = False
        print("PPO 环境加载成功 (带 VecNormalize)")
    except Exception as e:
        print(f"警告: 无法加载 VecNormalize ({e})。PPO 表现可能会很差。")

    # --- B. 准备 Baseline 环境 (纯净版) ---
    # Baseline 不需要归一化，直接看原始数值
    def make_raw_env():
        return FiveG_SLA_Env()

    env_base = DummyVecEnv([make_raw_env])

    # --- C. 加载 Agents ---
    ppo_agent = PPO.load(model_path, env=env_ppo)
    static_agent = StaticAgent()
    heuristic_agent = HeuristicAgent()

    # --- D. 运行评估 ---
    # 注意：evaluate_agent 内部会固定 seed，保证流量一致
    res_ppo = evaluate_agent("PPO (RL)", ppo_agent, env_ppo, N_TEST_STEPS, is_ppo=True)
    res_static = evaluate_agent("Static", static_agent, env_base, N_TEST_STEPS)
    res_heuristic = evaluate_agent("Heuristic", heuristic_agent, env_base, N_TEST_STEPS)

    # --- E. 画图 ---
    plot_comparison(res_ppo, res_static, res_heuristic)


def plot_comparison(r_ppo, r_static, r_rule):
    # --- 1. 数据准备 ---
    throughputs = [r_static[0], r_rule[0], r_ppo[0]]
    violations = [r_static[1], r_rule[1], r_ppo[1]]
    labels = ['Static', 'Heuristic', 'PPO (Ours)']

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use('ggplot')

    # 这里的 figsize 只要比例对就行，布局靠后面自动调整
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 2. 自动计算 Y 轴上限 (关键步骤) ---
    # 我们不写死 90 或 100，而是找出数据中的最大值，然后乘以 1.2 (留出 20% 的顶部空间放数字)
    max_viol = max(violations) if max(violations) > 0 else 1.0
    max_thr = max(throughputs)

    # 左轴上限 (至少 100，如果有异常值则更高，并留出 15% 空间)
    ylim_viol = max(105, max_viol * 1.15)
    # 右轴上限 (最大吞吐量 * 1.2)
    ylim_thr = max_thr * 1.2

    # --- 3. 绘制柱状图 ---
    # 左轴：违约率 (红)
    color_viol = '#d62728'
    rects1 = ax1.bar(x - width / 2, violations, width, label='SLA Violation Rate', color=color_viol, alpha=0.85)
    ax1.set_ylabel('SLA Violation Rate (%)', color=color_viol, fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_viol)
    ax1.set_ylim(0, ylim_viol)  # <--- 使用自动计算的上限

    # 右轴：吞吐量 (蓝)
    ax2 = ax1.twinx()
    color_thr = '#1f77b4'
    rects2 = ax2.bar(x + width / 2, throughputs, width, label='System Throughput', color=color_thr, alpha=0.85)
    ax2.set_ylabel('Avg Throughput (Mbps)', color=color_thr, fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_thr)
    ax2.set_ylim(0, ylim_thr)  # <--- 使用自动计算的上限

    # --- 4. 自动标注数值 ---
    # 使用 padding 参数自动控制距离，而不是手动调 xytext
    def autolabel(rects, ax, fmt):
        for rect in rects:
            height = rect.get_height()
            if height > 0:  # 只有大于0才标，避免0.0挤在一起难看
                ax.annotate(fmt.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),  # 距离柱子顶端 5 个点
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=11, fontweight='bold', color='black')

    autolabel(rects1, ax1, '{:.1f}%')
    autolabel(rects2, ax2, '{:.1f}')

    # X 轴设置
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')

    # --- 5. 图例优化 (放在顶部外面) ---
    # 标题稍微高一点
    ax1.set_title('Performance Comparison in Congested Scenario', fontsize=15, pad=50)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 放在图表区域的正上方 (bbox_to_anchor 控制绝对位置)
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center',
               bbox_to_anchor=(0.5, 1.02),  # (0.5, 1.0) 是图表顶部边缘，1.02 就是往上一点点
               ncol=2, fontsize=12, frameon=True, facecolor='white')

    # --- 6. 自动布局调整 (最重要的一步) ---
    # 这一行命令会让 matplotlib 自动计算所有元素的位置，防止遮挡
    # rect=[0, 0, 1, 0.95] 的意思是：图表只占用从底部到 95% 高度的空间，留出顶部 5% 给标题和图例
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig("./models_formal/final_comparison_auto.png", dpi=300)
    print("自动排版对比图已保存至 ./models_formal/final_comparison_auto.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()
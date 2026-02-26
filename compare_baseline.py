import os
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
# 1. 定义 Baseline Agents (针对 100MHz 场景优化)
# ==========================================

class StaticAgent:
    """
    静态分配 (针对 100MHz 宏站场景):
    策略：eMBB 75%, URLLC 15%, mMTC 10%
    注意：我们环境现在改用了线性裁剪映射: weights = clip((action + 1)/2, 0.01, 1.0)
    所以如果要达到 0.75, 0.15, 0.10 的比例，
    Action 对应的应该是 (weight * 2) - 1
    Action: [0.5, -0.7, -0.8]
    """

    def predict(self, obs, deterministic=True):
        return np.array([[0.5, -0.7, -0.8]]), None


class HeuristicAgent:
    """
    启发式代理 (改良版):
    避免了原版“用力过猛”饿死 eMBB 的问题。
    """
    def __init__(self, real_env):
        self.real_env = real_env

    def predict(self, obs, deterministic=True):
        urllc_queue = self.real_env.queues[1]
        
        # 提高阈值，不再神经过敏。只在积压超过 0.05Mb (约1个TTI的突发量) 时才启动紧急模式
        if urllc_queue > 0.05:
            # 紧急模式: 保证 eMBB 不死，给予 URLLC 足够的排空带宽
            # eMBB 50%, URLLC 40%, mMTC 10%
            # weights = [0.5, 0.4, 0.1]
            return np.array([[0.0, -0.2, -0.8]]), None
        else:
            # 平时模式: URLLC 给足 10% 防护垫
            # eMBB 80%, URLLC 10%, mMTC 10%
            # weights = [0.8, 0.1, 0.1]
            return np.array([[0.6, -0.8, -0.8]]), None


# ==========================================
# 2. 评估函数 (基于物理指标)
# ==========================================

def evaluate_agent(agent_name, agent, env, steps, is_ppo=False):
    print(f"正在评估: {agent_name} ...")

    # 强制重置环境和种子，保证三种算法遇到的流量序列完全一致
    env.seed(2026)
    obs = env.reset()
    # 同时固定 Numpy 的全局种子，因为环境里的 np.random 会用到
    np.random.seed(2026)

    total_throughput = 0.0
    total_violations = 0

    for _ in range(steps):
        # 预测动作
        action, _ = agent.predict(obs, deterministic=True)

        # 执行
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]  # VecEnv 包了一层

        # --- 核心：提取真实物理指标 ---
        # 1. 获取真实环境
        real_env = env.envs[0]

        # 2. 计算 SLA 违约
        # 新版环境的 violations 存的是连续归一化偏离度
        # 只要有任何切片违约 (值 > 0.001 排除浮点误差)，就记作该时刻发生违约
        if np.sum(info['violations']) > 0.001:
            total_violations += 1

        # 3. 计算实际总吞吐量 (Mbps)
        # 我们已经在 env_5g_sla.py 里加了 'throughput'
        total_throughput += info['throughput']

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
    real_baseline_env = env_base.envs[0]  # <--- 在这里提取真实的基站环境对象

    # --- C. 加载 Agents ---
    ppo_agent = PPO.load(model_path, env=env_ppo)
    static_agent = StaticAgent()
    heuristic_agent = HeuristicAgent(real_baseline_env)

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
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    # --- 7. 保存到结果文件夹 ---
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plt.savefig(f"{results_dir}/final_comparison_auto.png", dpi=300)
    print(f"自动排版对比图已保存至 {results_dir}/final_comparison_auto.png")
    plt.show()

if __name__ == "__main__":
    run_comparison()
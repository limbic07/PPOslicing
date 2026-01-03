import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 0. 路径与环境配置
# ==========================================
current_script_path = os.path.abspath(__file__)
analysis_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(analysis_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# 导入
from env_5g_sla import FiveG_SLA_Env
from baseline_agents import StaticAgent, HeuristicAgent

models_dir = os.path.join(project_root, "models_formal")
model_path = os.path.join(models_dir, "best_model.zip")
stats_path = os.path.join(models_dir, "vec_normalize.pkl")

# 配置
N_SEEDS = 20  # 蒙特卡洛次数
TEST_STEPS = 500  # 每轮步数


# ==========================================
# 1. 核心评估引擎 (修改版：增加原始数据采集)
# ==========================================
def evaluate_scenarios(agent_ppo, agent_static, agent_heuristic, load_factors=[1.0]):
    results = []
    raw_latencies = []  # 新增：用于存储每一步的原始时延数据

    def make_env_factory(factor):
        class FixedLoadEnv(FiveG_SLA_Env):
            def reset(self, seed=None, options=None):
                super().reset(seed=seed, options=options)
                self.load_factor = factor
                self._update_state()
                return self.state, {}

            def _update_state(self):
                super()._update_state()

        return FixedLoadEnv

    for factor in load_factors:
        print(f"⚡ Testing Load Factor: {factor}x ...")

        for seed in range(N_SEEDS):
            # --- PPO 环境 ---
            env_ppo = DummyVecEnv([lambda: make_env_factory(factor)()])
            try:
                env_ppo = VecNormalize.load(stats_path, env_ppo)
                env_ppo.training = False
                env_ppo.norm_reward = False
            except Exception as e:
                pass

            # --- Baseline 环境 ---
            env_base = DummyVecEnv([lambda: make_env_factory(factor)()])

            env_ppo.seed(seed)
            env_base.seed(seed)

            # --- 运行测试 ---
            def run_episode(agent, env, name):
                obs = env.reset()
                violations = 0
                throughput = 0

                # 临时列表存储本局时延
                episode_latencies = []

                for _ in range(TEST_STEPS):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, infos = env.step(action)
                    info = infos[0]

                    if np.sum(info['violations']) > 0:
                        violations += 1
                    throughput += info.get('throughput', 0)

                    # 采集 URLLC 时延 (转换为 ms)
                    delay_ms = info.get('est_delay_urllc', 0) * 1000.0
                    episode_latencies.append({
                        'Algorithm': name,
                        'Load': factor,
                        'Latency_ms': delay_ms
                    })

                # 将本局的时延数据加入总表
                raw_latencies.extend(episode_latencies)

                return {
                    'Algorithm': name,
                    'Load': factor,
                    'Violation Rate': (violations / TEST_STEPS) * 100,
                    'Throughput': throughput / TEST_STEPS
                }

            results.append(run_episode(agent_ppo, env_ppo, 'PPO (Ours)'))
            results.append(run_episode(agent_static, env_base, 'Static'))
            results.append(run_episode(agent_heuristic, env_base, 'Heuristic'))

    # 返回两个 DataFrame：统计表 和 原始数据表
    return pd.DataFrame(results), pd.DataFrame(raw_latencies)


# ==========================================
# 2. 绘图: 标准对比 (柱状图)
# ==========================================
def plot_bar_comparison(df):
    df_10 = df[df['Load'] == 1.0].groupby('Algorithm').mean().reset_index()

    order = ['Static', 'Heuristic', 'PPO (Ours)']
    df_10['Algorithm'] = pd.Categorical(df_10['Algorithm'], categories=order, ordered=True)
    df_10 = df_10.sort_values('Algorithm')

    labels = df_10['Algorithm'].tolist()
    viols = df_10['Violation Rate'].tolist()
    thrs = df_10['Throughput'].tolist()

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 7))

    x = np.arange(len(labels))
    width = 0.35

    ylim_viol = max(105, max(viols) * 1.2)
    ylim_thr = max(thrs) * 1.2

    color_viol = '#d62728'
    rects1 = ax1.bar(x - width / 2, viols, width, label='SLA Violation Rate', color=color_viol, alpha=0.85)
    ax1.set_ylabel('SLA Violation Rate (%)', color=color_viol, fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_viol)
    ax1.set_ylim(0, ylim_viol)

    ax2 = ax1.twinx()
    color_thr = '#1f77b4'
    rects2 = ax2.bar(x + width / 2, thrs, width, label='System Throughput', color=color_thr, alpha=0.85)
    ax2.set_ylabel('Avg Throughput (Mbps)', color=color_thr, fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_thr)
    ax2.set_ylim(0, ylim_thr)

    def autolabel(rects, ax, fmt):
        for rect in rects:
            height = rect.get_height()
            if height >= 0:
                ax.annotate(fmt.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    autolabel(rects1, ax1, '{:.2f}%')
    autolabel(rects2, ax2, '{:.1f}')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Comparison (Load=1.0, Avg of {N_SEEDS} Runs)', fontsize=15, pad=40)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(models_dir, "final_comparison_bar.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 柱状图已保存: {save_path}")
    plt.close()


# ==========================================
# 3. 绘图: 鲁棒性曲线 (0.5 - 1.2)
# ==========================================
def plot_robustness_line(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=df,
        x='Load',
        y='Violation Rate',
        hue='Algorithm',
        style='Algorithm',
        markers=True,
        dashes=False,
        linewidth=3,
        markersize=9,
        errorbar=None,
        palette=['#1f77b4', 'gray', '#ff7f0e']
    )

    plt.title("Robustness Analysis: Violation Rate vs. System Load", fontsize=15, fontweight='bold')
    plt.xlabel("Traffic Load Factor (1.0 = Normal Congestion)", fontsize=13)
    plt.ylabel("SLA Violation Rate (%)", fontsize=13)
    plt.ylim(-2, 30)

    plt.legend(title="", fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(models_dir, "final_robustness_line.png")
    plt.savefig(save_path, dpi=300)
    print(f"📈 鲁棒性图已保存: {save_path}")
    plt.close()


# ==========================================
# 4. 新增绘图: URLLC 时延 CDF (累积分布)
# ==========================================
def plot_latency_cdf(df_raw):
    """
    绘制 Load=1.0 下的 URLLC 时延 CDF 曲线
    """
    # 仅筛选 Load = 1.0 的数据
    df_plot = df_raw[df_raw['Load'] == 1.0].copy()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6))

    # 绘制 CDF
    sns.ecdfplot(
        data=df_plot,
        x='Latency_ms',
        hue='Algorithm',
        linewidth=2.5,
        palette=['#1f77b4', 'gray', '#ff7f0e']  # 保持颜色一致: PPO蓝, Static灰, Heuristic橙
    )

    # 绘制 SLA 警戒线 (2ms)
    plt.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='SLA Limit (2ms)')

    plt.title("CDF of URLLC Latency (Load=1.0)", fontsize=15, fontweight='bold')
    plt.xlabel("Estimated Latency (ms)", fontsize=13)
    plt.ylabel("Cumulative Probability", fontsize=13)

    # 优化坐标轴显示，聚焦于关键区域 (0 - 5ms)
    plt.xlim(0, 5.0)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = os.path.join(models_dir, "final_latency_cdf.png")
    plt.savefig(save_path, dpi=300)
    print(f"📉 CDF 图已保存: {save_path}")
    plt.close()


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    print(f"🚀 开始综合评估 (Runs={N_SEEDS})...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 临时环境加载模型
    temp_env = DummyVecEnv([lambda: FiveG_SLA_Env()])
    ppo_model = PPO.load(model_path)

    static_agent = StaticAgent()
    heuristic_agent = HeuristicAgent()

    # 有效区间扫描
    factors = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]

    # 获取 统计数据 和 原始时延数据
    df_results, df_raw_latencies = evaluate_scenarios(ppo_model, static_agent, heuristic_agent, factors)

    print("正在生成图表...")
    plot_bar_comparison(df_results)
    plot_robustness_line(df_results)

    # 新增：绘制 CDF
    plot_latency_cdf(df_raw_latencies)

    print("✅ 所有评估完成！请查看 models_formal 文件夹下的图片。")
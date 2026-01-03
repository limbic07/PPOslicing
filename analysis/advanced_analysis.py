import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_5g_sla import FiveG_SLA_Env
import os
import sys

# --- 1. 获取路径魔法 ---
# 获取当前脚本文件的绝对路径 (D:/.../PPOslicing/analysis/test_formal.py)
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录 (D:/.../PPOslicing/analysis)
analysis_dir = os.path.dirname(current_script_path)
# 获取项目根目录 (D:/.../PPOslicing) - 即上一级目录
project_root = os.path.dirname(analysis_dir)

# --- 2. 关键：把根目录加入 Python 搜索路径 ---
# 这样你才能直接 `from env_5g_sla import ...`，不管你在哪里运行脚本
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 3. 定义资源路径 ---
# 使用 os.path.join 拼接，而不是手写字符串
models_dir = os.path.join(project_root, "models_formal")
model_path = os.path.join(models_dir, "best_model.zip")
stats_path = os.path.join(models_dir, "vec_normalize.pkl")

# --- 4. 检查 ---
if not os.path.exists(models_dir):
    raise FileNotFoundError(f"错误：找不到模型文件夹，请检查路径: {models_dir}")
# --- 配置 ---
model_path = "../models_formal/best_model.zip"
stats_path = "../models_formal/vec_normalize.pkl"
TEST_STEPS = 2000


def get_data(agent_type, env, model=None):
    print(f"正在采集数据: {agent_type} ...")
    obs = env.reset()
    latencies = []
    throughputs = []

    # Static/Heuristic Agent 定义 (搬运之前的逻辑)
    class StaticAgent:
        def predict(self, obs, deterministic=True):
            return np.array([[0.5, 0.0, -2.0]]), None

    class HeuristicAgent:
        def predict(self, obs, deterministic=True):
            urllc_queue = obs[0][4]
            if urllc_queue > 0.005:
                return np.array([[-0.1, 0.0, -2.0]]), None
            else:
                return np.array([[0.8, 0.0, -2.0]]), None

    if agent_type == 'Static':
        agent = StaticAgent()
    elif agent_type == 'Heuristic':
        agent = HeuristicAgent()
    else:
        agent = model

    for _ in range(TEST_STEPS):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        info = infos[0]
        real_env = env.envs[0]

        # 1. 收集 eMBB 速率 (用于箱线图)
        # 这里需要 env info 里有 throughput，或者手动算。
        # 假设 info 里有，或者我们可以用 real_env 算 eMBB 的 served
        # 简单起见，我们重新算一下 eMBB 获得的速率
        raw_action = action[0]
        exp_a = np.exp(raw_action)
        ratios = exp_a / np.sum(exp_a)
        se_embb = real_env.state[6]
        # eMBB Capacity (Mbps)
        cap_embb = (ratios[0] * 20e6 * se_embb) / 1e6
        throughputs.append(cap_embb)

        # 2. 收集 URLLC 延迟 (用于 CDF)
        # 估算: Queue / Capacity
        q_urllc = real_env.state[4]  # Mb
        se_urllc = real_env.state[7]
        cap_urllc = (ratios[1] * 20e6 * se_urllc) / 1e6

        if cap_urllc > 0.01:
            lat = (q_urllc / cap_urllc) * 1000  # ms
        else:
            lat = 10.0  # 没带宽，延迟很大

        # 只记录有数据时的延迟，避免 0 延迟太多影响观测
        if q_urllc > 0:
            latencies.append(lat)

    return latencies, throughputs


def plot_advanced():
    # 准备环境
    env_ppo = DummyVecEnv([lambda: FiveG_SLA_Env()])
    env_ppo = VecNormalize.load(stats_path, env_ppo)
    env_ppo.training = False
    env_ppo.norm_reward = False

    env_base = DummyVecEnv([lambda: FiveG_SLA_Env()])  # 原始环境给 Baseline

    # 加载模型
    model = PPO.load(model_path, env=env_ppo)

    # 获取数据
    # 注意：固定 Seed 保证公平
    env_ppo.seed(2025);
    env_base.seed(2025)
    lat_ppo, thr_ppo = get_data("PPO", env_ppo, model)

    env_base.seed(2025)
    lat_static, thr_static = get_data("Static", env_base)

    env_base.seed(2025)
    lat_heuristic, thr_heuristic = get_data("Heuristic", env_base)

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图 1: URLLC 延迟 CDF (累积分布)
    # 只要看谁的曲线最快达到 1.0，且最靠左，谁就最强
    sns.ecdfplot(data=lat_ppo, label='PPO', ax=axes[0], linewidth=2, color='#1f77b4')
    sns.ecdfplot(data=lat_heuristic, label='Heuristic', ax=axes[0], linewidth=2, linestyle='--', color='#ff7f0e')
    sns.ecdfplot(data=lat_static, label='Static', ax=axes[0], linewidth=2, linestyle=':', color='#d62728')

    axes[0].axvline(2.0, color='red', linestyle='-.', alpha=0.5, label='SLA Limit (2ms)')
    axes[0].set_title("CDF of URLLC Latency", fontsize=14)
    axes[0].set_xlabel("Latency (ms)", fontsize=12)
    axes[0].set_ylabel("Cumulative Probability", fontsize=12)
    axes[0].set_xlim(0, 10)  # 关注前10ms
    axes[0].legend()

    # 图 2: eMBB 吞吐量箱线图 (稳定性)
    # 制作 DataFrame 方便 seaborn 画图
    import pandas as pd
    data_box = pd.DataFrame({
        'Algorithm': ['PPO'] * len(thr_ppo) + ['Heuristic'] * len(thr_heuristic) + ['Static'] * len(thr_static),
        'Throughput (Mbps)': thr_ppo + thr_heuristic + thr_static
    })

    sns.boxplot(x='Algorithm', y='Throughput (Mbps)', data=data_box, ax=axes[1], palette="Set2")
    axes[1].set_title("Distribution of eMBB Throughput", fontsize=14)
    axes[1].axhline(40.0, color='red', linestyle='--', label='GBR (40Mbps)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("./models_formal/advanced_analysis.png", dpi=300)
    print("高级分析图已保存至 ./models_formal/advanced_analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_advanced()
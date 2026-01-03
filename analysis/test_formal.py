import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,VecFrameStack


# ==========================================
# 0. 核心修正：路径配置 (稳健版)
# ==========================================
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取 analysis 目录
analysis_dir = os.path.dirname(current_script_path)
# 获取项目根目录 (PPOslicing/)
project_root = os.path.dirname(analysis_dir)

# 将根目录加入 Python 搜索路径，以便导入 env_5g_sla
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入环境
from env_5g_sla import FiveG_SLA_Env

# 定义资源路径
models_dir = os.path.join(project_root, "models_formal")
model_path = os.path.join(models_dir, "best_model.zip")
stats_path = os.path.join(models_dir, "vec_normalize.pkl")


def run_test():
    print(f"Loading model from: {model_path}")
    print(f"Loading stats from: {stats_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    # --- 1. 重建环境并加载统计数据 (Crucial Step!) ---
    # 必须先创建一个 DummyVecEnv
    env = DummyVecEnv([lambda: FiveG_SLA_Env()])
    env = VecFrameStack(env, n_stack=4)
    # 然后加载训练时的归一化参数 (Mean/Var)
    try:
        env = VecNormalize.load(stats_path, env)
    except Exception as e:
        print(f"错误：无法加载统计文件 {stats_path}. 请检查文件是否存在。")
        raise e

    # 测试模式设置：
    # training=False: 不要更新均值方差，保持静止
    # norm_reward=False: 我们想看原始的真实奖励，不要归一化的奖励
    env.training = False
    env.norm_reward = False

    # --- 2. 加载模型 ---
    model = PPO.load(model_path, env=env)

    # --- 3. 运行仿真 (Run Simulation) ---
    obs = env.reset()

    # 用于记录数据的列表
    history = {
        "action_ratios": [],  # 带宽比例
        "demands": [],  # 流量需求
        "served": [],  # 实际服务量
        "queues": [],  # 队列堆积
        "rewards": []
    }

    print("Running simulation episode...")
    for step in range(200):  # 跑一个回合 (200步)
        # deterministic=True: 使用确定的最优策略，不要随机探索
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        # 提取真实环境中的 Info (因为 VecEnv 会把 info 包一层 list)
        info = infos[0]

        # --- 记录数据用于画图 ---
        # 反推 Action 对应的比例 (Softmax)
        raw_action = action[0]  # 取出 batch 中的第一个
        exp_action = np.exp(raw_action)
        ratios = exp_action / np.sum(exp_action)

        # 记录
        history["action_ratios"].append(ratios)
        history["rewards"].append(rewards[0])

        # 获取环境内部状态
        real_env = env.envs[0]
        history["demands"].append(real_env.state[0:3].copy())  # 原始 Demand
        history["queues"].append(info["queue_sizes"])

        if dones[0]:
            break

    # --- 4. 数据可视化 (Plotting) ---
    plot_results(history)


def plot_results(history):
    ratios = np.array(history["action_ratios"])  # Shape: (200, 3)
    demands = np.array(history["demands"])  # Shape: (200, 3)
    queues = np.array(history["queues"])  # Shape: (200, 3)

    steps = range(len(ratios))

    # 设置字体，防止中文乱码 (如果你是中文系统)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- 图 1: 带宽分配比例 (核心结果) ---
    # 展示 Agent 如何动态调整三个切片的“蛋糕”大小
    axes[0].stackplot(steps, ratios[:, 0], ratios[:, 1], ratios[:, 2],
                      labels=['eMBB', 'URLLC', 'mMTC'],
                      colors=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    axes[0].set_title("Dynamic Bandwidth Allocation (Action)", fontsize=14)
    axes[0].set_ylabel("Bandwidth Ratio")
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # --- 图 2: URLLC 流量 vs 带宽响应 (细节分析) ---
    # 重点展示：当 URLLC 突发时，Agent 是否给了带宽？
    ax2 = axes[1]
    ax2.plot(steps, demands[:, 1], 'r--', label='URLLC Demand (Mbps)', linewidth=1.5)
    # 计算实际给 URLLC 分了多少带宽 (假设总量 20MHz, 简单映射为处理能力)
    # 这里画带宽比例作为参考
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, ratios[:, 1], 'orange', label='Allocated Ratio', linewidth=2)

    ax2.set_title("Agent Response to URLLC Burst Traffic", fontsize=14)
    ax2.set_ylabel("Demand (Mbps)")
    ax2_twin.set_ylabel("Allocated Ratio")

    # 合并图例
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # --- 图 3: 队列堆积情况 (SLA 验证) ---
    # 展示队列是否被控制住了
    axes[2].plot(steps, queues[:, 0], label='eMBB Queue')
    axes[2].plot(steps, queues[:, 1], label='URLLC Queue', linewidth=2)
    axes[2].plot(steps, queues[:, 2], label='mMTC Queue')
    axes[2].set_title("Queue Backlog Evolution (Congestion Control)", fontsize=14)
    axes[2].set_ylabel("Queue Size (Mb)")
    axes[2].set_xlabel("Simulation Steps (TTI)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # --- 修正：保存路径 ---
    save_path = os.path.join(models_dir, "test_result_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"结果图已保存至 {save_path}")
    plt.show()


if __name__ == "__main__":
    run_test()
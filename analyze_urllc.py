import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_5g_sla import FiveG_SLA_Env

# --- 配置路径 ---
model_path = "./models_formal/best_model.zip"
stats_path = "./models_formal/vec_normalize.pkl"


def analyze_urllc():
    print("正在加载模型进行 URLLC 专项分析...")

    # 1. 环境重建 (必须与训练一致)
    env = DummyVecEnv([lambda: FiveG_SLA_Env()])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # 2. 加载模型
    model = PPO.load(model_path, env=env)

    # 3. 运行仿真 (跑 500 步，捕捉更多突发情况)
    obs = env.reset()

    # 数据记录容器
    history = {
        "demand": [],  # URLLC 流量需求 (Mbps)
        "capacity": [],  # URLLC 获得的实际处理能力 (Mbps)
        "queue": [],  # URLLC 队列积压 (Mb)
        "latency": [],  # 估算延迟 (ms)
        "se": []  # 信道质量 (SE)
    }

    SIM_STEPS = 500

    for step in range(SIM_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)

        # --- 手动提取/计算 URLLC 的详细物理层数据 ---
        # 1. 获取真实环境实例
        real_env = env.envs[0]

        # 2. 获取状态数据
        # State Indices: [0-2 Demand, 3-5 Queue, 6-8 SE]
        # URLLC is index 1 (in 0,1,2), index 4 (in 3,4,5), index 7 (in 6,7,8)
        d_urllc = real_env.state[1]  # Demand
        q_urllc = real_env.state[4]  # Queue
        se_urllc = real_env.state[7]  # Spectral Efficiency

        # 3. 计算分配的带宽和能力
        # 反推 Action -> Ratio
        raw_action = action[0]
        exp_action = np.exp(raw_action)
        ratios = exp_action / np.sum(exp_action)
        urllc_ratio = ratios[1]

        # 计算获得的容量 Capacity (Mbps) = Ratio * 20MHz * SE / 1e6
        cap_urllc = (urllc_ratio * 20e6 * se_urllc) / 1e6

        # 4. 估算实时延迟 (Little's Law approximation)
        # Delay (ms) = Queue (Mb) / Capacity (Mbps) * 1000
        if cap_urllc > 0.01:
            lat_ms = (q_urllc / cap_urllc) * 1000
        else:
            lat_ms = 0  # 避免除以0，如果没有数据也就没有延迟

        # 记录
        history["demand"].append(d_urllc)
        history["capacity"].append(cap_urllc)
        history["queue"].append(q_urllc)
        history["latency"].append(lat_ms)
        history["se"].append(se_urllc)

        if dones[0]:
            obs = env.reset()

    # --- 4. 绘图与统计 ---
    plot_urllc_details(history, SIM_STEPS)


def plot_urllc_details(data, steps):
    t = range(steps)

    # 统计数据
    avg_lat = np.mean(data['latency'])
    max_lat = np.max(data['latency'])
    violations = np.sum(np.array(data['latency']) > 2.0)  # 假设 2ms SLA

    print("=" * 30)
    print(f"URLLC 性能统计 ({steps} TTI):")
    print(f"平均延迟: {avg_lat:.4f} ms")
    print(f"最大延迟: {max_lat:.4f} ms")
    print(f"SLA违约次数 (>2ms): {violations} 次")
    print("=" * 30)

    # 绘图设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 兼容中文
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 子图 1: 供需博弈 (Capacity vs Demand)
    # 观察：红线起来的时候，绿线有没有包住它？
    axes[0].plot(t, data['demand'], 'r--', label='Traffic Demand (Mbps)', alpha=0.7)
    axes[0].fill_between(t, 0, data['capacity'], color='green', alpha=0.3, label='Allocated Capacity')
    axes[0].plot(t, data['capacity'], 'g', linewidth=1.5)
    axes[0].set_title("URLLC: Supply vs Demand (供需匹配)", fontsize=14)
    axes[0].set_ylabel("Rate (Mbps)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 子图 2: 队列积压
    axes[1].plot(t, data['queue'], color='#ff7f0e', linewidth=2)
    axes[1].set_title("URLLC: Queue Backlog (队列积压)", fontsize=14)
    axes[1].set_ylabel("Data Size (Mb)")
    axes[1].grid(True, alpha=0.3)

    # 子图 3: 实时延迟 (关键 SLA 指标)
    axes[2].plot(t, data['latency'], color='blue', linewidth=1.5, label='Real-time Latency')
    # 画一条 2ms 的红线 (SLA 阈值)
    axes[2].axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='SLA Limit (2ms)')
    axes[2].set_title(f"URLLC: Latency Performance (Avg: {avg_lat:.2f}ms)", fontsize=14)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_xlabel("Time (TTI)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    # 限制 y 轴范围，让图好看点，除非爆表了
    if max_lat < 10:
        axes[2].set_ylim(0, 10)

    plt.tight_layout()
    
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    save_path = f"{results_dir}/urllc_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"URLLC 分析图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    analyze_urllc()
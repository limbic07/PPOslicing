import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,VecFrameStack

# ==========================================
# 0. 路径配置 (绝对路径)
# ==========================================
current_script_path = os.path.abspath(__file__)
analysis_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(analysis_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from env_5g_sla import FiveG_SLA_Env

models_dir = os.path.join(project_root, "models_formal")
model_path = os.path.join(models_dir, "best_model.zip")
stats_path = os.path.join(models_dir, "vec_normalize.pkl")


# ==========================================
# 1. 定义 Baseline Agents
# ==========================================
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


# ==========================================
# 2. 多轮平均测试逻辑 (Monte Carlo)
# ==========================================
def run_averaged_sensitivity():
    print("正在进行多轮平均负载敏感度分析 (Monte Carlo Simulation)...")

    # 配置
    load_factors = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]  # X轴点
    N_SEEDS = 10  # 每个点跑多少轮 (建议 5-10 轮)
    TEST_STEPS = 500  # 每轮跑多少步

    # 存储所有原始数据，用于生成 DataFrame
    # 结构: {'Algorithm': [], 'Load': [], 'Violation': []}
    data_records = []

    # 准备环境生成器
    def make_scaled_env(factor):
        """动态生成不同负载的环境"""

        class ScaledEnv(FiveG_SLA_Env):
            def _update_state(self):
                super()._update_state()
                # 暴力缩放流量 (State 0,1,2)
                self.state[0] *= factor
                self.state[1] *= factor
                self.state[2] *= factor

        return ScaledEnv()

    # 1. 加载 PPO 模型 (只加载一次)
    # 注意：这里先用标准环境加载模型，后面测试时再换环境
    temp_env = DummyVecEnv([lambda: FiveG_SLA_Env()])
    temp_env = VecFrameStack(temp_env, n_stack=4)
    try:
        temp_env = VecNormalize.load(stats_path, temp_env)
        ppo_model = PPO.load(model_path)
        print("PPO 模型加载成功")
    except Exception as e:
        print(f"Error loading PPO: {e}")
        return

    # 2. 实例化 Baseline
    static_agent = StaticAgent()
    heuristic_agent = HeuristicAgent()

    # 3. 开始大循环
    for factor in load_factors:
        print(f"Testing Load Factor: {factor}x ...")

        for seed in range(N_SEEDS):
            # --- 准备环境 ---
            # 这里的关键是：每次循环都要重新创建环境，并设置不同的种子

            # PPO 环境 (带归一化)
            # 注意：我们要用 load 出来的统计参数，但应用到 ScaledEnv 上
            env_ppo = DummyVecEnv([lambda: make_scaled_env(factor)])
            env_ppo = VecNormalize.load(stats_path, env_ppo)
            env_ppo.training = False
            env_ppo.norm_reward = False
            env_ppo.seed(seed)  # 设置随机种子

            # Baseline 环境 (原始)
            env_base = DummyVecEnv([lambda: make_scaled_env(factor)])
            env_base.seed(seed)  # 保证 Baseline 面对的流量和 PPO 一模一样

            # --- 跑测试函数 ---
            def eval_once(agent, env):
                obs = env.reset()
                violations = 0
                for _ in range(TEST_STEPS):
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, infos = env.step(action)
                    if np.sum(infos[0]['violations']) > 0:
                        violations += 1
                return (violations / TEST_STEPS) * 100

            # 记录 PPO
            v_ppo = eval_once(ppo_model, env_ppo)
            data_records.append({'Algorithm': 'PPO (Ours)', 'Load': factor, 'Violation': v_ppo})

            # 记录 Static
            v_static = eval_once(static_agent, env_base)
            data_records.append({'Algorithm': 'Static', 'Load': factor, 'Violation': v_static})

            # 记录 Heuristic
            v_heuristic = eval_once(heuristic_agent, env_base)
            data_records.append({'Algorithm': 'Heuristic', 'Load': factor, 'Violation': v_heuristic})

    # ==========================================
    # 3. 使用 Seaborn 绘图 (带置信区间)
    # ==========================================
    print("正在绘图...")

    # 转换为 DataFrame
    df = pd.DataFrame(data_records)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # 核心绘图代码
    # errorbar='sd' 表示阴影范围是 标准差 (Standard Deviation)
    # 也可以改成 'ci', 95 表示 95% 置信区间
    sns.lineplot(
        data=df,
        x='Load',
        y='Violation',
        hue='Algorithm',
        style='Algorithm',
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        errorbar=None,
        palette=['#1f77b4', 'gray', '#ff7f0e']  # 蓝，灰，橙
    )

    plt.title("Robustness Analysis: Violation Rate vs. System Load (Averaged)", fontsize=16)
    plt.xlabel("Traffic Load Factor (1.0 = Normal Congestion)", fontsize=14)
    plt.ylabel("SLA Violation Rate (%)", fontsize=14)

    # 设置 Y 轴范围 (根据数据自动调整，或者手动锁定)
    plt.ylim(-2, 105)

    # 优化图例
    plt.legend(title="", fontsize=12, loc='upper left')

    # 保存
    save_path = os.path.join(models_dir, "robustness_averaged.png")
    plt.savefig(save_path, dpi=300)
    print(f"平均分析图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_averaged_sensitivity()
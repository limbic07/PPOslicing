import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_5g_sla import FiveG_SLA_Env

# --- 配置 ---
model_path = "../models_formal/best_model.zip"
stats_path = "../models_formal/vec_normalize.pkl"
SIM_STEPS = 200


# ==========================================
# 1. 定义压力测试环境 (关键修改!)
# ==========================================
class StressTestEnv(FiveG_SLA_Env):
    """
    继承原环境，但强制注入“恶意”突发流量。
    不看随机概率，只看人为设定。
    """

    def _update_state(self):
        # 先调用父类生成基础随机数据
        super()._update_state()

        # --- 强制注入逻辑 ---
        # 在第 50ms 到 100ms 之间，制造超级拥塞
        if 50 <= self.current_step <= 100:
            # 强制 URLLC 需求飙升到 50 Mbps
            # (注: 20MHz带宽总容量约80Mbps。Static只给35%即28Mbps，必崩)
            self.state[1] = 50.0

            # eMBB 保持 60 Mbps (加上 URLLC 总需求 110Mbps > 80Mbps)
            # 这就是绝对的“资源不足”时刻
            self.state[0] = 60.0
        else:
            # 平时：URLLC 保持安静，方便观察回落过程
            self.state[1] = 0.0


# ==========================================
# 2. 定义 Baseline Agents
# ==========================================
class StaticAgent:
    """ Static: eMBB 60%, URLLC 35%, mMTC 5% """

    def predict(self, obs, deterministic=True):
        return np.array([[0.5, 0.0, -2.0]]), None


class HeuristicAgent:
    """ Heuristic V6 """

    def predict(self, obs, deterministic=True):
        urllc_queue = obs[0][4]
        if urllc_queue > 0.005:
            # 紧急: eMBB 45%, URLLC 50%
            return np.array([[-0.1, 0.0, -2.0]]), None
        else:
            # 平时: eMBB 65%, URLLC 30%
            return np.array([[0.8, 0.0, -2.0]]), None


# ==========================================
# 3. 仿真运行逻辑
# ==========================================
def run_simulation(agent, env_class):
    # 使用压力环境
    env = DummyVecEnv([lambda: env_class()])

    # 如果是 PPO，需要加载归一化参数
    # 注意：虽然流量是注入的，但 Agent 看到的 State 需要经过同样的归一化处理才能懂
    if isinstance(agent, PPO):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    obs = env.reset()
    urllc_queues = []

    for _ in range(SIM_STEPS):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        info = infos[0]

        # 记录 URLLC 队列积压量
        urllc_queues.append(info['queue_sizes'][1])

    return urllc_queues


def plot_superiority():
    # 加载 PPO
    print("Loading PPO...")
    ppo_agent = PPO.load(model_path)
    static_agent = StaticAgent()
    heuristic_agent = HeuristicAgent()

    print("Running Stress Test...")
    # 使用 StressTestEnv
    q_ppo = run_simulation(ppo_agent, StressTestEnv)
    q_static = run_simulation(static_agent, StressTestEnv)
    q_heuristic = run_simulation(heuristic_agent, StressTestEnv)

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6))

    t = range(SIM_STEPS)

    # 绘制曲线
    ax.plot(t, q_static, color='gray', linestyle='--', linewidth=2, label='Static (35% Alloc)', alpha=0.6)
    ax.plot(t, q_heuristic, color='#ff7f0e', linestyle='-.', linewidth=2, label='Heuristic (50% Alloc)')
    ax.plot(t, q_ppo, color='#1f77b4', linewidth=3, label='PPO (Dynamic Alloc)')

    # 标注突发区间
    ax.axvspan(50, 100, color='red', alpha=0.1, label='Injected Burst (50 Mbps)')

    # 装饰
    ax.set_title("Stress Test: Queue Response to Extreme Burst", fontsize=16)
    ax.set_xlabel("Time (TTI)", fontsize=14)
    ax.set_ylabel("URLLC Queue Backlog (Mb)", fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 设置 Y 轴范围，让差别更明显
    # 只要 Static 飞起来了，就限制一下视图，不然 PPO 贴地太紧看不清
    max_q = max(q_static)
    if max_q > 2.0:
        ax.set_ylim(-0.1, max_q * 1.1)

    plt.tight_layout()
    plt.savefig("./models_formal/stress_test_plot.png", dpi=300)
    print("压力测试图已保存至 ./models_formal/stress_test_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_superiority()
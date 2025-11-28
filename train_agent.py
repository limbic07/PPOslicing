from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env_5g_ra import FiveGResourceAllocationEnv
from env_5g_sla import FiveG_SLA_Env
# 1. 初始化环境
env = FiveG_SLA_Env()

# 2. 检查环境是否合规 (Debug)
check_env(env)
print("Environment check passed!")

# 3. 创建 PPO 模型
# Learning rate 0.0003 是 PPO 的经典默认值
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    tensorboard_log="./logs/",
    device="cpu"  # <--- 添加这一行
)

# 4. 训练
print("Training started...")
# 建议先跑 100,000步，毕设仿真可以跑 500k-1M 步
model.learn(total_timesteps=100000)

# 5. 保存
model.save("5g_ra_model")
print("Model saved.")
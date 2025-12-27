import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 引入SLA 环境
from env_5g_sla import FiveG_SLA_Env


def make_env():
    """
    Utility function to create environment with Monitor wrapper.
    Monitor wrapper is essential for tracking rewards in TensorBoard.
    """
    env = FiveG_SLA_Env()
    # Monitor 用于记录每一步的数据，方便画图
    return Monitor(env)


if __name__ == "__main__":
    # --- 1. 配置路径 (Setup Paths) ---
    log_dir = "./logs_formal/"  # TensorBoard 日志目录
    models_dir = "./models_formal/"  # 模型保存目录

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # --- 2. 创建环境 (Create Environments) ---
    # 训练环境：使用 VecNormalize 进行归一化
    # 为什么？因为流量是 1000Mbps，SLA 违约是 0/1，数值差异巨大，归一化能极大加速收敛。
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 评估环境：用于在训练中途测试模型好坏
    # 注意：评估环境也需要同样的归一化设置
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- 3. 定义回调函数 (Callbacks) ---
    # 核心功能：每 10,000 步测试一次，如果效果是历史最好的，就保存到 best_model.zip
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10  # 每次测试跑10个回合取平均
    )

    # --- 4. 定义网络架构 (Network Architecture) ---
    # 默认是 [64, 64]，对于 SLA 这种复杂逻辑，建议加宽加深到 [256, 256]
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # --- 5. 初始化 PPO 模型 (Init Model) ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # 经典学习率，如果波动大可调小到 1e-4
        n_steps=2048,  # 每次更新采样的步数
        batch_size=64,  # 批次大小
        gamma=0.99,  # 折扣因子
        gae_lambda=0.95,  # GAE 参数
        clip_range=0.2,  # PPO 裁剪范围
        policy_kwargs=policy_kwargs,  # 使用更大的网络
        tensorboard_log=log_dir,
        device="cpu"  # 强制使用 CPU，速度更快
    )

    # --- 6. 开始正式训练 (Start Training) ---
    print("🚀 Starting Formal Training...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Best model will be saved to: {models_dir}")

    # 步数：500,000
    TOTAL_TIMESTEPS = 500,000

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # --- 7. 保存最终结果 (Save Final) ---
    # 保存最后的模型
    model.save(f"{models_dir}/final_model")
    # 关键！保存归一化的统计参数 (均值方差)，否则将来加载模型时预测会不准
    env.save(f"{models_dir}/vec_normalize.pkl")

    print("✅ Training Finished!")
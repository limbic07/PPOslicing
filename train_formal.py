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
    env_base = DummyVecEnv([make_env])
    eval_env_base = DummyVecEnv([make_env])

    latest_model_path = os.path.join(models_dir, "latest_model.zip")
    latest_stats_path = os.path.join(models_dir, "latest_vec_normalize.pkl")

    if os.path.exists(latest_model_path) and os.path.exists(latest_stats_path):
        print(f"\n[INFO] Found existing model at {latest_model_path}. Resuming training...")
        
        # 1. 恢复环境归一化统计数据 (极其重要)
        env = VecNormalize.load(latest_stats_path, env_base)
        env.training = True 
        env.norm_reward = True

        eval_env = VecNormalize.load(latest_stats_path, eval_env_base)
        eval_env.training = False
        eval_env.norm_reward = False

        # 2. 恢复模型参数，并指定 tensorboard 继续记录
        model = PPO.load(latest_model_path, env=env, device="cpu", tensorboard_log=log_dir)
        print("[INFO] Successfully loaded checkpoint! The model will continue learning from where it left off.")
    else:
        print("\n[INFO] No existing checkpoint found. Starting fresh training...")
        
        # 训练环境：使用 VecNormalize 进行归一化
        env = VecNormalize(env_base, norm_obs=True, norm_reward=True, clip_obs=10.)
        eval_env = VecNormalize(eval_env_base, norm_obs=True, norm_reward=True, clip_obs=10.)

        # 默认是 [64, 64]，对于 SLA 这种复杂逻辑，建议加宽加深到 [256, 256]
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

        # 初始化新的 PPO 模型
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,  # 经典学习率
            n_steps=2048,  # 每次更新采样的步数
            batch_size=64,  # 批次大小
            gamma=0.99,  # 折扣因子
            gae_lambda=0.95,  # GAE 参数
            clip_range=0.2,  # PPO 裁剪范围
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            device="cpu"  # 恢复为使用 CPU 进行训练
        )

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

    # --- 4. 开始正式训练 (Start Training) ---
    print("\n[INFO] Starting Formal Training...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Best model will be saved to: {models_dir}")
    print("[TIP] You can press Ctrl+C at any time to interrupt. The progress will be saved automatically!\n")

    # 步数设置
    TOTAL_TIMESTEPS = 200_000

    try:
        # reset_num_timesteps=False 保证断点续训时，TensorBoard 上的 X 轴步数是连续的，不会清零
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Received KeyboardInterrupt (Ctrl+C)! Saving and safely aborting...")
    finally:
        # --- 5. 保存最终结果或断点 (Save Final/Checkpoint) ---
        print("[INFO] Saving the latest model and environment stats...")
        
        # 1. 保存断点 (用于后续自动恢复)
        model.save(latest_model_path)
        env.save(latest_stats_path)
        
        # 2. 顺便覆盖 final_model 和 vec_normalize.pkl，兼容你原有的测试代码
        model.save(f"{models_dir}/final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        
        print("[SUCCESS] Checkpoint saved successfully!")
        print(f"-> Location: {latest_model_path}")
        print("-> You can run `python train_formal.py` again to resume training from this exact point.")
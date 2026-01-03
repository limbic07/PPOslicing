import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env_5g_sla import FiveG_SLA_Env


def make_env():
    env = FiveG_SLA_Env()
    return Monitor(env)


if __name__ == "__main__":
    log_dir = "./logs_formal/"
    models_dir = "./models_formal/"

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(models_dir): os.makedirs(models_dir)

    # 训练环境 (无 Stack)
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 评估环境
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0002,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device="cpu"
    )

    print("🚀 开始训练 (Dynamic Load, No Stack)...")
    # 建议至少跑 50万 - 100万步
    model.learn(total_timesteps=500_000, callback=eval_callback)

    model.save(f"{models_dir}/final_model")
    env.save(f"{models_dir}/vec_normalize.pkl")
    print("✅ 训练完成！")
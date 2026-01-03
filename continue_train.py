import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env_5g_sla import FiveG_SLA_Env

# --- 配置路径 ---
# 读取旧模型和旧统计数据的路径
old_models_dir = "./models_formal/"
old_model_path = os.path.join(old_models_dir, "best_model.zip")
old_stats_path = os.path.join(old_models_dir, "vec_normalize.pkl")

# 新的日志路径 (可选：你可以存到同一个文件夹，也可以新建)
log_dir = "./logs_formal/"


def make_env():
    env = FiveG_SLA_Env()
    return Monitor(env)


if __name__ == "__main__":
    print(f"🔄 正在加载旧模型和环境参数: {old_model_path}")

    # --- 1. 重建环境并加载统计参数 (关键!) ---
    # 先创建一个空环境
    env = DummyVecEnv([make_env])

    # 加载之前的均值和方差 (VecNormalize)
    # training=True: 我们要继续训练，所以要继续更新统计数据
    # norm_reward=True: 奖励也继续归一化
    env = VecNormalize.load(old_stats_path, env)
    env.training = True
    env.norm_reward = True

    # 同时也需要为评估环境加载同样的参数
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load(old_stats_path, eval_env)
    eval_env.training = True
    eval_env.norm_reward = True

    # ... (前面的代码不变) ...

    # --- 2. 加载模型 ---
    model = PPO.load(old_model_path, env=env, device="cpu")

    # ==========================================
    # 🛠️ 【核心修复】手动校准时间步
    # ==========================================
    # 你必须知道上一轮训练总共跑了多少步 (比如 500,000)
    # 或者去 TensorBoard 看一眼最后一步是多少
    PREVIOUS_TOTAL_STEPS = 500_000  # <--- 请修改为你实际上一次结束时的总步数

    print(f"校准步数: 模型记录步数 {model.num_timesteps} -> 强制修正为 {PREVIOUS_TOTAL_STEPS}")
    model.num_timesteps = PREVIOUS_TOTAL_STEPS




    # --- 3. 设置回调函数 ---
    # 继续保存最好的模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=old_models_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    # --- 4. 继续训练 ---
    MORE_TIMESTEPS = 200_000
    print(f"🚀 开始续训 (追加 {MORE_TIMESTEPS} 步)...")

    # reset_num_timesteps=False 必须保留
    # tb_log_name 建议保持一致，这样会写在同一个 PPO_x 文件夹下(如果没被占用)
    # 或者你可以指定一个新的名字，TensorBoard 会自动把它们连起来显示
    model.learn(total_timesteps=MORE_TIMESTEPS,
                callback=eval_callback,
                reset_num_timesteps=False)  # 这里的 False 配合上面的手动修改才有效


    # --- 5. 保存结果 ---
    model.save(f"{old_models_dir}/final_model_extended")
    env.save(f"{old_models_dir}/vec_normalize.pkl")  # 覆盖更新统计文件

    print("✅ 续训完成！模型已更新。")
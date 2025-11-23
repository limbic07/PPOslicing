"""
PPO Algorithm Hyperparameter Configuration
PPO算法超参数配置
"""


class PPOHyperparameters:
    # Network Architecture / 网络结构
    STATE_DIM = 15  # State space dimension / 状态空间维度
    ACTION_DIM = 3  # Action space dimension / 动作空间维度
    HIDDEN_DIM = 256  # Hidden layer dimension / 隐藏层维度

    # PPO Parameters / PPO算法参数
    LEARNING_RATE_ACTOR = 1e-4  # Actor network learning rate / Actor网络学习率
    LEARNING_RATE_CRITIC = 1e-3  # Critic network learning rate / Critic网络学习率
    GAMMA = 0.99  # Discount factor / 折扣因子
    LAMBDA = 0.95  # GAE parameter / GAE参数
    EPS_CLIP = 0.2  # PPO clip parameter / PPO裁剪参数
    K_EPOCHS = 10  # Number of PPO epochs / PPO训练轮数
    CRITIC_DISCOUNT = 1.0  # Critic loss weight / Critic损失权重
    ENTROPY_BETA = 0.01  # Entropy bonus coefficient / 熵奖励系数

    # Training Parameters / 训练参数
    MAX_EPISODES = 10000  # Maximum training episodes / 最大训练回合数
    MAX_STEPS = 1000  # Maximum steps per episode / 每回合最大步数
    BATCH_SIZE = 64  # Training batch size / 训练批次大小
    UPDATE_EVERY = 100  # Update model every N episodes / 每N回合更新模型

    # Environment Parameters / 环境参数
    TOTAL_RESOURCES = 100  # Total available resources / 总可用资源
    NUM_SLICES = 3  # Number of network slices / 网络切片数量


class TrainingConfig:
    """Training Configuration / 训练配置"""
    SAVE_INTERVAL = 100  # Model save interval / 模型保存间隔
    LOG_INTERVAL = 10  # Logging interval / 日志记录间隔
    MODEL_SAVE_PATH = "models/"  # Model save path / 模型保存路径
    RESULTS_SAVE_PATH = "results/"  # Results save path / 结果保存路径
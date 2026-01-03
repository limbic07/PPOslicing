import numpy as np


class StaticAgent:
    """
    静态分配代理 (Static Allocation Agent)
    策略: 固定带宽比例，不随环境变化。

    默认配置 (针对拥塞场景优化):
    - eMBB: 60% (保 GBR)
    - URLLC: 35% (应对突发)
    - mMTC: 5%

    对应动作 (Softmax前): [0.5, 0.0, -2.0]
    """

    def __init__(self, action_logits=None):
        if action_logits is None:
            # 默认动作
            self.action = np.array([[0.5, 0.0, -2.0]])
        else:
            self.action = np.array([action_logits])

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """
        遵循 Stable-Baselines3 的 predict 接口签名
        """
        # 无论 obs 是什么，返回固定的动作
        # batch_size = obs.shape[0]
        # return np.tile(self.action, (batch_size, 1)), None

        # 简单起见，假设 batch_size=1 (DummyVecEnv)
        return self.action, None


class HeuristicAgent:
    """
    启发式代理 (Heuristic Agent - Weighted Queue-Aware)
    策略: 基于队列积压的加权动态分配。

    公式: Score_k = Base_k + (Weight_k * Queue_k)
    特点:
    1. Base_k 保证空闲时的基础带宽。
    2. Weight_k 决定了对拥塞的敏感度 (URLLC 敏感度最高)。
    """

    def __init__(self):
        # 基础权重 (Base): 类似 Static 的保底
        self.base_scores = np.array([0.6, 0.35, 0.05])

        # 动态权重 (Sensitivity): 对队列积压的反应强度
        # URLLC 设为 10.0，表示只要有一点积压，得分就飙升
        self.queue_weights = np.array([1.0, 10.0, 1.0])

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """
        根据当前观测值动态计算动作
        """
        # 1. 提取队列信息 (Obs 维度 9，索引 3,4,5 是队列)
        # obs shape is usually (1, 9)
        queues = obs[0, 3:6]  # [q_embb, q_urllc, q_mmtc]

        # 2. 计算得分 (Score)
        # score = base + weight * queue
        scores = self.base_scores + (self.queue_weights * queues)

        # 3. 转换为 Action (Logits)
        # 因为环境内部会做 Softmax: exp(action) / sum
        # 我们希望 exp(action) 正比于 scores
        # 所以 action = log(scores)
        # 添加极小值防止 log(0)
        action = np.log(np.maximum(scores, 1e-6))

        return np.array([action]), None
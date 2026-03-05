import argparse
import os
import warnings

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns
from ray.tune.registry import register_env

from checkpoint_utils import rank_checkpoints_by_metric
from multi_cell_env import MultiCell_5G_SLA_Env

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

TRAIN_SEEDS = [2026, 2027, 2028]
EXPERIMENT_DIRS = [f"./ray_results/MAPPO_5G_Slicing_seed{seed}" for seed in TRAIN_SEEDS]
ENV_CONFIG = {
    "penalty_weight": 0.7,
    "urllc_warning_ratio": 0.65,
    "urllc_softplus_slope": 12.0,
    "urllc_warning_gain": 1.0,
    "urllc_overflow_gain": 6.0,
    "urllc_exp_coeff": 2.5,
    "urllc_penalty_cap_factor": 20.0,
    "embb_penalty_quad_gain": 1.2,
    "embb_penalty_cap_factor": 10.0,
    "ici_gain": 0.65,
    "se_modifier_floor": 0.3,
}


def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def compute_action(algo, obs: np.ndarray, policy_id: str) -> np.ndarray:
    module = algo.get_module(policy_id)
    obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        module_out = module.forward_inference({Columns.OBS: obs_batch})
        dist_cls = module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS]).to_deterministic()
        action = action_dist.sample()[0].cpu().numpy().astype(np.float32)

    return np.clip(action, -1.0, 1.0)


def build_eval_algo(eval_seed: int):
    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=ENV_CONFIG)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=eval_seed)
        .rl_module(model_config_dict={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
        .multi_agent(
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(observation_filter="MeanStdFilter", num_env_runners=0)
        .learners(num_learners=0)
    )
    return config.build()


def resolve_checkpoint(manual_checkpoint: str | None):
    if manual_checkpoint:
        return manual_checkpoint

    ranked_checkpoints = rank_checkpoints_by_metric(EXPERIMENT_DIRS)
    if not ranked_checkpoints:
        raise FileNotFoundError(
            "No ranked checkpoints found under configured seed dirs. "
            "Please run train_marl.py first."
        )
    return ranked_checkpoints[0]["checkpoint_path"]


def run_smoke_test(steps: int, eval_seed: int, checkpoint_path: str | None):
    ray.init(ignore_reinit_error=True)
    algo = build_eval_algo(eval_seed=eval_seed)
    ckpt = resolve_checkpoint(checkpoint_path)
    algo.restore(ckpt)
    print(f"Loaded checkpoint: {ckpt}")

    env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
    obs, _ = env.reset(seed=eval_seed)
    done = {"__all__": False}

    center_rewards = []
    center_urllc_delay_ms = []
    executed_steps = 0

    while not done["__all__"] and executed_steps < steps:
        actions = {}
        for agent_id, agent_obs in obs.items():
            if not np.all(np.isfinite(agent_obs)):
                raise RuntimeError(f"Non-finite observation at step={executed_steps}, agent={agent_id}")
            policy_id = policy_mapping_fn(agent_id)
            action = compute_action(algo, agent_obs, policy_id=policy_id)
            if not np.all(np.isfinite(action)):
                raise RuntimeError(f"Non-finite action at step={executed_steps}, agent={agent_id}")
            actions[agent_id] = action

        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        if "BS_0" not in rewards:
            raise RuntimeError("Missing BS_0 reward in env output.")

        reward_center = float(rewards["BS_0"])
        delay_center_ms = float(infos["BS_0"]["est_urllc_delay"] * 1000.0)

        if not np.isfinite(reward_center) or not np.isfinite(delay_center_ms):
            raise RuntimeError(f"Non-finite metric at step={executed_steps}.")

        center_rewards.append(reward_center)
        center_urllc_delay_ms.append(delay_center_ms)
        done = terminateds
        executed_steps += 1

    algo.stop()
    ray.shutdown()

    if executed_steps == 0:
        raise RuntimeError("Smoke test executed zero steps.")

    print("Smoke test passed.")
    print(f"Executed steps: {executed_steps}")
    print(f"Center cumulative reward: {np.sum(center_rewards):.3f}")
    print(f"Center mean URLLC delay (ms): {np.mean(center_urllc_delay_ms):.3f}")
    print(f"Center max URLLC delay (ms): {np.max(center_urllc_delay_ms):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Quick smoke test for trained MAPPO checkpoint.")
    parser.add_argument("--steps", type=int, default=50, help="Maximum rollout steps for the test.")
    parser.add_argument("--eval-seed", type=int, default=2026, help="Evaluation seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, auto-picks best from seed dirs.",
    )
    args = parser.parse_args()

    run_smoke_test(steps=args.steps, eval_seed=args.eval_seed, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

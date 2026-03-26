import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns
from ray.tune.registry import register_env

from checkpoint_utils import rank_checkpoints_by_metric
from ippo_rl_module import (
    CENTRALIZED_CRITIC_GLOBAL_DIM,
    DEFAULT_INITIAL_ACTION_LOG_STD,
    build_initialized_rl_module_spec,
)
from multi_cell_env import MultiCell_5G_SLA_Env

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_memory_usage_threshold"] = "1.0"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

logging.getLogger("ray").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_EVAL_SEED = 2026
DEFAULT_TRAIN_SEEDS = [2026, 2027, 2028]
DEFAULT_ENV_PROFILE = "balanced"
DEFAULT_ALGO_MODE = "mappo"
DEFAULT_MIN_BEST_CHECKPOINT_ITER = 50

IPPO_EXPERIMENT_TAGS = {
    "harsh": "harsh_ippo_v2",
    "balanced": "balanced_ippo_v5",
}
MAPPO_EXPERIMENT_TAGS = {
    "harsh": "harsh_mappo_ctde_v3",
    "balanced": "balanced_mappo_ctde_v5",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO/MAPPO checkpoint.")
    parser.add_argument(
        "--algo-mode",
        choices=["ippo", "mappo"],
        default=DEFAULT_ALGO_MODE,
        help="ippo: pure local baseline; mappo: CTDE mode.",
    )
    parser.add_argument(
        "--env-profile",
        choices=["harsh", "balanced"],
        default=DEFAULT_ENV_PROFILE,
        help="Environment profile used during training.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Evaluation seed for rollout.",
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SEEDS,
        help="Training seeds used to search checkpoint directories.",
    )
    parser.add_argument(
        "--min-best-iter",
        type=int,
        default=DEFAULT_MIN_BEST_CHECKPOINT_ITER,
        help="Minimum training iteration when selecting best checkpoint.",
    )
    return parser.parse_args()


def resolve_algo_mode(algo_mode: str):
    mode = str(algo_mode).lower()
    if mode == "mappo":
        return {
            "observation_mode": "neighbor_augmented",
            "use_centralized_critic": True,
            "cooperative_alpha": None,
            "neighbor_liability_beta": None,
            "neighbor_dividend_gamma": None,
        }
    return {
        "observation_mode": "pure_local",
        "use_centralized_critic": False,
        "cooperative_alpha": 1.0,
        "neighbor_liability_beta": 0.0,
        "neighbor_dividend_gamma": 0.0,
    }


def resolve_experiment_tag(env_profile: str, algo_mode: str) -> str:
    profile = str(env_profile).lower()
    mode = str(algo_mode).lower()
    tags = MAPPO_EXPERIMENT_TAGS if mode == "mappo" else IPPO_EXPERIMENT_TAGS
    if profile not in tags:
        raise ValueError(f"Unsupported env_profile={env_profile!r}")
    return tags[profile]


def build_env_config(env_profile: str, algo_mode: str) -> dict:
    mode_cfg = resolve_algo_mode(algo_mode)
    profile_overrides = MultiCell_5G_SLA_Env._get_env_profile_overrides(env_profile)
    env_config = {
        "env_profile": env_profile,
        "observation_mode": mode_cfg["observation_mode"],
        "use_centralized_critic": mode_cfg["use_centralized_critic"],
        "centralized_critic_global_dim": CENTRALIZED_CRITIC_GLOBAL_DIM,
        "action_softmax_temperature": float(profile_overrides.get("action_softmax_temperature", 1.0)),
    }
    if mode_cfg["cooperative_alpha"] is not None:
        env_config["cooperative_alpha"] = mode_cfg["cooperative_alpha"]
    if mode_cfg["neighbor_liability_beta"] is not None:
        env_config["neighbor_liability_beta"] = mode_cfg["neighbor_liability_beta"]
    if mode_cfg["neighbor_dividend_gamma"] is not None:
        env_config["neighbor_dividend_gamma"] = mode_cfg["neighbor_dividend_gamma"]
    return env_config


def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def compute_action_new_stack(algo, obs: np.ndarray, policy_id: str) -> np.ndarray:
    module = algo.get_module(policy_id)
    obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        module_out = module.forward_inference({Columns.OBS: obs_batch})
        dist_cls = module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS]).to_deterministic()
        action = action_dist.sample()[0].cpu().numpy().astype(np.float32)

    return np.clip(action, -1.0, 1.0)


def run_test():
    args = parse_args()
    experiment_tag = resolve_experiment_tag(args.env_profile, args.algo_mode)
    experiment_dirs = [
        f"./ray_results/MAPPO_5G_Slicing_{experiment_tag}_seed{seed}"
        for seed in args.train_seeds
    ]
    env_config = build_env_config(args.env_profile, args.algo_mode)
    mode_cfg = resolve_algo_mode(args.algo_mode)

    print(
        f"Testing mode={args.algo_mode}, env_profile={args.env_profile}, "
        f"observation_mode={mode_cfg['observation_mode']}, "
        f"use_centralized_critic={mode_cfg['use_centralized_critic']}, "
        f"cooperative_alpha={mode_cfg['cooperative_alpha'] if mode_cfg['cooperative_alpha'] is not None else 'profile'}, "
        f"neighbor_liability_beta={mode_cfg['neighbor_liability_beta'] if mode_cfg['neighbor_liability_beta'] is not None else 'profile'}, "
        f"neighbor_dividend_gamma={mode_cfg['neighbor_dividend_gamma'] if mode_cfg['neighbor_dividend_gamma'] is not None else 'profile'}, "
        f"experiment_tag={experiment_tag}, eval_seed={args.eval_seed}"
    )

    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=args.eval_seed)
        .rl_module(
            rl_module_spec=build_initialized_rl_module_spec(
                env_config["action_softmax_temperature"],
                fcnet_hiddens=[256, 256],
                fcnet_activation="tanh",
                initial_action_log_std=DEFAULT_INITIAL_ACTION_LOG_STD,
                observation_mode=env_config["observation_mode"],
                include_ici_features=bool(env_config.get("neighbor_augmented_include_ici_features", False)),
                use_centralized_critic=bool(env_config.get("use_centralized_critic", False)),
                critic_global_dim=int(
                    env_config.get("centralized_critic_global_dim", CENTRALIZED_CRITIC_GLOBAL_DIM)
                ),
            )
        )
        .multi_agent(
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(observation_filter="MeanStdFilter", num_env_runners=0)
        .learners(num_learners=0)
    )

    algo = config.build()
    ranked_checkpoints = rank_checkpoints_by_metric(
        experiment_dirs,
        min_training_iteration=args.min_best_iter,
        fallback_to_any=False,
    )
    if not ranked_checkpoints:
        raise FileNotFoundError(
            f"No ranked checkpoints found in {args.algo_mode.upper()} seed experiment dirs. "
            "Please run train_marl.py first."
        )

    restore_errors = []
    restored_checkpoint = None
    for item in ranked_checkpoints:
        checkpoint_path = item["checkpoint_path"]
        score = item.get("episode_return_mean")
        iteration = item.get("training_iteration")
        urllc_violation = item.get("center_urllc_violations")
        embb_violation = item.get("center_embb_violations")
        urllc_delay_ms = item.get("center_urllc_delay_ms")
        quality_score = item.get("quality_score")
        base_tp = item.get("center_reward_base_tp")
        print(
            f"Trying checkpoint: {checkpoint_path} "
            f"(iter={iteration}, urllc_viol={urllc_violation}, "
            f"embb_viol={embb_violation}, urllc_delay_ms={urllc_delay_ms}, "
            f"base_tp={base_tp}, quality={quality_score}, episode_return_mean={score})"
        )
        try:
            algo.restore(checkpoint_path)
            restored_checkpoint = checkpoint_path
            print(
                f"Loaded best available checkpoint: {checkpoint_path} "
                f"(iter={iteration}, urllc_viol={urllc_violation}, "
                f"embb_viol={embb_violation}, urllc_delay_ms={urllc_delay_ms}, "
                f"base_tp={base_tp}, quality={quality_score}, episode_return_mean={score})"
            )
            break
        except Exception as exc:  # noqa: PERF203
            restore_errors.append(f"{checkpoint_path} -> {exc}")

    if restored_checkpoint is None:
        error_preview = "\n".join(restore_errors[:3])
        raise RuntimeError(
            "Failed to restore any ranked checkpoint.\n"
            f"Sample restore errors:\n{error_preview}"
        )

    env = MultiCell_5G_SLA_Env(config=env_config)
    obs, _ = env.reset(seed=args.eval_seed)

    rewards_history = {agent: [] for agent in env.agents}
    throughput_history = {agent: [] for agent in env.agents}
    urllc_delay_history = {agent: [] for agent in env.agents}

    done = {"__all__": False}
    step = 0

    print("Running evaluation...")
    while not done["__all__"] and step < 200:
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            actions[agent_id] = compute_action_new_stack(algo, agent_obs, policy_id=policy_id)

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for agent in env.agents:
            if agent in rewards:
                rewards_history[agent].append(rewards[agent])
                throughput_history[agent].append(infos[agent]["throughput"])
                urllc_delay_history[agent].append(infos[agent]["est_urllc_delay"] * 1000)

        done = terminateds
        step += 1

    print("Plotting results...")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(throughput_history["BS_0"], label="Center Cell (BS_0) Throughput")
    avg_edge_throughput = np.mean([throughput_history[f"BS_{i}"] for i in range(1, 7)], axis=0)
    plt.plot(avg_edge_throughput, label="Avg Edge Cell Throughput")
    plt.title("Throughput Comparison (Center vs Edge)")
    plt.xlabel("Step")
    plt.ylabel("Throughput (Mbps)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(urllc_delay_history["BS_0"], label="BS_0 URLLC Delay", color="red")
    plt.axhline(y=2.0, color="black", linestyle="--", label="2ms Deadline")
    plt.title("Center Cell URLLC Delay")
    plt.xlabel("Step")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(np.cumsum(rewards_history["BS_0"]), label="BS_0 Reward")
    avg_edge_reward = np.mean([np.cumsum(rewards_history[f"BS_{i}"]) for i in range(1, 7)], axis=0)
    plt.plot(avg_edge_reward, label="Avg Edge Reward")
    plt.title("Cumulative Cooperative Rewards")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)

    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("./results/marl_evaluation.png")
    print("Results saved to ./results/marl_evaluation.png")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    run_test()

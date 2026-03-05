import os
import time
import warnings

import matplotlib.pyplot as plt
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

EPS = 1e-5
PF_MA_ALPHA = 0.9
ROLLOUT_STEPS = 200
TRAIN_SEEDS = [2026, 2027, 2028]
EVAL_SEEDS = [2026, 2027, 2028, 2029]
MAPPO_EXPERIMENT_DIRS = [f"./ray_results/MAPPO_5G_Slicing_seed{seed}" for seed in TRAIN_SEEDS]

ALGORITHMS = [
    ("static", "Static Equal Share"),
    ("priority", "Priority Heuristic"),
    ("max_weight", "Max-Weight"),
    ("pf", "Proportional Fair"),
    ("mappo", "MAPPO (Proposed)"),
]

PLOT_COLORS = {
    "static": "#1f77b4",
    "priority": "#ff7f0e",
    "max_weight": "#2ca02c",
    "pf": "#d62728",
    "mappo": "#9467bd",
}

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

MAPPO_ALGO = None


def env_creator(env_config):
    return MultiCell_5G_SLA_Env(config=env_config)


register_env("MultiCell_5G_SLA_Env", env_creator)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "center_policy" if agent_id == "BS_0" else "edge_policy"


def compute_actions_batched_new_stack(algo, obs_dict):
    obs_by_policy = {"center_policy": [], "edge_policy": []}
    agent_ids_by_policy = {"center_policy": [], "edge_policy": []}

    for agent_id, agent_obs in obs_dict.items():
        policy_id = policy_mapping_fn(agent_id)
        obs_by_policy[policy_id].append(agent_obs)
        agent_ids_by_policy[policy_id].append(agent_id)

    actions = {}
    for policy_id in ("center_policy", "edge_policy"):
        if not obs_by_policy[policy_id]:
            continue

        module = algo.get_module(policy_id)
        obs_batch = torch.as_tensor(np.stack(obs_by_policy[policy_id], axis=0), dtype=torch.float32)

        with torch.no_grad():
            module_out = module.forward_inference({Columns.OBS: obs_batch})
            dist_cls = module.get_inference_action_dist_cls()
            action_dist = dist_cls.from_logits(module_out[Columns.ACTION_DIST_INPUTS]).to_deterministic()
            action_batch = action_dist.sample().cpu().numpy().astype(np.float32)

        action_batch = np.clip(action_batch, -1.0, 1.0)
        for idx, agent_id in enumerate(agent_ids_by_policy[policy_id]):
            actions[agent_id] = action_batch[idx]

    return actions


def ratios_to_action(ratios: np.ndarray) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=np.float32)
    ratios = np.clip(ratios, 1e-8, None)
    ratio_sum = float(np.sum(ratios))
    if ratio_sum <= 0.0:
        ratios = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    else:
        ratios = ratios / ratio_sum

    weights = np.clip(ratios, 0.01, 1.0)
    action = (weights * 2.0) - 1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def compute_jain_fairness(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    numerator = float(np.sum(x) ** 2)
    denominator = float(n * np.sum(x**2))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def init_mappo_algo() -> str:
    global MAPPO_ALGO

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("MultiCell_5G_SLA_Env", env_config=ENV_CONFIG)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .debugging(seed=EVAL_SEEDS[0])
        .rl_module(model_config_dict={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
        .multi_agent(
            policies={"center_policy", "edge_policy"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(observation_filter="MeanStdFilter", num_env_runners=0)
        .learners(num_learners=0)
    )

    MAPPO_ALGO = config.build()

    ranked_checkpoints = rank_checkpoints_by_metric(MAPPO_EXPERIMENT_DIRS)
    if not ranked_checkpoints:
        raise FileNotFoundError(
            "No ranked checkpoint found under MAPPO seed experiment dirs. "
            "Please run train_marl.py first."
        )

    restore_errors = []
    for item in ranked_checkpoints:
        checkpoint_path = item["checkpoint_path"]
        score = item.get("episode_return_mean")
        iteration = item.get("training_iteration")
        urllc_violation = item.get("center_urllc_violations")
        urllc_delay_ms = item.get("center_urllc_delay_ms")
        print(
            f"Trying checkpoint: {checkpoint_path} "
            f"(iter={iteration}, urllc_viol={urllc_violation}, "
            f"urllc_delay_ms={urllc_delay_ms}, episode_return_mean={score})"
        )
        try:
            MAPPO_ALGO.restore(checkpoint_path)
            print(f"Loaded MAPPO checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as exc:  # noqa: PERF203
            restore_errors.append(f"{checkpoint_path} -> {exc}")

    error_preview = "\n".join(restore_errors[:3])
    raise RuntimeError(
        "No compatible checkpoint could be restored with current stack.\n"
        f"Sample restore errors:\n{error_preview}"
    )


def run_evaluation(env, algo_name, seed):
    if algo_name == "mappo" and MAPPO_ALGO is None:
        raise RuntimeError("MAPPO_ALGO is not initialized. Call init_mappo_algo() first.")

    obs, _ = env.reset(seed=seed)
    done = {"__all__": False}

    center_reward = []
    center_urllc_delay_ms = []
    center_embb_shortfall = []
    center_throughput_mbps = []
    system_throughput_mbps = []
    inference_overhead_total_ms = []
    inference_overhead_per_agent_ms = []

    embb_cumulative_throughput = {agent: 0.0 for agent in env.agents}

    pf_avg_throughput = {
        agent: np.full(3, 1.0, dtype=np.float32) for agent in env.agents
    }

    # SLA success counters: slice order [eMBB, URLLC, mMTC].
    sla_ok_sys = np.zeros(3, dtype=np.float64)
    sla_total_sys = 0
    sla_ok_bs0 = np.zeros(3, dtype=np.float64)
    sla_total_bs0 = 0

    steps = 0
    while not done["__all__"] and steps < ROLLOUT_STEPS:
        actions = {}

        if algo_name == "static":
            static_ratios = np.array([0.33, 0.33, 0.34], dtype=np.float32)
            for agent in env.agents:
                actions[agent] = ratios_to_action(static_ratios)

        elif algo_name == "priority":
            for agent in env.agents:
                if env.queues[agent][1] > 0.05:
                    actions[agent] = np.array([-0.8, 1.0, -0.8], dtype=np.float32)
                else:
                    actions[agent] = np.array([0.8, -0.5, -0.8], dtype=np.float32)

        elif algo_name == "max_weight":
            for agent in env.agents:
                queue_vec = np.maximum(env.queues[agent], 0.0)
                se_vec = np.maximum(env.current_se[agent], 0.0)
                weights = queue_vec * se_vec
                if float(np.sum(weights)) <= EPS:
                    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                ratios = weights / np.sum(weights)
                actions[agent] = ratios_to_action(ratios)

        elif algo_name == "pf":
            for agent in env.agents:
                # Estimated rate if the slice gets full bandwidth.
                rate_est_mbps = (env.total_bandwidth * np.maximum(env.current_se[agent], 0.0)) / 1e6
                weights = rate_est_mbps / (pf_avg_throughput[agent] + EPS)
                if float(np.sum(weights)) <= EPS:
                    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                ratios = weights / np.sum(weights)
                actions[agent] = ratios_to_action(ratios)

        elif algo_name == "mappo":
            start = time.perf_counter()
            actions = compute_actions_batched_new_stack(MAPPO_ALGO, obs)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            inference_overhead_total_ms.append(elapsed_ms)
            inference_overhead_per_agent_ms.append(elapsed_ms / max(len(obs), 1))

        else:
            raise ValueError(f"Unknown algorithm name: {algo_name}")

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        step_system_tp_mbps = 0.0
        for agent in env.agents:
            if agent not in infos:
                continue
            step_system_tp_mbps += float(infos[agent]["throughput"])

            violations = np.asarray(infos[agent]["violations"], dtype=np.float64)
            sla_ok_sys += (violations <= 0.0).astype(np.float64)
            sla_total_sys += 1
            if agent == "BS_0":
                sla_ok_bs0 += (violations <= 0.0).astype(np.float64)
                sla_total_bs0 += 1

            embb_tp_mbps = float(infos[agent]["throughput_slices_mbps"][0])
            embb_cumulative_throughput[agent] += embb_tp_mbps

            if algo_name == "pf":
                inst_tp = np.asarray(infos[agent]["throughput_slices_mbps"], dtype=np.float32)
                pf_avg_throughput[agent] = (
                    PF_MA_ALPHA * pf_avg_throughput[agent] + (1.0 - PF_MA_ALPHA) * inst_tp
                )

        center_reward.append(float(rewards["BS_0"]))
        center_urllc_delay_ms.append(float(infos["BS_0"]["est_urllc_delay"] * 1000.0))
        center_embb_shortfall.append(float(env.state["BS_0"][13]))
        center_throughput_mbps.append(float(infos["BS_0"]["throughput"]))
        system_throughput_mbps.append(step_system_tp_mbps)

        done = terminateds
        steps += 1

    embb_vec = np.array([embb_cumulative_throughput[agent] for agent in env.agents], dtype=np.float64)
    jfi_embb = compute_jain_fairness(embb_vec)
    sla_sys_success = sla_ok_sys / max(float(sla_total_sys), 1.0)
    sla_bs0_success = sla_ok_bs0 / max(float(sla_total_bs0), 1.0)

    result = {
        "reward": np.asarray(center_reward, dtype=np.float32),
        "cum_reward": np.cumsum(np.asarray(center_reward, dtype=np.float32)),
        "urllc_delay_ms": np.asarray(center_urllc_delay_ms, dtype=np.float32),
        "embb_shortfall": np.asarray(center_embb_shortfall, dtype=np.float32),
        "center_throughput_mbps": np.asarray(center_throughput_mbps, dtype=np.float32),
        "system_throughput_mbps": np.asarray(system_throughput_mbps, dtype=np.float32),
        "sla_sys_success_rate": np.asarray(sla_sys_success, dtype=np.float64),
        "sla_bs0_success_rate": np.asarray(sla_bs0_success, dtype=np.float64),
        "jfi_embb": float(jfi_embb),
        "mean_inference_total_ms": (
            float(np.mean(inference_overhead_total_ms)) if inference_overhead_total_ms else np.nan
        ),
        "p95_inference_total_ms": (
            float(np.percentile(inference_overhead_total_ms, 95)) if inference_overhead_total_ms else np.nan
        ),
        "mean_inference_per_agent_ms": (
            float(np.mean(inference_overhead_per_agent_ms)) if inference_overhead_per_agent_ms else np.nan
        ),
    }
    return result


def stack_metric(run_list, metric_key):
    data = np.full((len(run_list), ROLLOUT_STEPS), np.nan, dtype=np.float32)
    for idx, run in enumerate(run_list):
        values = run[metric_key]
        valid_len = min(len(values), ROLLOUT_STEPS)
        data[idx, :valid_len] = values[:valid_len]
    return data


def aggregate_results(results_by_algo):
    aggregated = {}
    for algo_key, run_list in results_by_algo.items():
        delay = stack_metric(run_list, "urllc_delay_ms")
        cum_reward = stack_metric(run_list, "cum_reward")
        shortfall = stack_metric(run_list, "embb_shortfall")
        center_tp = stack_metric(run_list, "center_throughput_mbps")
        system_tp = stack_metric(run_list, "system_throughput_mbps")

        fairness = np.asarray([run["jfi_embb"] for run in run_list], dtype=np.float64)
        inf_total_ms = np.asarray([run["mean_inference_total_ms"] for run in run_list], dtype=np.float64)
        inf_total_ms_finite = inf_total_ms[np.isfinite(inf_total_ms)]
        inf_p95_total_ms = np.asarray([run["p95_inference_total_ms"] for run in run_list], dtype=np.float64)
        inf_p95_total_ms_finite = inf_p95_total_ms[np.isfinite(inf_p95_total_ms)]
        inf_per_agent_ms = np.asarray([run["mean_inference_per_agent_ms"] for run in run_list], dtype=np.float64)
        inf_per_agent_ms_finite = inf_per_agent_ms[np.isfinite(inf_per_agent_ms)]
        sla_sys = np.asarray([run["sla_sys_success_rate"] for run in run_list], dtype=np.float64)
        sla_bs0 = np.asarray([run["sla_bs0_success_rate"] for run in run_list], dtype=np.float64)

        aggregated[algo_key] = {
            "delay_mean": np.nanmean(delay, axis=0),
            "delay_std": np.nanstd(delay, axis=0),
            "cum_reward_mean": np.nanmean(cum_reward, axis=0),
            "cum_reward_std": np.nanstd(cum_reward, axis=0),
            "shortfall_mean": np.nanmean(shortfall, axis=0),
            "shortfall_std": np.nanstd(shortfall, axis=0),
            "center_tp_mean": np.nanmean(center_tp, axis=0),
            "center_tp_std": np.nanstd(center_tp, axis=0),
            "system_tp_mean": np.nanmean(system_tp, axis=0),
            "system_tp_std": np.nanstd(system_tp, axis=0),
            "fairness_mean": float(np.nanmean(fairness)),
            "fairness_std": float(np.nanstd(fairness)),
            "inference_total_ms_mean": (
                float(np.mean(inf_total_ms_finite)) if inf_total_ms_finite.size > 0 else np.nan
            ),
            "inference_total_ms_std": (
                float(np.std(inf_total_ms_finite)) if inf_total_ms_finite.size > 0 else np.nan
            ),
            "inference_p95_total_ms_mean": (
                float(np.mean(inf_p95_total_ms_finite)) if inf_p95_total_ms_finite.size > 0 else np.nan
            ),
            "inference_p95_total_ms_std": (
                float(np.std(inf_p95_total_ms_finite)) if inf_p95_total_ms_finite.size > 0 else np.nan
            ),
            "inference_per_agent_ms_mean": (
                float(np.mean(inf_per_agent_ms_finite)) if inf_per_agent_ms_finite.size > 0 else np.nan
            ),
            "inference_per_agent_ms_std": (
                float(np.std(inf_per_agent_ms_finite)) if inf_per_agent_ms_finite.size > 0 else np.nan
            ),
            "sla_sys_mean": np.nanmean(sla_sys, axis=0),
            "sla_sys_std": np.nanstd(sla_sys, axis=0),
            "sla_bs0_mean": np.nanmean(sla_bs0, axis=0),
            "sla_bs0_std": np.nanstd(sla_bs0, axis=0),
        }

    return aggregated


def plot_with_band(ax, x_axis, mean, std, label, color):
    ax.plot(x_axis, mean, label=label, linewidth=2.0, color=color)
    ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.18, linewidth=0.0)


def run_baselines():
    print("Initializing MAPPO checkpoint...")
    loaded_checkpoint = init_mappo_algo()
    print(f"Using checkpoint: {loaded_checkpoint}")

    results_by_algo = {key: [] for key, _ in ALGORITHMS}

    for seed in EVAL_SEEDS:
        print(f"\n=== Evaluation seed={seed} ===")
        for algo_key, algo_label in ALGORITHMS:
            env = MultiCell_5G_SLA_Env(config=ENV_CONFIG)
            run = run_evaluation(env, algo_key, seed)
            results_by_algo[algo_key].append(run)

            print(
                f"[{algo_label}] "
                f"JFI={run['jfi_embb']:.4f}, "
                f"sys_tp_mean={np.mean(run['system_throughput_mbps']):.2f} Mbps, "
                f"mean_delay_ms={np.mean(run['urllc_delay_ms']):.3f}, "
                f"cum_reward={run['cum_reward'][-1]:.3f}"
            )
            if algo_key == "mappo":
                print(
                    f"[{algo_label}] "
                    f"inference_total={run['mean_inference_total_ms']:.4f} ms, "
                    f"inference_per_agent={run['mean_inference_per_agent_ms']:.4f} ms, "
                    f"inference_p95_total={run['p95_inference_total_ms']:.4f} ms"
                )

    aggregated = aggregate_results(results_by_algo)

    print("\n=== Summary over 4 seeds (mean ± std) ===")
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        fairness_text = f"JFI={stats['fairness_mean']:.4f} ± {stats['fairness_std']:.4f}"
        sys_tp_text = f"SysTP={np.mean(stats['system_tp_mean']):.2f} Mbps"
        delay_text = f"Delay={np.mean(stats['delay_mean']):.3f} ms"
        reward_text = f"FinalCumReward={stats['cum_reward_mean'][-1]:.3f}"
        sla_sys_text = (
            "SLA_sys[eMBB/URLLC/mMTC]="
            f"{stats['sla_sys_mean'][0]*100:.1f}%/"
            f"{stats['sla_sys_mean'][1]*100:.1f}%/"
            f"{stats['sla_sys_mean'][2]*100:.1f}%"
        )
        print(f"{algo_label:>20}: {fairness_text}, {sys_tp_text}, {delay_text}, {reward_text}")
        print(f"{' ':>22}{sla_sys_text}")
        if algo_key == "mappo":
            print(
                " " * 22
                + "Inference total="
                + f"{stats['inference_total_ms_mean']:.4f} ± {stats['inference_total_ms_std']:.4f} ms, "
                + "per-agent="
                + f"{stats['inference_per_agent_ms_mean']:.4f} ± {stats['inference_per_agent_ms_std']:.4f} ms, "
                + "p95(total)="
                + f"{stats['inference_p95_total_ms_mean']:.4f} ± {stats['inference_p95_total_ms_std']:.4f} ms "
                + "(TTI=0.5ms)"
            )

    x = np.arange(ROLLOUT_STEPS)
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    ax1 = axes[0, 0]
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        plot_with_band(
            ax1,
            x,
            stats["delay_mean"],
            stats["delay_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax1.axhline(y=2.0, color="black", linestyle="--", linewidth=1.5, label="SLA Deadline (2ms)")
    ax1.set_title("URLLC Delay (mean ± std)")
    ax1.set_xlabel("Time Step (TTI)")
    ax1.set_ylabel("Delay (ms)")
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[0, 1]
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        plot_with_band(
            ax2,
            x,
            stats["cum_reward_mean"],
            stats["cum_reward_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax2.set_title("Cumulative Reward (mean ± std, BS_0)")
    ax2.set_xlabel("Time Step (TTI)")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True)
    ax2.legend()

    ax3 = axes[1, 0]
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        plot_with_band(
            ax3,
            x,
            stats["shortfall_mean"],
            stats["shortfall_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax3.set_title("eMBB Shortfall (mean ± std, BS_0)")
    ax3.set_xlabel("Time Step (TTI)")
    ax3.set_ylabel("Shortfall (Mbps)")
    ax3.grid(True)
    ax3.legend()

    ax4 = axes[1, 1]
    for algo_key, algo_label in ALGORITHMS:
        stats = aggregated[algo_key]
        plot_with_band(
            ax4,
            x,
            stats["system_tp_mean"],
            stats["system_tp_std"],
            algo_label,
            PLOT_COLORS[algo_key],
        )
    ax4.set_title("System Throughput (mean ± std, 7 cells)")
    ax4.set_xlabel("Time Step (TTI)")
    ax4.set_ylabel("Throughput (Mbps)")
    ax4.grid(True)
    ax4.legend()

    ax5 = axes[2, 0]
    labels = [label for _, label in ALGORITHMS]
    fairness_means = [aggregated[key]["fairness_mean"] for key, _ in ALGORITHMS]
    fairness_stds = [aggregated[key]["fairness_std"] for key, _ in ALGORITHMS]
    bars = ax5.bar(labels, fairness_means, yerr=fairness_stds, capsize=4.0, color=[PLOT_COLORS[k] for k, _ in ALGORITHMS])
    ax5.set_title("Jain Fairness Index (eMBB Throughput)")
    ax5.set_ylabel("JFI")
    ax5.set_ylim(0.0, 1.05)
    ax5.grid(True, axis="y")
    ax5.tick_params(axis="x", rotation=20)

    mappo_stats = aggregated["mappo"]
    ax5.text(
        0.02,
        0.92,
        (
            "MAPPO inference overhead:\n"
            f"total={mappo_stats['inference_total_ms_mean']:.4f} ms\n"
            f"per-agent={mappo_stats['inference_per_agent_ms_mean']:.4f} ms\n"
            f"p95(total)={mappo_stats['inference_p95_total_ms_mean']:.4f} ms"
        ),
        transform=ax5.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    for bar, score in zip(bars, fairness_means):
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax6 = axes[2, 1]
    x_idx = np.arange(len(ALGORITHMS))
    width = 0.25
    embb_sla = np.array([aggregated[key]["sla_sys_mean"][0] for key, _ in ALGORITHMS], dtype=np.float64)
    urllc_sla = np.array([aggregated[key]["sla_sys_mean"][1] for key, _ in ALGORITHMS], dtype=np.float64)
    mmtc_sla = np.array([aggregated[key]["sla_sys_mean"][2] for key, _ in ALGORITHMS], dtype=np.float64)

    ax6.bar(x_idx - width, embb_sla, width=width, label="eMBB SLA success", color="#1f77b4")
    ax6.bar(x_idx, urllc_sla, width=width, label="URLLC SLA success", color="#ff7f0e")
    ax6.bar(x_idx + width, mmtc_sla, width=width, label="mMTC SLA success", color="#2ca02c")
    ax6.set_xticks(x_idx, labels)
    ax6.set_ylim(0.0, 1.05)
    ax6.set_ylabel("Success Rate")
    ax6.set_title("System SLA Success Rate (mean over seeds)")
    ax6.tick_params(axis="x", rotation=20)
    ax6.grid(True, axis="y")
    ax6.legend()

    os.makedirs("./results", exist_ok=True)
    plt.tight_layout()
    output_path = "./results/marl_baseline_comparison_multiseed.png"
    plt.savefig(output_path)
    print(f"Saved comparison plots to {output_path}")

    if MAPPO_ALGO is not None:
        MAPPO_ALGO.stop()
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    run_baselines()

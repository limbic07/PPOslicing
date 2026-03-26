import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import compare_marl_baseline as base


OUTPUT_PREFIX = "./results/urllc_delay_cdf"
TARGET_ALGOS = [
    ("priority", "Priority Heuristic"),
    ("ippo", "IPPO (Baseline, pure_local)"),
    ("mappo", "MAPPO (Proposed, CTDE)"),
]


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def empirical_cdf(values: np.ndarray):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    arr = np.sort(arr)
    y = np.arange(1, arr.size + 1, dtype=np.float64) / float(arr.size)
    return arr, y


def summarize(values: np.ndarray):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "max_ms": float("nan"),
            "share_lt_0_1ms": float("nan"),
            "share_lt_0_5ms": float("nan"),
            "share_lt_1_0ms": float("nan"),
            "share_lt_2_0ms": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "share_lt_0_1ms": float(np.mean(arr < 0.1)),
        "share_lt_0_5ms": float(np.mean(arr < 0.5)),
        "share_lt_1_0ms": float(np.mean(arr < 1.0)),
        "share_lt_2_0ms": float(np.mean(arr < 2.0)),
    }


def collect_delays_for_algo(algo_key: str, learned_evaluator=None):
    system_delays_ms = []
    center_delays_ms = []

    for seed in base.EVAL_SEEDS:
        if algo_key in base.LEARNED_METHOD_BY_KEY:
            env = base.MultiCell_5G_SLA_Env(config=learned_evaluator["env_config"])
        else:
            env = base.MultiCell_5G_SLA_Env(config=base.HEURISTIC_ENV_CONFIG)

        obs, reset_infos = env.reset(seed=seed)
        done = {"__all__": False}
        learned_runner = None
        learned_episode = None
        learned_shared_data = None

        if algo_key in base.LEARNED_METHOD_BY_KEY:
            learned_runner, learned_episode, learned_shared_data = base.build_ippo_episode_context(
                learned_evaluator["algo"], obs, reset_infos
            )

        steps = 0
        while not done["__all__"] and steps < base.ROLLOUT_STEPS:
            actions = {}
            extra_model_outputs = None

            if algo_key == "priority":
                normal_ratios = np.array([0.6621907, 0.29754147, 0.04026786], dtype=np.float32)
                emergency_ratios = np.array([0.4435102, 0.4901546, 0.0663352], dtype=np.float32)
                for agent in env.agents:
                    agent_obs = obs[agent]
                    urllc_queue_feature = float(agent_obs[4])
                    ratios = emergency_ratios if urllc_queue_feature > 0.005 else normal_ratios
                    actions[agent] = base.ratios_to_action(ratios, env.action_softmax_temperature)
            elif algo_key in base.LEARNED_METHOD_BY_KEY:
                policy_actions, actions, extra_model_outputs = base.compute_actions_batched(
                    learned_evaluator["algo"],
                    learned_runner,
                    learned_episode,
                    learned_shared_data,
                )
            else:
                raise ValueError(f"Unsupported algo_key={algo_key}")

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            if algo_key in base.LEARNED_METHOD_BY_KEY:
                learned_episode.add_env_step(
                    obs,
                    policy_actions,
                    rewards,
                    infos=infos,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    extra_model_outputs=extra_model_outputs,
                )

            for agent in env.agents:
                delay_ms = float(infos[agent].get("est_urllc_delay", 0.0) * 1000.0)
                system_delays_ms.append(delay_ms)
                if agent == "BS_0":
                    center_delays_ms.append(delay_ms)

            done = terminateds
            steps += 1

    return np.asarray(system_delays_ms, dtype=np.float64), np.asarray(center_delays_ms, dtype=np.float64)


def plot_cdf(delay_dict, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for algo_key, algo_label in TARGET_ALGOS:
        values = delay_dict[algo_key]
        x, y = empirical_cdf(values)
        ax.plot(x, y, label=algo_label, linewidth=2.0, color=base.PLOT_COLORS[algo_key])
    ax.axvline(2.0, color="black", linestyle="--", linewidth=1.5, label="2 ms SLA")
    ax.set_title(title)
    ax.set_xlabel("URLLC Delay (ms)")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    base.validate_seed_split()
    learned_evaluators_by_algo = base.init_learned_evaluators()

    system_delay_dict = {}
    center_delay_dict = {}
    summary = {"selection_mode": base.LEARNED_SELECTION_MODE, "algorithms": {}}

    for algo_key, algo_label in TARGET_ALGOS:
        learned_evaluator = None
        if algo_key in base.LEARNED_METHOD_BY_KEY:
            learned_evaluator = learned_evaluators_by_algo[algo_key][0]

        system_delays_ms, center_delays_ms = collect_delays_for_algo(
            algo_key, learned_evaluator=learned_evaluator
        )
        system_delay_dict[algo_key] = system_delays_ms
        center_delay_dict[algo_key] = center_delays_ms
        summary["algorithms"][algo_key] = {
            "label": algo_label,
            "system_delay_ms": summarize(system_delays_ms),
            "center_delay_ms": summarize(center_delays_ms),
        }
        if learned_evaluator is not None:
            summary["algorithms"][algo_key]["selected_checkpoint"] = str(learned_evaluator["checkpoint_path"])
            summary["algorithms"][algo_key]["selected_iteration"] = int(learned_evaluator["training_iteration"])

    system_plot = f"{OUTPUT_PREFIX}_system.png"
    center_plot = f"{OUTPUT_PREFIX}_bs0.png"
    summary_path = f"{OUTPUT_PREFIX}_summary.json"
    ensure_parent(system_plot)
    ensure_parent(center_plot)
    ensure_parent(summary_path)

    plot_cdf(system_delay_dict, "URLLC Delay CDF (All Cells)", system_plot)
    plot_cdf(center_delay_dict, "URLLC Delay CDF (BS_0)", center_plot)

    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {system_plot}")
    print(f"Saved: {center_plot}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

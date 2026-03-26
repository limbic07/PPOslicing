import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ray

import analyze_policy_behavior as apb
import compare_marl_baseline as cb


DEFAULT_EVAL_SEEDS = [3026, 3027, 3028, 3029]
DEFAULT_ROLLOUT_STEPS = 200
DEFAULT_EDGE_AGENT = "BS_1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate midterm-showcase plots proving MAPPO cooperation channels are active."
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL_SEEDS,
        help="Evaluation seeds.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help="Max rollout steps per evaluation seed.",
    )
    parser.add_argument(
        "--edge-agent",
        type=str,
        default=DEFAULT_EDGE_AGENT,
        help="Edge agent id to inspect (e.g., BS_1 ... BS_6).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="./results/coop_showcase",
        help="Output prefix for CSV/JSON/PNG.",
    )
    return parser.parse_args()


def _select_evaluator(algo_key: str) -> Dict:
    evaluators = cb.LEARNED_EVALUATORS.get(algo_key, [])
    if not evaluators:
        raise RuntimeError(f"No evaluator loaded for algo={algo_key}.")

    def ranking_key(item: Dict):
        quality = item.get("quality_score")
        iter_ = item.get("training_iteration")
        total_viol = item.get("center_total_sla_violations")
        return (
            -1e9 if quality is None else float(quality),
            -1e9 if iter_ is None else float(iter_),
            1e9 if total_viol is None else -float(total_viol),
        )

    return max(evaluators, key=ranking_key)


def _ensure_parent(path_prefix: str):
    parent = os.path.dirname(path_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sla_success(rows: List[Dict], algo: str, agent: str) -> Dict[str, float]:
    selected = [r for r in rows if r["algo"] == algo and r["agent"] == agent]
    if not selected:
        return {"eMBB": np.nan, "URLLC": np.nan, "mMTC": np.nan}
    embb_ok = np.mean([float(r["viol_embb"]) <= 0.0 for r in selected])
    urllc_ok = np.mean([float(r["viol_urllc"]) <= 0.0 for r in selected])
    mmtc_ok = np.mean([float(r["viol_mmtc"]) <= 0.0 for r in selected])
    return {"eMBB": float(embb_ok), "URLLC": float(urllc_ok), "mMTC": float(mmtc_ok)}


def make_figure(rows: List[Dict], output_png: str) -> Dict:
    colors = {"ippo": "#1f77b4", "mappo": "#d62728"}

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # BS_0 SLA bar chart only.
    sla_ippo = _sla_success(rows, "ippo", "BS_0")
    sla_mappo = _sla_success(rows, "mappo", "BS_0")
    slice_names = ["eMBB", "URLLC", "mMTC"]
    ippo_vals = [sla_ippo[name] for name in slice_names]
    mappo_vals = [sla_mappo[name] for name in slice_names]
    idx = np.arange(len(slice_names))
    width = 0.35
    ax.bar(idx - width / 2.0, ippo_vals, width=width, color=colors["ippo"], label="IPPO")
    ax.bar(idx + width / 2.0, mappo_vals, width=width, color=colors["mappo"], label="MAPPO")
    ax.set_xticks(idx, slice_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("BS_0 SLA Success Rate")
    ax.set_ylabel("Success Rate")
    ax.grid(True, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close(fig)

    coop_summary = {
        "bs0_sla_success": {"ippo": sla_ippo, "mappo": sla_mappo},
    }
    return coop_summary


def main():
    args = parse_args()
    _ensure_parent(args.output_prefix)

    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        cb.init_learned_evaluators()

        selected = {}
        evaluators = {}
        for algo in ("ippo", "mappo"):
            evaluator = _select_evaluator(algo)
            evaluators[algo] = evaluator
            selected[algo] = {
                "train_seed": int(evaluator["train_seed"]),
                "training_iteration": int(evaluator["training_iteration"]),
                "quality_score": (
                    None if evaluator.get("quality_score") is None else float(evaluator["quality_score"])
                ),
                "checkpoint_path": str(evaluator["checkpoint_path"]),
            }
            print(
                f"Using {algo.upper()} checkpoint: seed={evaluator['train_seed']}, "
                f"iter={evaluator['training_iteration']}, quality={evaluator.get('quality_score')}, "
                f"path={evaluator['checkpoint_path']}"
            )

        rows: List[Dict] = []
        for algo in ("ippo", "mappo"):
            for eval_seed in args.eval_seeds:
                print(f"Rollout {algo.upper()} eval_seed={eval_seed} ...")
                rows.extend(
                    apb.run_behavior_rollout(
                        algo_key=algo,
                        evaluator=evaluators[algo],
                        eval_seed=int(eval_seed),
                        rollout_steps=int(args.rollout_steps),
                        edge_agent=args.edge_agent,
                    )
                )

        csv_path = f"{args.output_prefix}_steps.csv"
        fig_path = f"{args.output_prefix}_figure.png"
        summary_path = f"{args.output_prefix}_summary.json"

        _write_csv(rows, csv_path)
        coop_summary = make_figure(
            rows=rows,
            output_png=fig_path,
        )

        output = {
            "eval_seeds": [int(s) for s in args.eval_seeds],
            "rollout_steps": int(args.rollout_steps),
            "edge_agent": args.edge_agent,
            "selected_checkpoints": selected,
            "coop_summary": coop_summary,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print("\n=== Coop Showcase Summary ===")
        print(json.dumps(coop_summary, indent=2))
        print(f"\nSaved step-level CSV: {csv_path}")
        print(f"Saved showcase figure: {fig_path}")
        print(f"Saved summary JSON: {summary_path}")
    finally:
        cb.stop_learned_evaluators()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

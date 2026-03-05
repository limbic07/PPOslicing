import json
import os
import pickle
from glob import glob
from pathlib import Path


def _extract_metrics(record):
    env_runners = record.get("env_runners", {}) if isinstance(record, dict) else {}
    custom_metrics = env_runners.get("custom_metrics", {}) if isinstance(env_runners, dict) else {}

    episode_return_mean = env_runners.get("episode_return_mean")
    if episode_return_mean is None:
        episode_return_mean = record.get("episode_reward_mean")

    center_urllc_viol = custom_metrics.get("center_urllc_violations")
    center_urllc_delay_ms = custom_metrics.get("center_urllc_delay_ms")

    metrics = {
        "episode_return_mean": float(episode_return_mean) if episode_return_mean is not None else None,
        "center_urllc_violations": float(center_urllc_viol) if center_urllc_viol is not None else None,
        "center_urllc_delay_ms": float(center_urllc_delay_ms) if center_urllc_delay_ms is not None else None,
    }
    return metrics


def _extract_training_metric(record):
    metrics = _extract_metrics(record)
    if metrics["episode_return_mean"] is not None:
        return metrics["episode_return_mean"]

    env_runners = record.get("env_runners", {})
    if isinstance(env_runners, dict):
        metric = env_runners.get("episode_return_mean")
        if metric is not None:
            return float(metric)

    metric = record.get("episode_reward_mean")
    if metric is not None:
        return float(metric)

    return None


def _load_metrics_by_iteration(trial_dir: Path):
    result_file = trial_dir / "result.json"
    if not result_file.exists() or result_file.stat().st_size == 0:
        return {}

    metrics_by_iteration = {}
    with result_file.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            training_iteration = record.get("training_iteration")
            if training_iteration is None:
                continue

            metrics = _extract_metrics(record)
            if any(metrics.values()):
                metrics_by_iteration[int(training_iteration)] = metrics

    return metrics_by_iteration


def _read_checkpoint_iteration(checkpoint_dir: Path):
    state_file = checkpoint_dir / "algorithm_state.pkl"
    if not state_file.exists():
        return None

    try:
        with state_file.open("rb") as file_obj:
            state = pickle.load(file_obj)
    except Exception:
        return None

    if isinstance(state, dict) and "training_iteration" in state:
        return int(state["training_iteration"])
    return None


def _normalize_experiment_dirs(experiment_dirs):
    if isinstance(experiment_dirs, (str, Path)):
        experiment_dirs = [str(experiment_dirs)]

    expanded = []
    for item in experiment_dirs:
        matches = glob(str(item))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(str(item))

    uniq = []
    seen = set()
    for path in expanded:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            uniq.append(norm)
    return uniq


def _ranking_key(item):
    # Lower URLLC violation first, then lower delay, then higher return.
    viol = item.get("center_urllc_violations")
    delay = item.get("center_urllc_delay_ms")
    ret = item.get("episode_return_mean")
    iter_ = item.get("training_iteration", -1)

    return (
        0 if viol is not None else 1,
        viol if viol is not None else float("inf"),
        0 if delay is not None else 1,
        delay if delay is not None else float("inf"),
        -(ret if ret is not None else float("-inf")),
        -iter_,
        -item.get("ctime", 0.0),
    )


def rank_checkpoints_by_metric(experiment_dirs):
    ranked = []
    for exp_dir_str in _normalize_experiment_dirs(experiment_dirs):
        exp_dir = Path(exp_dir_str)
        if not exp_dir.exists():
            continue

        for trial_dir in sorted(exp_dir.glob("PPO_*")):
            if not trial_dir.is_dir():
                continue
            metrics_by_iteration = _load_metrics_by_iteration(trial_dir)
            if not metrics_by_iteration:
                continue

            for checkpoint_dir in sorted(trial_dir.glob("checkpoint_*")):
                if not checkpoint_dir.is_dir():
                    continue

                training_iteration = _read_checkpoint_iteration(checkpoint_dir)
                if training_iteration is None:
                    continue

                metrics = metrics_by_iteration.get(training_iteration)
                if metrics is None:
                    continue

                ranked.append(
                    {
                        "checkpoint_path": str(checkpoint_dir),
                        "training_iteration": training_iteration,
                        "episode_return_mean": metrics.get("episode_return_mean"),
                        "center_urllc_violations": metrics.get("center_urllc_violations"),
                        "center_urllc_delay_ms": metrics.get("center_urllc_delay_ms"),
                        "experiment_dir": str(exp_dir),
                        "trial_dir": str(trial_dir),
                        "ctime": os.path.getctime(checkpoint_dir),
                    }
                )

    ranked.sort(key=_ranking_key)
    return ranked

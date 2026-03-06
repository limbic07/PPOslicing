import json
import math
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
    center_embb_viol = custom_metrics.get("center_embb_violations")
    center_mmtc_viol = custom_metrics.get("center_mmtc_violations")
    center_embb_sla_ok = custom_metrics.get("center_embb_sla_ok")
    center_urllc_sla_ok = custom_metrics.get("center_urllc_sla_ok")
    center_mmtc_sla_ok = custom_metrics.get("center_mmtc_sla_ok")
    center_reward_base_tp = custom_metrics.get("center_reward_base_tp")
    system_throughput_mbps = custom_metrics.get("system_throughput_mbps")

    metrics = {
        "episode_return_mean": float(episode_return_mean) if episode_return_mean is not None else None,
        "center_urllc_violations": float(center_urllc_viol) if center_urllc_viol is not None else None,
        "center_urllc_delay_ms": float(center_urllc_delay_ms) if center_urllc_delay_ms is not None else None,
        "center_embb_violations": float(center_embb_viol) if center_embb_viol is not None else None,
        "center_mmtc_violations": float(center_mmtc_viol) if center_mmtc_viol is not None else None,
        "center_embb_sla_ok": float(center_embb_sla_ok) if center_embb_sla_ok is not None else None,
        "center_urllc_sla_ok": float(center_urllc_sla_ok) if center_urllc_sla_ok is not None else None,
        "center_mmtc_sla_ok": float(center_mmtc_sla_ok) if center_mmtc_sla_ok is not None else None,
        "center_reward_base_tp": float(center_reward_base_tp) if center_reward_base_tp is not None else None,
        "system_throughput_mbps": float(system_throughput_mbps) if system_throughput_mbps is not None else None,
    }
    metrics["center_total_sla_violations"] = _sum_available_violations(
        metrics.get("center_embb_violations"),
        metrics.get("center_urllc_violations"),
        metrics.get("center_mmtc_violations"),
    )
    metrics["quality_score"] = _compute_quality_score(metrics)
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


def _normalize_base_tp(base_tp):
    if base_tp is None:
        return None
    return max(0.0, min(float(base_tp), 1.0))


def _sum_available_violations(*violations):
    available = [max(float(v), 0.0) for v in violations if v is not None]
    if not available:
        return None
    return float(sum(available))


def _select_throughput_metric(metrics):
    # Prefer explicit system throughput; fallback to center throughput proxy for old logs.
    system_tp = metrics.get("system_throughput_mbps")
    if system_tp is not None:
        return float(system_tp)

    base_tp = metrics.get("center_reward_base_tp")
    if base_tp is not None:
        return float(base_tp)
    return None


def _normalize_throughput_metric(metrics):
    system_tp = metrics.get("system_throughput_mbps")
    if system_tp is not None:
        return math.tanh(max(float(system_tp), 0.0) / 2000.0)
    return _normalize_base_tp(metrics.get("center_reward_base_tp"))


def _normalize_violation(violation, decay):
    if violation is None:
        return None
    violation = max(float(violation), 0.0)
    return math.exp(-decay * violation)


def _normalize_urllc_delay(delay_ms, budget_ms=2.0, decay=1.0):
    if delay_ms is None:
        return None
    delay_ms = max(float(delay_ms), 0.0)
    delay_excess_ms = max(delay_ms - budget_ms, 0.0)
    return math.exp(-decay * delay_excess_ms)


def _normalize_episode_return(episode_return_mean, scale=1000.0):
    if episode_return_mean is None:
        return None
    return 0.5 * (1.0 + math.tanh(float(episode_return_mean) / scale))


def _compute_quality_score(metrics):
    # Violation-first quality summary:
    # total SLA violation is dominant, throughput is secondary.
    total_violation = metrics.get("center_total_sla_violations")
    violation_quality = _normalize_violation(total_violation, decay=1.0)
    throughput_quality = _normalize_throughput_metric(metrics)
    return_quality = _normalize_episode_return(metrics.get("episode_return_mean"))

    components = [
        (0.80, violation_quality),
        (0.15, throughput_quality),
        (0.05, return_quality),
    ]
    available = [(weight, value) for weight, value in components if value is not None]
    if not available:
        return None

    total_weight = sum(weight for weight, _ in available)
    weighted_sum = sum(weight * value for weight, value in available)
    return weighted_sum / total_weight


def _ranking_key(item):
    # Violation-first selection: lowest total SLA violation wins.
    # Throughput is the secondary criterion (system throughput preferred).
    total_viol = item.get("center_total_sla_violations")
    system_tp = item.get("system_throughput_mbps")
    throughput_metric = _select_throughput_metric(item)
    quality = item.get("quality_score")
    urllc_viol = item.get("center_urllc_violations")
    embb_viol = item.get("center_embb_violations")
    mmtc_viol = item.get("center_mmtc_violations")
    delay = item.get("center_urllc_delay_ms")
    ret = item.get("episode_return_mean")
    iter_ = item.get("training_iteration", -1)

    return (
        0 if total_viol is not None else 1,
        total_viol if total_viol is not None else float("inf"),
        0 if system_tp is not None else 1,
        -(system_tp if system_tp is not None else float("-inf")),
        0 if throughput_metric is not None else 1,
        -(throughput_metric if throughput_metric is not None else float("-inf")),
        0 if quality is not None else 1,
        -(quality if quality is not None else float("-inf")),
        0 if urllc_viol is not None else 1,
        urllc_viol if urllc_viol is not None else float("inf"),
        0 if delay is not None else 1,
        delay if delay is not None else float("inf"),
        0 if embb_viol is not None else 1,
        embb_viol if embb_viol is not None else float("inf"),
        0 if mmtc_viol is not None else 1,
        mmtc_viol if mmtc_viol is not None else float("inf"),
        -(ret if ret is not None else float("-inf")),
        -iter_,
        -item.get("ctime", 0.0),
    )


def _collect_ranked_checkpoints(experiment_dirs, min_training_iteration):
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
                if training_iteration < min_training_iteration:
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
                        "center_embb_violations": metrics.get("center_embb_violations"),
                        "center_mmtc_violations": metrics.get("center_mmtc_violations"),
                        "center_total_sla_violations": metrics.get("center_total_sla_violations"),
                        "center_embb_sla_ok": metrics.get("center_embb_sla_ok"),
                        "center_urllc_sla_ok": metrics.get("center_urllc_sla_ok"),
                        "center_mmtc_sla_ok": metrics.get("center_mmtc_sla_ok"),
                        "center_reward_base_tp": metrics.get("center_reward_base_tp"),
                        "system_throughput_mbps": metrics.get("system_throughput_mbps"),
                        "quality_score": metrics.get("quality_score"),
                        "experiment_dir": str(exp_dir),
                        "trial_dir": str(trial_dir),
                        "ctime": os.path.getctime(checkpoint_dir),
                    }
                )

    ranked.sort(key=_ranking_key)
    return ranked


def rank_checkpoints_by_metric(experiment_dirs, min_training_iteration=0, fallback_to_any=False):
    ranked = _collect_ranked_checkpoints(experiment_dirs, min_training_iteration=min_training_iteration)
    if ranked or not fallback_to_any or min_training_iteration <= 0:
        return ranked
    return _collect_ranked_checkpoints(experiment_dirs, min_training_iteration=0)

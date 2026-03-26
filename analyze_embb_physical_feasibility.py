import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ray

import compare_mappo_variants as variants

DEFAULT_SUMMARY_JSON = './results/mappo_variant_comparison_balanced_summary.json'
DEFAULT_TARGET_KEY = 'mappo_current'
DEFAULT_REFERENCE_KEY = 'ippo'
DEFAULT_OUTPUT_PREFIX = './results/embb_physical_feasibility'

AGENT_ORDER = [f'BS_{idx}' for idx in range(7)]


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze whether eMBB SLA violations are physically unavoidable.')
    parser.add_argument('--summary-json', type=str, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument('--target-key', type=str, default=DEFAULT_TARGET_KEY)
    parser.add_argument('--reference-key', type=str, default=DEFAULT_REFERENCE_KEY)
    parser.add_argument('--output-prefix', type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument('--rollout-steps', type=int, default=variants.base.ROLLOUT_STEPS)
    return parser.parse_args()


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_selection_summary(path: str):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['selection_summary']


def get_method_by_key(algo_key: str):
    for method in variants.LEARNED_METHODS:
        if method['algo_key'] == algo_key:
            return method
    raise KeyError(algo_key)


def build_evaluator(algo_key: str, selected_item: dict):
    method = get_method_by_key(algo_key)
    env_config = variants.base.LEARNED_ENV_CONFIGS[algo_key]
    checkpoint_path = selected_item['selected_checkpoint']
    trial_dir = str(Path(checkpoint_path).parent.parent)
    observation_filter = selected_item.get('selected_observation_filter') or variants.base._resolve_trial_observation_filter(trial_dir)
    algo = variants.base.build_learned_eval_algo(observation_filter=observation_filter, env_config=env_config)
    algo.restore(checkpoint_path)
    return {
        'algo_key': algo_key,
        'label': method['label'],
        'algo': algo,
        'env_config': dict(env_config),
        'checkpoint_path': checkpoint_path,
        'training_iteration': int(selected_item.get('selected_iteration', -1)),
    }


def _simulate_eval_with_candidate(history_values, candidate_tp, window_tti):
    hist = deque(history_values, maxlen=window_tti)
    hist.append(float(candidate_tp))
    return float(np.mean(hist)) if hist else 0.0


def _max_embb_tp_mbps(env, agent_id, ratio_dict, mode='actual_ici_full_local_embb'):
    cf_ratios = {k: np.asarray(v, dtype=np.float32).copy() for k, v in ratio_dict.items()}
    if mode == 'actual_ici_full_local_embb':
        cf_ratios[agent_id] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        se_mod = env._calculate_interference_and_sinr(cf_ratios)[agent_id][0]
    elif mode == 'zero_ici_full_local_embb':
        se_mod = 1.0
    else:
        raise ValueError(mode)

    effective_se = float(env.current_se[agent_id][0] * se_mod)
    service_rate_mbps = float((env.total_bandwidth * effective_se) / 1e6)
    arrivals_mb = float(env.state[agent_id][0] * env.duration_tti)
    queue_mb = float(env.queues[agent_id][0])
    demand_mb = queue_mb + arrivals_mb
    service_cap_mb = service_rate_mbps * env.duration_tti
    served_mb = min(service_cap_mb, demand_mb)
    return float(served_mb / env.duration_tti)


def analyze_rollout(evaluator, rollout_steps):
    rows = []
    for seed in variants.base.EVAL_SEEDS:
        env = variants.base.MultiCell_5G_SLA_Env(config=evaluator['env_config'])
        obs, reset_infos = env.reset(seed=seed)
        runner, episode, shared_data = variants.base.build_ippo_episode_context(evaluator['algo'], obs, reset_infos)
        done = {'__all__': False}
        step = 0
        while not done['__all__'] and step < rollout_steps:
            pre_histories = {agent: list(env.embb_tp_history[agent]) for agent in env.agents}
            pre_queues = {agent: env.queues[agent].copy() for agent in env.agents}
            pre_state = {agent: env.state[agent].copy() for agent in env.agents}

            policy_actions, env_actions, extra_model_outputs = variants.base.compute_actions_batched(
                evaluator['algo'], runner, episode, shared_data
            )
            ratio_dict = {agent_id: env._action_to_ratios(action).astype(np.float32) for agent_id, action in env_actions.items()}

            obs, rewards, terminateds, truncateds, infos = env.step(env_actions)
            episode.add_env_step(
                obs,
                policy_actions,
                rewards,
                infos=infos,
                terminateds=terminateds,
                truncateds=truncateds,
                extra_model_outputs=extra_model_outputs,
            )

            for agent_id in env.agents:
                info = infos[agent_id]
                actual_eval = float(info.get('embb_eval_tp_mbps', 0.0))
                actual_flag = float(info.get('violation_flags', np.zeros(3, dtype=np.float32))[0])
                max_tp_actual_ici = _max_embb_tp_mbps(env, agent_id, ratio_dict, mode='actual_ici_full_local_embb')
                max_tp_zero_ici = _max_embb_tp_mbps(env, agent_id, ratio_dict, mode='zero_ici_full_local_embb')
                max_eval_actual_ici = _simulate_eval_with_candidate(pre_histories[agent_id], max_tp_actual_ici, env.embb_sla_window_tti)
                max_eval_zero_ici = _simulate_eval_with_candidate(pre_histories[agent_id], max_tp_zero_ici, env.embb_sla_window_tti)
                gbr = float(env.sla_props['embb_gbr'])
                rows.append({
                    'algo_key': evaluator['algo_key'],
                    'algo_label': evaluator['label'],
                    'seed': int(seed),
                    'step': int(step),
                    'agent': agent_id,
                    'is_center': int(agent_id == 'BS_0'),
                    'actual_embb_eval_tp_mbps': actual_eval,
                    'actual_embb_flag': actual_flag,
                    'pre_queue_embb': float(pre_queues[agent_id][0]),
                    'pre_arrival_embb': float(pre_state[agent_id][0]),
                    'pre_se_embb': float(pre_state[agent_id][6]),
                    'actual_ratio_embb': float(ratio_dict[agent_id][0]),
                    'max_tp_actual_ici_full_local_embb_mbps': max_tp_actual_ici,
                    'max_tp_zero_ici_full_embb_mbps': max_tp_zero_ici,
                    'max_eval_actual_ici_full_local_embb_mbps': max_eval_actual_ici,
                    'max_eval_zero_ici_full_embb_mbps': max_eval_zero_ici,
                    'physically_unavoidable_actual_ici': float(max_eval_actual_ici < gbr),
                    'physically_unavoidable_zero_ici': float(max_eval_zero_ici < gbr),
                    'recoverable_by_local_full_embb': float(max_eval_actual_ici >= gbr),
                    'recoverable_by_ideal_zero_ici': float(max_eval_zero_ici >= gbr),
                    'gbr_mbps': gbr,
                })
            done = terminateds
            step += 1
    return rows


def summarize(rows):
    def mean(sub, key):
        return float(np.mean([r[key] for r in sub])) if sub else float('nan')

    out = {}
    for algo in sorted({r['algo_key'] for r in rows}):
        sub = [r for r in rows if r['algo_key'] == algo]
        viol = [r for r in sub if r['actual_embb_flag'] > 0.5]
        center = [r for r in viol if r['is_center'] == 1]
        edge = [r for r in viol if r['is_center'] == 0]
        per_cell = {}
        for agent in AGENT_ORDER:
            a = [r for r in viol if r['agent'] == agent]
            if not a:
                continue
            per_cell[agent] = {
                'violation_count': len(a),
                'share_unavoidable_actual_ici': mean(a, 'physically_unavoidable_actual_ici'),
                'share_unavoidable_zero_ici': mean(a, 'physically_unavoidable_zero_ici'),
                'mean_actual_ratio_embb': mean(a, 'actual_ratio_embb'),
                'mean_max_eval_actual_ici_full_local_embb_mbps': mean(a, 'max_eval_actual_ici_full_local_embb_mbps'),
                'mean_max_eval_zero_ici_full_embb_mbps': mean(a, 'max_eval_zero_ici_full_embb_mbps'),
            }
        out[algo] = {
            'overall': {
                'total_rows': len(sub),
                'total_embb_violations': len(viol),
                'share_embb_violations': len(viol) / max(len(sub), 1),
                'share_unavoidable_actual_ici_among_violations': mean(viol, 'physically_unavoidable_actual_ici'),
                'share_unavoidable_zero_ici_among_violations': mean(viol, 'physically_unavoidable_zero_ici'),
                'share_recoverable_by_local_full_embb_among_violations': mean(viol, 'recoverable_by_local_full_embb'),
                'share_recoverable_by_ideal_zero_ici_among_violations': mean(viol, 'recoverable_by_ideal_zero_ici'),
                'center_share_unavoidable_actual_ici': mean(center, 'physically_unavoidable_actual_ici'),
                'edge_share_unavoidable_actual_ici': mean(edge, 'physically_unavoidable_actual_ici'),
                'center_share_unavoidable_zero_ici': mean(center, 'physically_unavoidable_zero_ici'),
                'edge_share_unavoidable_zero_ici': mean(edge, 'physically_unavoidable_zero_ici'),
                'mean_actual_ratio_embb_on_violations': mean(viol, 'actual_ratio_embb'),
                'mean_max_eval_actual_ici_full_local_embb_mbps': mean(viol, 'max_eval_actual_ici_full_local_embb_mbps'),
                'mean_max_eval_zero_ici_full_embb_mbps': mean(viol, 'max_eval_zero_ici_full_embb_mbps'),
            },
            'per_cell': per_cell,
        }
    return out


def save_csv(path, rows):
    ensure_parent(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, summaries, target_key, reference_key):
    ensure_parent(path)
    labels = [reference_key, target_key]
    vals_actual = [summaries[k]['overall']['share_unavoidable_actual_ici_among_violations'] for k in labels]
    vals_zero = [summaries[k]['overall']['share_unavoidable_zero_ici_among_violations'] for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, vals_actual, width, label='Actual ICI upper bound')
    ax.bar(x + width/2, vals_zero, width, label='Zero-ICI upper bound')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Share among eMBB violations')
    ax.set_title('Physically unavoidable share of eMBB violations')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    variants.configure_base_module()
    args = parse_args()
    selection_summary = load_selection_summary(args.summary_json)
    required = [args.reference_key, args.target_key]
    evaluators = {}
    for key in required:
        evaluators[key] = build_evaluator(key, selection_summary[key])

    all_rows = []
    try:
        for key in required:
            rows = analyze_rollout(evaluators[key], args.rollout_steps)
            all_rows.extend(rows)
    finally:
        for key in required:
            evaluators[key]['algo'].stop()
        ray.shutdown()

    summaries = summarize(all_rows)
    comparison = {
        'delta_unavoidable_actual_ici_share': summaries[args.target_key]['overall']['share_unavoidable_actual_ici_among_violations'] - summaries[args.reference_key]['overall']['share_unavoidable_actual_ici_among_violations'],
        'delta_unavoidable_zero_ici_share': summaries[args.target_key]['overall']['share_unavoidable_zero_ici_among_violations'] - summaries[args.reference_key]['overall']['share_unavoidable_zero_ici_among_violations'],
    }
    csv_path = f'{args.output_prefix}_steps.csv'
    json_path = f'{args.output_prefix}_summary.json'
    plot_path = f'{args.output_prefix}_unavoidable_share.png'
    save_csv(csv_path, all_rows)
    ensure_parent(json_path)
    Path(json_path).write_text(json.dumps({
        'target_key': args.target_key,
        'reference_key': args.reference_key,
        'summaries': summaries,
        'comparison': comparison,
        'artifacts': {
            'csv': csv_path,
            'plot': plot_path,
        },
    }, indent=2), encoding='utf-8')
    make_plot(plot_path, summaries, args.target_key, args.reference_key)
    print(f'Saved csv: {csv_path}')
    print(f'Saved summary json: {json_path}')
    print(f'Saved plot: {plot_path}')
    print(json.dumps({'comparison': comparison, 'target': summaries[args.target_key]['overall'], 'reference': summaries[args.reference_key]['overall']}, indent=2))


if __name__ == '__main__':
    main()

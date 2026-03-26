import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import compare_mappo_variants as variants
from multi_cell_env import MultiCell_5G_SLA_Env

DEFAULT_OUTPUT_PREFIX = './results/tti20_feasibility'


# This script is an oracle-style feasibility probe, not a formal proof solver.
# It uses a per-step joint optimizer under the current ICI rule and 20-TTI rolling
# eMBB SLA, then measures how often the system can keep all SLA constraints green.


def parse_args():
    parser = argparse.ArgumentParser(
        description='20-TTI feasibility probe under current ICI and SLA rules.'
    )
    parser.add_argument('--output-prefix', type=str, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument('--rollout-steps', type=int, default=variants.base.ROLLOUT_STEPS)
    parser.add_argument('--optimizer-maxiter', type=int, default=80)
    parser.add_argument('--seeds', type=int, nargs='*', default=variants.base.EVAL_SEEDS)
    parser.add_argument('--warmup-window', type=int, default=20)
    return parser.parse_args()


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _build_env_config():
    variants.configure_base_module()
    return dict(variants.base.LEARNED_ENV_CONFIGS['mappo_current'])


def _vector_to_ratio_dict(x):
    ratios = {}
    for idx, agent in enumerate([f'BS_{i}' for i in range(7)]):
        embb = float(x[2 * idx])
        urllc = float(x[2 * idx + 1])
        mmtc = max(0.0, 1.0 - embb - urllc)
        ratios[agent] = np.array([embb, urllc, mmtc], dtype=np.float32)
    return ratios


def _ratios_to_env_actions(ratio_dict, temperature):
    actions = {}
    eps = 1e-8
    for agent, ratios in ratio_dict.items():
        r = np.clip(np.asarray(ratios, dtype=np.float32), eps, 1.0)
        logits = np.log(r)
        action = logits / max(float(temperature), eps)
        actions[agent] = action.astype(np.float32)
    return actions


def _simulate_candidate(env, ratio_dict):
    se_modifiers = env._calculate_interference_and_sinr(ratio_dict)
    agent_metrics = {}
    total_score = 0.0
    total_penalty = 0.0

    for agent in env.agents:
        ratios = ratio_dict[agent]
        bw_allocated = ratios * env.total_bandwidth
        effective_se = env.current_se[agent] * se_modifiers[agent]
        service_rate_mbps = (bw_allocated * effective_se) / 1e6
        service_capacity_mb = service_rate_mbps * env.duration_tti

        arrivals_mb = env.state[agent][0:3] * env.duration_tti
        q_before = env.queues[agent] + arrivals_mb
        served_mb = np.minimum(service_capacity_mb, q_before)
        q_after = q_before - served_mb
        throughput_slices_mbps = served_mb / env.duration_tti

        embb_hist = deque(env.embb_tp_history[agent], maxlen=env.embb_sla_window_tti)
        embb_hist.append(float(throughput_slices_mbps[0]))
        embb_eval_tp = float(np.mean(embb_hist)) if embb_hist else 0.0

        safe_urllc_rate = max(float(service_rate_mbps[1]), 0.1)
        urllc_delay = float(q_after[1] / safe_urllc_rate)
        mmtc_queue = float(q_after[2])

        embb_gbr = float(env.sla_props['embb_gbr'])
        urllc_max_delay = float(env.sla_props['urllc_max_delay'])
        mmtc_max_queue = float(env.sla_props['mmtc_max_queue'])

        embb_deficit = max(0.0, embb_gbr - embb_eval_tp) / max(embb_gbr, 1e-6)
        urllc_violation = max(0.0, urllc_delay - urllc_max_delay) / max(urllc_max_delay, 1e-6)
        mmtc_violation = max(0.0, mmtc_queue - mmtc_max_queue) / max(mmtc_max_queue, 1e-6)

        current_embb_eval = float(np.mean(env.embb_tp_history[agent])) if env.embb_tp_history[agent] else 0.0
        current_deficit = max(0.0, embb_gbr - current_embb_eval) / max(embb_gbr, 1e-6)
        queue_weight = min(float(env.queues[agent][0]) / 2.0, 1.0)
        weight = 1.0 + 2.0 * current_deficit + 0.5 * queue_weight

        score = weight * min(embb_eval_tp / max(embb_gbr, 1e-6), 1.2)
        if embb_eval_tp >= embb_gbr:
            score += 1.5

        penalty = 200.0 * (urllc_violation ** 2) + 120.0 * (mmtc_violation ** 2)
        total_score += score
        total_penalty += penalty
        agent_metrics[agent] = {
            'q_after': q_after.astype(np.float32),
            'tp_slices': throughput_slices_mbps.astype(np.float32),
            'embb_eval_tp': float(embb_eval_tp),
            'embb_deficit': float(embb_deficit),
            'urllc_delay': float(urllc_delay),
            'mmtc_queue': float(mmtc_queue),
            'se_modifier': se_modifiers[agent].astype(np.float32),
        }

    return total_score - total_penalty, agent_metrics


def _initial_guesses(env):
    guesses = []
    # Static-like
    static = []
    for _ in env.agents:
        static.extend([0.34, 0.33])
    guesses.append(np.array(static, dtype=np.float64))

    # Priority-ish
    priority = []
    for _ in env.agents:
        priority.extend([0.78, 0.18])
    guesses.append(np.array(priority, dtype=np.float64))

    # More aggressive eMBB
    aggressive = []
    for _ in env.agents:
        aggressive.extend([0.86, 0.10])
    guesses.append(np.array(aggressive, dtype=np.float64))

    return guesses


def _choose_oracle_ratios(env, prev_x, maxiter):
    n_agents = len(env.agents)
    bounds = []
    constraints = []
    for idx in range(n_agents):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])
        constraints.append(
            {
                'type': 'ineq',
                'fun': lambda x, i=idx: 1.0 - x[2 * i] - x[2 * i + 1],
            }
        )

    starts = []
    if prev_x is not None:
        starts.append(np.asarray(prev_x, dtype=np.float64))
    starts.extend(_initial_guesses(env))

    best_x = starts[0]
    best_obj = -np.inf

    def objective(x):
        ratio_dict = _vector_to_ratio_dict(x)
        score, _ = _simulate_candidate(env, ratio_dict)
        return -float(score)

    for x0 in starts:
        res = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': int(maxiter), 'ftol': 1e-4, 'disp': False},
        )
        cand_x = np.asarray(res.x if res.success else x0, dtype=np.float64)
        cand_score = -float(objective(cand_x))
        if cand_score > best_obj:
            best_obj = cand_score
            best_x = cand_x

    return _vector_to_ratio_dict(best_x), best_x


def rollout_oracle(env_config, seeds, rollout_steps, maxiter):
    rows = []
    for seed in seeds:
        env = MultiCell_5G_SLA_Env(config=env_config)
        obs, infos = env.reset(seed=seed)
        prev_x = None
        for step in range(rollout_steps):
            ratio_dict, prev_x = _choose_oracle_ratios(env, prev_x, maxiter)
            env_actions = _ratios_to_env_actions(ratio_dict, env.action_softmax_temperature)
            obs, rewards, terminateds, truncateds, infos = env.step(env_actions)

            system_all_green = 1
            for agent in env.agents:
                info = infos[agent]
                flags = np.asarray(info.get('violation_flags', np.zeros(3, dtype=np.float32)), dtype=np.float32)
                all_green = float(np.all(flags <= 0.0))
                system_all_green = int(system_all_green and all_green > 0.5)
                rows.append(
                    {
                        'seed': int(seed),
                        'step': int(step),
                        'agent': agent,
                        'is_center': int(agent == 'BS_0'),
                        'ratio_embb': float(ratio_dict[agent][0]),
                        'ratio_urllc': float(ratio_dict[agent][1]),
                        'ratio_mmtc': float(ratio_dict[agent][2]),
                        'embb_eval_tp_mbps': float(info.get('embb_eval_tp_mbps', 0.0)),
                        'urllc_delay_ms': float(info.get('est_urllc_delay', 0.0) * 1000.0),
                        'mmtc_queue': float(info.get('queue_sizes', np.zeros(3, dtype=np.float32))[2]),
                        'embb_flag': float(flags[0]),
                        'urllc_flag': float(flags[1]),
                        'mmtc_flag': float(flags[2]),
                        'reward': float(rewards.get(agent, 0.0)),
                        'throughput_mbps': float(info.get('throughput', 0.0)),
                    }
                )
            if terminateds.get('__all__', False):
                break
    return rows


def summarize(rows, warmup_window):
    summary = {}
    total_rows = len(rows)
    system_tp = float(np.mean([r['throughput_mbps'] for r in rows])) if rows else float('nan')
    summary['overall'] = {
        'total_rows': total_rows,
        'system_sla_success': {
            'embb': 1.0 - float(np.mean([r['embb_flag'] for r in rows])) if rows else float('nan'),
            'urllc': 1.0 - float(np.mean([r['urllc_flag'] for r in rows])) if rows else float('nan'),
            'mmtc': 1.0 - float(np.mean([r['mmtc_flag'] for r in rows])) if rows else float('nan'),
        },
        'mean_system_tp_agent_mbps': system_tp,
        'mean_ratio_embb': float(np.mean([r['ratio_embb'] for r in rows])) if rows else float('nan'),
        'mean_ratio_urllc': float(np.mean([r['ratio_urllc'] for r in rows])) if rows else float('nan'),
    }

    per_cell = {}
    for agent in [f'BS_{i}' for i in range(7)]:
        sub = [r for r in rows if r['agent'] == agent]
        per_cell[agent] = {
            'embb_sla_success': 1.0 - float(np.mean([r['embb_flag'] for r in sub])) if sub else float('nan'),
            'urllc_sla_success': 1.0 - float(np.mean([r['urllc_flag'] for r in sub])) if sub else float('nan'),
            'mmtc_sla_success': 1.0 - float(np.mean([r['mmtc_flag'] for r in sub])) if sub else float('nan'),
            'mean_ratio_embb': float(np.mean([r['ratio_embb'] for r in sub])) if sub else float('nan'),
            'mean_embb_eval_tp_mbps': float(np.mean([r['embb_eval_tp_mbps'] for r in sub])) if sub else float('nan'),
        }
    summary['per_cell'] = per_cell

    # System full-green at each step.
    step_groups = {}
    for r in rows:
        step_groups.setdefault((r['seed'], r['step']), []).append(r)
    step_records = []
    for (seed, step), group in sorted(step_groups.items()):
        system_full_green = int(all((g['embb_flag'] <= 0.0 and g['urllc_flag'] <= 0.0 and g['mmtc_flag'] <= 0.0) for g in group))
        step_records.append({'seed': seed, 'step': step, 'system_full_green': system_full_green})
    summary['system_step_full_green_share'] = float(np.mean([r['system_full_green'] for r in step_records])) if step_records else float('nan')

    # 20-TTI window-end feasibility: end step is green for every agent and warmup satisfied.
    valid_window_records = [r for r in step_records if r['step'] >= max(warmup_window - 1, 0)]
    summary['window_end_full_green_share'] = float(np.mean([r['system_full_green'] for r in valid_window_records])) if valid_window_records else float('nan')
    summary['hard_failed_window_end_count'] = int(sum(1 for r in valid_window_records if r['system_full_green'] == 0))
    summary['valid_window_end_count'] = int(len(valid_window_records))
    return summary, step_records


def save_csv(path, rows):
    ensure_parent(path)
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_window_success(path, step_records):
    ensure_parent(path)
    if not step_records:
        return
    plt.figure(figsize=(10, 4.8))
    for seed in sorted({r['seed'] for r in step_records}):
        sub = [r for r in step_records if r['seed'] == seed]
        xs = [r['step'] for r in sub]
        ys = [r['system_full_green'] for r in sub]
        plt.plot(xs, ys, label=f'seed={seed}', alpha=0.8)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Step')
    plt.ylabel('System Full-Green')
    plt.title('20-TTI Feasibility Probe: System Full-SLA Success over Time')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_per_cell_embb(path, summary):
    ensure_parent(path)
    items = summary['per_cell']
    labels = list(items.keys())
    vals = [items[k]['embb_sla_success'] for k in labels]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, vals, color='#8c564b')
    plt.ylim(0.0, 1.0)
    plt.ylabel('eMBB SLA Success')
    plt.title('20-TTI Feasibility Probe: Per-Cell eMBB SLA Success')
    plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    env_config = _build_env_config()
    rows = rollout_oracle(env_config, args.seeds, args.rollout_steps, args.optimizer_maxiter)
    summary, step_records = summarize(rows, args.warmup_window)

    csv_path = f'{args.output_prefix}_steps.csv'
    json_path = f'{args.output_prefix}_summary.json'
    plot1 = f'{args.output_prefix}_window_success.png'
    plot2 = f'{args.output_prefix}_per_cell_embb.png'

    save_csv(csv_path, rows)
    ensure_parent(json_path)
    Path(json_path).write_text(
        json.dumps(
            {
                'env_profile': variants.ENV_PROFILE,
                'seeds': args.seeds,
                'rollout_steps': args.rollout_steps,
                'optimizer_maxiter': args.optimizer_maxiter,
                'summary': summary,
                'artifacts': {
                    'csv': csv_path,
                    'plots': [plot1, plot2],
                },
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    plot_window_success(plot1, step_records)
    plot_per_cell_embb(plot2, summary)

    print(f'Saved csv: {csv_path}')
    print(f'Saved summary json: {json_path}')
    print(f'Saved plots: {plot1}, {plot2}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

from collections import Counter
import math

from torch import nn

from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.phase0_4b_evaluation import factorized_metrics
from src.phase0.phase0_4b_protocol import SFTSample
from src.phase0.phase0_4c_evaluation import aggregate_metrics


def behavior_statistics(rows: list[dict]) -> dict:
    count = len(rows)
    actions = [r['parsed_action'] for r in rows]
    points = [r['predicted_waypoints'] for r in rows if r['prediction_valid']]
    consistency = [sum(r['parsed_action'][axis] == r['trajectory_projection'][f'{axis}_action']
                       for axis in ('longitudinal', 'lateral')) / 2
                   for r in rows if r['parsed_action'] is not None and r['trajectory_projection'] is not None]
    return {
        'sample_count': count,
        'stop_fraction': sum(a is not None and a['longitudinal'] == 'stop' for a in actions) / count,
        'keep_fraction': sum(a is not None and a['longitudinal'] == 'keep' for a in actions) / count,
        'straight_fraction': sum(a is not None and a['lateral'] == 'straight' for a in actions) / count,
        'mean_endpoint_displacement_m': sum(math.hypot(*p[-1]) for p in points) / len(points) if points else None,
        'displacement_sample_count': len(points),
        'consistency_mean': sum(consistency) / len(consistency) if consistency else None,
        'consistency_sample_count': len(consistency),
        'action_parsing_success_rate': sum(a is not None for a in actions) / count,
        'trajectory_parsing_success_rate': len(points) / count,
        'invalid_rate': 1 - len(points) / count,
        'generation_status_counts': dict(Counter(r['invalid_reason'] or 'valid' for r in rows)),
        'generated_tokens_by_turn': [sum(len(s['token_ids']) for r in rows for s in r['assistant_spans']
                                         if s['turn'] == turn) for turn in (1, 2)],
    }


def evaluate_population(model: nn.Module, runner: Stage2Runner, samples: list[SFTSample],
                        records: dict) -> tuple[list[dict], dict]:
    if not samples or any(s.split != 'validation' for s in samples):
        raise ValueError('GRPO monitoring requires validation observations')
    rows, metrics = evaluate(model, runner, samples, records, progress_interval=16)
    metrics['behavior'] = behavior_statistics(rows)
    return rows, metrics


def compare_predictions(baseline: list[dict], current: list[dict]) -> dict:
    first = {r['sample_token']: r for r in baseline}
    second = {r['sample_token']: r for r in current}
    if len(first) != len(baseline) or len(second) != len(current) or first.keys() != second.keys():
        raise ValueError('comparison requires identical unique validation population')
    for token, row in first.items():
        if any(row[key] != second[token][key] for key in
               ('scene_token', 'split', 'target', 'target_waypoints', 'trajectory_valid_mask')):
            raise ValueError('comparison target or split mismatch')
    common = [token for token in first if first[token]['prediction_valid'] and second[token]['prediction_valid']]
    paired = {}
    for key in ('ade_3s_m', 'fde_3s_m'):
        tokens = [t for t in common if first[t][key] is not None and second[t][key] is not None]
        before = sum(first[t][key] for t in tokens) / len(tokens) if tokens else None
        after = sum(second[t][key] for t in tokens) / len(tokens) if tokens else None
        paired[key] = {'sample_count': len(tokens), 'baseline': before, 'current': after,
                       'delta': after - before if tokens else None}
    before, after = behavior_statistics(baseline), behavior_statistics(current)
    deltas = {key: after[key] - before[key] if before[key] is not None and after[key] is not None else None
              for key in ('stop_fraction', 'keep_fraction', 'straight_fraction',
                          'mean_endpoint_displacement_m', 'consistency_mean', 'invalid_rate')}
    consistency_up = deltas['consistency_mean'] is not None and deltas['consistency_mean'] > 0
    return {'population_count': len(first), 'common_valid_count': len(common),
            'baseline': {'action': factorized_metrics(baseline),
                         'trajectory': aggregate_metrics(baseline, 'predicted_action'), 'behavior': before},
            'current': {'action': factorized_metrics(current),
                        'trajectory': aggregate_metrics(current, 'predicted_action'), 'behavior': after},
            'paired_errors': paired, 'behavior_delta': deltas,
            'diagnostic_signals': {
                'stop_fraction_increased': deltas['stop_fraction'] > 0,
                'keep_fraction_increased': deltas['keep_fraction'] > 0,
                'straight_fraction_increased': deltas['straight_fraction'] > 0,
                'endpoint_displacement_decreased': deltas['mean_endpoint_displacement_m'] is not None
                                                   and deltas['mean_endpoint_displacement_m'] < 0,
                'invalid_rate_increased': deltas['invalid_rate'] > 0,
                'consistency_up_and_error_worse': consistency_up and any(
                    p['delta'] is not None and p['delta'] > 0 for p in paired.values()),
            },
            'interpretation': 'descriptive signals only; inspect magnitudes and sample-level cases; '
                              'not significance or automatic reward-hacking verdict',
            'continuation': 'user_review_required; no automatic full training'}

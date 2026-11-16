from __future__ import annotations

from collections import Counter
import math

from src.phase0.drivema_grpo_evaluation import behavior_statistics
from src.phase0.phase0_4b_evaluation import factorized_metrics
from src.phase0.phase0_4c_evaluation import aggregate_metrics


CONTRACT_FIELDS = ('scene_token', 'split', 'target', 'target_waypoints', 'trajectory_valid_mask')


def align_predictions(reference: list[dict], other: list[dict]) -> list[dict]:
    first = {row['sample_token']: row for row in reference}
    second = {row['sample_token']: row for row in other}
    if len(first) != len(reference) or len(second) != len(other) or first.keys() != second.keys():
        raise ValueError('comparison requires identical unique population; missing predictions are an incomplete run')
    for token, row in first.items():
        if row['split'] != 'validation' or any(row[k] != second[token][k] for k in CONTRACT_FIELDS):
            raise ValueError(f'comparison contract mismatch: {token}')
    return [second[row['sample_token']] for row in reference]


def performance(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return {'action': factorized_metrics(rows), 'trajectory': aggregate_metrics(rows, 'predicted_action'),
            'behavior': behavior_statistics(rows)}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def paired_sensitivity(clean: list[dict], perturbed: list[dict]) -> dict:
    other = align_predictions(clean, perturbed)
    common = [(a, b) for a, b in zip(clean, other) if a['prediction_valid'] and b['prediction_valid']]
    errors = {}
    for key in ('ade_3s_m', 'fde_3s_m'):
        pairs = [(a[key], b[key]) for a, b in common if a[key] is not None and b[key] is not None]
        errors[key] = {'sample_count': len(pairs), 'clean': _mean([a for a, _ in pairs]),
                       'perturbed': _mean([b for _, b in pairs]), 'delta': _mean([b - a for a, b in pairs])}
    distances = [sum(math.dist(p, q) for p, q in zip(a['predicted_waypoints'], b['predicted_waypoints'],
                                                    strict=True)) / 6 for a, b in common]
    actions = [(a['parsed_action'], b['parsed_action']) for a, b in zip(clean, other)
               if a['parsed_action'] is not None and b['parsed_action'] is not None]
    changes = {axis: {'sample_count': len(actions), 'changed_count': sum(a[axis] != b[axis] for a, b in actions),
                      'rate': _mean([float(a[axis] != b[axis]) for a, b in actions])}
               for axis in ('longitudinal', 'lateral')}
    changes['joint'] = {'sample_count': len(actions), 'changed_count': sum(a != b for a, b in actions),
                        'rate': _mean([float(a != b) for a, b in actions])}
    transitions = {}
    for label, valid in (('trajectory', lambda r: r['prediction_valid']),
                         ('action', lambda r: r['parsed_action'] is not None)):
        counts = Counter(f'{"valid" if valid(a) else "invalid"}_to_{"valid" if valid(b) else "invalid"}'
                         for a, b in zip(clean, other))
        transitions[label] = {f'{a}_to_{b}': counts[f'{a}_to_{b}']
                              for a in ('valid', 'invalid') for b in ('valid', 'invalid')}
    return {'population_count': len(clean), 'common_valid_count': len(common), 'paired_errors': errors,
            'trajectory_change': {'sample_count': len(distances), 'mean_distance_m': _mean(distances)},
            'action_change': changes, 'validity_transitions': transitions,
            'clean': performance(clean), 'perturbed': performance(other)}


def condition_report(clean: list[dict], perturbed: list[dict]) -> dict:
    other = align_predictions(clean, perturbed)
    active = [i for i, row in enumerate(other) if row['visual_intervention']['applicable']]
    groups = {}
    for index, row in enumerate(clean):
        target = row['target']
        label = f'{target["longitudinal"]}/{target["lateral"]}'
        groups.setdefault(label, []).append(index)
    return {
        'overall': paired_sensitivity(clean, other),
        'intervened_only': paired_sensitivity([clean[i] for i in active], [other[i] for i in active]),
        'not_applicable_count': len(clean) - len(active),
        'gt_action_groups_intervened_only': {
            label: paired_sensitivity([clean[i] for i in indices if i in active],
                                      [other[i] for i in indices if i in active])
            for label, indices in groups.items()},
    }


def difference_in_differences(sft_clean: list[dict], sft_changed: list[dict],
                              grpo_clean: list[dict], grpo_changed: list[dict]) -> dict:
    tracks = [sft_clean] + [align_predictions(sft_clean, rows)
                            for rows in (sft_changed, grpo_clean, grpo_changed)]
    result = {}
    for scope in ('overall', 'intervened_only'):
        errors = {}
        for key in ('ade_3s_m', 'fde_3s_m'):
            rows = [group for group in zip(*tracks)
                    if all(r['prediction_valid'] and r[key] is not None for r in group)
                    and (scope == 'overall' or all(group[i]['visual_intervention']['applicable'] for i in (1, 3)))]
            sft = [b[key] - a[key] for a, b, c, d in rows]
            grpo = [d[key] - c[key] for a, b, c, d in rows]
            errors[key] = {'sample_count': len(rows), 'sft_delta': _mean(sft), 'grpo_delta': _mean(grpo),
                           'difference_in_differences': _mean([b - a for a, b in zip(sft, grpo)])}
        result[scope] = errors
    return {'paired_errors': result, 'denominator': 'same four-way valid samples with valid GT horizon',
            'interpretation': 'descriptive test-time sensitivity; not a modality causal effect or visual ability ranking'}

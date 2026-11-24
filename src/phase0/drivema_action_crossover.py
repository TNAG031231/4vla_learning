from __future__ import annotations

from collections import Counter
from statistics import mean

from src.phase0.drivema_action_oracle_metrics import TRAJECTORY_ERRORS
from src.phase0.drivema_grpo_diagnostics import paired_statistics
from src.phase0.drivema_visual_metrics import align_predictions
from src.phase0.phase0_4b_protocol import ActionTarget, parse_action


def _context(row: dict) -> tuple[list, list, list]:
    action, trajectory = row['assistant_spans']
    ids = row['input_ids']
    if not (row['conditioning_type'] == 'predicted_action' and row['prediction_valid']
            and row['fde_3s_m'] is not None and parse_action(row['action_context']) == row['parsed_action']
            and row['parsed_action'] is not None and (action['turn'], trajectory['turn']) == (1, 2)
            and 0 < action['start'] < action['end'] < trajectory['start'] < trajectory['end'] == len(ids)
            and ids[action['start']:action['end']] == action['token_ids']):
        raise ValueError(f"invalid saved action or trajectory for {row['sample_token']}")
    return ids[:action['start']], action['token_ids'], ids[action['end']:trajectory['start']]


def crossover_plan(baseline: list[dict], feedback: list[dict], oracle: list[dict]) -> list[dict]:
    feedback = align_predictions(baseline, feedback)
    cached = {r['sample_token']: r for r in oracle}
    wrong = {r['sample_token'] for r in baseline
             if r['parsed_action'] != {axis: r['target'][axis] for axis in ('longitudinal', 'lateral')}}
    if len(cached) != len(oracle) or cached.keys() != wrong:
        raise ValueError('oracle cache must cover the unique baseline joint-action errors')
    rows = []
    for old, new in zip(baseline, feedback, strict=True):
        token = old['sample_token']
        prefix, old_ids, suffix = _context(old)
        new_prefix, new_ids, new_suffix = _context(new)
        if prefix != new_prefix or suffix != new_suffix:
            raise ValueError(f'observation or trajectory prompt differs for {token}')
        source, points = 'replay', None
        if old_ids == new_ids:
            source, points = 'baseline', old['predicted_waypoints']
        elif token in cached and new['action_context'] == ActionTarget(**old['target']).training_text():
            row = cached[token]
            if not (row['split'] == 'validation' and row['conditioning_type'] == 'gt_action_diagnostic'
                    and row['predicted_action'] == old['parsed_action'] and row['oracle_action'] == new['parsed_action']
                    and row['baseline_waypoints'] == old['predicted_waypoints'] and row['invalid_reason'] is None
                    and all(row[k] == old[k] for k in ('scene_token', 'target_waypoints', 'trajectory_valid_mask'))):
                raise ValueError(f'oracle comparison differs from archived predictions for {token}')
            source, points = 'oracle', row['oracle_waypoints']
        arms = {'baseline': {k: old[k] for k in TRAJECTORY_ERRORS},
                'feedback': {k: new[k] for k in TRAJECTORY_ERRORS}, 'mixed': None}
        if source == 'baseline':
            arms['mixed'] = arms['baseline']
        elif source == 'oracle':
            arms['mixed'] = {k: cached[token][f'oracle_{k}'] for k in TRAJECTORY_ERRORS}
        rows.append({'sample_token': token, 'scene_token': old['scene_token'], 'split': 'validation',
                     'source': source, 'baseline_action': old['parsed_action'], 'feedback_action': new['parsed_action'],
                     'mixed_waypoints': points, **arms})
    return rows


def summarize_plan(rows: list[dict]) -> dict:
    known = [r for r in rows if r['mixed'] is not None]
    return {
        'population': len(rows), 'scenes': len({r['scene_token'] for r in rows}),
        'sources': dict(Counter(r['source'] for r in rows)),
        'replay_scenes': len({r['scene_token'] for r in rows if r['source'] == 'replay'}),
        'observed_total_delta_m': {k: mean(r['feedback'][k] - r['baseline'][k] for r in rows)
                                   for k in TRAJECTORY_ERRORS},
        'known_contribution_m': {
            k: {'action': sum(r['mixed'][k] - r['baseline'][k] for r in known) / len(rows),
                'decoder': sum(r['feedback'][k] - r['mixed'][k] for r in known) / len(rows)}
            for k in TRAJECTORY_ERRORS},
        'known_count': len(known), 'contribution_denominator': len(rows),
        'oracle_cache_condition': 'exact replacement action token IDs verified with source tokenizer at evaluation',
    }


def decomposition(rows: list[dict]) -> dict:
    if any(row['mixed'] is None for row in rows):
        raise ValueError('crossover results incomplete')
    valid = [all(row['mixed'][k] is not None for k in TRAJECTORY_ERRORS) for row in rows]
    estimates = {}
    for name, first, second in [('action', 'baseline', 'mixed'), ('decoder', 'mixed', 'feedback'),
                                ('total', 'baseline', 'feedback')]:
        pairs = []
        for row, common_valid in zip(rows, valid, strict=True):
            common = {'prediction_valid': common_valid, 'scene_token': row['scene_token'], 'weighted_error_m': None}
            pairs.append(({**common, **row[first]}, {**common, **row[second]}))
        result = paired_statistics(pairs, repetitions=10_000, seed=20261116, reference_label=first, current_label=second)
        estimates[name] = {k: result[k] for k in TRAJECTORY_ERRORS}
    return {'population': len(rows), 'scenes': len({r['scene_token'] for r in rows}),
            'sources': dict(Counter(r['source'] for r in rows)), 'paired': estimates,
            'common_valid_count': sum(valid), 'mixed_invalid_count': len(valid) - sum(valid),
            'full_population_decomposition': all(valid),
            'bootstrap': {'unit': 'scene', 'repetitions': 10_000, 'seed': 20261116},
            'mixed': 'feedback predicted action with fixed additive decoder; deltas are second minus first'}

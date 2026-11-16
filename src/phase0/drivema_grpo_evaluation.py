from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

from torch import nn

from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.phase0_4b_evaluation import factorized_metrics
from src.phase0.phase0_4b_protocol import SFTSample
from src.phase0.phase0_4c_evaluation import aggregate_metrics


def load_baseline_predictions(path: Path, samples: list[SFTSample], records: dict) -> list[dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    by_token = {row['sample_token']: row for row in rows}
    if len(by_token) != len(rows):
        raise ValueError('duplicate baseline sample_token')
    if set(by_token) != {sample.sample_token for sample in samples}:
        raise ValueError('baseline validation population differs: missing or unexpected samples')
    for sample in samples:
        row = by_token[sample.sample_token]
        record = records[sample.sample_token]
        expected = {'scene_token': sample.scene_token, 'split': 'validation',
                    'target': asdict(sample.target), 'target_waypoints': record['future_waypoints'],
                    'trajectory_valid_mask': record['trajectory_valid_mask']}
        if sample.split != 'validation' or record['split'] != 'validation':
            raise ValueError('baseline intake requires validation samples')
        for key, value in expected.items():
            if row[key] != value:
                raise ValueError(f'baseline {sample.sample_token}: {key} mismatch')
    return [by_token[sample.sample_token] for sample in samples]


def baseline_provenance(path: Path, checkpoint: str, model_id: str, revision: str,
                        inference: dict) -> dict:
    metadata_path = path.parent / 'metadata.json'
    summary_path = path.parent / 'training_summary.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    checked, unavailable = {}, []
    sources = [(metadata.get('provenance', {}), {'model_id': model_id, 'revision': revision}),
               (metadata.get('config', {}), {'trajectory_max_new_tokens': inference['trajectory_max_new_tokens']}),
               (metadata.get('inference', {}), {'batch_size': inference['batch_size'],
                                               'optimized': inference['optimized'],
                                               'action_generation': inference['action_generation']})]
    for source, expected in sources:
        for key, value in expected.items():
            if key not in source:
                unavailable.append(key)
            elif source[key] != value:
                raise ValueError(f'baseline provenance {key} mismatch')
            else:
                checked[key] = source[key]
    if 'best' in summary:
        selected = summary['best']['checkpoint']
        expected_parts = Path(checkpoint).parts
        if Path(selected).parts[-len(expected_parts):] != expected_parts:
            raise ValueError('baseline provenance SFT checkpoint mismatch')
        checked['checkpoint'] = selected
    else:
        unavailable.append('checkpoint')
    return {'predictions': str(path.resolve()), 'checked': checked, 'unavailable': unavailable,
            'metadata': str(metadata_path) if metadata_path.exists() else None,
            'training_summary': str(summary_path) if summary_path.exists() else None}


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

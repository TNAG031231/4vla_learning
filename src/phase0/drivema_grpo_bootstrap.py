from __future__ import annotations

import numpy as np

from src.phase0.drivema_grpo_evaluation import behavior_statistics
from src.phase0.phase0_4b_evaluation import factorized_metrics


def action_behavior_bootstrap(reference: list[dict], current: list[dict], repetitions: int, seed: int) -> dict:
    scenes = sorted({r['scene_token'] for r in reference})
    if len(scenes) < 2:
        return {'scene_count': len(scenes), 'intervals': None}
    draws = np.random.default_rng(seed).integers(len(scenes), size=(repetitions, len(scenes)))
    estimates = []
    for rows in (reference, current):
        groups = [[r for r in rows if r['scene_token'] == scene] for scene in scenes]
        actions = [factorized_metrics(group) for group in groups]
        behavior = [behavior_statistics(group) for group in groups]
        values = {}
        for axis in ('longitudinal', 'lateral'):
            counts = np.array([[a[axis]['effective_sample_count']] for a in actions])
            support = np.array([[a[axis]['target_distribution'][c] for c in a[axis]['classes']] for a in actions])
            predicted = np.array([[a[axis]['prediction_distribution'][c] for c in a[axis]['classes']] for a in actions])
            tp = np.array([np.diag(a[axis]['confusion_matrix']) for a in actions])
            denominator = (support + predicted)[draws].sum(axis=1)
            f1 = np.divide(2 * tp[draws].sum(axis=1), denominator,
                           out=np.zeros_like(denominator, dtype=float), where=denominator != 0).mean(axis=1)
            f1[counts[draws].sum(axis=(1, 2)) == 0] = np.nan
            values[f'{axis}_macro_f1'] = f1
        joint_counts = np.array([a['joint']['effective_sample_count'] for a in actions])
        joint_correct = np.array([(a['joint_accuracy'] or 0.) * n for a, n in zip(actions, joint_counts)])
        consistency_counts = np.array([b['consistency_sample_count'] for b in behavior])
        consistency_sum = np.array([(b['consistency_mean'] or 0.) * n for b, n in zip(behavior, consistency_counts)])
        totals = np.array([len(group) for group in groups])
        invalid = np.array([b['invalid_rate'] * n for b, n in zip(behavior, totals)])
        for name, numerator, denominator in (
            ('joint_accuracy', joint_correct, joint_counts),
            ('consistency_mean', consistency_sum, consistency_counts), ('invalid_rate', invalid, totals)):
            den = denominator[draws].sum(axis=1)
            values[name] = np.divide(numerator[draws].sum(axis=1), den,
                                     out=np.full(repetitions, np.nan), where=den != 0)
        estimates.append(values)
    intervals = {}
    for name, before in estimates[0].items():
        delta = estimates[1][name] - before
        valid = delta[np.isfinite(delta)]
        intervals[name] = {'delta_95ci': np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                           'defined_replicates': len(valid), 'undefined_replicates': repetitions - len(valid)}
    return {'scene_count': len(scenes), 'intervals': intervals, 'repetitions': repetitions, 'seed': seed,
            'method': 'paired scene-cluster percentile bootstrap; same scene draws for both arms; current minus reference',
            'macro_f1': 'recompute from pooled class counts; frozen classes, invalid predictions remain false negatives',
            'consistency': 'existing evaluator denominator: rows with parsed action and projected trajectory'}

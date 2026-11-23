from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from src.phase0.drivema_grpo_evaluation import compare_predictions
from src.phase0.drivema_grpo_bootstrap import action_behavior_bootstrap
from src.phase0.drivema_stage2_protocol import trajectory_to_meta_action
from src.phase0.drivema_stage3_rewards import RewardConfig, assign_advantages
from src.phase0.drivema_visual_metrics import align_predictions
from src.phase0.phase0_4c_evaluation import prediction_metrics


ERRORS = ('ade_3s_m', 'fde_3s_m', 'weighted_error_m')


def mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def paired_statistics(pairs: list[tuple[dict, dict]], *, repetitions: int, seed: int,
                      reference_label: str = 'sft', current_label: str = 'grpo') -> dict:
    result = {}
    for key in ERRORS:
        eligible = [(a, b) for a, b in pairs if a['prediction_valid'] and b['prediction_valid']
                    and a[key] is not None and b[key] is not None]
        delta = np.array([b[key] - a[key] for a, b in eligible])
        scenes = sorted({a['scene_token'] for a, _ in eligible})
        intervals = None
        if len(scenes) >= 2:
            counts = np.array([sum(a['scene_token'] == s for a, _ in eligible) for s in scenes])
            sums = np.array([sum(b[key] - a[key] for a, b in eligible if a['scene_token'] == s)
                             for s in scenes])
            wins = np.array([sum(b[key] < a[key] for a, b in eligible if a['scene_token'] == s)
                             for s in scenes])
            draws = np.random.default_rng(seed).integers(len(scenes), size=(repetitions, len(scenes)))
            denominator = counts[draws].sum(axis=1)
            intervals = {
                'mean_delta_m': np.quantile(sums[draws].sum(axis=1) / denominator, [.025, .975]).tolist(),
                'improved_fraction': np.quantile(wins[draws].sum(axis=1) / denominator, [.025, .975]).tolist(),
            }
        result[key] = {
            'sample_count': len(eligible), 'excluded_count': len(pairs) - len(eligible),
            'scene_count': len(scenes), f'{reference_label}_mean_m': mean([a[key] for a, _ in eligible]),
            f'{current_label}_mean_m': mean([b[key] for _, b in eligible]), 'mean_delta_m': mean(delta.tolist()),
            'median_delta_m': float(np.median(delta)) if len(delta) else None,
            'delta_quantiles_05_25_75_95_m': np.quantile(delta, [.05, .25, .75, .95]).tolist() if len(delta) else None,
            'improved': int((delta < 0).sum()), 'tied': int((delta == 0).sum()),
            'worsened': int((delta > 0).sum()),
            'improved_fraction': float((delta < 0).mean()) if len(delta) else None,
            'cluster_bootstrap_95ci': intervals,
        }
    return result


def validation_diagnostics(baseline: list[dict], current: list[dict], reward: dict,
                           repetitions: int, seed: int, *, reference_label: str = 'sft',
                           current_label: str = 'grpo') -> dict:
    options = {'repetitions': repetitions, 'seed': seed,
               'reference_label': reference_label, 'current_label': current_label}
    current = align_predictions(baseline, current)
    max_difference = 0.0
    for row in baseline + current:
        computed = prediction_metrics(
            torch.tensor(row['predicted_waypoints']) if row['prediction_valid'] else None,
            torch.tensor(row['target_waypoints']), torch.tensor(row['trajectory_valid_mask'], dtype=torch.bool))
        if computed['prediction_valid'] != row['prediction_valid']:
            raise ValueError('saved prediction validity differs from evaluator')
        for key in ERRORS[:2]:
            saved, actual = row.get(key), computed.get(key)
            if saved is None or actual is None:
                if saved != actual:
                    raise ValueError('saved horizon availability differs from evaluator')
            else:
                if not math.isfinite(saved) or not math.isclose(saved, actual, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError('saved error differs from evaluator')
                max_difference = max(max_difference, abs(saved - actual))
        row['weighted_error_m'] = (
            reward['ade_weight'] * row['ade_3s_m'] + reward['fde_weight'] * row['fde_3s_m']
            if row['prediction_valid'] and row['fde_3s_m'] is not None else None)
    pairs = list(zip(baseline, current, strict=True))
    grouped = {}
    for axis in ('scene', 'longitudinal', 'lateral', 'joint'):
        subsets = {}
        for pair in pairs:
            row = pair[0]
            target = row['target']
            labels = {a: target[a] if target[f'{a}_valid'] else 'unavailable'
                      for a in ('longitudinal', 'lateral')}
            label = (row['scene_token'] if axis == 'scene' else
                     f'{labels["longitudinal"]}/{labels["lateral"]}' if axis == 'joint' else labels[axis])
            subsets.setdefault(label, []).append(pair)
        grouped[axis] = {label: paired_statistics(values, **options)
                         for label, values in sorted(subsets.items())}
    return {
        'comparison': compare_predictions(baseline, current),
        'action_behavior_cluster_bootstrap': action_behavior_bootstrap(baseline, current, repetitions, seed),
        'metric_recomputation_max_abs_difference_m': max_difference,
        'overall': paired_statistics(pairs, **options), 'groups': grouped,
        'bootstrap': {'unit': 'scene; uniformly resample scenes with replacement, retain all their rows',
                      'estimand': 'sample-weighted mean paired delta and improved fraction',
                      'repetitions': repetitions, 'seed': seed, 'interval': 'percentile 95%',
                      'single_scene': 'CI unavailable; descriptive point estimates only',
                      'scope': 'validation scene sampling uncertainty, not training-seed uncertainty'},
        'delta_definition': f'{current_label} minus {reference_label}; negative is better; exact saved-value ties',
        'subgroups': 'GT factorized labels; descriptive, no multiplicity-adjusted significance claims',
    }


def ranks(values: list[float]) -> np.ndarray:
    array = np.array(values)
    return np.array([1 + np.sum(array < x) + (np.sum(array == x) - 1) / 2 for x in array])


def rank_relation(scores: list[float], errors: list[float]) -> dict:
    counts = Counter()
    for i, j in combinations(range(len(scores)), 2):
        dr, de = scores[i] - scores[j], errors[i] - errors[j]
        label = ('both_tied' if dr == de == 0 else 'reward_tied' if dr == 0 else
                 'error_tied' if de == 0 else 'reversed' if dr * de > 0 else 'concordant')
        counts[label] += 1
    rho = None
    if len(set(scores)) > 1 and len(set(errors)) > 1:
        rho = float(np.corrcoef(ranks(scores), ranks([-e for e in errors]))[0, 1])
    best = [e for s, e in zip(scores, errors) if s == max(scores)] if scores else []
    return {'candidate_count': len(scores), 'spearman_reward_vs_negative_error': rho,
            'pairs': {k: counts[k] for k in ('concordant', 'reversed', 'reward_tied', 'error_tied', 'both_tied')},
            'top_tie_averaged_regret_m': mean([e - min(errors) for e in best])}


def rollout_diagnostics(groups: list[dict], history: list[dict], train_tokens: list[str], reward: dict) -> dict:
    config = RewardConfig(**reward)
    tokens = Counter(g['sample_token'] for g in groups)
    rows, generated = [], Counter()
    effective = [0, 0]
    actionable = [0, 0]
    for group, update in zip(groups, history, strict=True):
        candidates = group['candidates']
        if (len(candidates) != 4 or group['split'] != 'train'
                or group['optimizer_step'] != update['step'] or group['sample_token'] != update['sample_token']
                or group['sample_token'] not in train_tokens
                or [c['candidate_index'] for c in candidates] != list(range(4))
                or any(c['sample_token'] != group['sample_token'] for c in candidates)):
            raise ValueError('rollout/history/train population mismatch')
        derived = deepcopy(candidates)
        statistics = assign_advantages(derived, config.advantage_eps,
                                       action_geometry_weight=config.action_geometry_weight)
        if statistics != group['statistics'] or statistics != update['group_statistics']:
            raise ValueError('saved group statistics differ from reward producer')
        if not np.allclose([c['turn_advantages'] for c in candidates],
                           [c['turn_advantages'] for c in derived], rtol=1e-12, atol=1e-12):
            raise ValueError('saved advantages differ from reward producer')
        for turn in (0, 1):
            active = any(c['turn_advantages'][turn] != 0 for c in derived)
            if active != update['effective_advantage_group'][turn]:
                raise ValueError('history effective group mismatch')
            effective[turn] += active
            actionable[turn] += any(c['turn_advantages'][turn] != 0 and any(
                s['turn'] == turn + 1 and s['token_ids'] for s in c['assistant_spans']) for c in candidates)
        if update['effective_advantage_groups'] != effective:
            raise ValueError('history cumulative effective counts mismatch')
        for candidate in candidates:
            action = candidate['parsed_action'] if candidate['turn_valid'][0] else None
            generated['invalid' if action is None else f'{action["longitudinal"]}/{action["lateral"]}'] += 1
            r = candidate['rewards']
            m = r['trajectory_metrics']
            expected = 0.0
            if m['prediction_valid']:
                error = (m['ade_3s_m'] if m['fde_3s_m'] is None else
                         config.ade_weight * m['ade_3s_m'] + config.fde_weight * m['fde_3s_m'])
                if not math.isclose(error, r['error_m'], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('rollout weighted error mismatch')
                expected = math.exp(-error / config.distance_scale_m)
            consistency = 0.0
            if m['prediction_valid'] and candidate['turn_complete'][0] and candidate['parsed_action'] is not None:
                projection = trajectory_to_meta_action(candidate['waypoints'])
                consistency = sum(projection[f'{axis}_action'] == candidate['parsed_action'][axis]
                                  for axis in ('longitudinal', 'lateral')) / 2
            if r['consistency'] != consistency:
                raise ValueError('saved consistency differs from trajectory projection')
            if (not math.isclose(r['trajectory'], expected, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(r['turn_rewards'][1], config.combine(expected, consistency),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('rollout reward formula mismatch')
        valid = [c['rewards'] for c in candidates if c['rewards']['trajectory_metrics']['prediction_valid']]
        relations = {}
        transitions = {}
        for key in ('error_m', *ERRORS[:2]):
            available = [r for r in valid if (r[key] if key == 'error_m' else r['trajectory_metrics'][key]) is not None]
            errors = [r[key] if key == 'error_m' else r['trajectory_metrics'][key] for r in available]
            relations[key] = {
                label: rank_relation([r['turn_rewards'][1] if label == 'combined' else r['trajectory']
                                      for r in available], errors)
                for label in ('combined', 'trajectory_only')}
            changes = Counter()
            for i, j in combinations(range(len(available)), 2):
                de = errors[i] - errors[j]
                if de == 0:
                    continue
                before = (available[i]['trajectory'] - available[j]['trajectory']) * de > 0
                after = (available[i]['turn_rewards'][1] - available[j]['turn_rewards'][1]) * de > 0
                changes[f'{"reversed" if before else "not_reversed"}_to_'
                        f'{"reversed" if after else "not_reversed"}'] += 1
            transitions[key] = dict(changes)
        top_changed = None
        if len(valid) >= 2:
            before = {i for i, r in enumerate(valid) if r['trajectory'] == max(x['trajectory'] for x in valid)}
            after = {i for i, r in enumerate(valid) if r['turn_rewards'][1] == max(x['turn_rewards'][1] for x in valid)}
            top_changed = before != after
        rows.append({'step': group['optimizer_step'], 'sample_token': group['sample_token'],
                     'valid_candidates': len(valid), 'top_set_changed': top_changed, 'relations': relations,
                     'consistency_pair_transitions': transitions})
    aggregate = {}
    for key in ('error_m', *ERRORS[:2]):
        aggregate[key] = {}
        for label in ('combined', 'trajectory_only'):
            values = [r['relations'][key][label] for r in rows]
            aggregate[key][label] = {
                'pair_counts': {k: sum(v['pairs'][k] for v in values) for k in values[0]['pairs']},
                'groups_with_reversal': sum(v['pairs']['reversed'] > 0 for v in values),
                'defined_spearman_groups': sum(v['spearman_reward_vs_negative_error'] is not None for v in values),
                'mean_group_spearman': mean([v['spearman_reward_vs_negative_error'] for v in values
                                             if v['spearman_reward_vs_negative_error'] is not None]),
                'mean_top_tie_averaged_regret_m': mean([v['top_tie_averaged_regret_m'] for v in values
                                                       if v['candidate_count'] >= 2]),
            }
    return {
        'group_count': len(groups), 'candidate_count': 4 * len(groups),
        'unique_observations': len(tokens), 'repeated_draws': len(groups) - len(tokens),
        'observation_multiplicity': dict(Counter(tokens.values())),
        'unique_train_scenes': len({g['scene_token'] for g in groups}),
        'eligible_train_count': len(train_tokens), 'observation_coverage_fraction': len(tokens) / len(train_tokens),
        'effective_advantage_groups': effective,
        'zero_variance_groups': [sum(g['statistics']['zero_variance'][turn] for g in groups) for turn in (0, 1)],
        'groups_with_nonzero_advantage_and_present_tokens': actionable,
        'generated_action_distribution': dict(sorted(generated.items())),
        'gt_training_action_distribution': None,
        'gt_training_action_unavailable_reason': 'rollout/history do not store GT actions; train manifest absent',
        'invalid_trajectory_candidates': sum(4 - r['valid_candidates'] for r in rows),
        'top_set_changed_groups': sum(r['top_set_changed'] is True for r in rows),
        'top_set_eligible_groups': sum(r['top_set_changed'] is not None for r in rows),
        'ranking': aggregate, 'per_group': rows,
        'consistency_pair_transitions': {
            key: dict(sum((Counter(r['consistency_pair_transitions'][key]) for r in rows), Counter()))
            for key in ('error_m', *ERRORS[:2])},
        'ranking_scope': 'within observed valid candidates; invalid excluded and counted separately; '
                         'positive Spearman means aligned; undefined constant ranks omitted with denominator; '
                         'trajectory-only is a fixed-candidate counterfactual, not retraining',
        'training_log_means': {k: mean([r[k] for r in history]) for k in
                               ('kl', 'gradient_norm', 'lora_parameter_delta_l2', 'clip_fraction')},
    }


def analyze_archive(path: Path, *, repetitions: int = 10000, seed: int = 20261116) -> dict:
    if repetitions < 1:
        raise ValueError('bootstrap repetitions must be positive')
    pilot, continuation = 'grpo_pilot_run01', 'grpo_continuation_to256_run01'
    full = 'grpo_full_validation_reuse_sft_run01'
    with ZipFile(path) as archive:
        def read(name: str) -> dict:
            with archive.open(name) as stream:
                return json.load(stream)

        def read_rows(name: str) -> list[dict]:
            with archive.open(name) as stream:
                return [json.loads(line) for line in stream]

        metadata = [read(d + '/metadata.json') for d in (pilot, continuation, full)]
        identity = metadata[0]['identity']
        if any(m['identity'] != identity for m in metadata[1:]):
            raise ValueError('run model/data/reward identity differs')
        if metadata[2]['step'] != 256 or metadata[2]['mode'] != 'evaluate-full':
            raise ValueError('requires step-256 full validation')
        if (Path(metadata[1]['resume']).parts[-2:] != (pilot, 'checkpoint_32')
                or Path(metadata[2]['checkpoint']).parts[-2:] != (continuation, 'checkpoint_256')):
            raise ValueError('checkpoint lineage differs')
        sft_metadata = read('stage2b_full_run01/metadata.json')
        if any(sft_metadata['provenance'][k] != metadata[2][k] for k in ('model_id', 'revision')):
            raise ValueError('historical SFT model/revision differs')
        baseline = read_rows(full + '/sft_full_predictions.jsonl')
        current = read_rows(full + '/evaluate-full_256_predictions.jsonl')
        if baseline != read_rows('stage2b_full_run01/final_validation_predictions.jsonl'):
            raise ValueError('reused baseline differs from original SFT predictions')
        if (len(baseline) != 3594 or len(metadata[2]['population_tokens']) != 3594
                or set(metadata[2]['population_tokens']) != {r['sample_token'] for r in baseline}):
            raise ValueError('incomplete full validation population')
        history = read_rows(pilot + '/training_history.jsonl') + read_rows(continuation + '/training_history.jsonl')
        if [r['step'] for r in history] != list(range(1, 257)):
            raise ValueError('history must contain exactly steps 1 through 256')
        groups = [read(f'{pilot if i <= 32 else continuation}/rollout_{i}.json') for i in range(1, 257)]
    if ({g['scene_token'] for g in groups} & {r['scene_token'] for r in baseline}
            or set(identity['train_tokens']) & {r['sample_token'] for r in baseline}):
        raise ValueError('train/validation overlap')
    reward = identity['rollout']['reward']
    return {
        'evidence': {'archive': path.name, 'run_directories': [pilot, continuation, full],
                     'model_id': metadata[2]['model_id'], 'revision': metadata[2]['revision'],
                     'initial_sft_checkpoint': identity['initial_sft_checkpoint'], 'evaluated_step': 256,
                     'execution_commits': [m['git_commit'] for m in metadata],
                     'saved_manifest_identities': identity['data_intake']['manifests'],
                     'sft_data_provenance': sft_metadata['provenance']['data'],
                     'saved_run_identity_equal': True, 'historical_sft_exact_equal': True,
                     'checkpoint_lineage': 'metadata paths agree; checkpoint binaries/state not supplied',
                     'test_records_read_by_diagnostics': 0,
                     'unavailable': ['raw train/validation manifests: saved identities cannot be rehashed',
                                     'GT train actions: actual class coverage unavailable',
                                     'checkpoint state/weights: physical checkpoint identity unavailable',
                                     'baseline_source.json and SFT training_summary.json: selected checkpoint and '
                                     'historical action decoding provenance not independently confirmed']},
        'reward': reward,
        'validation': validation_diagnostics(baseline, current, reward, repetitions, seed),
        'training': rollout_diagnostics(groups, history, identity['train_tokens'], reward),
    }

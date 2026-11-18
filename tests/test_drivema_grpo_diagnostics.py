from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import diagnose_drivema_grpo as cli
from src.phase0.drivema_grpo_diagnostics import (
    ERRORS, paired_statistics, rank_relation, rollout_diagnostics, validation_diagnostics,
)
from src.phase0.drivema_stage3_rewards import assign_advantages, candidate_rewards
from src.phase0.drivema_stage3a import load_config
from src.phase0.phase0_4b_protocol import ActionTarget
from src.phase0.phase0_4c_evaluation import prediction_metrics
from src.phase0.drivema_stage2_protocol import trajectory_to_meta_action


@pytest.fixture
def reward():
    return load_config(ROOT / 'configs/drivema_stage3a.yaml').reward


def prediction(token, scene, error, mask=None):
    target = [[0., 0.]] * 6
    mask = mask if mask is not None else [True] * 6
    points = [[error, 0.]] * 6 if error is not None else None
    return {'sample_token': token, 'scene_token': scene, 'split': 'validation',
            'target': {'longitudinal': 'stop', 'lateral': 'straight',
                       'longitudinal_valid': True, 'lateral_valid': True},
            'target_waypoints': target, 'trajectory_valid_mask': mask,
            'conditioning_type': 'predicted_action',
            'parsed_action': {'longitudinal': 'stop', 'lateral': 'straight'},
            'trajectory_projection': trajectory_to_meta_action(points) if points else None,
            'assistant_spans': [],
            **prediction_metrics(torch.tensor(points) if points else None,
                                 torch.tensor(target), torch.tensor(mask))}


def test_scene_bootstrap_retains_clusters_and_sample_weighting():
    pairs = []
    for i, (scene, delta) in enumerate([('large', -1.)] * 9 + [('small', 9.)]):
        a = {'scene_token': scene, 'prediction_valid': True, **dict.fromkeys(ERRORS, 10.)}
        b = {**a, **dict.fromkeys(ERRORS, 10. + delta)}
        pairs.append((a, b))
    result = paired_statistics(pairs, repetitions=10000, seed=123)
    for value in result.values():
        assert value['mean_delta_m'] == 0
        assert value['improved_fraction'] == .9
        assert value['cluster_bootstrap_95ci']['mean_delta_m'] == [-1., 9.]
        assert value['cluster_bootstrap_95ci']['improved_fraction'] == [0., 1.]
    assert result == paired_statistics(pairs, repetitions=10000, seed=123)


def test_pairing_horizons_invalid_and_target_groups(reward):
    from dataclasses import asdict
    a = [prediction('a', 's1', 1), prediction('b', 's2', 2, [True] * 5 + [False]),
         prediction('c', 's2', None)]
    b = [prediction('a', 's1', 0), prediction('b', 's2', 3, [True] * 5 + [False]),
         prediction('c', 's2', 1)]
    result = validation_diagnostics(a, list(reversed(b)), asdict(reward), 100, 1)
    ade = result['overall']['ade_3s_m']
    assert ade['sample_count'] == 2 and ade['excluded_count'] == 1
    assert ade['improved'] == ade['worsened'] == 1
    assert result['overall']['fde_3s_m']['sample_count'] == 1
    assert result['overall']['weighted_error_m']['sample_count'] == 1
    assert result['overall']['fde_3s_m']['cluster_bootstrap_95ci'] is None
    assert result['groups']['joint']['stop/straight']['ade_3s_m'] == ade


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'gt', 'split', 'metric'])
def test_external_prediction_mismatch_fails(change, reward):
    from dataclasses import asdict
    a = [prediction('a', 's1', 1), prediction('b', 's2', 2)]
    b = deepcopy(a)
    if change == 'duplicate':
        b.append(b[0])
    if change == 'missing':
        b.pop()
    if change == 'gt':
        b[0]['target_waypoints'][0][0] = 99
    if change == 'split':
        b[0]['split'] = 'test'
    if change == 'metric':
        b[0]['ade_3s_m'] = 99
    with pytest.raises(ValueError):
        validation_diagnostics(a, b, asdict(reward), 100, 1)


def test_rank_ties_constant_and_reversal():
    result = rank_relation([2., 1., 1.], [1., 2., 3.])
    assert result['pairs']['concordant'] == 2
    assert result['pairs']['reward_tied'] == 1
    assert result['spearman_reward_vs_negative_error'] == pytest.approx(np.sqrt(.75))
    assert rank_relation([1., 1.], [0., 2.])['top_tie_averaged_regret_m'] == 1
    assert rank_relation([1., 1.], [0., 2.])['spearman_reward_vs_negative_error'] is None
    assert rank_relation([0., 1.], [0., 2.])['pairs']['reversed'] == 1
    assert rank_relation([], [])['top_tie_averaged_regret_m'] is None


def group_and_history(reward):
    candidates = []
    for index, (points, action) in enumerate([
        ([[1., 0.]] * 6, {'longitudinal': 'keep', 'lateral': 'left'}),
        ([[1.1, 0.]] * 6, {'longitudinal': 'stop', 'lateral': 'straight'}),
        ([[1.1, 0.]] * 6, {'longitudinal': 'stop', 'lateral': 'straight'}),
        (None, None),
    ]):
        candidate = {'parsed_action': action, 'waypoints': points, 'sample_token': 'train-a',
                     'candidate_index': index, 'turn_complete': [action is not None, points is not None],
                     'turn_valid': [action is not None, points is not None], 'turn_mask': [True, points is not None],
                     'assistant_spans': [{'turn': 1, 'token_ids': [1]}] +
                                        ([{'turn': 2, 'token_ids': [2]}] if points else [])}
        candidate['rewards'] = candidate_rewards(candidate, ActionTarget('stop', 'straight', True, True),
                                                 [[0., 0.]] * 6, [True] * 6, reward)
        candidates.append(candidate)
    statistics = assign_advantages(candidates, reward.advantage_eps)
    group = {'optimizer_step': 1, 'sample_token': 'train-a', 'scene_token': 'train-scene',
             'split': 'train', 'candidates': candidates, 'statistics': statistics}
    effective = [not x for x in statistics['zero_variance']]
    history = {'step': 1, 'sample_token': 'train-a', 'group_statistics': statistics,
               'effective_advantage_group': effective, 'effective_advantage_groups': list(map(int, effective)),
               'kl': 0., 'gradient_norm': 1., 'lora_parameter_delta_l2': .1, 'clip_fraction': 0.}
    return group, history


def test_producer_rewards_consistency_conflict_and_coverage(reward):
    from dataclasses import asdict
    g, h = group_and_history(reward)
    second, update = deepcopy(g), deepcopy(h)
    second['optimizer_step'] = update['step'] = 2
    update['effective_advantage_groups'] = [2 * x for x in h['effective_advantage_groups']]
    result = rollout_diagnostics([g, second], [h, update], ['train-a', 'train-b'], asdict(reward))
    assert result['unique_observations'] == 1 and result['repeated_draws'] == 1
    assert result['observation_coverage_fraction'] == .5
    assert result['invalid_trajectory_candidates'] == 2
    assert result['top_set_changed_groups'] == 2
    assert result['ranking']['error_m']['combined']['pair_counts']['reversed'] == 4
    assert result['ranking']['error_m']['trajectory_only']['pair_counts']['reversed'] == 0
    assert result['gt_training_action_distribution'] is None


@pytest.mark.parametrize('change', ['step', 'reward', 'advantage', 'statistics'])
def test_rollout_history_contract_failure(change, reward):
    from dataclasses import asdict
    g, h = group_and_history(reward)
    if change == 'step':
        h['step'] = 2
    if change == 'reward':
        g['candidates'][0]['rewards']['error_m'] += 1
    if change == 'advantage':
        g['candidates'][0]['turn_advantages'][0] += 1
    if change == 'statistics':
        h = {**h, 'group_statistics': {}}
    with pytest.raises(ValueError):
        rollout_diagnostics([g], [h], ['train-a'], asdict(reward))


def test_cli_writes_json_and_refuses_overwrite(tmp_path, monkeypatch):
    calls = []
    def analyze(path, **kwargs):
        calls.append((path, kwargs))
        return {'evidence': 'producer-backed functions tested separately'}
    monkeypatch.setattr(cli, 'analyze_archive', analyze)
    output = tmp_path / 'result.json'
    args = ['--archive', str(tmp_path / 'inputs.zip'), '--output', str(output)]
    assert cli.main(args) == 0
    assert json.loads(output.read_text())['evidence']
    with pytest.raises(SystemExit):
        cli.main(args)
    assert len(calls) == 1


@pytest.fixture
def archive(tmp_path, reward):
    from types import SimpleNamespace
    from zipfile import ZipFile
    from src.phase0.drivema_grpo_train import load_config as training_config, run_identity
    from src.phase0.drivema_grpo_diagnostics import analyze_archive
    rollout = load_config(ROOT / 'configs/drivema_stage3a.yaml')
    runner = SimpleNamespace(action_generation={'do_sample': False}, trajectory_max_new_tokens=192,
                             optimized=True, inference_batch_size=1, compute_log_probs=False)
    metadata = {'model_id': 'fixture', 'revision': 'fixture', 'initial_sft_checkpoint': 'checkpoint_3564',
                'git_commit': 'fixture', 'data_intake': {'manifests': {}, 'test_records_read': 0}}
    identity = run_identity(training_config(ROOT / 'configs/drivema_grpo.yaml'), rollout,
                            [SimpleNamespace(sample_token='train-a')],
                            [], runner, metadata)
    metadata['identity'] = identity
    baseline = [prediction(f'v{i}', f'scene-{i % 2}', 1.) for i in range(3594)]
    current = [prediction(f'v{i}', f'scene-{i % 2}', .5) for i in range(3594)]
    pilot, continuation, full = 'grpo_pilot_run01', 'grpo_continuation_to256_run01', 'grpo_full_validation_reuse_sft_run01'
    contents = {
        f'{pilot}/metadata.json': json.dumps({**metadata, 'resume': None}),
        f'{continuation}/metadata.json': json.dumps({**metadata, 'resume': f'{pilot}/checkpoint_32'}),
        f'{full}/metadata.json': json.dumps({**metadata, 'step': 256, 'mode': 'evaluate-full',
                                            'checkpoint': f'{continuation}/checkpoint_256',
                                            'population_tokens': [r['sample_token'] for r in baseline]}),
        'stage2b_full_run01/metadata.json': json.dumps({'provenance': {'model_id': 'fixture',
                                                                      'revision': 'fixture', 'data': {}}}),
    }
    def jsonl(values):
        return '\n'.join(json.dumps(v) for v in values) + '\n'
    contents['stage2b_full_run01/final_validation_predictions.jsonl'] = jsonl(baseline)
    contents[f'{full}/sft_full_predictions.jsonl'] = jsonl(baseline)
    contents[f'{full}/evaluate-full_256_predictions.jsonl'] = jsonl(current)
    histories = {pilot: [], continuation: []}
    for step in range(1, 257):
        group, history = group_and_history(reward)
        group['optimizer_step'] = history['step'] = step
        history['effective_advantage_groups'] = [step * n for n in history['effective_advantage_groups']]
        directory = pilot if step <= 32 else continuation
        histories[directory].append(history)
        contents[f'{directory}/rollout_{step}.json'] = json.dumps(group)
    for directory, history in histories.items():
        contents[f'{directory}/training_history.jsonl'] = jsonl(history)
    def save(name, overrides=None):
        path = tmp_path / name
        with ZipFile(path, 'w') as z:
            for key, value in {**contents, **(overrides or {})}.items():
                z.writestr(key, value)
        return path
    return save, contents, analyze_archive


def test_archive_real_producer_helpers_through_cli(archive, tmp_path):
    save, _, _ = archive
    path = save('inputs.zip')
    output = tmp_path / 'analysis.json'
    assert cli.main(['--archive', str(path), '--output', str(output), '--bootstrap-repetitions', '30']) == 0
    result = json.loads(output.read_text())
    assert result['validation']['overall']['ade_3s_m']['sample_count'] == 3594
    assert result['validation']['overall']['ade_3s_m']['mean_delta_m'] == -.5
    assert result['training']['repeated_draws'] == 255
    assert result['training']['consistency_pair_transitions']['error_m']['not_reversed_to_reversed'] == 512


@pytest.mark.parametrize('failure', ['history_gap', 'identity', 'lineage', 'full_population'])
def test_archive_intake_rejects_incomplete_or_mixed_runs(archive, failure):
    save, contents, analyze = archive
    directory = 'grpo_continuation_to256_run01'
    if failure == 'history_gap':
        key = f'{directory}/training_history.jsonl'
        changed = '\n'.join(contents[key].splitlines()[1:])
    else:
        key = f'{directory}/metadata.json' if failure != 'full_population' else 'grpo_full_validation_reuse_sft_run01/metadata.json'
        m = json.loads(contents[key])
        if failure == 'identity':
            m['identity']['revision'] = 'different'
        if failure == 'lineage':
            m['resume'] = 'wrong/checkpoint_32'
        if failure == 'full_population':
            m['population_tokens'].pop()
        changed = json.dumps(m)
    with pytest.raises(ValueError):
        analyze(save('bad.zip', {key: changed}), repetitions=30)

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_reward_shape import main
from src.phase0.drivema_grpo_replay import replay_identity, reward_shape_replay
from src.phase0.drivema_grpo_train import group_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from test_drivema_grpo import arguments, case, config, low512_control, processor, records, runner, samples
from test_drivema_grpo_diagnostics import group_and_history, prediction, reward


@pytest.fixture
def evaluations(tmp_path, low512_control, reward):
    low, run = tmp_path / 'low512', tmp_path / 'reward_shape'
    low.mkdir()
    run.mkdir()
    tokens = [f'v{i}' for i in range(3594)]
    parent = '/training/worker_a_continuation/checkpoint_512'
    control = {**low512_control, 'population_tokens': tokens, 'checkpoint': parent}
    for label, error in (('low512', 1.), ('additive', .75), ('multiplicative', .5)):
        step, directory, full = 512, low, control
        if label != 'low512':
            step, directory = 768, run / f'{label}_full'
            directory.mkdir()
            replay = reward_shape_replay(control, label)
            identity = replay_identity(replay['baseline_identity'], replay)
            (run / label).mkdir()
            write_json(run / label / 'metadata.json', {'identity': identity, 'resume': parent})
            config = replace(reward, composition=label)
            group, _ = group_and_history(config)
            metrics = group_metrics(group, config)
            history = [{**metrics, 'step': s['optimizer_step'], 'sample_token': s['sample_token']}
                       for s in replay['steps'][512:]]
            write_predictions(run / label / 'training_history.jsonl', history)
            full = {**control, 'step': step, 'identity': identity, 'observation_replay': replay,
                    'checkpoint': str(Path('/training/reward_shape') / label / 'checkpoint_768')}
        write_json(directory / 'metadata.json', full)
        template = prediction('v', 's', error)
        rows = [{**template, 'sample_token': token, 'scene_token': f's{i % 140}'} for i, token in enumerate(tokens)]
        if label == 'multiplicative':
            rows.reverse()
        write_predictions(directory / f'evaluate-full_{step}_predictions.jsonl', rows)
    return low, run


def test_report_compares_both_continuations_with_each_other_and_low512(evaluations):
    low, run = evaluations
    assert main(['--low512-validation', str(low), '--run-root', str(run), '--bootstrap-repetitions', '20']) == 0
    report = json.loads((run / 'comparison.json').read_text())
    assert (report['observations'], report['scenes']) == (3594, 140)
    assert report['primary']['multiplicative']['ade_3s_m'] == .5
    for signal in report['training_signal'].values():
        assert signal['steps'] == 256 and signal['zero_variance_groups'] == {'action': 0, 'trajectory': 0}
        assert signal['ade_reward_ranking'] == {'non_tied_pairs': 512, 'additive_reversals': 512,
                                               'multiplicative_reversals': 512, 'additive_reversal_rate': 1.,
                                               'multiplicative_reversal_rate': 1.}
    for name, delta in (('additive_minus_low512', -.25), ('multiplicative_minus_low512', -.5),
                        ('multiplicative_minus_additive', -.25)):
        paired = report['paired'][name]
        for metric in ('ade_3s_m', 'fde_3s_m'):
            assert paired[metric]['mean_delta_m'] == delta
            assert paired[metric]['cluster_bootstrap_95ci']['mean_delta_m'] == [delta, delta]
        for metric in ('joint_accuracy', 'consistency_mean', 'invalid_rate'):
            expected = .5 if metric == 'consistency_mean' and name.startswith('multiplicative_') else 0.
            assert paired[metric + '_delta'] == expected
            interval = paired['action_behavior_cluster_bootstrap']['intervals'][metric]
            assert interval['delta_95ci'] == [expected, expected]


@pytest.mark.parametrize('mismatch', ['parent', 'reward'])
def test_report_rejects_unmatched_continuations(evaluations, mismatch):
    low, run = evaluations
    if mismatch == 'parent':
        path = run / 'multiplicative/metadata.json'
        metadata = json.loads(path.read_text())
        metadata['resume'] = '/other/worker_a_continuation/checkpoint_512'
    else:
        path = run / 'multiplicative_full/metadata.json'
        metadata = json.loads(path.read_text())
        metadata['identity'] = json.loads((run / 'additive_full/metadata.json').read_text())['identity']
    write_json(path, metadata)
    with pytest.raises(ValueError, match='same Low-512 checkpoint|identity mismatch'):
        main(['--low512-validation', str(low), '--run-root', str(run), '--bootstrap-repetitions', '20'])
    assert not (run / 'comparison.json').exists()

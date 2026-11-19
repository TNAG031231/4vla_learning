from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_grpo as cli
from scripts import report_drivema_low512_ce as report_cli
from src.phase0 import drivema_grpo_train as training
from src.phase0.drivema_grpo_replay import low512_replay, replay_identity
from src.phase0.drivema_grpo_schedules import coverage_schedules
from src.phase0.drivema_stage2b import select_subset
from src.phase0.phase0_4c_full_train import write_predictions
from test_drivema_grpo import (
    arguments, case, config, controlled_rollout, dual_model, lora_model, processor, records, runner,
    samples, tiny_qwen, validation_predict, assert_nested_equal,
)
from test_drivema_grpo_diagnostics import prediction


@pytest.fixture
def low_control(arguments, config, tmp_path):
    original = arguments['train'][0]
    record = arguments['records'][original.sample_token]
    train = [replace(original, sample_token=f'train-{i}') for i in range(512)]
    arguments['train'] = train
    arguments['records'].update({s.sample_token: {**record, 'sample_token': s.sample_token} for s in train})
    runner = arguments['runner']
    runner.compute_log_probs = False
    rollout = arguments['rollout_config']
    monitor = select_subset(arguments['validation'], config.monitor_samples, rollout.seed, 'validation')
    baseline = training.run_identity(config, rollout, train, monitor, runner, arguments['metadata'])
    original_steps = [{'optimizer_step': i, 'sample_token': s.sample_token, 'scene_token': s.scene_token}
                      for i, s in enumerate(train[:256], 1)]
    schedules = coverage_schedules({'baseline_identity': baseline, 'steps': original_steps, 'source_archive': 'fixture'},
                                   train, .5, {}, 20261116)
    replay = schedules['repeat256']
    control = {'mode': 'evaluate-full', 'step': 512, 'observation_replay': replay,
               'identity': replay_identity(baseline, replay)}
    arguments['metadata']['observation_replay'] = low512_replay(control)
    directory = tmp_path / 'low512'
    directory.mkdir()
    (directory / 'metadata.json').write_text(json.dumps(control))
    return directory


@pytest.mark.parametrize('arm', ['grpo_ce', 'ce_only'])
def test_low_schedule_updates_and_resume(arm, low_control, arguments, config, validation_predict,
                                         dual_model, controlled_rollout, monkeypatch, tmp_path):
    selected = training.load_config(ROOT / f'configs/drivema_low512_{arm}.yaml')
    assert selected.max_steps == 512 and 256 in selected.evaluation_steps
    cfg = replace(selected, max_steps=3, monitor_samples=config.monitor_samples, evaluation_steps=[0, 1, 3])
    monkeypatch.setattr(training, 'rollout_group', controlled_rollout[0])
    other = deepcopy(dual_model)
    training.fit(model=dual_model, config=cfg, output=tmp_path / 'single', **arguments)
    training.fit(model=other, config=replace(cfg, max_steps=1), output=tmp_path / 'pilot', **arguments)
    checkpoint = tmp_path / 'pilot/checkpoint_1'
    training.fit(model=other, config=cfg, output=tmp_path / 'continued', resume=checkpoint, **arguments)
    assert_nested_equal(dual_model.state_dict(), other.state_dict())
    rows = [json.loads(line) for line in (tmp_path / 'single/training_history.jsonl').read_text().splitlines()]
    assert [r['sample_token'] for r in rows] == ['train-0', 'train-0', 'train-1']
    for row in rows:
        assert row['trajectory_ce'] > 0 and row['trajectory_tokens'] > 0 and row['lora_parameter_delta_l2'] > 0
        assert row['trajectory_ce_weight'] == selected.trajectory_ce_weight
    if arm == 'ce_only':
        assert not controlled_rollout[1]
        assert all(r['loss'] == r['trajectory_ce'] and 'kl' not in r for r in rows)
        assert not list((tmp_path / 'single').glob('rollout_*.json'))
    else:
        assert len(controlled_rollout[1]) == 6
        assert all(r['loss'] == pytest.approx(r['policy_loss'] + cfg.beta * r['kl'] + .1 * r['trajectory_ce'])
                   for r in rows)
    cross = replace(cfg, objective='ce_only' if arm == 'grpo_ce' else 'grpo_ce',
                    trajectory_ce_weight=1. if arm == 'grpo_ce' else .1)
    with pytest.raises(ValueError, match='identity'):
        training.fit(model=other, config=cross, output=tmp_path / 'cross', resume=checkpoint, **arguments)


def test_low512_source_and_supervision_intake(low_control, arguments, config, dual_model, tmp_path):
    control = json.loads((low_control / 'metadata.json').read_text())
    control['observation_replay']['experiment']['arm'] = 'unique512'
    with pytest.raises(ValueError, match='Low-512'):
        low512_replay(control)
    path = ROOT / 'configs/drivema_low512_ce_only.yaml'
    with pytest.raises(SystemExit):
        cli.main(['--config', str(path), '--dry-run'])
    arguments['records']['train-255']['trajectory_valid_mask'] = [True] * 5 + [False]
    cfg = replace(training.load_config(path), monitor_samples=config.monitor_samples)
    with pytest.raises(ValueError, match='train-255'):
        training.fit(model=dual_model, config=cfg, output=tmp_path / 'bad', **arguments)
    assert not (tmp_path / 'bad').exists()


def test_final_report_pairs_all_three_metrics(low_control, tmp_path):
    control = json.loads((low_control / 'metadata.json').read_text())
    control['population_tokens'] = [f'v{i}' for i in range(3594)]
    (low_control / 'metadata.json').write_text(json.dumps(control))
    for name, error, weight in [('low512', 1., 0.), ('grpo_ce', .5, .1), ('ce_only', .75, 1.)]:
        directory = tmp_path / name
        if weight:
            directory.mkdir()
            identity = {**control['identity'], 'trajectory_supervision': {'objective': name, 'weight': weight}}
            (directory / 'metadata.json').write_text(json.dumps({**control, 'identity': identity}))
        template = prediction('v', 's', error)
        rows = [{**template, 'sample_token': token, 'scene_token': f's{i % 140}'}
                for i, token in enumerate(control['population_tokens'])]
        if name == 'grpo_ce':
            rows[0]['parsed_action'] = None
        write_predictions(directory / 'evaluate-full_512_predictions.jsonl', rows)
    output = tmp_path / 'comparison.json'
    assert report_cli.main(['--low512', str(low_control), '--grpo-ce', str(tmp_path / 'grpo_ce'),
                            '--ce-only', str(tmp_path / 'ce_only'), '--bootstrap-repetitions', '8',
                            '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['primary']['ce_only'] == {'ade_3s_m': .75, 'fde_3s_m': .75, 'joint_accuracy': 1.}
    paired = result['paired']['grpo_ce_minus_low512']
    for key in ('ade_3s_m', 'fde_3s_m'):
        assert paired[key]['mean_delta_m'] == -.5
        assert paired[key]['cluster_bootstrap_95ci']['mean_delta_m'] == [-.5, -.5]
    assert paired['joint_accuracy_delta'] == pytest.approx(-1 / 3594)
    assert paired['joint_accuracy_bootstrap']['defined_replicates'] == 8
    with pytest.raises(ValueError, match='matched final checkpoint'):
        report_cli.compare_arms(low_control, tmp_path / 'ce_only', tmp_path / 'grpo_ce', 8)

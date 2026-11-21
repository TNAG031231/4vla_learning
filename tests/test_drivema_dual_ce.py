from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_dual_ce as launch
from scripts.report_drivema_dual_ce import historical_runs, report
from src.phase0 import drivema_grpo_train as training
from src.phase0.drivema_grpo_replay import extend_low512, low512_replay
from src.phase0.drivema_stage2b import select_subset
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from test_drivema_grpo import (
    ScalarAdapters, arguments, case, config, processor, records, runner, samples, scalar_training, validation_predict,
)
from test_drivema_grpo_diagnostics import prediction
from test_drivema_low512_ce import low_control


def test_checkpoint_boundary_and_replay(low_control, config, arguments, scalar_training, validation_predict, tmp_path):
    control = json.loads((low_control / 'metadata.json').read_text())
    schedule = extend_low512(low512_replay(control))
    prefix, tail = schedule['steps'][:512], schedule['steps'][512:]
    assert [(r['sample_token'], r['scene_token'], r['optimizer_step']) for r in prefix] == [
        (r['sample_token'], r['scene_token'], r['optimizer_step']) for r in control['observation_replay']['steps']]
    assert [r['sample_token'] for r in tail] == [r['sample_token'] for r in prefix if r['visit'] == 1]
    assert all(r['total_visits'] == 3 and r['visit'] == 3 for r in tail)

    model = ScalarAdapters()
    low_cfg = replace(config, max_steps=512, evaluation_steps=[0, 512], checkpoint_interval=512)
    training.fit(model=model, config=low_cfg, output=tmp_path / 'pure', **arguments)
    low_checkpoint = tmp_path / 'pure/checkpoint_512'
    assert torch.load(low_checkpoint / 'unified_state.pt', weights_only=True)['identity'] == control['identity']
    arguments['metadata']['observation_replay'] = schedule
    early = replace(training.load_config(launch.CONFIG), monitor_samples=config.monitor_samples)
    late = replace(early, dual_ce_first_step=513)
    a_model = ScalarAdapters()
    training.fit(model=a_model, config=replace(early, max_steps=512), output=tmp_path / 'early', **arguments)
    a_checkpoint = tmp_path / 'early/checkpoint_512'
    with pytest.raises(ValueError, match='original pure GRPO'):
        training.fit(model=model, config=late, output=tmp_path / 'wrong', resume=a_checkpoint,
                     low512_parent_identity=control['identity'], **arguments)
    with pytest.raises(ValueError, match='requires the Low-512 checkpoint'):
        training.fit(model=model, config=late, output=tmp_path / 'fresh', **arguments)

    for label, cfg, checkpoint in [('A', early, a_checkpoint), ('B', late, low_checkpoint)]:
        restored = ScalarAdapters()
        restored.load_state_dict(torch.load(checkpoint / 'adapter/fixture.pt', weights_only=True))
        summary = training.fit(model=restored, config=cfg, output=tmp_path / label, resume=checkpoint,
                               low512_parent_identity=control['identity'], **arguments)
        assert summary['optimizer_steps_this_run'] == 256 and summary['last_step'] == 768
        state = torch.load(tmp_path / label / 'checkpoint_768/training_state.pt', weights_only=True)
        assert state['identity']['dual_supervision'] == cfg.dual_supervision()
        assert all(float(s['step']) == 768 for s in state['optimizer']['state'].values())
        history = [json.loads(s) for s in (tmp_path / label / 'training_history.jsonl').read_text().splitlines()]
        assert [(r['step'], r['sample_token']) for r in history] == [(r['optimizer_step'], r['sample_token']) for r in tail]


def test_reports_use_complete_matched_checkpoints(low_control, arguments, config, tmp_path):
    low = low_control
    control = json.loads((low / 'metadata.json').read_text())
    tokens = [f'v{i}' for i in range(3594)]
    control['population_tokens'] = tokens
    write_json(low / 'metadata.json', control)
    original = control['identity']
    trajectory = tmp_path / 'drivema/pr66_ce_fixture'
    meta = tmp_path / 'drivema/meta_ce_fixture'
    run = tmp_path / 'dual'
    run.mkdir()
    arguments['metadata']['observation_replay'] = extend_low512(low512_replay(control))
    cfg = replace(training.load_config(launch.CONFIG), monitor_samples=config.monitor_samples)
    monitor = select_subset(arguments['validation'], cfg.monitor_samples, arguments['rollout_config'].seed, 'validation')

    def save(directory, step, identity, error, *, sft=False):
        directory.mkdir(exist_ok=True, parents=True)
        write_json(directory / 'metadata.json', {**control, 'step': step, 'identity': identity})
        row = prediction('v', 'scene', error)
        rows = [{**row, 'sample_token': token, 'scene_token': f's{i % 140}'} for i, token in enumerate(tokens)]
        label = 'sft_full' if sft else f'evaluate-full_{step}'
        write_predictions(directory / f'{label}_predictions.jsonl', rows)

    save(low, 512, original, 1.)
    save(low, 512, original, 1.25, sft=True)
    for name, folder, key, weight in [('grpo_ce', trajectory, 'trajectory_supervision', .1),
                                      ('ce_only', trajectory, 'trajectory_supervision', 1.),
                                      ('grpo_meta_ce', meta, 'meta_action_supervision', .1),
                                      ('meta_ce_only', meta, 'meta_action_supervision', 1.)]:
        save(folder / f'{name}_full', 512, {**original, key: {'objective': name, 'weight': weight}}, .75)
    assert historical_runs(tmp_path, None, None) == (trajectory, meta)
    for name, step, first, error in [('A', 512, 1, .5), ('A', 768, 1, .25), ('B', 768, 513, .5)]:
        identity = training.run_identity(replace(cfg, dual_ce_first_step=first), arguments['rollout_config'],
                                         arguments['train'], monitor, arguments['runner'], arguments['metadata'])
        save(run / f'{name}{step}_full', step, identity, error)
    mid = report(run, low, 512, cfg, trajectory, meta, repetitions=4)
    assert set(mid['primary']) == {'sft', 'low512', 'grpo_ce', 'ce_only', 'grpo_meta_ce', 'meta_ce_only', 'A'}
    assert mid['paired']['A_minus_low512']['ade_3s_m']['mean_delta_m'] == -.5
    final = report(run, low, 768, cfg, trajectory, meta, repetitions=4)
    assert final['paired']['A_minus_B']['fde_3s_m']['mean_delta_m'] == -.25
    assert final['paired']['A_minus_B']['joint_accuracy_delta'] == 0.
    assert final == json.loads((run / 'comparison_768.json').read_text())
    save(run / 'B768_full', 768, original, .5)
    with pytest.raises(ValueError, match='objective or population mismatch'):
        report(run, low, 768, cfg, trajectory, meta, repetitions=2)


def test_worker_uses_per_arm_checkpoints_and_reports_before_a_tail(tmp_path, monkeypatch):
    events = []

    def execute(command, **kwargs):
        def option(key):
            return command[command.index(key) + 1] if key in command else None

        assert option('--local-model-dir') == str(tmp_path)
        events.append((option('--output-relative-dir'), option('--dual-ce-first-step'),
                       option('--resume'), option('--checkpoint'), option('--max-steps')))

    monkeypatch.setattr(launch.subprocess, 'run', execute)
    monkeypatch.setattr(launch, 'historical_runs', lambda *args: (tmp_path, None))
    monkeypatch.setattr(launch, 'report', lambda run, low, step, *args: events.append(step))
    parent = tmp_path / 'original/checkpoint_512'
    run = tmp_path / 'run'
    run.mkdir()
    for arm in ('A', 'B'):
        assert launch.main(['--arm', arm, '--derived-root', str(tmp_path), '--dataset-root', str(tmp_path),
                            '--local-model-dir', str(tmp_path), '--low512-checkpoint', str(parent),
                            '--run-relative-dir', 'run']) == 0
    a512 = str(run / 'A_to512/checkpoint_512')
    assert events == [
        ('run/A_to512', '1', None, None, '512'), ('run/A512_full', '1', None, a512, None), 512,
        ('run/A_to768', '1', a512, None, '768'),
        ('run/A768_full', '1', None, str(run / 'A_to768/checkpoint_768'), None),
        ('run/B_to768', '513', str(parent), None, '768'),
        ('run/B768_full', '513', None, str(run / 'B_to768/checkpoint_768'), None)]

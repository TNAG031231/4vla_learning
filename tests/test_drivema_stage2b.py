from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0 import drivema_stage2b as training
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.phase0_4b_protocol import ActionTarget
from scripts import run_drivema_stage2b as cli
from test_drivema_stage2_sft import case, records, samples, processor, runner, tiny_qwen, lora_model, scripted_generate


def test_fixed_subset_action_scene_coverage_and_split(samples):
    population = [replace(samples['train'][0], sample_token=f'train-{i}', scene_token=f'scene-{i % 140}',
                          target=ActionTarget(('keep', 'stop', 'accelerate', 'decelerate')[i % 4],
                                              ('left', 'straight', 'right')[i % 3], True, True)) for i in range(1000)]
    first = training.select_subset(population, 512, 7, 'train')
    assert first == training.select_subset(list(reversed(population)), 512, 7, 'train')
    assert len({s.sample_token for s in first}) == 512
    assert len({s.target.training_text() for s in first}) == 12
    assert len({s.scene_token for s in first}) == 140
    validation = [replace(s, split='validation', sample_token='val-' + s.sample_token) for s in population]
    second = training.select_subset(validation, 128, 7, 'validation')
    assert not {s.sample_token for s in first} & {s.sample_token for s in second}
    with pytest.raises(ValueError, match='split'):
        training.select_subset(population, 128, 7, 'validation')
    with pytest.raises(ValueError):
        training.select_subset([], 1, 7, 'test')


def test_full_config_schedule_and_cli(samples, capsys):
    path = ROOT / 'configs/drivema_stage2b_full.yaml'
    config = training.load_config(path)
    pilot = training.load_config(ROOT / 'configs/drivema_stage2b_pilot.yaml')
    assert config == replace(pilot, train_samples=None, checkpoint_interval=891,
                             validation_interval=891, output_relative_dir='drivema/stage2b_full_run01')
    population = [replace(samples['train'][0], sample_token=f'train-{i}') for i in range(14253)]
    selected = training.select_subset(population, config.train_samples, config.seed, 'train')
    groups = training.epoch_batches(selected, config, 0)
    assert len(selected) == len({s.sample_token for group in groups for s in group}) == 14253
    assert len(groups) * config.epochs == 3564
    assert len(groups[-1]) == 1
    assert list(range(config.validation_interval, len(groups) + 1, config.validation_interval)) == [891, 1782, 2673, 3564]
    assert cli.main(['--config', str(path), '--mode', 'full', '--epochs', '1', '--learning-rate', '0.0001',
                     '--output-relative-dir', config.output_relative_dir, '--baseline-predictions',
                     'waypoint.jsonl', 'no-action.jsonl', 'mlp.jsonl', '--dry-run']) == 0
    assert training.TrainingConfig(**json.loads(capsys.readouterr().out)) == config


def test_selection_and_action_regression():
    def metrics(invalid, ade):
        return {'trajectory': {'invalid_prediction_count': invalid, 'ade_3s_m': ade, 'fde_3s_m': ade}}
    assert training.selection_key(metrics(0, 4), 2) < training.selection_key(metrics(1, .1), 1)
    assert training.selection_key(metrics(0, 1), 1) < training.selection_key(metrics(0, 1), 2)
    assert training.selection_key(metrics(2, None), 1)[1] == float('inf')
    original = {'longitudinal_macro_f1': .8, 'lateral_macro_f1': .7, 'joint_accuracy': .6}
    assert training.action_change(original, {**original, 'joint_accuracy': .5})['any_decline']


def test_real_lora_training_resume_matches_uninterrupted(lora_model, runner, samples, records, processor,
                                                        monkeypatch, tmp_path):
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', scripted_generate(processor,
        [samples['train'][0].target.training_text(), serialize_trajectory(records['train'][0]['future_waypoints'])], []))
    config = replace(training.load_config(ROOT / 'configs/drivema_stage2b_pilot.yaml'), train_samples=4,
                     monitor_samples=1, gradient_accumulation_steps=2, checkpoint_interval=1, validation_interval=1)
    fresh_base = deepcopy(lora_model).unload()
    targets = {r['sample_token']: r for rows in records.values() for r in rows}
    arguments = dict(runner=runner, train=samples['train'], monitor=samples['validation'][:1],
                     records=targets, config=config, provenance={'evidence': 'CPU_synthetic'})
    full = training.fit(model=lora_model, output=tmp_path / 'full', **arguments)
    checkpoint = tmp_path / 'full/checkpoint_1'
    resumed_model = PeftModel.from_pretrained(fresh_base, checkpoint / 'adapter', is_trainable=True)
    resumed = training.fit(model=resumed_model, output=tmp_path / 'resume', resume=checkpoint, **arguments)
    assert full['optimizer_steps'] == resumed['optimizer_steps'] == 2
    for (name, original), (_, restored) in zip(lora_model.named_parameters(), resumed_model.named_parameters()):
        if 'lora_' in name:
            torch.testing.assert_close(original, restored, atol=0, rtol=0)
    logs = [json.loads(line) for line in (tmp_path / 'full/training_history.jsonl').read_text().splitlines()]
    assert len(logs) == 2 and sum(r['sample_count'] for r in logs) == 4
    assert all(r['lora_step_delta_l2'] > 0 and r['action_ce'] > 0 and r['trajectory_ce'] > 0 for r in logs)
    assert all(r['peak_allocated_bytes'] is None for r in logs)
    with pytest.raises(FileExistsError):
        training.fit(model=resumed_model, output=tmp_path / 'resume', resume=checkpoint, **arguments)
    with pytest.raises(ValueError, match='identities'):
        training.fit(model=resumed_model, output=tmp_path / 'bad', resume=checkpoint,
                     **{**arguments, 'config': replace(config, learning_rate=.9)})


def test_microbatch_and_partial_groups(samples):
    config = replace(training.load_config(ROOT / 'configs/drivema_stage2b_pilot.yaml'),
                     batch_size=2, gradient_accumulation_steps=1)
    groups = training.epoch_batches(samples['train'][:3], config, 0)
    assert list(map(len, groups)) == [2, 1]
    assert len({s.sample_token for g in groups for s in g}) == 3


def test_cli_requires_pilot_informed_full_parameters(capsys):
    assert cli.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['train_samples'] == 512
    with pytest.raises(SystemExit):
        cli.main(['--mode', 'full', '--dry-run'])
    assert cli.main(['--mode', 'full', '--epochs', '2', '--learning-rate', '0.00005',
                     '--output-relative-dir', 'drivema/future-full', '--baseline-predictions',
                     'waypoint.jsonl', 'no-action.jsonl', 'mlp.jsonl', '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['train_samples'] is None


def test_baseline_alignment_recomputes_metrics_and_rejects_mismatch(lora_model, runner, processor, samples,
                                                                  records, tmp_path, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    from src.phase0.drivema_stage2_sft import evaluate
    from src.phase0.phase0_4c_full_train import write_predictions
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', scripted_generate(processor,
        [samples['train'][0].target.training_text(), serialize_trajectory(records['train'][0]['future_waypoints'])], []))
    selected = samples['validation'][:1]
    targets = {r['sample_token']: r for r in records['validation'][:1]}
    rows, _ = evaluate(lora_model, runner, selected, targets)
    paths = []
    for track in ('predicted_action', 'no_action_neutral_context', 'ego_history_mlp'):
        path = tmp_path / f'{track}.jsonl'
        baseline = [{**r, 'conditioning_type': track,
                     'target_waypoints': torch.tensor(r['target_waypoints']).tolist()} for r in rows]
        write_predictions(path, baseline)
        paths.append(path)
    result = cli.compare_baselines(rows, targets, paths)
    assert result['sample_count'] == 1
    assert all(p['ade_3s_m']['mean_delta_m'] == 0 for p in result['pairwise'].values())
    altered = json.loads(paths[0].read_text())
    altered['scene_token'] = 'wrong-scene'
    write_predictions(paths[0], [altered])
    with pytest.raises(ValueError, match='mismatch'):
        cli.compare_baselines(rows, targets, paths)


@pytest.mark.parametrize('options,expected', [([], 1), (['--inference-batch-size', '2'], 2),
                                             (['--inference-batch-size', '4'], 4)])
def test_cli_inference_default_and_explicit_batches(options, expected, monkeypatch, capsys):
    parse = cli.argparse.ArgumentParser.parse_args
    observed = []
    def capture(parser, *args, **kwargs):
        namespace = parse(parser, *args, **kwargs)
        observed.append(namespace.inference_batch_size)
        return namespace
    monkeypatch.setattr(cli.argparse.ArgumentParser, 'parse_args', capture)
    assert cli.main(['--dry-run', *options]) == 0
    assert observed == [expected]
    assert 'inference_batch_size' not in json.loads(capsys.readouterr().out)

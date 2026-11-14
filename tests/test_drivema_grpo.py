from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_grpo as cli
from src.phase0 import drivema_grpo_train as training
from src.phase0 import drivema_stage3a
from src.phase0.drivema_grpo_evaluation import compare_predictions, evaluate_population
from src.phase0.drivema_stage2_protocol import trajectory_to_meta_action
from src.phase0.drivema_stage2b import select_subset
from test_drivema_stage2_sft import case, records, samples, processor, runner, tiny_qwen, lora_model
from test_drivema_stage3b import dual_model, controlled_rollout


@pytest.fixture
def config():
    return replace(training.load_config(ROOT / 'configs/drivema_grpo.yaml'), monitor_samples=2)


@pytest.fixture
def arguments(samples, records, runner):
    return dict(runner=runner, train=samples['train'], validation=samples['validation'],
                records={r['sample_token']: r for rows in records.values() for r in rows},
                rollout_config=drivema_stage3a.load_config(ROOT / 'configs/drivema_stage3a.yaml'),
                metadata={'model_id': 'tiny', 'revision': 'random',
                          'initial_sft_checkpoint': 'synthetic-sft', 'data_intake': {'test_records_read': 0}})


@pytest.fixture
def validation_predict(runner, arguments, monkeypatch):
    calls = []

    def predict(model, sample):
        calls.append(sample.sample_token)
        # Deliberately consume all CPU RNGs: evaluation must restore each stream.
        random.random()
        np.random.random()
        torch.rand(1)
        points = arguments['records'][sample.sample_token]['future_waypoints']
        return {'waypoints': points, 'parsed_action': {'longitudinal': 'keep', 'lateral': 'straight'},
                'conditioning_type': 'predicted_action', 'invalid_reason': None,
                'trajectory_projection': trajectory_to_meta_action(points),
                'assistant_spans': [{'turn': 1, 'token_ids': [1, 2]}, {'turn': 2, 'token_ids': [3, 2]}]}
    monkeypatch.setattr(runner, 'predict', predict)
    return calls


class ScalarAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({name: nn.Linear(1, 1, bias=False) for name in ('default', 'reference')})
        for layer in self.lora_A.values():
            nn.init.constant_(layer.weight, .5)

    def set_adapter(self, adapter):
        self.active_adapter = adapter

    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs == {'gradient_checkpointing_kwargs': {'use_reentrant': False}}

    def save_pretrained(self, path, selected_adapters):
        path.mkdir()
        assert selected_adapters == ['default', 'reference']
        torch.save(self.state_dict(), path / 'fixture.pt')


@pytest.fixture
def scalar_training(monkeypatch):
    def rollout(**kwargs):
        draw = float(torch.rand(1)) + random.random() + np.random.random()
        return {'sample_token': kwargs['sample'].sample_token, 'draw': draw,
                'candidates': [{'rewards': {'action': draw, 'trajectory': .5, 'consistency': .25,
                                           'turn_rewards': [draw, .625]},
                                'turn_advantages': [1., -1.], 'assistant_spans': []}],
                'statistics': {'zero_variance': [False, False], 'invalid_rate': [0., 0.]}}

    def update(model, runner, sample, images, group, optimizer, config):
        optimizer.zero_grad()
        before = model.lora_A['default'].weight.detach().clone()
        loss = (model.lora_A['default'].weight - group['draw']).square().sum()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_([model.lora_A['default'].weight], config.max_grad_norm)
        optimizer.step()
        return {'loss': float(loss.detach()), 'policy_loss': float(loss.detach()), 'kl': 0.,
                'gradient_norm': float(gradient),
                'lora_parameter_delta_l2': float((model.lora_A['default'].weight.detach() - before).norm())}
    monkeypatch.setattr(training, 'rollout_group', rollout)
    monkeypatch.setattr(training, 'update_group', update)


def read_json(path):
    return json.loads(path.read_text())


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_default_32_and_resume_33_match_continuous(config, arguments, validation_predict,
                                                scalar_training, tmp_path):
    pilot, continuous, resumed = [tmp_path / name for name in ('pilot', 'continuous', 'resumed')]
    first = ScalarAdapters()
    result = training.fit(model=first, config=config, output=pilot, **arguments)
    assert result['optimizer_steps_this_run'] == result['last_step'] == 32
    assert (pilot / 'checkpoint_32/training_state.pt').is_file()
    assert (pilot / 'checkpoint_32/unified_state.pt').is_file()
    assert not (pilot / 'update_33.json').exists()
    assert read_json(pilot / 'monitor_32_comparison.json')['continuation'].startswith('user_review_required')
    bigger = replace(config, max_steps=35, evaluation_steps=[0, 32, 34])
    full_model = ScalarAdapters()
    training.fit(model=full_model, config=bigger, output=continuous, **arguments)
    restored = ScalarAdapters()
    restored.load_state_dict(torch.load(pilot / 'checkpoint_32/adapter/fixture.pt', weights_only=True))
    result = training.fit(model=restored, config=bigger, output=resumed, resume=pilot / 'checkpoint_32', **arguments)
    assert result['optimizer_steps_this_run'] == 3
    assert not (resumed / 'update_1.json').exists()
    assert_nested_equal(full_model.state_dict(), restored.state_dict())
    assert restored.lora_A['reference'].weight.item() == .5
    assert not restored.lora_A['reference'].weight.requires_grad
    for step in (33, 34, 35):
        assert read_json(continuous / f'rollout_{step}.json') == read_json(resumed / f'rollout_{step}.json')
    for filename in ('training_state.pt', 'unified_state.pt'):
        full = torch.load(continuous / f'checkpoint_35/{filename}', weights_only=True)
        resumed_state = torch.load(resumed / f'checkpoint_35/{filename}', weights_only=True)
        assert_nested_equal(full, resumed_state)
    monitor = read_json(pilot / 'monitor_tokens.json')
    assert monitor == read_json(resumed / 'monitor_tokens.json')
    assert all(validation_predict[i:i + 2] == monitor for i in range(0, len(validation_predict), 2))
    with pytest.raises(FileExistsError):
        training.fit(model=ScalarAdapters(), config=config, output=pilot, **arguments)


def test_tiny_qwen_real_update_resume(config, arguments, validation_predict, dual_model,
                                       controlled_rollout, monkeypatch, tmp_path):
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    monkeypatch.setattr(training, 'rollout_group', controlled_rollout[0])
    config = replace(config, max_steps=2, evaluation_steps=[0, 1, 2], checkpoint_interval=1)
    full = tmp_path / 'full'
    training.fit(model=dual_model, config=config, output=full, **arguments)
    expected = {n: p.detach().clone() for n, p in dual_model.named_parameters()}
    unloaded = dual_model.unload()
    base = Qwen3VLForConditionalGeneration(deepcopy(unloaded.config))
    base.load_state_dict(unloaded.state_dict())
    checkpoint = full / 'checkpoint_1'
    restored = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=True)
    restored.load_adapter(checkpoint / 'adapter/reference', adapter_name='reference', is_trainable=False)
    resumed = tmp_path / 'resumed'
    training.fit(model=restored, config=config, output=resumed, resume=checkpoint, **arguments)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
        if '.reference.' in name:
            assert not parameter.requires_grad and parameter.grad is None
    assert read_json(full / 'rollout_2.json') == read_json(resumed / 'rollout_2.json')
    assert read_json(resumed / 'update_2.json')['lora_parameter_delta_l2'] > 0
    a = torch.load(full / 'checkpoint_2/training_state.pt', weights_only=True)
    b = torch.load(resumed / 'checkpoint_2/training_state.pt', weights_only=True)
    assert_nested_equal(a, b)


def test_full_intake_never_truncates(records, monkeypatch, tmp_path):
    rows = []
    for index in range(14253):
        row = deepcopy(records['train'][index % len(records['train'])])
        row['sample_token'] = f'train-{index:05}'
        row['historical_sample_tokens'][-1] = row['sample_token']
        rows.append(row)
    reads = []

    def load(*args, split):
        reads.append(split)
        return rows if split == 'train' else records['validation']
    monkeypatch.setattr(drivema_stage3a, 'load_temporal_records', load)
    monkeypatch.setattr(training, 'load_temporal_records', load)
    directory = tmp_path / 'phase_0_4/temporal_waypoint_v0_1'
    directory.mkdir(parents=True)
    for split, values in (('train', rows), ('validation', records['validation'])):
        (directory / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in values))
    train, validation, targets, intake = training.prepare_data(ROOT, tmp_path)
    assert len(train) == intake['train']['eligible_records'] == 14253
    assert len(validation) == len(records['validation'])
    assert reads == ['train', 'validation']
    assert len(targets) == 14253 + len(validation)
    assert intake['test_records_read'] == 0
    assert intake['manifests']['train']['sha256']


def test_validation_rng_and_paired_errors(arguments, validation_predict, tmp_path):
    model = ScalarAdapters()
    before = training.capture_rng()
    rows, metrics = training.evaluate_and_save(model, arguments['runner'], arguments['validation'],
                                              arguments['records'], tmp_path, 'eval')
    assert_nested_equal(before, training.capture_rng())
    assert metrics['trajectory']['ade_3s_m'] == 0
    assert 'per_class' in metrics['action']['longitudinal']
    changed = deepcopy(rows)
    changed[0].update(prediction_valid=False, predicted_waypoints=None, invalid_reason='invalid_trajectory_text',
                      trajectory_projection=None)
    changed[1]['ade_3s_m'], changed[1]['fde_3s_m'] = 3., 6.
    changed[2]['fde_3s_m'] = rows[2]['fde_3s_m'] = None
    comparison = compare_predictions(rows, list(reversed(changed)))
    assert comparison['common_valid_count'] == len(rows) - 1
    assert comparison['paired_errors']['ade_3s_m']['current'] == pytest.approx(3 / (len(rows) - 1))
    assert comparison['paired_errors']['fde_3s_m']['sample_count'] == len(rows) - 2
    assert comparison['diagnostic_signals']['invalid_rate_increased']
    with pytest.raises(ValueError, match='population'):
        compare_predictions(rows, changed[:-1])
    changed[1]['target_waypoints'][0][0] += 1
    with pytest.raises(ValueError, match='target'):
        compare_predictions(rows, changed)
    with pytest.raises(ValueError, match='validation'):
        evaluate_population(model, arguments['runner'], arguments['train'], arguments['records'])


@pytest.mark.parametrize('change', [
    {'learning_rate': 2e-6}, {'beta': .02}, {'monitor_samples': 3}, {'max_steps': 16},
    {'evaluation_steps': [0, 16, 32]}, {'evaluation_steps': [0, 64]},
])
def test_incompatible_resume(config, arguments, change):
    monitor = select_subset(arguments['validation'], config.monitor_samples, arguments['rollout_config'].seed, 'validation')
    identity = training.run_identity(config, arguments['rollout_config'], arguments['train'], monitor,
                                     arguments['runner'], arguments['metadata'])
    state = {'identity': identity, 'config': asdict(config), 'step': 32}
    altered = replace(config, max_steps=64, **{k: v for k, v in change.items() if k != 'max_steps'})
    if 'max_steps' in change:
        altered = replace(altered, max_steps=change['max_steps'])
    new_identity = training.run_identity(altered, arguments['rollout_config'], arguments['train'], monitor,
                                         arguments['runner'], arguments['metadata'])
    with pytest.raises(ValueError):
        training.validate_resume(state, new_identity, altered)


@pytest.mark.parametrize('part', ['train_tokens', 'data_intake', 'rollout', 'initial_sft_checkpoint'])
def test_identity_changes_rejected(config, arguments, part):
    monitor = select_subset(arguments['validation'], 2, arguments['rollout_config'].seed, 'validation')
    identity = training.run_identity(config, arguments['rollout_config'], arguments['train'], monitor,
                                     arguments['runner'], arguments['metadata'])
    changed = deepcopy(identity)
    changed[part] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        training.validate_resume({'identity': identity, 'config': asdict(config), 'step': 32},
                                 changed, replace(config, max_steps=64))


def test_cli_config_and_validation(tmp_path, capsys):
    assert cli.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['training']['max_steps'] == 32
    assert cli.main(['--dry-run', '--max-steps', '256', '--evaluation-steps', '0', '32', '128', '256']) == 0
    assert json.loads(capsys.readouterr().out)['training']['max_steps'] == 256
    values = yaml.safe_load((ROOT / 'configs/drivema_grpo.yaml').read_text())
    for key, value in [('max_steps', 0), ('learning_rate', float('nan')), ('epsilon', 1),
                       ('evaluation_steps', [32, 0]), ('checkpoint_interval', 0)]:
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump({**values, key: value}))
        with pytest.raises(ValueError):
            training.load_config(path)
    with pytest.raises(SystemExit):
        cli.main(['--mode', 'evaluate-full', '--dry-run'])


def test_cli_training_resume_and_both_evaluations(dual_model, controlled_rollout, processor, runner,
                                                 arguments, validation_predict, config, monkeypatch, tmp_path):
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration
    initial = tmp_path / arguments['rollout_config'].checkpoint_relative_path / 'adapter'
    dual_model.save_pretrained(initial, selected_adapters=['default'])
    unloaded = dual_model.unload()

    def load_base(*args):
        model = Qwen3VLForConditionalGeneration(deepcopy(unloaded.config))
        model.load_state_dict(unloaded.state_dict())
        return model
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=load_base,
                              image_loader=lambda _: Image.new('RGB', (64, 64)), package_version=lambda _: 'test')
    monkeypatch.setattr(cli, 'default_runtime_dependencies', lambda: runtime)
    monkeypatch.setattr(cli, 'collect_git_provenance', lambda _: SimpleNamespace(commit='synthetic-test'))
    monkeypatch.setattr(cli, 'Stage2Runner', lambda *a, **kw: runner)
    monkeypatch.setattr(cli, 'prepare_data', lambda *a: (
        arguments['train'], arguments['validation'], arguments['records'], arguments['metadata']['data_intake']))
    monkeypatch.setattr(training, 'rollout_group', controlled_rollout[0])
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(asdict(replace(config, max_steps=1, evaluation_steps=[0, 1]))))
    common = ['--config', str(path), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path)]
    assert cli.main(common) == 0
    checkpoint = tmp_path / config.output_relative_dir / 'checkpoint_1'
    assert cli.main(common + ['--resume', str(checkpoint), '--max-steps', '2',
                              '--evaluation-steps', '0', '1', '2', '--output-relative-dir', 'resumed']) == 0
    assert read_json(tmp_path / 'resumed/summary.json')['optimizer_steps_this_run'] == 1
    assert cli.main(common + ['--mode', 'evaluate-monitor', '--checkpoint', str(checkpoint),
                              '--output-relative-dir', 'monitor-eval']) == 0
    assert cli.main(common + ['--mode', 'evaluate-full', '--checkpoint', str(checkpoint),
                              '--output-relative-dir', 'full-eval']) == 0
    result = read_json(tmp_path / 'full-eval/evaluate-full_1_comparison.json')
    assert result['population_count'] == len(arguments['validation'])
    assert result['common_valid_count'] == len(arguments['validation'])
    assert (tmp_path / 'full-eval/sft_full_predictions.jsonl').exists()
    assert not (tmp_path / 'full-eval/training_history.jsonl').exists()
    assert not (tmp_path / 'monitor-eval/training_history.jsonl').exists()

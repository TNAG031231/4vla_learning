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
from src.phase0.drivema_grpo_evaluation import (
    baseline_provenance, compare_predictions, evaluate_population, load_baseline_predictions,
)
from src.phase0.drivema_grpo_replay import replay_identity, reward_shape_replay
from src.phase0.drivema_grpo_schedules import coverage_schedules
from src.phase0.phase0_4c_full_train import write_predictions
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
        sample = kwargs['sample']
        return {'sample_token': sample.sample_token, 'scene_token': sample.scene_token, 'draw': draw,
                'candidates': [{'rewards': {'action': draw, 'trajectory': .5, 'consistency': .25,
                                           'turn_rewards': [draw, .625],
                                           'trajectory_metrics': {'prediction_valid': True, 'ade_3s_m': .5}},
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


@pytest.fixture
def low512_control(arguments, config):
    sample = arguments['train'][0]
    train = [replace(sample, sample_token=f'train-{i}', scene_token=f'scene-{i % 16}') for i in range(512)]
    arguments['train'] = train
    for row in train:
        arguments['records'][row.sample_token] = arguments['records'][sample.sample_token]
    runner, rollout = arguments['runner'], arguments['rollout_config']
    runner.compute_log_probs = False
    monitor = select_subset(arguments['validation'], config.monitor_samples, rollout.seed, 'validation')
    baseline = training.run_identity(config, rollout, train, monitor, runner, arguments['metadata'])
    original = {'source_archive': 'fixture.zip', 'baseline_identity': baseline, 'steps': [
        {'optimizer_step': i, 'sample_token': row.sample_token, 'scene_token': row.scene_token}
        for i, row in enumerate(train[:256], 1)]}
    low = coverage_schedules(original, train, .5, {'rationale': 'matched fixture'}, 20261116)['repeat256']
    return {'mode': 'evaluate-full', 'step': 512, 'observation_replay': low,
            'identity': replay_identity(baseline, low), 'population_tokens': [v.sample_token for v in arguments['validation']]}


def test_low512_reward_shape_continuation(low512_control, config, arguments, scalar_training,
                                          validation_predict, tmp_path):
    low = low512_control
    original = low['observation_replay']
    cfg = replace(config, max_steps=512, evaluation_steps=[0, 512], checkpoint_interval=512)
    original_metadata = arguments['metadata']
    parent = ScalarAdapters()
    arguments['metadata'] = {**original_metadata, 'observation_replay': original}
    training.fit(model=parent, config=cfg, output=tmp_path / 'low', **arguments)
    checkpoint = tmp_path / 'low/checkpoint_512'
    state = torch.load(checkpoint / 'unified_state.pt', weights_only=True)
    assert state['step'] == 512 and state['identity'] == low['identity']
    assert 'composition' not in state['identity']['rollout']['reward']
    assert 'action_geometry_weight' not in state['identity']['rollout']['reward']
    for arm, shape, weight in [('additive', 'additive', 0.), ('multiplicative', 'multiplicative', 0.),
                               ('feedback', 'additive', 1.38)]:
        replay = reward_shape_replay(low, shape, action_geometry_weight=weight)
        assert replay['steps'][:512] == original['steps']
        assert [r['sample_token'] for r in replay['steps'][512:]] == [
            r['sample_token'] for r in original['steps'] if r['visit'] == 1]
        assert all(r['visit'] == r['total_visits'] == 3 for r in replay['steps'][512:])
        arguments['metadata'] = {**original_metadata, 'observation_replay': replay}
        rollout = arguments['rollout_config']
        arguments['rollout_config'] = replace(rollout, reward=replace(rollout.reward, composition=shape,
                                                                     action_geometry_weight=weight))
        child = deepcopy(parent)
        kwargs = {**arguments, 'low512_identity': low['identity']}
        cfg = replace(cfg, max_steps=513)
        training.fit(model=child, config=cfg, output=tmp_path / arm, resume=checkpoint, **kwargs)
        first = tmp_path / arm / 'checkpoint_513'
        summary = training.fit(model=child, config=replace(cfg, max_steps=768), output=tmp_path / f'{arm}_again',
                               resume=first, **kwargs)
        assert summary['last_step'] == 768 and summary['optimizer_steps_this_run'] == 255
        final_rollout = read_json(tmp_path / f'{arm}_again/rollout_768.json')
        assert final_rollout['sample_token'] == replay['steps'][-1]['sample_token']
        child_state = torch.load(first / 'unified_state.pt', weights_only=True)
        assert child_state['identity'] == replay_identity(original['baseline_identity'], replay)
        assert child_state['identity']['rollout']['reward']['composition'] == shape
        assert child_state['identity']['rollout']['reward'].get('action_geometry_weight', 0.) == weight
        first_rollout = read_json(tmp_path / arm / 'rollout_513.json')
        assert first_rollout['sample_token'] == replay['steps'][512]['sample_token']
        if arm != 'additive':
            with pytest.raises(ValueError, match='identity'):
                training.fit(model=child, config=replace(cfg, max_steps=514), output=tmp_path / 'wrong-arm',
                             resume=tmp_path / 'additive/checkpoint_513', **kwargs)
    with pytest.raises(ValueError, match='original Low-512'):
        training.fit(model=child, config=cfg, output=tmp_path / 'wrong-parent', resume=checkpoint,
                     **{**kwargs, 'low512_identity': original['baseline_identity']})


def test_action_geometry_cli_uses_low512_additive_replay(low512_control, config, tmp_path, capsys):
    (tmp_path / 'metadata.json').write_text(json.dumps(low512_control))
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(asdict(replace(config, max_steps=768))))
    args = ['--config', str(config_path), '--resume', 'checkpoint_512', '--dry-run',
            '--action-geometry-weight', '1.38']
    with pytest.raises(SystemExit):
        cli.main(args)
    matched = args + ['--low512-validation', str(tmp_path), '--reward-shape']
    with pytest.raises(SystemExit):
        cli.main(matched + ['multiplicative'])
    assert cli.main(matched + ['additive']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['rollout']['reward']['action_geometry_weight'] == 1.38
    assert plan['observation_replay']['steps'] == 768


def test_action_geometry_feedback_reaches_optimizer(dual_model, controlled_rollout, runner, samples, records,
                                                   config):
    from src.phase0.phase0_4b_protocol import ActionTarget

    rollout = training.training_rollout_config(ROOT, config)
    rollout = replace(rollout, reward=replace(rollout.reward, action_geometry_weight=1.38))
    sample = replace(samples['train'][0], target=ActionTarget('accelerate', 'right', True, True))
    images = runner.images(sample)
    group = controlled_rollout[0](model=dual_model, processor=runner.processor, sample=sample, images=images,
                                 record=records['train'][0], config=rollout, device='cpu')
    signals = training.group_metrics(group, rollout.reward)
    increments = signals['group_statistics']['action_geometry_feedback']
    assert signals['group_statistics']['zero_variance'][0]
    assert signals['effective_advantage_group'][0] and any(value != 0 for value in increments)

    optimizer = torch.optim.AdamW(dual_model.parameters(), lr=config.learning_rate)
    update = training.update_group(dual_model, runner, sample, images, group, optimizer, config)
    assert update['lora_parameter_delta_l2'] > 0
    for item, feedback in zip(update['candidates'], increments, strict=True):
        assert item['turns'][0]['policy_loss'] == pytest.approx(-feedback, abs=1e-6)


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
    local_model = tmp_path / 'local-qwen'
    local_model.mkdir()

    def load_base(source, revision, dtype, attention, local_only):
        assert source == str(local_model) and local_only
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
    common = ['--config', str(path), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path),
              '--local-model-dir', str(local_model)]
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
    calls = []
    original_evaluate = cli.evaluate_and_save

    def current_only(*args, **kwargs):
        assert kwargs.get('adapter', 'default') == 'default'
        calls.append(len(args[2]))
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(cli, 'evaluate_and_save', current_only)
    saved = tmp_path / 'full-eval/sft_full_predictions.jsonl'
    assert cli.main(common + ['--mode', 'evaluate-full', '--checkpoint', str(checkpoint),
                              '--baseline-predictions', str(saved),
                              '--output-relative-dir', 'reused-eval']) == 0
    assert calls == [len(arguments['validation'])]
    assert read_json(tmp_path / 'reused-eval/evaluate-full_1_comparison.json') == result
    assert read_json(tmp_path / 'reused-eval/baseline_source.json')['predictions'] == str(saved)


@pytest.fixture
def saved_baseline(arguments, validation_predict, tmp_path):
    rows, _ = evaluate_population(ScalarAdapters(), arguments['runner'], arguments['validation'], arguments['records'])
    path = tmp_path / 'final_validation_predictions.jsonl'
    write_predictions(path, rows)
    return path, rows


def test_saved_full_population_and_paired_metrics(saved_baseline, arguments):
    path, source = saved_baseline
    rows, samples, records = [], [], {}
    for index in range(3594):
        token = f'validation-{index}'
        sample = replace(arguments['validation'][index % len(source)], sample_token=token)
        row = deepcopy(source[index % len(source)])
        row['sample_token'] = token
        rows.append(row)
        samples.append(sample)
        records[token] = {'split': 'validation', 'future_waypoints': row['target_waypoints'],
                          'trajectory_valid_mask': row['trajectory_valid_mask']}
    write_predictions(path, list(reversed(rows)))
    loaded = load_baseline_predictions(path, samples, records)
    assert loaded == rows
    current = deepcopy(rows)
    current[0]['ade_3s_m'] = 3.594
    comparison = compare_predictions(loaded, current)
    assert comparison['common_valid_count'] == 3594
    assert comparison['paired_errors']['ade_3s_m']['delta'] == pytest.approx(.001)
    assert comparison['baseline']['action']['longitudinal_macro_f1'] is not None
    assert comparison['baseline']['behavior']['trajectory_parsing_success_rate'] == 1


@pytest.mark.parametrize('field', ['sample_token', 'scene_token', 'split', 'target',
                                    'target_waypoints', 'trajectory_valid_mask', 'duplicate', 'missing'])
def test_saved_baseline_rejects_population_and_gt_errors(saved_baseline, arguments, field):
    path, rows = saved_baseline
    if field == 'duplicate':
        rows.append(rows[0])
    elif field == 'missing':
        rows.pop()
    else:
        rows[0][field] = 'wrong'
    write_predictions(path, rows)
    with pytest.raises(ValueError):
        load_baseline_predictions(path, arguments['validation'], arguments['records'])


def test_saved_baseline_rejects_malformed_json(saved_baseline, arguments):
    path, _ = saved_baseline
    path.write_text('{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        load_baseline_predictions(path, arguments['validation'], arguments['records'])


@pytest.mark.parametrize('mismatch', [None, 'model_id', 'revision', 'trajectory_max_new_tokens',
                                      'batch_size', 'optimized', 'action_generation', 'checkpoint'])
def test_available_sft_provenance(saved_baseline, mismatch):
    path, _ = saved_baseline
    checkpoint = 'drivema/stage2b_full_run01/checkpoint_3564'
    inference = {'trajectory_max_new_tokens': 192, 'batch_size': 1, 'optimized': True,
                 'action_generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 24}}
    missing = baseline_provenance(path, checkpoint, 'model', 'revision', inference)
    assert set(missing['unavailable']) == {'model_id', 'revision', 'trajectory_max_new_tokens',
                                          'batch_size', 'optimized', 'action_generation', 'checkpoint'}
    metadata = {'provenance': {'model_id': 'model', 'revision': 'revision'},
                'config': {'trajectory_max_new_tokens': 192},
                'inference': {key: value for key, value in inference.items() if key != 'trajectory_max_new_tokens'}}
    summary = {'best': {'checkpoint': '/old-derived-root/' + checkpoint, 'step': 3564}}
    for section in metadata.values():
        if mismatch in section:
            section[mismatch] = 'wrong'
    if mismatch == 'checkpoint':
        summary['best']['checkpoint'] = '/old-derived-root/drivema/other_run/checkpoint_3564'
    (path.parent / 'metadata.json').write_text(json.dumps(metadata))
    (path.parent / 'training_summary.json').write_text(json.dumps(summary))
    if mismatch:
        with pytest.raises(ValueError, match='provenance'):
            baseline_provenance(path, checkpoint, 'model', 'revision', inference)
    else:
        checked = baseline_provenance(path, checkpoint, 'model', 'revision', inference)
        assert checked['unavailable'] == []
        assert checked['checked']['checkpoint'] == summary['best']['checkpoint']


@pytest.mark.parametrize('mode', ['train', 'evaluate-monitor'])
def test_baseline_argument_only_full(mode):
    with pytest.raises(SystemExit):
        cli.main(['--mode', mode, '--baseline-predictions', 'baseline.jsonl', '--dry-run'])

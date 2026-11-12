from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_drivema_stage3a as cli
from src.phase0 import drivema_stage3a as rollout
from src.phase0.drivema_stage2_protocol import assistant_log_probs, serialize_trajectory
from src.phase0.drivema_stage3_rewards import candidate_rewards, assign_advantages, summarize_groups
from src.phase0.phase0_4b_protocol import ActionTarget
from test_drivema_stage2_sft import case, records, samples, processor, runner, tiny_qwen, lora_model


@pytest.fixture
def config():
    return rollout.load_config(ROOT / 'configs/drivema_stage3a.yaml')


def reward_candidate(action, points):
    return {'parsed_action': action, 'waypoints': points, 'turn_complete': [True, points is not None]}


@pytest.mark.parametrize('action,expected', [
    ({'longitudinal': 'keep', 'lateral': 'straight'}, 1.),
    ({'longitudinal': 'stop', 'lateral': 'straight'}, .5),
    ({'longitudinal': 'stop', 'lateral': 'left'}, 0.), (None, 0.),
])
def test_rewards_correct_partial_wrong_invalid(config, action, expected):
    points = [[float(i), 0.] for i in range(1, 7)]
    c = reward_candidate(action, points)
    result = candidate_rewards(c, ActionTarget('keep', 'straight', True, True), points, [True] * 6,
                               config.reward)
    assert result['action'] == expected
    assert result['trajectory'] == 1
    assert result['consistency'] == expected
    assert result['turn_rewards'] == [expected, 1 + .5 * expected]
    result = candidate_rewards(reward_candidate(action, None), ActionTarget('keep', 'straight', True, True),
                               points, [True] * 6, config.reward)
    assert result['trajectory'] == result['consistency'] == 0


def test_partial_gt_and_missing_endpoint(config):
    points = [[float(i), 0.] for i in range(1, 7)]
    predicted = [[x + 1, y] for x, y in points]
    predicted[-1][0] += 50
    c = reward_candidate({'longitudinal': 'stop', 'lateral': 'straight'}, predicted)
    result = candidate_rewards(c, ActionTarget(None, 'straight', False, True), points,
                               [True] * 5 + [False], config.reward)
    assert result['action'] == 1
    assert result['trajectory'] == pytest.approx(math.exp(-1))
    assert result['trajectory_metrics']['fde_3s_m'] is None
    assert result['trajectory_rule'] == 'masked_ADE_only_missing_3s_endpoint'
    full = candidate_rewards(c, ActionTarget('keep', None, True, False), points, [True] * 6, config.reward)
    assert full['action'] == 0
    assert full['error_m'] == pytest.approx(.7 * (56 / 6) + .3 * 51)
    with pytest.raises(ValueError, match='supervised'):
        candidate_rewards(c, ActionTarget(None, None, False, False), points, [True] * 6, config.reward)
    c['turn_complete'][0] = False
    assert candidate_rewards(c, ActionTarget('stop', None, True, False), points, [True] * 6,
                             config.reward)['action'] == 0


def test_independent_advantages_zero_variance_and_missing_turn(config):
    candidates = [{'rewards': {'turn_rewards': [a, b]}, 'turn_valid': [True, i != 3],
                   'turn_mask': [True, i != 3]} for i, (a, b) in enumerate(zip([1, .5, 0, 0], [0, 1, .5, 0]))]
    result = assign_advantages(candidates, config.reward.advantage_eps)
    expected = torch.tensor([[1., 0.], [.5, 1.], [0., .5], [0., 0.]], dtype=torch.float64)
    expected = (expected - expected.mean(0)) / (expected.std(0) + config.reward.advantage_eps)
    torch.testing.assert_close(torch.tensor([c['turn_advantages'] for c in candidates], dtype=torch.float64), expected)
    assert result['invalid_rate'] == result['missing_turn_rate'] == [0, .25]
    assert candidates[3]['turn_advantages'][1] < 0
    for c in candidates:
        c['rewards']['turn_rewards'] = [1., 0.]
    result = assign_advantages(candidates, config.reward.advantage_eps)
    assert result['zero_variance'] == [True, True]
    assert all(c['turn_advantages'] == [0, 0] for c in candidates)
    summary = summarize_groups([{'candidates': candidates, 'statistics': result}])
    assert summary['zero_variance_group_rate'] == [1, 1]


def scripted_candidates(processor, responses, calls):
    pending = iter(responses)

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs['use_model_defaults'] is False
        calls.append(kwargs['input_ids'][0].tolist())
        text, eos = next(pending)
        tokens = processor.tokenizer(text, add_special_tokens=False)['input_ids']
        if eos:
            tokens.append(processor.tokenizer.eos_token_id)
        sequence = torch.cat((kwargs['input_ids'], torch.tensor([tokens])), dim=1)
        score = torch.zeros(1, self.config.text_config.vocab_size)
        return SimpleNamespace(sequences=sequence, scores=tuple(score for _ in tokens))
    return generate


@pytest.mark.parametrize('index', [0, 3])
def test_four_candidate_context_spans_and_policy_contract(config, processor, lora_model, runner,
                                                         samples, records, monkeypatch, tmp_path, index):
    from transformers import Qwen3VLForConditionalGeneration
    points = [[float(i), 0.] for i in range(1, 7)]
    actions = ['longitudinal=keep; lateral=straight', 'longitudinal=stop; lateral=straight',
               'longitudinal=keep; lateral=left', 'bad action']
    trajectories = [serialize_trajectory(points), serialize_trajectory([[0., 0.]] * 6), 'bad trajectory']
    responses = [(actions[0], True), (trajectories[0], True), (actions[1], True), (trajectories[1], True),
                 (actions[2], True), (trajectories[2], False), (actions[3], True)]
    calls = []
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', scripted_candidates(processor, responses, calls))
    sample = replace(samples['train'][index], target=ActionTarget('decelerate', 'right', True, True))
    record = deepcopy(records['train'][index])
    record['future_waypoints'] = [[98765., -45678.]] * 6
    images = runner.images(sample)
    group = rollout.rollout_group(model=lora_model, processor=processor, sample=sample, images=images,
                                  record=record, config=config, device='cpu')
    assert len(group['candidates']) == 4
    assert len({c['sample_token'] for c in group['candidates']}) == 1
    assert len({tuple(c['assistant_spans'][0]['token_ids']) for c in group['candidates']}) == 4
    assert len(calls) == 7
    for c in group['candidates']:
        inputs, spans = rollout.replay_inputs(processor, sample.observation, images, c, 'cpu')
        assert inputs['image_grid_thw'].shape[0] == (1 if index == 0 else 3)
        for saved in c['assistant_spans']:
            assert saved['prefix_ids'] in calls
            assert saved['token_ids'] == c['input_ids'][saved['start']:saved['end']]
            assert len(saved['old_token_log_probs']) == saved['end'] - saved['start']
            assert saved['eos'] == (saved['token_ids'][-1] == processor.tokenizer.eos_token_id)
        targets = rollout.policy_targets(c, 'cpu')
        current = torch.cat(assistant_log_probs(lora_model.eval(), inputs, spans))
        reference = torch.cat(assistant_log_probs(lora_model, inputs, spans)).detach()
        assert current.shape == reference.shape == targets['old_log_probs'].shape == targets['advantages'].shape
        assert current.requires_grad
        assert not targets['old_log_probs'].requires_grad and not targets['advantages'].requires_grad
        assert torch.isfinite((current - targets['old_log_probs']).exp()).all()
        assert torch.equal(targets['logit_positions'], targets['target_positions'] - 1)
        assert targets['loss_mask'].all()
        assert set(targets['turn_ids'].tolist()) == ({1, 2} if c['turn_mask'][1] else {1})
        if c['turn_mask'][1]:
            first, second = c['assistant_spans']
            assert second['prefix_ids'][first['start']:first['end']] == first['token_ids']
    for call in calls:
        text = processor.tokenizer.decode(call)
        assert sample.target.training_text() not in text
        assert '98765' not in text and '45678' not in text
    assert group['candidates'][3]['turn_mask'] == [True, False]
    assert group['candidates'][3]['trajectory_text'] is None
    assert group['candidates'][2]['parser_status'][1] == 'length'
    assert group['statistics']['invalid_rate'] == [.25, .5]
    path = tmp_path / 'group.json'
    path.write_text(json.dumps(group, allow_nan=False))
    assert json.loads(path.read_text())['candidates'] == group['candidates']
    assert '| 3 | bad action | None |' in rollout.group_markdown(group)
    tampered = deepcopy(group['candidates'][0])
    tampered['assistant_spans'][0]['token_ids'][0] = 0
    with pytest.raises(ValueError, match='span mismatch'):
        rollout.replay_inputs(processor, sample.observation, images, tampered, 'cpu')


@pytest.mark.parametrize('index', [0, 3])
def test_real_sampling_raw_distribution_replay_and_no_updates(config, processor, lora_model, runner,
                                                           samples, records, monkeypatch, index):
    from transformers import Qwen3VLForConditionalGeneration
    config = replace(config, action_max_new_tokens=3, trajectory_max_new_tokens=3)
    seen_processors = []
    original = Qwen3VLForConditionalGeneration._get_logits_processor

    def inspect(self, *args, **kwargs):
        processors = original(self, *args, **kwargs)
        seen_processors.append([type(p).__name__ for p in processors])
        return processors
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, '_get_logits_processor', inspect)
    lora_model.generation_config.top_k = 1
    lora_model.generation_config.repetition_penalty = 2.0
    before = {n: p.detach().clone() for n, p in lora_model.named_parameters()}
    sample = samples['train'][index]
    arguments = dict(model=lora_model, processor=processor, sample=sample, images=runner.images(sample),
                     record=records['train'][index], config=config, device='cpu')
    torch.manual_seed(config.seed)
    group = rollout.rollout_group(**arguments)
    torch.manual_seed(config.seed)
    assert rollout.rollout_group(**arguments) == group
    assert seen_processors and all(not processors for processors in seen_processors)
    assert len({tuple(c['assistant_spans'][0]['token_ids']) for c in group['candidates']}) > 1
    for c in group['candidates']:
        inputs, spans = rollout.replay_inputs(processor, sample.observation, runner.images(sample), c, 'cpu')
        current = torch.cat(assistant_log_probs(lora_model, inputs, spans))
        old = rollout.policy_targets(c, 'cpu')['old_log_probs']
        torch.testing.assert_close(current.detach(), old, atol=2e-5, rtol=1e-5)
        assert not c['turn_valid'][0] and c['rewards']['turn_rewards'] == [0, 0]
    assert all(torch.equal(p, before[n]) and p.grad is None for n, p in lora_model.named_parameters())


def test_train_intake_partial_targets_and_split(config, records, monkeypatch):
    rows = deepcopy(records['train'])
    rows[0].update(longitudinal_action=None, longitudinal_action_valid=False, factorized_action_joint_valid=False)
    rows[1].update(longitudinal_action=None, longitudinal_action_valid=False, lateral_action=None,
                   lateral_action_valid=False, factorized_action_joint_valid=False)
    splits = []

    def read(*args, split):
        splits.append(split)
        return rows
    monkeypatch.setattr(rollout, 'load_temporal_records', read)
    selected, _, intake = rollout.prepare_train(ROOT, Path('unused'))
    assert len(selected) == 3 and intake['excluded_no_action_supervision'] == 1
    assert splits == ['train'] and intake['test_records_read'] == 0
    assert any(not s.target.longitudinal_valid for s in selected)
    rows[0]['split'] = 'validation'
    with pytest.raises(ValueError, match='split'):
        rollout.prepare_train(ROOT, Path('unused'))


def test_real_second_turn_sampling_after_controlled_action(config, processor, lora_model, runner,
                                                         samples, records, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    original = Qwen3VLForConditionalGeneration.generate
    calls = []
    action = 'longitudinal=keep; lateral=straight'
    first = scripted_candidates(processor, [(action, True)] * 4, [])
    config = replace(config, trajectory_max_new_tokens=3)

    def mixed(self, **kwargs):
        if kwargs['generation_config'].max_new_tokens == config.action_max_new_tokens:
            return first(self, **kwargs)
        calls.append(kwargs['input_ids'][0].tolist())
        return original(self, **kwargs)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', mixed)
    sample = samples['train'][3]
    images = runner.images(sample)
    torch.manual_seed(config.seed)
    group = rollout.rollout_group(model=lora_model, processor=processor, sample=sample, images=images,
                                  record=records['train'][3], config=config, device='cpu')
    assert len(calls) == 4
    for c, prefix in zip(group['candidates'], calls, strict=True):
        assert c['turn_mask'] == [True, True]
        assert c['assistant_spans'][1]['prefix_ids'] == prefix
        inputs, spans = rollout.replay_inputs(processor, sample.observation, images, c, 'cpu')
        current = assistant_log_probs(lora_model, inputs, spans)[1].detach()
        old = torch.tensor(c['assistant_spans'][1]['old_token_log_probs'])
        torch.testing.assert_close(current, old, atol=2e-5, rtol=1e-5)
        assert c['waypoints'] is None
        assert c['rewards']['trajectory'] == c['rewards']['consistency'] == 0


def test_readable_four_candidate_example(config, processor, lora_model, runner, samples, records,
                                        monkeypatch, capsys):
    from transformers import Qwen3VLForConditionalGeneration
    points = [[float(i), 0.] for i in range(1, 7)]
    responses = [('longitudinal=keep; lateral=straight', True), (serialize_trajectory(points), True),
                 ('longitudinal=stop; lateral=straight', True), (serialize_trajectory([[0., 0.]] * 6), True),
                 ('longitudinal=keep; lateral=left', True), (serialize_trajectory(points), True),
                 ('invalid', True)]
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', scripted_candidates(processor, responses, []))
    sample = replace(samples['train'][0], target=ActionTarget('keep', 'straight', True, True))
    record = {**records['train'][0], 'future_waypoints': points}
    group = rollout.rollout_group(model=lora_model, processor=processor, sample=sample, images=runner.images(sample),
                                  record=record, config=config, device='cpu')
    assert [c['rewards']['action'] for c in group['candidates']] == [1, .5, .5, 0]
    assert [c['rewards']['consistency'] for c in group['candidates']] == [1, 1, .5, 0]
    assert group['candidates'][1]['rewards']['trajectory'] == pytest.approx(math.exp(-4.25))
    with capsys.disabled():
        print('\nControlled synthetic example (not real sampled 4B output):\n' + rollout.group_markdown(group))


def test_cli_real_sampling_persistence(config, records, processor, lora_model, monkeypatch, tmp_path, capsys):
    from PIL import Image
    from peft import PeftModel
    assert cli.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['top_k'] == 0
    config = replace(config, action_max_new_tokens=2, trajectory_max_new_tokens=2, train_samples=1)
    path = tmp_path / 'config.yaml'
    from dataclasses import asdict
    path.write_text(yaml.safe_dump(asdict(config)))
    monkeypatch.setattr(rollout, 'load_temporal_records', lambda *args, split: records[split])
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=lambda *a: lora_model,
                              image_loader=lambda path: Image.new('RGB', (64, 64)), package_version=lambda _: 'test')
    monkeypatch.setattr(cli, 'default_runtime_dependencies', lambda: runtime)
    monkeypatch.setattr(cli, 'collect_git_provenance', lambda _: SimpleNamespace(commit='synthetic-test'))
    loaded = []

    def load(base, path, is_trainable):
        loaded.append((path, is_trainable))
        return base
    monkeypatch.setattr(PeftModel, 'from_pretrained', load)
    args = ['--config', str(path), '--dataset-root', str(tmp_path), '--derived-root', str(tmp_path)]
    assert cli.main(args) == 0
    output = tmp_path / config.output_relative_dir
    metadata = json.loads((output / 'metadata.json').read_text())
    group = json.loads((output / 'groups.jsonl').read_text())
    assert loaded == [(tmp_path / config.checkpoint_relative_path / 'adapter', False)]
    assert metadata['optimizer_updates'] == 0 and metadata['data_intake']['test_records_read'] == 0
    assert len(group['candidates']) == 4
    assert json.loads((output / 'summary.json').read_text())['candidate_count'] == 4
    assert (output / 'example.md').read_text() == rollout.group_markdown(group)
    assert metadata['generation_config'][0]['top_k'] == 0
    with pytest.raises(ValueError, match='new directory'):
        cli.main(args)


@pytest.mark.parametrize('key,value', [('top_k', 50), ('temperature', .7), ('group_size', 1),
                                      ('checkpoint_relative_path', 'wrong'), ('train_samples', 0)])
def test_config_rejects_changed_sampling_contract(key, value, tmp_path):
    values = yaml.safe_load((ROOT / 'configs/drivema_stage3a.yaml').read_text())
    values[key] = value
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError):
        rollout.load_config(path)

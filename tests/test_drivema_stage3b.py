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

from scripts import run_drivema_stage3b as cli
from src.phase0 import drivema_stage3b as training
from src.phase0.drivema_stage2_protocol import assistant_log_probs, serialize_trajectory
from src.phase0.drivema_stage3a import load_config, policy_targets, replay_inputs, rollout_group
from src.phase0.phase0_4b_protocol import ActionTarget
from test_drivema_stage2_sft import case, records, samples, processor, runner, tiny_qwen, lora_model
from test_drivema_stage3a import scripted_candidates


def targets(old, advantages, turns, mask=None):
    return {'old_log_probs': torch.tensor(old, requires_grad=True),
            'advantages': torch.tensor(advantages, requires_grad=True),
            'turn_ids': torch.tensor(turns),
            'loss_mask': torch.tensor(mask if mask is not None else [True] * len(turns))}


@pytest.mark.parametrize('advantage,ratio,expected,gradient', [
    (1., 1., -1., -1.), (-1., 1., 1., 1.),
    (1., 1.4, -1.2, 0.), (-1., .6, .8, 0.),
    (1., .6, -.6, -.6), (-1., 1.4, 1.4, 1.4),
])
def test_clipped_value_direction_and_detach(advantage, ratio, expected, gradient):
    current = torch.tensor([ratio]).log().requires_grad_()
    reference = current.detach().clone().requires_grad_()
    target = targets([0.], [advantage], [1])
    loss, _ = training.grpo_loss(current, reference, target, .2, .01)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert current.grad.item() == pytest.approx(gradient)
    assert reference.grad is target['old_log_probs'].grad is target['advantages'].grad is None


def test_kl_reduction_mask_and_zero_advantage():
    current = torch.zeros(4, requires_grad=True)
    ref = torch.tensor([.5, .5, -.5, 10.])
    target = targets([0.] * 4, [0.] * 4, [1, 1, 2, 2], [True, True, True, False])
    loss, metrics = training.grpo_loss(current, ref, target, .2, .01)
    expected = ((torch.expm1(ref[0]) - ref[0]) + (torch.expm1(ref[2]) - ref[2])) / 2
    torch.testing.assert_close(loss, .01 * expected)
    assert metrics['policy_loss'] == 0 and metrics['kl'] > 0
    assert [t['tokens'] for t in metrics['turns']] == [2, 1]
    loss.backward()
    assert current.grad[0] < 0 and current.grad[2] > 0 and current.grad[3] == 0
    target = targets([0.] * 4, [1., 1., -1., 100.], [1, 1, 2, 2], [True, True, True, False])
    loss, _ = training.grpo_loss(current, current.detach(), target, .2, 0.)
    assert loss.item() == 0  # Equal turn weights, despite unequal token counts.
    target['loss_mask'][2] = False
    loss, metrics = training.grpo_loss(current, current.detach(), target, .2, 0.)
    assert loss.item() == -1 and len(metrics['turns']) == 1


def test_zero_advantage_identity_and_nonfinite():
    current = torch.zeros(2, requires_grad=True)
    target = targets([0., 0.], [0., 0.], [1, 1])
    loss, metrics = training.grpo_loss(current, current, target, .2, .01)
    loss.backward()
    assert loss.item() == metrics['kl'] == 0 and current.grad.eq(0).all()
    with pytest.raises(FloatingPointError):
        training.grpo_loss(current + 1000, current, target, .2, .01)


@pytest.fixture
def dual_model(lora_model, tmp_path):
    # A nonzero synthetic SFT adapter makes an accidental base reference observable.
    with torch.no_grad():
        for name, p in lora_model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.02)
    lora_model.save_pretrained(tmp_path / 'sft')
    lora_model.load_adapter(tmp_path / 'sft', adapter_name='reference', is_trainable=False)
    training.activate_policy(lora_model, 'default')
    return lora_model


@pytest.fixture
def controlled_rollout(monkeypatch, processor):
    from transformers import Qwen3VLForConditionalGeneration
    original_generate = Qwen3VLForConditionalGeneration.generate
    calls = []

    def produce(**kwargs):
        responses = [('longitudinal=keep; lateral=straight', True),
                     (serialize_trajectory([[float(i), 0.] for i in range(1, 7)]), True),
                     ('longitudinal=stop; lateral=straight', True),
                     (serialize_trajectory([[0., 0.]] * 6), True),
                     ('longitudinal=keep; lateral=left', True), ('bad trajectory', True),
                     ('bad action', True)]
        monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate',
                            scripted_candidates(processor, responses, []))
        group = rollout_group(**kwargs)
        monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', original_generate)
        # Controlled text fixture with true behavior probabilities from the same policy.
        with torch.no_grad():
            for c in group['candidates']:
                batch, spans = replay_inputs(processor, kwargs['sample'].observation,
                                              kwargs['images'], c, 'cpu')
                for span, probs in zip(c['assistant_spans'], assistant_log_probs(kwargs['model'], batch, spans)):
                    span['old_token_log_probs'] = probs.tolist()
        calls.append(deepcopy(group))
        return group
    monkeypatch.setattr(training, 'rollout_group', produce)
    return produce, calls


def test_accumulation_matches_joint_backward_and_freezes_reference(dual_model, controlled_rollout,
                                                                  runner, samples, records):
    model = dual_model
    config = training.load_config(ROOT / 'configs/drivema_stage3b_smoke.yaml')
    config = replace(config, learning_rate=1e-3, max_grad_norm=100.)
    rollout = load_config(ROOT / 'configs/drivema_stage3a.yaml')
    sample = replace(samples['train'][0], target=ActionTarget('keep', 'straight', True, True))
    images = runner.images(sample)
    group = controlled_rollout[0](model=model, processor=runner.processor, sample=sample, images=images,
                                 record=records['train'][0], config=rollout, device='cpu')
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = []
    for c in group['candidates']:
        batch, spans = replay_inputs(runner.processor, sample.observation, images, c, 'cpu')
        training.activate_policy(model, 'reference')
        with torch.no_grad():
            ref = torch.cat(assistant_log_probs(model, batch, spans))
        training.activate_policy(model, 'default', backward=True)
        current = torch.cat(assistant_log_probs(model, batch, spans))
        torch.testing.assert_close(current, ref)
        losses.append(training.grpo_loss(current, ref, policy_targets(c, 'cpu'), .2, .01)[0])
    torch.stack(losses).mean().backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    result = training.update_group(model, runner, sample, images, group, optimizer, config)
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], atol=2e-6, rtol=1e-4)
        else:
            assert torch.equal(p, before[name]) and p.grad is None
    assert result['lora_parameter_delta_l2'] > 0 and result['gradient_norm'] > 0
    assert result['kl'] == pytest.approx(0., abs=1e-7)
    assert max(c['current_old_abs_max'] for c in result['candidates']) < 1e-4
    assert len(result['candidates'][-1]['turns']) == 1
    assert result['candidates'][2]['turns'][1]['tokens'] > 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if '.reference.' in n)


def test_real_sampling_fresh_online_loop_save_reload(dual_model, runner, samples, records, tmp_path):
    rollout = replace(load_config(ROOT / 'configs/drivema_stage3a.yaml'),
                      action_max_new_tokens=2, trajectory_max_new_tokens=2)
    config = training.load_config(ROOT / 'configs/drivema_stage3b_smoke.yaml')
    output = tmp_path / 'real'
    summary = training.fit(model=dual_model, runner=runner, samples=samples['train'],
                           records={r['sample_token']: r for r in records['train']}, rollout_config=rollout,
                           config=config, output=output, metadata={'model_id': 'tiny', 'revision': 'random'})
    assert summary['optimizer_steps_this_run'] == 2
    first = json.loads((output / 'rollout_1.json').read_text())
    second = json.loads((output / 'rollout_2.json').read_text())
    assert first['optimizer_step'] == 1 and second['optimizer_step'] == 2
    assert first['candidates'][0]['input_ids'] != second['candidates'][0]['input_ids']
    diagnostic = json.loads((output / 'update_1.json').read_text())
    assert max(c['current_old_abs_max'] for c in diagnostic['candidates']) < 1e-4
    assert diagnostic['kl'] == pytest.approx(0., abs=1e-7)
    assert (output / 'checkpoint_2/adapter/reference/adapter_model.safetensors').exists()


def test_controlled_online_resume_matches_uninterrupted(dual_model, controlled_rollout, runner,
                                                       samples, records, tmp_path):
    from peft import PeftModel
    config = replace(training.load_config(ROOT / 'configs/drivema_stage3b_smoke.yaml'), learning_rate=1e-3)
    rollout = load_config(ROOT / 'configs/drivema_stage3a.yaml')
    arguments = dict(runner=runner, samples=samples['train'], records={r['sample_token']: r for r in records['train']},
                     rollout_config=rollout, metadata={'model_id': 'tiny', 'revision': 'random'})
    output = tmp_path / 'full'
    training.fit(model=dual_model, config=config, output=output, **arguments)
    expected = {n: p.detach().clone() for n, p in dual_model.named_parameters()}
    checkpoint = output / 'checkpoint_1'
    from transformers import Qwen3VLForConditionalGeneration
    unloaded = dual_model.unload()
    base = Qwen3VLForConditionalGeneration(deepcopy(unloaded.config))
    base.load_state_dict(unloaded.state_dict())
    restored = PeftModel.from_pretrained(base, checkpoint / 'adapter', is_trainable=True)
    restored.load_adapter(checkpoint / 'adapter/reference', adapter_name='reference', is_trainable=False)
    result = training.fit(model=restored, config=config, output=tmp_path / 'resumed', resume=checkpoint, **arguments)
    assert result['optimizer_steps_this_run'] == 1
    for n, p in restored.named_parameters():
        torch.testing.assert_close(p, expected[n], atol=0, rtol=0)
    assert controlled_rollout[1][1] == controlled_rollout[1][2]
    full = json.loads((output / 'update_2.json').read_text())
    resumed = json.loads((tmp_path / 'resumed/update_2.json').read_text())
    assert full['candidates'] == resumed['candidates']
    assert full['lora_parameter_delta_l2'] > 0


def test_cli_config_and_limits(tmp_path, capsys):
    import yaml
    assert cli.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['training']['max_steps'] == 2
    values = yaml.safe_load((ROOT / 'configs/drivema_stage3b_smoke.yaml').read_text())
    for key, value in [('max_steps', 3), ('beta', -1), ('epsilon', 0), ('learning_rate', float('nan'))]:
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump({**values, key: value}))
        with pytest.raises(ValueError):
            training.load_config(path)


def test_cli_loads_sft_twice_and_persists_training(dual_model, controlled_rollout, samples, records,
                                                 processor, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration
    rollout = load_config(ROOT / 'configs/drivema_stage3a.yaml')
    sft = tmp_path / rollout.checkpoint_relative_path / 'adapter'
    dual_model.save_pretrained(sft, selected_adapters=['default'])
    unloaded = dual_model.unload()
    base = Qwen3VLForConditionalGeneration(deepcopy(unloaded.config))
    base.load_state_dict(unloaded.state_dict())
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=lambda *a: base,
                              image_loader=lambda _: Image.new('RGB', (64, 64)), package_version=lambda _: 'test')
    monkeypatch.setattr(cli, 'default_runtime_dependencies', lambda: runtime)
    monkeypatch.setattr(cli, 'collect_git_provenance', lambda _: SimpleNamespace(commit='synthetic-test'))
    monkeypatch.setattr(cli, 'prepare_train', lambda *a: (
        samples['train'], {r['sample_token']: r for r in records['train']}, {'test_records_read': 0}))
    arguments = ['--dataset-root', str(tmp_path), '--derived-root', str(tmp_path)]
    assert cli.main(arguments) == 0
    output = tmp_path / 'drivema/stage3b_smoke_run01'
    update = json.loads((output / 'update_1.json').read_text())
    assert update['kl'] == pytest.approx(0., abs=1e-7)
    assert update['lora_parameter_delta_l2'] > 0
    assert all(c['current_reference_abs_max'] < 1e-4 for c in update['candidates'])
    assert all(all(c['attention_mask']) for c in update['candidates'])
    assert update['turn_token_counts'][1] > 0
    assert set(update['reward_means']) == {'action', 'trajectory', 'consistency'}
    with pytest.raises(ValueError, match='new directory'):
        cli.main(arguments)

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import diagnose_drivema_stage3b_gradients as cli
from scripts import run_drivema_stage3b as training_cli
from src.phase0.drivema_stage2_protocol import assistant_log_probs
from src.phase0.drivema_stage3a import load_config, policy_targets, replay_inputs
from src.phase0.drivema_stage3b import activate_policy, load_config as load_grpo_config
from src.phase0.drivema_stage3b_gradient_diagnostic import compare_gradients, diagnose_gradients
from test_drivema_stage2_sft import case, records, samples, processor, runner, tiny_qwen, lora_model
from test_drivema_stage3b import dual_model, controlled_rollout


@pytest.fixture
def saved_group(dual_model, controlled_rollout, runner, samples, records):
    sample = samples['train'][0]
    group = controlled_rollout[0](model=dual_model, processor=runner.processor, sample=sample,
                                 images=runner.images(sample), record=records['train'][0],
                                 config=load_config(ROOT / 'configs/drivema_stage3a.yaml'), device='cpu')
    group['optimizer_step'] = 1
    return json.loads(json.dumps(group))


def test_gradient_comparison_direction_and_zero_norm():
    a = torch.tensor([1., 0.])
    assert compare_gradients(a, -a)['cosine_similarity'] == -1
    assert compare_gradients(a, -a)['relative_l2_difference'] == 2
    assert compare_gradients(a, a.flip(0))['angle_degrees'] == 90
    result = compare_gradients(a * 0, a)
    assert result['relative_l2_difference'] is result['cosine_similarity'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('drift', [0., .11])
def test_fixed_group_gradient_scale_and_no_weight_update(dual_model, saved_group, runner,
                                                        samples, monkeypatch, drift):
    from transformers import Qwen3VLForConditionalGeneration

    def forbidden(*args, **kwargs):
        pytest.fail('diagnostic must not sample, optimize or save model weights')

    monkeypatch.setattr(Qwen3VLForConditionalGeneration, 'generate', forbidden)
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    monkeypatch.setattr(dual_model, 'save_pretrained', forbidden)
    for candidate in saved_group['candidates']:
        for span in candidate['assistant_spans']:
            span['old_token_log_probs'] = [p + drift for p in span['old_token_log_probs']]
    before = {n: p.detach().clone() for n, p in dual_model.named_parameters()}
    original = deepcopy(saved_group)
    sample = samples['train'][0]
    result = diagnose_gradients(dual_model, runner, sample, runner.images(sample), saved_group,
                                load_grpo_config(ROOT / 'configs/drivema_stage3b_smoke.yaml'))
    comparison = result['comparison']
    scale = torch.exp(torch.tensor(-drift)).item()
    assert comparison['norm_a'] > 0
    assert comparison['norm_b'] / comparison['norm_a'] == pytest.approx(scale, abs=2e-5)
    assert comparison['cosine_similarity'] == pytest.approx(1., abs=1e-8)
    assert comparison['relative_l2_difference'] == pytest.approx(1 - scale, abs=2e-5)
    assert result['surrogates']['A']['kl'] == result['surrogates']['B']['kl']
    assert all(result['checks'].values())
    assert saved_group == original
    for name, parameter in dual_model.named_parameters():
        assert torch.equal(parameter, before[name]) and parameter.grad is None
        assert parameter.requires_grad == ('lora_' in name and '.default.' in name)
    for candidate, audit in zip(saved_group['candidates'], result['candidates'], strict=True):
        positions = [p for span in candidate['assistant_spans'] for p in range(span['start'], span['end'])]
        assert audit['target_positions'] == positions
        assert audit['logit_positions'] == [p - 1 for p in positions]
        assert all(audit['loss_mask']) and all(audit['attention_mask'])
        assert audit['assistant_spans'] == candidate['assistant_spans']
    assert len(result['candidates'][-1]['surrogates']['A']['turns']) == 1
    assert len(result['candidates'][2]['surrogates']['A']['turns']) == 2  # Invalid text retains credit.
    json.dumps(result, allow_nan=False)
    if drift == 0:
        # At identical SFT probabilities, ratio=1 and KL gradient=0: direct -A*logp is an oracle.
        activate_policy(dual_model, 'default', backward=True)
        losses = []
        for candidate in saved_group['candidates']:
            batch, spans = replay_inputs(runner.processor, sample.observation, runner.images(sample),
                                          candidate, 'cpu')
            current = torch.cat(assistant_log_probs(dual_model, batch, spans))
            targets = policy_targets(candidate, 'cpu')
            turns = [(-current[targets['turn_ids'].eq(t)] * targets['advantages'][targets['turn_ids'].eq(t)]).mean()
                     for t in (1, 2) if targets['turn_ids'].eq(t).any()]
            losses.append(torch.stack(turns).mean())
        torch.stack(losses).mean().backward()
        expected = torch.cat([p.grad.flatten().double() for p in dual_model.parameters() if p.grad is not None])
        assert comparison['norm_a'] == pytest.approx(float(expected.norm()), rel=2e-5)


@pytest.mark.parametrize('fault', ['step', 'observation', 'eos', 'mask', 'probability_length', 'nonfinite'])
def test_external_rollout_contract(dual_model, saved_group, runner, samples, fault):
    candidate = saved_group['candidates'][0]
    if fault == 'step':
        saved_group['optimizer_step'] = 2
    elif fault == 'observation':
        saved_group['observation']['frame_texts'][0] += 'changed'
    elif fault == 'eos':
        candidate['assistant_spans'][0]['eos'] = False
    elif fault == 'mask':
        candidate['turn_mask'][1] = False
    elif fault == 'probability_length':
        candidate['assistant_spans'][0]['old_token_log_probs'].pop()
    else:
        candidate['turn_advantages'][0] = float('nan')
    sample = samples['train'][0]
    with pytest.raises(ValueError):
        diagnose_gradients(dual_model, runner, sample, runner.images(sample), saved_group,
                           load_grpo_config(ROOT / 'configs/drivema_stage3b_smoke.yaml'))


def test_cli_consumes_training_producer_and_saves_json(dual_model, controlled_rollout, samples,
                                                      records, processor, tmp_path, monkeypatch):
    from PIL import Image
    rollout = load_config(ROOT / 'configs/drivema_stage3a.yaml')
    dual_model.save_pretrained(tmp_path / rollout.checkpoint_relative_path / 'adapter',
                               selected_adapters=['default'])
    base = dual_model.unload()
    runtime = SimpleNamespace(device_selector=lambda _: 'cpu', dtype_selector=lambda _: torch.bfloat16,
                              processor_loader=lambda *a: processor, model_loader=lambda *a: deepcopy(base),
                              image_loader=lambda _: Image.new('RGB', (64, 64)), package_version=lambda _: 'test')
    for entry in (training_cli, cli):
        monkeypatch.setattr(entry, 'default_runtime_dependencies', lambda: runtime)
        monkeypatch.setattr(entry, 'prepare_train', lambda *a: (
            samples['train'], {r['sample_token']: r for r in records['train']}, {'test_records_read': 0}))
        monkeypatch.setattr(entry, 'collect_git_provenance', lambda _: SimpleNamespace(commit='synthetic-test'))
    roots = ['--dataset-root', str(tmp_path), '--derived-root', str(tmp_path)]
    assert training_cli.main(roots) == 0
    source = tmp_path / 'drivema/stage3b_smoke_run01/rollout_1.json'
    output = tmp_path / 'gradient_sensitivity.json'
    assert cli.main([*roots, '--rollout', str(source), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['metadata']['input_rollout'] == str(source)
    assert result['metadata']['source_metadata']['resume'] is None
    assert result['metadata']['data_intake']['test_records_read'] == 0
    assert result['comparison']['cosine_similarity'] == pytest.approx(1., abs=1e-8)
    assert result['optimizer_steps'] == result['resampled_candidates'] == 0
    with pytest.raises(ValueError, match='new file'):
        cli.main([*roots, '--rollout', str(source), '--output', str(output)])

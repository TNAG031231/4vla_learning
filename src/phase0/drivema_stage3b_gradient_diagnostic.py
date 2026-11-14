from __future__ import annotations

from dataclasses import asdict
import json
import math

import torch
from torch import nn

from src.phase0.drivema_stage2_protocol import assistant_log_probs
from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage3a import SCHEMA_VERSION, policy_targets, replay_inputs
from src.phase0.drivema_stage3b import GRPOConfig, activate_policy, grpo_loss, probability_diagnostics
from src.phase0.phase0_4b_protocol import SFTSample


def compare_gradients(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    norm_a, norm_b = float(a.norm()), float(b.norm())
    difference = float((b - a).norm())
    cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a and norm_b else None
    return {'norm_a': norm_a, 'norm_b': norm_b, 'difference_l2': difference,
            'relative_l2_difference': difference / norm_a if norm_a else None,
            'cosine_similarity': cosine,
            'angle_degrees': math.degrees(math.acos(max(-1., min(1., cosine)))) if cosine is not None else None,
            'definition': 'relative L2 = ||g_B-g_A|| / ||g_A||; zero-norm comparisons are null'}


def diagnose_gradients(model: nn.Module, runner: Stage2Runner, sample: SFTSample,
                       images: list, group: dict, config: GRPOConfig) -> dict:
    if (group['schema_version'] != SCHEMA_VERSION or group['optimizer_step'] != 1
            or group['split'] != 'train' or sample.split != 'train'
            or group['sample_token'] != sample.sample_token or group['scene_token'] != sample.scene_token
            or group['observation'] != json.loads(json.dumps(asdict(sample.observation)))
            or len(group['candidates']) != 4):
        raise ValueError('diagnostic requires the saved Step 1 G=4 train group and identical observation')
    eos = runner.processor.tokenizer.eos_token_id
    for candidate in group['candidates']:
        spans = candidate['assistant_spans']
        if ([s['turn'] for s in spans] != list(range(1, len(spans) + 1))
                or len(spans) not in (1, 2) or candidate['turn_mask'] != [True, len(spans) == 2]
                or candidate['sample_token'] != sample.sample_token):
            raise ValueError('candidate turn mask/span mismatch')
        previous_end = 0
        for span in spans:
            ids = span['token_ids']
            ended = bool(ids) and ids[-1] == eos
            if (not ids or not 0 < span['start'] < span['end'] <= len(candidate['input_ids'])
                    or span['start'] < previous_end or len(ids) != span['end'] - span['start']
                    or len(span['old_token_log_probs']) != len(ids) or eos in ids[:-1]
                    or span['eos'] != ended or candidate['turn_complete'][span['turn'] - 1] != ended
                    or span['termination'] != ('eos' if ended else 'length')):
                raise ValueError('candidate token span/EOS mismatch')
            previous_end = span['end']
        targets = policy_targets(candidate, 'cpu')
        if not all(torch.isfinite(targets[k]).all() for k in ('old_log_probs', 'advantages')):
            raise ValueError('nonfinite saved probabilities/advantages')
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    activate_policy(model, 'default', backward=True)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    versions = {n: p._version for n, p in model.named_parameters()}
    before = [p.detach().cpu().clone() for _, p in parameters]
    gradients = {key: [torch.zeros_like(p, device='cpu', dtype=torch.float32) for _, p in parameters]
                 for key in ('A', 'B')}
    old_probs = []
    with torch.no_grad():
        for candidate in group['candidates']:
            inputs, spans = replay_inputs(runner.processor, sample.observation, images, candidate, runner.device)
            old_probs.append(torch.cat(assistant_log_probs(model, inputs, spans)).cpu())
            del inputs
    diagnostics = []
    for candidate, old_cpu in zip(group['candidates'], old_probs, strict=True):
        inputs, spans = replay_inputs(runner.processor, sample.observation, images, candidate, runner.device)
        targets = policy_targets(candidate, runner.device)
        generation = targets['old_log_probs']
        old = old_cpu.to(runner.device)
        activate_policy(model, 'reference')
        with torch.no_grad():
            reference = torch.cat(assistant_log_probs(model, inputs, spans))
        activate_policy(model, 'default', backward=True)
        current = torch.cat(assistant_log_probs(model, inputs, spans))
        metrics = {}
        for label, denominator in (('A', old), ('B', generation)):
            model.zero_grad(set_to_none=True)
            loss, metrics[label] = grpo_loss(current, reference, {**targets, 'old_log_probs': denominator},
                                           config.epsilon, config.beta)
            (loss / len(group['candidates'])).backward(retain_graph=label == 'A')
            for (_, parameter), accumulated in zip(parameters, gradients[label], strict=True):
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('nonfinite LoRA gradient')
                    accumulated.add_(parameter.grad.detach().float().cpu())
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('base/reference received gradients')
        diagnostics.append({
            'candidate_index': candidate['candidate_index'], 'surrogates': metrics,
            'current_tf': current.detach().cpu().tolist(), 'old_tf': old_cpu.tolist(),
            'generation': generation.cpu().tolist(), 'reference_tf': reference.cpu().tolist(),
            'probability_diagnostics': probability_diagnostics(current, old, generation, reference, targets['turn_ids']),
            'target_positions': targets['target_positions'].cpu().tolist(),
            'logit_positions': targets['logit_positions'].cpu().tolist(),
            'turn_ids': targets['turn_ids'].cpu().tolist(), 'loss_mask': targets['loss_mask'].cpu().tolist(),
            'attention_mask': inputs['attention_mask'][0].cpu().tolist(),
            'assistant_spans': candidate['assistant_spans'], 'rewards': candidate['rewards'],
            'turn_advantages': candidate['turn_advantages'],
        })
        del loss, current, reference, inputs
    model.zero_grad(set_to_none=True)
    if (any(p._version != versions[n] for n, p in model.named_parameters())
            or any(not torch.equal(p.detach().cpu(), previous)
                   for (_, p), previous in zip(parameters, before, strict=True))):
        raise RuntimeError('diagnostic changed model weights')
    vectors = {key: torch.cat([g.flatten() for g in values]) for key, values in gradients.items()}
    if not all(torch.isfinite(v).all() for v in vectors.values()):
        raise FloatingPointError('nonfinite accumulated gradient')
    activate_policy(model, 'default')
    return {
        'comparison': compare_gradients(vectors['A'], vectors['B']),
        'surrogates': {key: {metric: sum(c['surrogates'][key][metric] for c in diagnostics) / 4
                            for metric in ('loss', 'policy_loss', 'kl')} for key in ('A', 'B')},
        'parameter_names': [n for n, _ in parameters], 'gradient_elements': vectors['A'].numel(),
        'checks': {'finite_gradients': True, 'weights_unchanged': True,
                   'base_reference_no_grad': True, 'span_eos_mask_checked': True},
        'optimizer_steps': 0, 'resampled_candidates': 0, 'gradient_clipping': False,
        'reduction': 'token mean per present turn; turn mean per candidate; candidate mean',
        'interpretation': 'B is not a strictly unbiased importance sampling correction. '
                          'This measures sensitivity between two specified surrogates on one fixed group; '
                          'it does not establish elimination of sampling bias or population-level behavior.',
        'candidates': diagnostics,
    }

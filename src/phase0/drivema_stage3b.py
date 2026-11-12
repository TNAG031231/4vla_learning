from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import yaml

from src.phase0.drivema_stage2_protocol import assistant_log_probs
from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage3a import RolloutConfig, policy_targets, replay_inputs, rollout_group
from src.phase0.drivema_stage3_rewards import summarize_groups
from src.phase0.phase0_4b_protocol import SFTSample
from src.phase0.phase0_4c_tiny_overfit import write_json


@dataclass(frozen=True)
class GRPOConfig:
    rollout_config: str
    output_relative_dir: str
    max_steps: int = 2
    epsilon: float = 0.2
    beta: float = 0.01
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True


def load_config(path: Path) -> GRPOConfig:
    config = GRPOConfig(**yaml.safe_load(path.read_text()))
    if type(config.max_steps) is not int or not 1 <= config.max_steps <= 2:
        raise ValueError('GPU smoke supports only 1–2 total optimizer steps')
    for name in ('epsilon', 'beta', 'learning_rate', 'weight_decay', 'max_grad_norm'):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 < config.epsilon < 1 or config.learning_rate == 0 or config.max_grad_norm == 0:
        raise ValueError('require 0 < epsilon < 1 and positive learning rate / gradient norm')
    return config


def grpo_loss(current: torch.Tensor, reference: torch.Tensor, targets: dict[str, torch.Tensor],
              epsilon: float, beta: float) -> tuple[torch.Tensor, dict]:
    old = targets['old_log_probs'].detach()
    advantage = targets['advantages'].detach()
    ratio = (current.float() - old.float()).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    difference = reference.detach().float() - current.float()
    kl = torch.expm1(difference) - difference
    turns = []
    for turn in (1, 2):
        mask = targets['loss_mask'] & targets['turn_ids'].eq(turn)
        if mask.any():
            turns.append((policy[mask].mean(), kl[mask].mean(), {
                'turn': turn, 'tokens': int(mask.sum()),
                'policy_loss': float(policy[mask].detach().mean()),
                'kl': float(kl[mask].detach().mean()),
                'clip_fraction': float((ratio[mask].detach().sub(1).abs() > epsilon).float().mean()),
            }))
    policy_mean = torch.stack([t[0] for t in turns]).mean()
    kl_mean = torch.stack([t[1] for t in turns]).mean()
    loss = policy_mean + beta * kl_mean
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite GRPO loss')
    return loss, {'loss': float(loss.detach()), 'policy_loss': float(policy_mean.detach()),
                  'kl': float(kl_mean.detach()), 'turns': [t[2] for t in turns]}


def activate_policy(model: nn.Module, adapter: str, *, backward: bool = False) -> None:
    model.set_adapter(adapter)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(adapter == 'default' and 'lora_' in name and '.default.' in name)
    model.train(backward)
    # Checkpointing requires train mode; all stochastic dropout remains disabled.
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.0


def update_group(model: nn.Module, runner: Stage2Runner, sample: SFTSample, images: list, group: dict,
                 optimizer: torch.optim.Optimizer, config: GRPOConfig) -> dict:
    activate_policy(model, 'default')
    parameters = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().float().cpu().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    diagnostics = []
    for candidate in group['candidates']:
        inputs, spans = replay_inputs(runner.processor, sample.observation, images, candidate, runner.device)
        targets = policy_targets(candidate, runner.device)
        activate_policy(model, 'reference')
        with torch.no_grad():
            reference = torch.cat(assistant_log_probs(model, inputs, spans))
        activate_policy(model, 'default', backward=True)
        current = torch.cat(assistant_log_probs(model, inputs, spans))
        loss, metrics = grpo_loss(current, reference, targets, config.epsilon, config.beta)
        old = targets['old_log_probs']
        detached = current.detach()
        ratios = (detached - old).exp()
        metrics.update(
            candidate_index=candidate['candidate_index'],
            current_log_probs=detached.cpu().tolist(), reference_log_probs=reference.cpu().tolist(),
            old_log_probs=old.cpu().tolist(), ratios=ratios.cpu().tolist(),
            current_old_abs_max=float((detached - old).abs().max()),
            current_reference_abs_max=float((detached - reference).abs().max()),
            ratio_quantiles=torch.quantile(ratios.float(), ratios.new_tensor([0., .05, .5, .95, 1.])).tolist(),
            attention_mask=inputs['attention_mask'][0].cpu().tolist(),
            assistant_spans=candidate['assistant_spans'],
            rewards=candidate['rewards'], turn_advantages=candidate['turn_advantages'])
        (loss / len(group['candidates'])).backward()
        diagnostics.append(metrics)
        del loss, current, reference, inputs, targets
    norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    delta = sum((p.detach().float().cpu() - previous).square().sum().item()
                for p, previous in zip(parameters, before, strict=True)) ** .5
    if not math.isfinite(delta):
        raise FloatingPointError('nonfinite LoRA parameter update')
    activate_policy(model, 'default')
    ratios = torch.tensor([r for d in diagnostics for r in d['ratios']])
    return {'clip_fraction': sum(sum(t['clip_fraction'] for t in d['turns']) / len(d['turns'])
                                 for d in diagnostics) / len(diagnostics),
            'ratio_quantiles': torch.quantile(ratios, torch.tensor([0., .05, .5, .95, 1.])).tolist(),
            'turn_token_counts': [sum(t['tokens'] for d in diagnostics for t in d['turns']
                                      if t['turn'] == turn) for turn in (1, 2)],
            'loss': sum(d['loss'] for d in diagnostics) / len(diagnostics),
            'policy_loss': sum(d['policy_loss'] for d in diagnostics) / len(diagnostics),
            'kl': sum(d['kl'] for d in diagnostics) / len(diagnostics),
            'gradient_norm': float(norm), 'lora_parameter_delta_l2': delta,
            'candidates': diagnostics}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: Path,
                    step: int, identity: dict) -> None:
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path / 'adapter', selected_adapters=['default', 'reference'])
    torch.save({'step': step, 'identity': identity, 'optimizer': optimizer.state_dict(),
                'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               path / 'training_state.pt')


def restore_training(optimizer: torch.optim.Optimizer, path: Path, identity: dict) -> int:
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True)
    if state['identity'] != identity:
        raise ValueError('resume training/model/data identity differs')
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def fit(*, model: nn.Module, runner: Stage2Runner, samples: list[SFTSample], records: dict,
        rollout_config: RolloutConfig,
        config: GRPOConfig, output: Path, metadata: dict, resume: Path | None = None) -> dict:
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    activate_policy(model, 'default')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    identity = {'training': {k: v for k, v in asdict(config).items()
                             if k not in ('output_relative_dir', 'max_steps')},
                'rollout': asdict(rollout_config), 'sample_tokens': [s.sample_token for s in samples],
                'model': metadata['model_id'], 'revision': metadata['revision']}
    torch.manual_seed(rollout_config.seed)
    start = restore_training(optimizer, resume, identity) if resume else 0
    if start >= config.max_steps:
        raise ValueError('resume checkpoint has already reached smoke step limit')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata.json', {**metadata, 'identity': identity, 'config': asdict(config),
                                        'resume': str(resume) if resume else None,
                                        'reduction': 'token mean per present turn; turn mean per candidate; candidate mean',
                                        'kl_estimator': 'expm1(logp_ref-logp_current)-(logp_ref-logp_current)',
                                        'old_policy': 'fresh rollout per optimizer step; sampled token probabilities fixed'})
    cuda = torch.device(runner.device).type == 'cuda'
    groups, updates = [], []
    for step in range(start + 1, config.max_steps + 1):
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sample = samples[int(torch.randint(len(samples), (1,)))]
        images = runner.images(sample)
        activate_policy(model, 'default')
        group = rollout_group(model=model, processor=runner.processor, sample=sample, images=images,
                              record=records[sample.sample_token], config=rollout_config, device=runner.device)
        group['optimizer_step'] = step
        write_json(output / f'rollout_{step}.json', group)
        update = update_group(model, runner, sample, images, group, optimizer, config)
        candidates = group['candidates']
        update['reward_means'] = {key: sum(c['rewards'][key] for c in candidates) / len(candidates)
                                  for key in ('action', 'trajectory', 'consistency')}
        update['trajectory_error_means'] = {}
        for key in ('ade_3s_m', 'fde_3s_m'):
            values = [c['rewards']['trajectory_metrics'][key] for c in candidates
                      if c['rewards']['trajectory_metrics']['prediction_valid']
                      and c['rewards']['trajectory_metrics'][key] is not None]
            update['trajectory_error_means'][key] = sum(values) / len(values) if values else None
            update['trajectory_error_means'][f'{key}_count'] = len(values)
        update['group_statistics'] = group['statistics']
        update['consistency_up_without_error_improvement'] = None
        if updates:
            previous = updates[-1]
            ade = update['trajectory_error_means']['ade_3s_m']
            old_ade = previous['trajectory_error_means']['ade_3s_m']
            if ade is not None and old_ade is not None:
                update['consistency_up_without_error_improvement'] = (
                    update['reward_means']['consistency'] > previous['reward_means']['consistency']
                    and ade >= old_ade)
        if cuda:
            torch.cuda.synchronize()
        update.update(step=step, sample_token=sample.sample_token, elapsed_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None)
        write_json(output / f'update_{step}.json', update)
        save_checkpoint(model, optimizer, output / f'checkpoint_{step}', step, identity)
        groups.append(group)
        updates.append(update)
        print(json.dumps({k: v for k, v in update.items() if k != 'candidates'}, allow_nan=False), flush=True)
    summary = {**summarize_groups(groups), 'optimizer_steps_this_run': len(updates),
               'last_step': updates[-1]['step'], 'gpu_alignment_status': 'inspect saved per-token diagnostics',
               'quality_improvement': 'not established by 1–2 steps; compare separate rewards and ADE/FDE',
               'updates': [{k: v for k, v in u.items() if k != 'candidates'} for u in updates]}
    write_json(output / 'summary.json', summary)
    return summary

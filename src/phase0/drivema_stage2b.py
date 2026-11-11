from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
import yaml

from src.phase0.drivema_stage2_protocol import assistant_log_probs
from src.phase0.drivema_stage2_sft import Stage2Runner, evaluate
from src.phase0.phase0_4b_protocol import SFTSample
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    checkpoint_interval: int
    validation_interval: int
    train_samples: int | None
    monitor_samples: int
    trajectory_max_new_tokens: int
    output_relative_dir: str
    selected_adapter_relative_path: str


def load_config(path: Path) -> TrainingConfig:
    config = TrainingConfig(**yaml.safe_load(path.read_text()))
    for name in ('epochs', 'batch_size', 'gradient_accumulation_steps', 'checkpoint_interval',
                 'validation_interval', 'monitor_samples', 'trajectory_max_new_tokens'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config.train_samples is not None and (type(config.train_samples) is not int or config.train_samples < 1):
        raise ValueError('train_samples must be positive or null for full training')
    for value in (config.learning_rate, config.max_grad_norm):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('optimizer settings must be positive and finite')
    if config.selected_adapter_relative_path != 'phase_0_4/structured_action_lora_full_v0_1/adapter_step_3564':
        raise ValueError('Stage 2B starts from the frozen step-3564 adapter')
    return config


def select_subset(samples: list[SFTSample], count: int | None, seed: int, split: str) -> list[SFTSample]:
    if split not in ('train', 'validation') or any(s.split != split for s in samples):
        raise ValueError('subset selection requires matching train/validation split')
    if count is None:
        return sorted(samples, key=lambda s: s.sample_token)
    if not 1 <= count <= len(samples):
        raise ValueError('insufficient samples for fixed subset')
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for sample in sorted(samples, key=lambda s: s.sample_token):
        groups[sample.target.training_text()][sample.scene_token].append(sample)
    actions = sorted(groups)
    rng.shuffle(actions)
    scenes = {}
    for action in actions:
        scenes[action] = sorted(groups[action])
        rng.shuffle(scenes[action])
        for pool in groups[action].values():
            rng.shuffle(pool)
    selected = []
    scene_counts = Counter()
    while len(selected) < count:
        for action in actions:
            if not scenes[action]:
                continue
            scene = min(scenes[action], key=lambda name: scene_counts[name])
            scenes[action].remove(scene)
            selected.append(groups[action][scene].pop())
            scene_counts[scene] += 1
            if groups[action][scene]:
                scenes[action].append(scene)
            if len(selected) == count:
                break
    return selected


def epoch_batches(samples: list[SFTSample], config: TrainingConfig, epoch: int) -> list[list[SFTSample]]:
    ordered = sorted(samples, key=lambda s: s.sample_token)
    random.Random(config.seed + epoch).shuffle(ordered)
    size = config.batch_size * config.gradient_accumulation_steps
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def selection_key(metrics: dict, step: int) -> tuple:
    trajectory = metrics['trajectory']
    return (trajectory['invalid_prediction_count'],
            trajectory['ade_3s_m'] if trajectory['ade_3s_m'] is not None else math.inf,
            trajectory['fde_3s_m'] if trajectory['fde_3s_m'] is not None else math.inf, step)


def action_change(initial: dict, current: dict) -> dict:
    keys = ('longitudinal_macro_f1', 'lateral_macro_f1', 'joint_accuracy')
    deltas = {key: current[key] - initial[key] for key in keys}
    return {'delta_vs_step3564': deltas, 'any_decline': any(v < 0 for v in deltas.values()),
            'interpretation': 'fixed_monitor_descriptive_comparison_not_significance_test'}


def fit(*, model: torch.nn.Module, runner: Stage2Runner, train: list[SFTSample], monitor: list[SFTSample],
        records: dict, config: TrainingConfig, output: Path, provenance: dict,
        resume: Path | None = None) -> dict:
    if any(s.split != 'train' for s in train) or any(s.split != 'validation' for s in monitor):
        raise ValueError('training/monitor split mismatch')
    runner.compute_log_probs = False
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not parameters or any('lora_' not in n for n in parameters):
        raise ValueError('only existing LoRA may be trained')
    identity = {'config': {k: v for k, v in asdict(config).items() if k != 'output_relative_dir'},
                'train_tokens': [s.sample_token for s in train], 'monitor_tokens': [s.sample_token for s in monitor]}
    optimizer = torch.optim.AdamW(parameters.values(), lr=config.learning_rate, weight_decay=0.0)
    state = None
    if resume is not None:
        state = torch.load(resume / 'training_state.pt', map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('resume config or sample identities differ')
        optimizer.load_state_dict(state['optimizer'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata.json', {**identity, 'provenance': provenance,
               'subset_coverage': {name: {'scene_count': len({s.scene_token for s in items}),
                                  'actions': dict(Counter(s.target.training_text() for s in items))}
                                   for name, items in (('train', train), ('monitor', monitor))},
               'resume': str(resume) if resume else None,
               'inference': {'batch_size': runner.inference_batch_size, 'optimized': runner.optimized,
                             'compute_log_probs': runner.compute_log_probs},
               'loss_reduction': 'mean_of_sample_assistant_token_mean_CE; both_turns_weight_1',
               'batch_execution': 'independent_single_sample_forwards; batch_size_graphs_per_backward'})
    if state is None:
        rows, initial = evaluate(model, runner, monitor, records, progress_interval=16)
        write_predictions(output / 'initial_monitor_predictions.jsonl', rows)
        best = None
        step, next_epoch, next_batch = 0, 0, 0
    else:
        initial, best, step = state['initial'], state['best'], state['step']
        next_epoch, next_batch = state['next_epoch'], state['next_batch']
        torch.set_rng_state(state['rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    write_json(output / 'initial_monitor_metrics.json', initial)
    total_steps = config.epochs * len(epoch_batches(train, config, 0))
    with (output / 'training_history.jsonl').open('w') as stream:
        for epoch in range(next_epoch, config.epochs):
            batches = epoch_batches(train, config, epoch)
            for batch_index, group in enumerate(batches):
                if epoch == next_epoch and batch_index < next_batch:
                    continue
                model.train()
                optimizer.zero_grad(set_to_none=True)
                before = {n: p.detach().clone() for n, p in parameters.items()}
                cuda = next(model.parameters()).is_cuda
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                totals, counts, sample_loss = [0., 0.], [0, 0], 0.
                for offset in range(0, len(group), config.batch_size):
                    losses = []
                    for sample in group[offset:offset + config.batch_size]:
                        inputs, spans = runner.training_batch(sample, records[sample.sample_token])
                        values = assistant_log_probs(model, inputs, spans)
                        loss = -torch.cat(values).mean()
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite training loss')
                        losses.append(loss)
                        sample_loss += float(loss.detach())
                        for turn, value in enumerate(values):
                            totals[turn] -= float(value.detach().sum())
                            counts[turn] += value.numel()
                    (torch.stack(losses).sum() / len(group)).backward()
                gradient = torch.nn.utils.clip_grad_norm_(parameters.values(), config.max_grad_norm,
                                                          error_if_nonfinite=True)
                optimizer.step()
                delta = sum(float((p.detach() - before[n]).float().square().sum()) for n, p in parameters.items()) ** .5
                if not math.isfinite(delta) or delta == 0:
                    raise ValueError('LoRA update missing or nonfinite')
                if cuda:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                step += 1
                entry = {'step': step, 'epoch': epoch + 1, 'sample_count': len(group),
                         'action_ce': totals[0] / counts[0], 'trajectory_ce': totals[1] / counts[1],
                         'loss': sample_loss / len(group), 'action_tokens': counts[0], 'trajectory_tokens': counts[1],
                         'gradient_norm': float(gradient), 'lora_step_delta_l2': delta,
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated() if cuda else None,
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else None,
                         'train_seconds': elapsed, 'samples_per_second': len(group) / elapsed,
                         'supervised_tokens_per_second': sum(counts) / elapsed}
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps(entry), flush=True)
                validate = step % config.validation_interval == 0 or step == total_steps
                checkpoint = output / f'checkpoint_{step}'
                if step % config.checkpoint_interval == 0 or validate:
                    checkpoint.mkdir()
                    model.save_pretrained(checkpoint / 'adapter')
                    runner.processor.save_pretrained(checkpoint / 'adapter')
                    if validate:
                        rows, metrics = evaluate(model, runner, monitor, records, progress_interval=16)
                        metrics['action_regression'] = action_change(initial['action'], metrics['action'])
                        write_predictions(checkpoint / 'monitor_predictions.jsonl', rows)
                        write_json(checkpoint / 'monitor_metrics.json', metrics)
                        if best is None or selection_key(metrics, step) < selection_key(best['metrics'], best['step']):
                            best = {'step': step, 'checkpoint': str(checkpoint.resolve()), 'metrics': metrics}
                    torch.save({'identity': identity, 'optimizer': optimizer.state_dict(), 'step': step,
                                'next_epoch': epoch + int(batch_index + 1 == len(batches)),
                                'next_batch': (batch_index + 1) % len(batches), 'initial': initial, 'best': best,
                                'rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all() if cuda else []}, checkpoint / 'training_state.pt')
    result = {'best': best, 'optimizer_steps': step, 'initial_monitor': initial,
              'selection_policy': 'invalid_count_then_ADE3_then_FDE3_then_earliest',
              'full_validation': 'pending_selected_checkpoint_evaluation'}
    write_json(output / 'training_summary.json', result)
    return result

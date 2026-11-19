from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import yaml

from src.phase0.drivema_grpo_evaluation import compare_predictions, evaluate_population
from src.phase0.drivema_grpo_replay import baseline_training_settings, replay_identity, validate_replay_configuration
from src.phase0.drivema_stage2_sft import Stage2Runner
from src.phase0.drivema_stage2b import select_subset
from src.phase0.drivema_stage3a import RolloutConfig, load_config as load_stage3a_config, prepare_train, rollout_group
from src.phase0.drivema_stage3b import GRPOConfig, activate_policy, restore_training, save_checkpoint, update_group
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import SFTSample
from src.phase0.phase0_4c_full_train import eligible_samples, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import load_config as load_ego_config, resolve_derived_path, sha256_file


OBJECTIVE_CE_WEIGHTS = {'grpo': 0., 'grpo_ce': .1, 'ce_only': 1.}


@dataclass(frozen=True)
class TrainingConfig(GRPOConfig):
    max_steps: int = 32
    monitor_samples: int = 128
    evaluation_steps: list[int] = field(default_factory=lambda: [0, 32])
    checkpoint_interval: int = 32
    consistency_weight: float = .5
    objective: str = 'grpo'
    trajectory_ce_weight: float = 0.


def load_config(path: Path) -> TrainingConfig:
    config = TrainingConfig(**yaml.safe_load(path.read_text()))
    validate_config(config)
    return config


def validate_config(config: TrainingConfig) -> None:
    if (config.objective not in OBJECTIVE_CE_WEIGHTS
            or config.trajectory_ce_weight != OBJECTIVE_CE_WEIGHTS[config.objective]):
        raise ValueError('objective and trajectory CE weight must match the frozen experiment arm')
    if type(config.consistency_weight) not in (int, float) or config.consistency_weight not in (0., .25, .5):
        raise ValueError('GRPO consistency weight supports only 0, 0.25 and the frozen 0.5 baseline')
    for name in ('max_steps', 'monitor_samples', 'checkpoint_interval'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (not isinstance(config.evaluation_steps, list)
            or any(type(s) is not int or s < 0 for s in config.evaluation_steps)
            or sorted(set(config.evaluation_steps)) != config.evaluation_steps):
        raise ValueError('evaluation_steps must be sorted unique nonnegative integers')
    for name in ('epsilon', 'beta', 'learning_rate', 'weight_decay', 'max_grad_norm'):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 < config.epsilon < 1 or config.learning_rate == 0 or config.max_grad_norm == 0:
        raise ValueError('invalid optimizer settings')


def training_rollout_config(repository: Path, config: TrainingConfig) -> RolloutConfig:
    frozen = load_stage3a_config(repository / config.rollout_config)
    return replace(frozen, reward=replace(frozen.reward, consistency_weight=config.consistency_weight))


def prepare_data(repository: Path, derived_root: Path) -> tuple[list, list, dict, dict]:
    train, records, intake = prepare_train(repository, derived_root)
    rows = load_temporal_records(repository, derived_root, split='validation')
    ego = load_ego_config(repository / 'configs/phase0_3_dataset_adapter.yaml')
    validation, validation_intake = eligible_samples(rows, ego, 'validation')
    if ({s.scene_token for s in train} & {s.scene_token for s in validation}
            or {s.sample_token for s in train} & {s.sample_token for s in validation}):
        raise ValueError('train/validation contamination')
    temporal = yaml.safe_load((repository / 'configs/phase0_4_temporal_dataset.yaml').read_text())
    directory = resolve_derived_path(derived_root, temporal['output_relative_dir'])
    # Reuse the existing external-artifact identity helper for resume intake.
    manifests = {split: {'relative_path': str(Path(temporal['output_relative_dir']) / f'{split}.jsonl'),
                         'sha256': sha256_file(directory / f'{split}.jsonl')}
                 for split in ('train', 'validation')}
    return train, validation, {**records, **{r['sample_token']: r for r in rows}}, {
        'train': intake, 'validation': validation_intake, 'manifests': manifests, 'test_records_read': 0}


def capture_rng() -> dict:
    numpy = np.random.get_state()
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(),
            'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:])}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))


@contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def run_identity(config: TrainingConfig, rollout: RolloutConfig, train: list[SFTSample],
                 monitor: list[SFTSample], runner: Stage2Runner, metadata: dict) -> dict:
    identity = {'training': baseline_training_settings(asdict(config)),
            'rollout': {k: v for k, v in asdict(rollout).items() if k not in ('train_samples', 'output_relative_dir')},
            'train_tokens': [s.sample_token for s in train], 'monitor_tokens': [s.sample_token for s in monitor],
            'sampling': 'torch.randint; uniform_with_replacement; one_observation_per_optimizer_step',
            'model': metadata['model_id'], 'revision': metadata['revision'],
            'initial_sft_checkpoint': metadata['initial_sft_checkpoint'],
            'data_intake': metadata['data_intake'],
            'validation': {'action_generation': runner.action_generation,
                           'trajectory_max_new_tokens': runner.trajectory_max_new_tokens,
                           'optimized': runner.optimized, 'batch_size': runner.inference_batch_size,
                           'compute_log_probs': runner.compute_log_probs}}
    replay = metadata.get('observation_replay')
    if config.objective != 'grpo' and (replay is None or replay.get('experiment', {}).get('arm') != 'repeat256'):
        raise ValueError('trajectory CE requires the Low-512 observation sequence')
    if config.consistency_weight != rollout.reward.consistency_weight:
        raise ValueError('training and actual rollout consistency weights differ')
    if config.consistency_weight != .5 and replay is None:
        raise ValueError('consistency ablation requires original observation replay')
    if replay is not None:
        validate_replay_configuration(asdict(config), rollout, replay)
        identity = replay_identity(identity, replay)
        if config.objective != 'grpo':
            identity['trajectory_supervision'] = {'objective': config.objective, 'weight': config.trajectory_ce_weight}
    return identity


def validate_resume(state: dict, identity: dict, config: TrainingConfig) -> None:
    if state['identity'] != identity:
        raise ValueError('resume training/model/data identity differs')
    previous = state['config']
    step = state['step']
    if config.max_steps < previous['max_steps'] or config.max_steps <= step:
        raise ValueError('resume requires an expanded cumulative step budget')
    old_schedule, new_schedule = set(previous['evaluation_steps']), set(config.evaluation_steps)
    if not old_schedule <= new_schedule or any(s <= step for s in new_schedule - old_schedule):
        raise ValueError('resume may only add future evaluation steps')


def evaluate_and_save(model: nn.Module, runner: Stage2Runner, samples: list[SFTSample], records: dict,
                      output: Path, label: str, baseline: list[dict] | None = None, *,
                      adapter: str = 'default') -> tuple[list, dict]:
    with isolated_rng():
        activate_policy(model, adapter)
        rows, metrics = evaluate_population(model, runner, samples, records)
    write_predictions(output / f'{label}_predictions.jsonl', rows)
    write_json(output / f'{label}_metrics.json', metrics)
    if baseline is not None:
        write_json(output / f'{label}_comparison.json', compare_predictions(baseline, rows))
    return rows, metrics


def group_metrics(group: dict) -> dict:
    candidates = group['candidates']
    means = {key: sum(c['rewards'][key] for c in candidates) / len(candidates)
             for key in ('action', 'trajectory', 'consistency')}
    return {'reward_means': means,
            'total_reward_mean': sum(sum(c['rewards']['turn_rewards']) for c in candidates) / len(candidates),
            'turn_advantages': [c['turn_advantages'] for c in candidates],
            'group_statistics': group['statistics'],
            'effective_advantage_group': [not zero for zero in group['statistics']['zero_variance']],
            'generated_tokens_by_turn': [sum(len(s['token_ids']) for c in candidates for s in c['assistant_spans']
                                             if s['turn'] == turn) for turn in (1, 2)]}


def fit(*, model: nn.Module, runner: Stage2Runner, train: list[SFTSample], validation: list[SFTSample],
        records: dict, rollout_config: RolloutConfig, config: TrainingConfig, output: Path,
        metadata: dict, resume: Path | None = None) -> dict:
    validate_config(config)
    if not train or any(s.split != 'train' for s in train):
        raise ValueError('GRPO requires nonempty train split')
    monitor = select_subset(validation, config.monitor_samples, rollout_config.seed, 'validation')
    runner.compute_log_probs = False
    identity = run_identity(config, rollout_config, train, monitor, runner, metadata)
    sequence = identity.get('observation_sequence')
    by_token = {sample.sample_token: sample for sample in train}
    if sequence is not None and any(by_token[row['sample_token']].scene_token != row['scene_token']
                                    for row in sequence):
        raise ValueError('replay observation scene differs from current train intake')
    if config.objective != 'grpo':
        for token in dict.fromkeys(row['sample_token'] for row in sequence):
            target = by_token[token].target
            if not (target.longitudinal_valid and target.lateral_valid and all(records[token]['trajectory_valid_mask'])):
                raise ValueError(f'trajectory CE requires a valid expert action and six valid waypoints: {token}')
    state = None
    if resume:
        state = torch.load(resume / 'unified_state.pt', map_location='cpu', weights_only=True)
        validate_resume(state, identity, config)
    output.mkdir(parents=True, exist_ok=False)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    activate_policy(model, 'default')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    torch.manual_seed(rollout_config.seed)
    random.seed(rollout_config.seed)
    np.random.seed(rollout_config.seed)
    start = restore_training(optimizer, resume, identity) if resume else 0
    if state is not None:
        if start != state['step']:
            raise ValueError('checkpoint training states disagree on global step')
        restore_rng(state['rng'])
        baseline, baseline_metrics = state['baseline'], state['baseline_metrics']
        effective = state['effective_advantage_groups']
    else:
        baseline, baseline_metrics = evaluate_and_save(model, runner, monitor, records, output, 'monitor_0')
        effective = [0, 0]
    write_predictions(output / 'baseline_predictions.jsonl', baseline)
    write_json(output / 'baseline_metrics.json', baseline_metrics)
    write_json(output / 'monitor_tokens.json', identity['monitor_tokens'])
    if sequence is not None:
        write_json(output / 'observation_replay.json', metadata['observation_replay'])
    write_json(output / 'metadata.json', {**metadata, 'identity': identity, 'config': asdict(config),
                                        'resume': str(resume) if resume else None,
                                        'reference': 'frozen Stage 2B SFT; saved adapter/reference',
                                        'full_validation': 'separate explicit evaluation command'})
    cuda = torch.device(runner.device).type == 'cuda'
    with (output / 'training_history.jsonl').open('x') as history:
        for step in range(start + 1, config.max_steps + 1):
            started = time.perf_counter()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            draw = int(torch.randint(len(train), (1,)))
            sample = train[draw] if sequence is None else by_token[sequence[step - 1]['sample_token']]
            images = runner.images(sample)
            activate_policy(model, 'default')
            record = records[sample.sample_token]
            group = None
            if config.objective != 'ce_only':
                group = rollout_group(model=model, processor=runner.processor, sample=sample, images=images,
                                      record=record, config=rollout_config, device=runner.device)
                group['optimizer_step'] = step
                if sequence is not None and any(group[key] != sequence[step - 1][key]
                                                for key in ('optimizer_step', 'sample_token', 'scene_token')):
                    raise ValueError('generated rollout differs from original observation step')
                write_json(output / f'rollout_{step}.json', group)
            update = update_group(model, runner, sample, images, group, optimizer, config,
                                  expert_record=record if config.objective != 'grpo' else None,
                                  ce_weight=config.trajectory_ce_weight)
            if group is not None:
                update.update(group_metrics(group))
                for turn in (0, 1):
                    effective[turn] += int(update['effective_advantage_group'][turn])
                update.update(effective_advantage_groups=effective.copy(),
                              effective_advantage_group_rate=[count / step for count in effective])
            if cuda:
                torch.cuda.synchronize()
            update.update(step=step, sample_token=sample.sample_token, scene_token=sample.scene_token,
                          elapsed_seconds=time.perf_counter() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None)
            write_json(output / f'update_{step}.json', update)
            compact = {k: v for k, v in update.items() if k != 'candidates'}
            history.write(json.dumps(compact, allow_nan=False) + '\n')
            history.flush()
            print(json.dumps(compact, allow_nan=False), flush=True)
            final = step == config.max_steps
            evaluate_step = step in config.evaluation_steps or final
            if step % config.checkpoint_interval == 0 or evaluate_step:
                checkpoint = output / f'checkpoint_{step}'
                save_checkpoint(model, optimizer, checkpoint, step, identity)
                torch.save({'identity': identity, 'config': asdict(config), 'step': step,
                            'rng': capture_rng(), 'baseline': baseline, 'baseline_metrics': baseline_metrics,
                            'effective_advantage_groups': effective.copy()}, checkpoint / 'unified_state.pt')
            if evaluate_step:
                evaluate_and_save(model, runner, monitor, records, output, f'monitor_{step}', baseline)
    summary = {'last_step': config.max_steps, 'optimizer_steps_this_run': config.max_steps - start,
               'train_sample_count': len(train), 'monitor_sample_count': len(monitor),
               'effective_advantage_groups': effective, 'continuation': 'user_review_required',
               'final_monitor_report': f'monitor_{config.max_steps}_comparison.json',
               'full_validation': 'not_run; use explicit evaluate-full command'}
    write_json(output / 'summary.json', summary)
    return summary

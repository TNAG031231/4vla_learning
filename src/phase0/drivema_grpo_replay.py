from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from zipfile import ZipFile

from src.phase0.drivema_stage3a import RolloutConfig


MUTABLE_TRAINING_FIELDS = ('max_steps', 'evaluation_steps', 'checkpoint_interval',
                           'output_relative_dir', 'rollout_config', 'consistency_weight')


def baseline_training_settings(training: dict) -> dict:
    return {k: v for k, v in training.items()
            if k not in (*MUTABLE_TRAINING_FIELDS, 'objective', 'trajectory_ce_weight')}


def low512_replay(metadata: dict) -> dict:
    replay = metadata['observation_replay']
    experiment = replay['experiment']
    steps = replay['steps']
    counts = Counter(row['sample_token'] for row in steps)
    if (metadata['mode'] != 'evaluate-full' or metadata['step'] != 512
            or experiment['kind'] != 'coverage' or experiment['arm'] != 'repeat256'
            or experiment['consistency_weight'] != .5
            or [row['optimizer_step'] for row in steps] != list(range(1, 513))
            or len(counts) != 256 or set(counts.values()) != {2}
            or metadata['identity'] != replay_identity(replay['baseline_identity'], replay)):
        raise ValueError('expected the original Low-512 full evaluation with its repeat256 schedule')
    return replay


def load_observation_replay(path: Path) -> dict:
    directories = ('grpo_pilot_run01', 'grpo_continuation_to256_run01')
    with ZipFile(path) as archive:
        metadata = [json.loads(archive.read(f'{d}/metadata.json')) for d in directories]
        identity = metadata[0]['identity']
        if metadata[1]['identity'] != identity or identity['rollout']['reward']['consistency_weight'] != .5:
            raise ValueError('replay requires the original matching weight-0.5 Stage 3D runs')
        if Path(metadata[1]['resume']).parts[-2:] != (directories[0], 'checkpoint_32'):
            raise ValueError('replay continuation must originate at baseline Pilot step 32')
        history = [json.loads(line) for d in directories
                   for line in archive.read(f'{d}/training_history.jsonl').splitlines()]
        if [r['step'] for r in history] != list(range(1, 257)):
            raise ValueError('replay requires every original step 1 through 256 exactly once')
        steps = []
        for row in history:
            step = row['step']
            directory = directories[0 if step <= 32 else 1]
            group = json.loads(archive.read(f'{directory}/rollout_{step}.json'))
            if (group['optimizer_step'] != step or group['sample_token'] != row['sample_token']
                    or group['split'] != 'train' or group['sample_token'] not in identity['train_tokens']
                    or len(group['candidates']) != 4):
                raise ValueError('original rollout/history step or train-token mismatch')
            steps.append({key: group[key] for key in ('optimizer_step', 'sample_token', 'scene_token')})
    return {'source_archive': path.name, 'source_directories': list(directories),
            'source_execution_commits': [m['git_commit'] for m in metadata],
            'baseline_identity': identity, 'steps': steps}


def validate_replay_configuration(training: dict, rollout: RolloutConfig, replay: dict) -> None:
    identity = replay['baseline_identity']
    fixed = baseline_training_settings(training)
    actual = asdict(rollout)
    actual = {k: v for k, v in actual.items() if k not in ('train_samples', 'output_relative_dir')}
    actual['reward']['consistency_weight'] = .5
    if fixed != identity['training'] or actual != identity['rollout']:
        raise ValueError('ablation must preserve baseline training and rollout settings except consistency weight')
    if training['max_steps'] > len(replay['steps']):
        raise ValueError('ablation budget exceeds the frozen observation schedule')
    if 'experiment' in replay and training['consistency_weight'] != replay['experiment']['consistency_weight']:
        raise ValueError('reward differs from the frozen experiment schedule')


def replay_identity(identity: dict, replay: dict) -> dict:
    baseline = deepcopy(identity)
    baseline['rollout']['reward']['consistency_weight'] = .5
    if baseline != replay['baseline_identity']:
        raise ValueError('replay model/data/train population/monitor/inference differs from baseline')
    result = {**identity, 'sampling': 'original_stage3d_observation_sequence; consume original randint slot',
              'observation_sequence': replay['steps']}
    if 'experiment' in replay:
        result['experiment'] = replay['experiment']
        if replay['experiment']['kind'] == 'coverage':
            result['sampling'] = 'frozen_coverage_observation_sequence; consume original randint slot'
    return result


def validate_replay_checkpoint(state: dict, replay: dict, consistency_weight: float) -> None:
    identity = state['identity']
    if (identity.get('observation_sequence') != replay['steps']
            or identity['rollout']['reward']['consistency_weight'] != consistency_weight
            or identity.get('experiment') != replay.get('experiment')):
        raise ValueError('checkpoint must belong to this reward ablation and observation sequence; old GRPO forbidden')


def load_observation_schedule(path: Path) -> dict:
    replay = json.loads(path.read_text())
    steps = replay['steps']
    if ([row['optimizer_step'] for row in steps] != list(range(1, len(steps) + 1))
            or len(steps) not in (256, 512)
            or any(row['sample_token'] not in replay['baseline_identity']['train_tokens'] for row in steps)):
        raise ValueError('invalid frozen observation schedule')
    return replay

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from zipfile import ZipFile

from src.phase0.drivema_stage3a import RolloutConfig


MUTABLE_TRAINING_FIELDS = ('max_steps', 'evaluation_steps', 'checkpoint_interval',
                           'output_relative_dir', 'rollout_config', 'consistency_weight')


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
    fixed = {k: v for k, v in training.items() if k not in MUTABLE_TRAINING_FIELDS}
    actual = asdict(rollout)
    actual = {k: v for k, v in actual.items() if k not in ('train_samples', 'output_relative_dir')}
    actual['reward']['consistency_weight'] = .5
    if fixed != identity['training'] or actual != identity['rollout']:
        raise ValueError('ablation must preserve baseline training and rollout settings except consistency weight')
    if training['max_steps'] > len(replay['steps']):
        raise ValueError('ablation budget exceeds the original 256 observations')


def replay_identity(identity: dict, replay: dict) -> dict:
    baseline = deepcopy(identity)
    baseline['rollout']['reward']['consistency_weight'] = .5
    if baseline != replay['baseline_identity']:
        raise ValueError('replay model/data/train population/monitor/inference differs from baseline')
    return {**identity, 'sampling': 'original_stage3d_observation_sequence; consume original randint slot',
            'observation_sequence': replay['steps']}


def validate_replay_checkpoint(state: dict, replay: dict, consistency_weight: float) -> None:
    identity = state['identity']
    if (identity.get('observation_sequence') != replay['steps']
            or identity['rollout']['reward']['consistency_weight'] != consistency_weight):
        raise ValueError('checkpoint must belong to this reward ablation and observation sequence; old GRPO forbidden')

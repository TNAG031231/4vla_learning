from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from zipfile import ZipFile

from src.phase0.drivema_stage3a import RolloutConfig


MUTABLE_TRAINING_FIELDS = ('max_steps', 'evaluation_steps', 'checkpoint_interval',
                           'output_relative_dir', 'rollout_config', 'consistency_weight')


def reward_shape_replay(low_metadata: dict, composition: str, *, action_geometry_weight: float = 0.) -> dict:
    if composition not in ('additive', 'multiplicative'):
        raise ValueError('unsupported reward shape')
    if not math.isfinite(action_geometry_weight) or action_geometry_weight < 0:
        raise ValueError('action geometry weight must be finite and nonnegative')
    if action_geometry_weight and composition != 'additive':
        raise ValueError('action geometry feedback requires the additive arm')
    replay = low_metadata['observation_replay']
    experiment, steps = replay['experiment'], replay['steps']
    visits = Counter(row['sample_token'] for row in steps)
    if (low_metadata['mode'] != 'evaluate-full' or low_metadata['step'] != 512
            or experiment['kind'] != 'coverage' or experiment['arm'] != 'repeat256'
            or experiment['consistency_weight'] != .5 or len(visits) != 256 or set(visits.values()) != {2}
            or [row['optimizer_step'] for row in steps] != list(range(1, 513))
            or low_metadata['identity'] != replay_identity(replay['baseline_identity'], replay)):
        raise ValueError('reward-shape continuation requires the original pure GRPO Low-512 full validation')
    first_visit = {row['sample_token']: row for row in steps if row['visit'] == 1}
    if first_visit.keys() != visits.keys():
        raise ValueError('expected one first visit per Low-512 observation')
    tail = [{**row, 'optimizer_step': step, 'visit': 3, 'total_visits': 3}
            for step, row in enumerate(first_visit.values(), 513)]
    child = {'kind': 'reward_shape', 'arm': composition, 'consistency_weight': .5,
             'parent_step': 512, 'parent_experiment': experiment}
    if action_geometry_weight:
        child['action_geometry_weight'] = action_geometry_weight
    return {**{k: v for k, v in replay.items() if k != 'coverage'}, 'steps': [*steps, *tail], 'experiment': child}


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
    experiment = replay.get('experiment', {})
    composition = experiment['arm'] if experiment.get('kind') == 'reward_shape' else 'additive'
    if rollout.reward.composition != composition:
        raise ValueError('rollout reward shape differs from the experiment arm')
    if rollout.reward.action_geometry_weight != experiment.get('action_geometry_weight', 0.):
        raise ValueError('rollout action geometry weight differs from the experiment arm')
    actual = {k: v for k, v in rollout.identity_fields().items() if k not in ('train_samples', 'output_relative_dir')}
    if experiment.get('kind') == 'reward_shape':
        actual['reward'].pop('composition', None)
        actual['reward'].pop('action_geometry_weight', None)
    actual['reward']['consistency_weight'] = .5
    if fixed != identity['training'] or actual != identity['rollout']:
        raise ValueError('ablation must preserve baseline training and rollout settings except consistency weight')
    if training['max_steps'] > len(replay['steps']):
        raise ValueError('ablation budget exceeds the frozen observation schedule')
    if 'experiment' in replay and training['consistency_weight'] != replay['experiment']['consistency_weight']:
        raise ValueError('reward differs from the frozen experiment schedule')


def replay_identity(identity: dict, replay: dict) -> dict:
    baseline = deepcopy(identity)
    if replay.get('experiment', {}).get('kind') == 'reward_shape':
        baseline['rollout']['reward'].pop('composition', None)
        baseline['rollout']['reward'].pop('action_geometry_weight', None)
    baseline['rollout']['reward']['consistency_weight'] = .5
    if baseline != replay['baseline_identity']:
        raise ValueError('replay model/data/train population/monitor/inference differs from baseline')
    result = {**identity, 'sampling': 'original_stage3d_observation_sequence; consume original randint slot',
              'observation_sequence': replay['steps']}
    if 'experiment' in replay:
        result['experiment'] = replay['experiment']
        if replay['experiment']['kind'] in ('coverage', 'reward_shape'):
            result['sampling'] = 'frozen_coverage_observation_sequence; consume original randint slot'
        if replay['experiment']['kind'] == 'reward_shape':
            result['rollout'] = {**identity['rollout'], 'reward': {
                **identity['rollout']['reward'], 'composition': replay['experiment']['arm']}}
            if 'action_geometry_weight' in replay['experiment']:
                result['rollout']['reward']['action_geometry_weight'] = replay['experiment']['action_geometry_weight']
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

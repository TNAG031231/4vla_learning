#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_grpo_ablation import audit_ablation
from src.phase0.drivema_grpo_replay import (
    load_observation_replay, load_observation_schedule, validate_replay_checkpoint, validate_replay_configuration,
)
from src.phase0.drivema_grpo_train import load_config, training_rollout_config
from src.phase0.drivema_parallel import run_concurrent


def gpu_snapshot(gpu: str) -> dict:
    result = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    return {'uuid': result[0].strip(), 'total_mib': int(result[1]), 'free_mib': int(result[2])}


def check_checkpoint(path: Path, replay: dict, weight: float, step: int) -> None:
    import torch
    state = torch.load(path / 'unified_state.pt', map_location='cpu', weights_only=True)
    validate_replay_checkpoint(state, replay, weight)
    training = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != step or training['step'] != step or training['identity'] != state['identity']:
        raise ValueError('checkpoint step differs from requested stage')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run two isolated processes concurrently on one GPU.')
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--stage', choices=('pilot', 'continue', 'evaluate'), required=True)
    parser.add_argument('--worktree-a', type=Path, required=True)
    parser.add_argument('--worktree-b', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--derived-root', type=Path, required=True)
    parser.add_argument('--run-relative-dir', required=True)
    parser.add_argument('--baseline-predictions', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--min-free-gpu-gib', type=float, default=24.)
    parser.add_argument('--min-free-disk-gib', type=float, default=40.)
    parser.add_argument('--pilot-approved', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'continue' and not args.pilot_approved:
        parser.error('continuation requires --pilot-approved after reviewing both real Pilot audits and memory telemetry')
    if args.stage == 'evaluate' and not args.baseline_predictions:
        parser.error('full validation requires verified historical --baseline-predictions')
    suite = json.loads(args.suite.read_text())
    if set(suite['workers']) != {'worker_a', 'worker_b'} or suite['pilot_steps'] != 32:
        raise ValueError('suite requires two workers and a 32-step Pilot')
    weights = [suite['workers'][w]['weight'] for w in ('worker_a', 'worker_b')]
    if ((suite['round'] == 'reward' and (weights != [0., .25] or suite['final_steps'] != 256))
            or (suite['round'] == 'coverage' and (weights[0] != weights[1] or suite['final_steps'] != 512))
            or suite['round'] not in ('reward', 'coverage')):
        raise ValueError('suite mixes experiment variables or budgets')
    original = load_observation_replay(Path(suite['baseline_archive']))
    trees = {'worker_a': args.worktree_a.resolve(), 'worker_b': args.worktree_b.resolve()}
    if any(a.is_relative_to(b) for a, b in ((trees['worker_a'], trees['worker_b']),
                                          (trees['worker_b'], trees['worker_a']))):
        raise ValueError('workers require distinct independent worktrees')
    root = (args.derived_root / args.run_relative_dir).resolve()
    if not root.is_relative_to(args.derived_root.resolve()) or any(root.is_relative_to(t) for t in (*trees.values(), ROOT)):
        raise ValueError('run directory must be outside worktrees and within derived root')
    final, pilot = suite['final_steps'], suite['pilot_steps']
    commands, schedules = {}, {}
    suffix = {'pilot': 'pilot', 'continue': 'continuation', 'evaluate': 'full_validation'}[args.stage]
    for worker, spec in suite['workers'].items():
        schedule = load_observation_schedule(args.suite.parent / spec['schedule'])
        arm = worker if suite['round'] == 'reward' else {'worker_a': 'repeat256', 'worker_b': 'unique512'}[worker]
        if (schedule['baseline_identity'] != original['baseline_identity']
                or (suite['round'] == 'reward' and schedule['steps'] != original['steps'])
                or schedule['experiment']['kind'] != suite['round'] or schedule['experiment']['arm'] != arm):
            raise ValueError('schedule does not preserve original baseline provenance/step alignment')
        config = load_config(args.suite.parent / spec['config'])
        validate_replay_configuration(asdict(config), training_rollout_config(ROOT, config), schedule)
        if config.consistency_weight != spec['weight'] or len(schedule['steps']) != final:
            raise ValueError('suite weight/budget differs from worker schedule')
        schedules[worker] = schedule
        output = root / f'{worker}_{suffix}'
        if output.exists():
            raise FileExistsError(output)
        command = ['conda', 'run', '--no-capture-output', '-n', 'codex4vla_env', 'python',
            'scripts/run_drivema_grpo.py', '--config', str((args.suite.parent / spec['config']).resolve()),
            '--observation-schedule', str((args.suite.parent / spec['schedule']).resolve()),
            '--dataset-root', str(args.dataset_root.resolve()), '--derived-root', str(args.derived_root.resolve()),
            '--output-relative-dir', str(output.relative_to(args.derived_root.resolve())),
            '--max-steps', str(pilot if args.stage == 'pilot' else final)]
        if args.stage == 'continue':
            command += ['--resume', str(root / f'{worker}_pilot/checkpoint_{pilot}')]
        if args.stage == 'evaluate':
            command += ['--mode', 'evaluate-full', '--checkpoint', str(root / f'{worker}_continuation/checkpoint_{final}'),
                        '--baseline-predictions', str(args.baseline_predictions.resolve())]
        commands[worker] = command
    if args.dry_run:
        print(json.dumps({'commands': commands, 'CUDA_VISIBLE_DEVICES': args.gpu,
                          'concurrent_processes': 2, 'gpu_execution': False,
                          'resource_and_checkpoint_checks': 'pending real launch'}, indent=2))
        return 0
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for tree in trees.values():
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=tree, text=True)
        if head != commit or dirty:
            raise ValueError('both worktrees must contain the same clean committed implementation as controller')
    for worker, schedule in schedules.items():
        spec = suite['workers'][worker]
        if args.stage != 'pilot':
            directories = [root / f'{worker}_pilot']
            step = pilot
            if args.stage == 'evaluate':
                directories.append(root / f'{worker}_continuation')
                step = final
            audit_ablation(schedule, directories, step, spec['weight'])
            check_checkpoint(directories[-1] / f'checkpoint_{step}', schedule, spec['weight'], step)
    resources = gpu_snapshot(args.gpu)
    free_disk = shutil.disk_usage(args.derived_root).free / 2**30
    if resources['free_mib'] < args.min_free_gpu_gib * 1024 or free_disk < args.min_free_disk_gib:
        raise ValueError(f'insufficient configured headroom: GPU={resources}; disk_free_gib={free_disk}')
    launch = root / f'launch_{args.stage}'
    launch.mkdir(parents=True, exist_ok=False)
    (launch / 'launch.json').write_text(json.dumps({'git_commit': commit, 'commands': commands,
        'worktrees': {w: str(t) for w, t in trees.items()}, 'CUDA_VISIBLE_DEVICES': args.gpu,
        'gpu_before': resources, 'disk_free_gib': free_disk}, indent=2) + '\n')
    run_concurrent(commands, trees, {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu}, launch,
                   lambda: {'gpu': gpu_snapshot(args.gpu), 'disk_free_gib': shutil.disk_usage(root).free / 2**30})
    if args.stage != 'evaluate':
        step = pilot if args.stage == 'pilot' else final
        for worker, schedule in schedules.items():
            directories = [root / f'{worker}_pilot']
            if args.stage == 'continue':
                directories.append(root / f'{worker}_continuation')
            audit = audit_ablation(schedule, directories, step, suite['workers'][worker]['weight'])
            check_checkpoint(directories[-1] / f'checkpoint_{step}', schedule, suite['workers'][worker]['weight'], step)
            (launch / f'{worker}_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'stage': args.stage, 'status': 'completed', 'evidence': str(launch)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

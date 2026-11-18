#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.phase0.drivema_grpo_replay import load_observation_replay
from src.phase0.drivema_grpo_schedules import coverage_schedules, reward_selection
from src.phase0.drivema_stage3a import prepare_train
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path, sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Freeze two-worker reward/coverage schedules; no model loading.')
    parser.add_argument('--round', choices=('reward', 'coverage'), required=True)
    parser.add_argument('--baseline-archive', required=True, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--derived-root', type=Path)
    parser.add_argument('--reward-results', type=Path)
    parser.add_argument('--consistency-weight', type=float, choices=(0., .25, .5))
    parser.add_argument('--selection-rationale')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    plan = yaml.safe_load((ROOT / 'configs/drivema_grpo_parallel.yaml').read_text())
    base = yaml.safe_load((ROOT / plan['training_config']).read_text())
    replay = load_observation_replay(args.baseline_archive)
    if args.round == 'reward':
        schedules = {}
        for arm, weight in zip(('worker_a', 'worker_b'), plan['reward_weights'], strict=True):
            schedules[arm] = deepcopy(replay)
            schedules[arm]['experiment'] = {'kind': 'reward', 'arm': arm, 'consistency_weight': weight}
        steps = plan['reward_steps']
    else:
        if args.dry_run and (args.derived_root is None or args.reward_results is None):
            print(json.dumps({'round': 'coverage', 'status': 'blocked_pending_reward_selection_and_train_manifest',
                              'missing': ['completed reward comparison + explicit weight/rationale',
                                          'original train-only manifest with scene and GT actions'],
                              'schedules_written': False}))
            return 0
        if None in (args.derived_root, args.reward_results, args.consistency_weight, args.selection_rationale):
            parser.error('coverage requires derived-root, reward-results, consistency-weight and selection-rationale')
        selection = reward_selection(json.loads(args.reward_results.read_text()), args.consistency_weight,
                                     args.selection_rationale)
        selection['source_report'] = args.reward_results.name
        manifest = replay['baseline_identity']['data_intake']['manifests']['train']
        if sha256_file(resolve_derived_path(args.derived_root, manifest['relative_path'])) != manifest['sha256']:
            raise ValueError('coverage train manifest differs from original Stage 3D')
        train, _, _ = prepare_train(ROOT, args.derived_root)
        pair = coverage_schedules(replay, train, args.consistency_weight, selection, plan['schedule_seed'])
        schedules = dict(zip(('worker_a', 'worker_b'), pair.values(), strict=True))
        steps = plan['coverage_steps']
    suite = {'round': args.round, 'final_steps': steps, 'pilot_steps': plan['pilot_steps'],
             'baseline_archive': str(args.baseline_archive.resolve()), 'workers': {}}
    for arm, schedule in schedules.items():
        weight = schedule['experiment']['consistency_weight']
        suite['workers'][arm] = {'weight': weight, 'config': f'{arm}.yaml', 'schedule': f'{arm}.json'}
    if args.dry_run:
        print(json.dumps({**suite, 'schedules_written': False,
                          'step_counts': {a: len(s['steps']) for a, s in schedules.items()}}, indent=2))
        return 0
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('experiment schedules must live outside the repository')
    args.output.mkdir(parents=True, exist_ok=False)
    for arm, schedule in schedules.items():
        config = {**base, 'consistency_weight': schedule['experiment']['consistency_weight'],
                  'max_steps': plan['pilot_steps'], 'evaluation_steps': sorted({0, 32, 64, 128, 256, steps}),
                  'output_relative_dir': f'drivema/parallel_{args.round}/{arm}_pilot'}
        (args.output / f'{arm}.yaml').write_text(yaml.safe_dump(config))
        (args.output / f'{arm}.json').write_text(json.dumps(schedule, indent=2) + '\n')
    (args.output / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
    print(json.dumps({'suite': str(args.output / 'suite.json'), 'status': 'schedules_frozen'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

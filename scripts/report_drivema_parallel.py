#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_grpo_ablation import main as report_worker, read_rows
from src.phase0.drivema_grpo_diagnostics import validation_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Paired scene-cluster comparison of both workers and saved baselines.')
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    suite = json.loads(args.suite.read_text())
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('reports must live outside repository')
    args.output.mkdir(parents=True, exist_ok=False)
    workers, predictions, coverage = {}, {}, {}
    for worker, spec in suite['workers'].items():
        output = args.output / f'{worker}.json'
        evaluation = args.run_root / f'{worker}_full_validation'
        report_worker(['--baseline-archive', suite['baseline_archive'], '--run-dirs',
            str(args.run_root / f'{worker}_pilot'), str(args.run_root / f'{worker}_continuation'),
            '--steps', str(suite['final_steps']), '--consistency-weight', str(spec['weight']),
            '--observation-schedule', str(args.suite.parent / spec['schedule']),
            '--evaluation-dir', str(evaluation), '--output', str(output),
            '--bootstrap-repetitions', str(args.bootstrap_repetitions)])
        workers[worker] = json.loads(output.read_text())
        predictions[worker] = read_rows(evaluation / f'evaluate-full_{suite["final_steps"]}_predictions.jsonl')
        schedule = json.loads((args.suite.parent / spec['schedule']).read_text())
        coverage[worker] = schedule.get('coverage')
    reward = workers['worker_a']['identity']['rollout']['reward']
    result = {'round': suite['round'], 'status': 'full_validation_complete', 'workers': workers,
        'weights': {str(spec['weight']): worker for worker, spec in suite['workers'].items()},
        'coverage': coverage,
        'worker_b_vs_worker_a': validation_diagnostics(predictions['worker_a'], predictions['worker_b'],
            reward, args.bootstrap_repetitions, 20261116, reference_label='worker_a', current_label='worker_b'),
        'interpretation': 'One run per arm with matched observations/budget; cross-seed stability remains unverified.'}
    if suite['round'] == 'reward':
        result['weights']['0.5'] = 'original_stage3d_256'
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

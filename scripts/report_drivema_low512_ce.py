#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_grpo_ablation import read_rows
from src.phase0.drivema_grpo_diagnostics import validation_diagnostics
from src.phase0.drivema_grpo_replay import low512_replay
from src.phase0.drivema_grpo_train import OBJECTIVE_CE_WEIGHTS


def compare_arms(low512: Path, grpo_ce: Path, ce_only: Path, repetitions: int) -> dict:
    control = json.loads((low512 / 'metadata.json').read_text())
    low512_replay(control)
    tokens = control['population_tokens']
    if len(set(tokens)) != 3594 or len(tokens) != 3594:
        raise ValueError('expected the complete 3594-observation validation population')
    arms = {}
    for name, directory in [('low512', low512), ('grpo_ce', grpo_ce), ('ce_only', ce_only)]:
        if name != 'low512':
            metadata = json.loads((directory / 'metadata.json').read_text())
            supervision = {'objective': name, 'weight': OBJECTIVE_CE_WEIGHTS[name]}
            if (metadata['identity'] != {**control['identity'], 'trajectory_supervision': supervision}
                    or metadata['step'] != 512 or metadata['mode'] != 'evaluate-full'
                    or metadata['population_tokens'] != tokens):
                raise ValueError(f'{name} requires the matched final checkpoint and full validation')
        rows = read_rows(directory / 'evaluate-full_512_predictions.jsonl')
        if [r['sample_token'] for r in rows] != tokens or len({r['scene_token'] for r in rows}) != 140:
            raise ValueError(f'{name} predictions differ from the full validation population')
        arms[name] = rows

    metrics, paired = {}, {}
    for reference, current in [('low512', 'grpo_ce'), ('low512', 'ce_only'), ('ce_only', 'grpo_ce')]:
        result = validation_diagnostics(arms[reference], arms[current], control['identity']['rollout']['reward'],
                                        repetitions, 20261116, reference_label=reference, current_label=current)
        comparison = result['comparison']
        metrics[reference], metrics[current] = comparison['baseline'], comparison['current']
        joint = [metrics[arm]['action']['joint_accuracy'] for arm in (reference, current)]
        paired[f'{current}_minus_{reference}'] = {
            'ade_3s_m': result['overall']['ade_3s_m'], 'fde_3s_m': result['overall']['fde_3s_m'],
            'joint_accuracy_delta': joint[1] - joint[0],
            'joint_accuracy_bootstrap': result['action_behavior_cluster_bootstrap']['intervals']['joint_accuracy'],
        }
    primary = {name: {'ade_3s_m': values['trajectory']['ade_3s_m'],
                      'fde_3s_m': values['trajectory']['fde_3s_m'],
                      'joint_accuracy': values['action']['joint_accuracy']}
               for name, values in metrics.items()}
    return {'checkpoint_step': 512, 'observations': len(tokens), 'scenes': 140, 'primary': primary,
            'metrics': metrics, 'paired': paired, 'bootstrap': result['bootstrap']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Paired Low-512, GRPO + trajectory CE and CE-only full validation.')
    for arm in ('low512', 'grpo-ce', 'ce-only'):
        parser.add_argument(f'--{arm}', type=Path, required=True, help='full validation output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(ROOT) or args.output.exists() or args.bootstrap_repetitions < 1:
        parser.error('provide a new report outside the repository and positive bootstrap repetitions')
    report = compare_arms(args.low512, args.grpo_ce, args.ce_only, args.bootstrap_repetitions)
    with args.output.open('x') as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['primary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

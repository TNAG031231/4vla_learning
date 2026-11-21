from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_grpo_ablation import read_rows
from src.phase0.drivema_grpo_diagnostics import validation_diagnostics
from src.phase0.drivema_grpo_replay import replay_identity, reward_shape_replay
from src.phase0.phase0_4c_tiny_overfit import write_json


def full_predictions(directory: Path, metadata: dict, step: int, tokens: set[str]) -> list[dict]:
    population = metadata['population_tokens']
    if (metadata['mode'] != 'evaluate-full' or metadata['step'] != step
            or len(population) != len(tokens) or set(population) != tokens):
        raise ValueError(f'{directory.name}: full-validation checkpoint or population mismatch')
    rows = read_rows(directory / f'evaluate-full_{step}_predictions.jsonl')
    if (len(rows) != len(tokens) or {row['sample_token'] for row in rows} != tokens
            or len({row['scene_token'] for row in rows}) != 140
            or any(row['split'] != 'validation' for row in rows)):
        raise ValueError(f'{directory.name}: expected the complete 3594-row, 140-scene validation population')
    return rows


def training_signal(directory: Path, tail: list[dict]) -> dict:
    history = read_rows(directory / 'training_history.jsonl')
    if ([(r['step'], r['sample_token']) for r in history]
            != [(s['optimizer_step'], s['sample_token']) for s in tail]):
        raise ValueError(f'{directory.name}: training history must match the scheduled steps 513 through 768')
    counts = {key: sum(row['ade_reward_ranking'][key] for row in history)
              for key in ('non_tied_pairs', 'additive_reversals', 'multiplicative_reversals')}
    pairs = counts['non_tied_pairs']
    for shape in ('additive', 'multiplicative'):
        counts[f'{shape}_reversal_rate'] = counts[f'{shape}_reversals'] / pairs if pairs else None
    return {'steps': len(history),
            'zero_variance_groups': {turn: sum(r['group_statistics']['zero_variance'][i] for r in history)
                                     for i, turn in enumerate(('action', 'trajectory'))},
            'ade_reward_ranking': counts}


def report(low512_validation: Path, run_root: Path, repetitions: int = 10000) -> dict:
    control = json.loads((low512_validation / 'metadata.json').read_text())
    tokens = set(control['population_tokens'])
    if len(tokens) != 3594:
        raise ValueError('expected 3594 unique Low-512 validation observations')
    predictions = {'low512': full_predictions(low512_validation, control, 512, tokens)}
    checkpoint_paths = {'low512': control['checkpoint']}
    parents = set()
    signals = {}
    for arm in ('additive', 'multiplicative'):
        train = json.loads((run_root / arm / 'metadata.json').read_text())
        directory = run_root / f'{arm}_full'
        full = json.loads((directory / 'metadata.json').read_text())
        replay = reward_shape_replay(control, arm)
        expected = replay_identity(replay['baseline_identity'], replay)
        if train['identity'] != expected or full['identity'] != expected:
            raise ValueError(f'{arm}: training or evaluation reward composition, schedule or identity mismatch')
        if Path(full['checkpoint']).parts[-2:] != (arm, 'checkpoint_768'):
            raise ValueError(f'{arm}: full evaluation must use the corresponding checkpoint_768')
        if (not train['resume'] or Path(train['resume']).name != 'checkpoint_512'
                or Path(train['resume']).parts[-2:] != Path(control['checkpoint']).parts[-2:]):
            raise ValueError(f'{arm}: training must resume from the Low-512 checkpoint')
        parents.add(train['resume'])
        checkpoint_paths[arm] = full['checkpoint']
        signals[arm] = training_signal(run_root / arm, replay['steps'][512:])
        predictions[arm] = full_predictions(directory, full, 768, tokens)
    if len(parents) != 1:
        raise ValueError('both training arms must resume from the same Low-512 checkpoint')
    if Path(checkpoint_paths['additive']).parents[1] != Path(checkpoint_paths['multiplicative']).parents[1]:
        raise ValueError('both full validations must use checkpoints from the same training run')

    primary, actions, paired = {}, {}, {}
    reward = control['identity']['rollout']['reward']
    for reference, current in combinations(predictions, 2):
        diagnostics = validation_diagnostics(predictions[reference], predictions[current], reward, repetitions,
                                             20261116, reference_label=reference, current_label=current)
        comparison = diagnostics['comparison']
        for label, role in ((reference, 'baseline'), (current, 'current')):
            scores = comparison[role]
            actions[label] = scores['action']
            primary[label] = {key: scores['trajectory'][key] for key in ('ade_3s_m', 'fde_3s_m')}
            primary[label].update(joint_accuracy=scores['action']['joint_accuracy'],
                                  consistency_mean=scores['behavior']['consistency_mean'],
                                  consistency_sample_count=scores['behavior']['consistency_sample_count'],
                                  invalid_rate=scores['behavior']['invalid_rate'])
        before, after = primary[reference]['joint_accuracy'], primary[current]['joint_accuracy']
        paired[f'{current}_minus_{reference}'] = {
            'ade_3s_m': diagnostics['overall']['ade_3s_m'],
            'fde_3s_m': diagnostics['overall']['fde_3s_m'],
            'joint_accuracy_delta': after - before if after is not None and before is not None else None,
            'consistency_mean_delta': comparison['behavior_delta']['consistency_mean'],
            'invalid_rate_delta': comparison['behavior_delta']['invalid_rate'],
            'action_behavior_cluster_bootstrap': diagnostics['action_behavior_cluster_bootstrap'],
        }
    result = {'observations': len(tokens), 'scenes': 140, 'parent_checkpoint': parents.pop(),
              'checkpoints': checkpoint_paths, 'primary': primary, 'action_metrics': actions, 'paired': paired,
              'bootstrap': diagnostics['bootstrap'], 'training_signal': signals,
              'training_signal_scope': 'Steps 513–768 on train observations. Each formula ranks the same valid '
                                       'candidate pairs within that arm, excluding pairs with equal ADE.'}
    write_json(run_root / 'comparison.json', result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare greedy full validation for both reward shapes and Low-512.')
    parser.add_argument('--low512-validation', required=True, type=Path)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions < 1 or args.run_root.resolve().is_relative_to(ROOT):
        parser.error('use positive bootstrap repetitions and a run directory outside the repository')
    result = report(args.low512_validation, args.run_root, args.bootstrap_repetitions)
    print(json.dumps(result['primary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

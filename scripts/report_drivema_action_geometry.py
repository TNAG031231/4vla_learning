from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_grpo_ablation import read_rows
from scripts.report_drivema_reward_shape import full_predictions
from src.phase0.drivema_grpo_diagnostics import validation_diagnostics
from src.phase0.drivema_grpo_replay import replay_identity, reward_shape_replay
from src.phase0.drivema_visual_metrics import align_predictions, paired_sensitivity
from src.phase0.phase0_4c_tiny_overfit import write_json


def action_contributions(reference: list[dict], current: list[dict]) -> dict:
    transitions = {key: [] for key in ('wrong_to_correct', 'correct_to_wrong', 'correct_to_correct',
                                        'wrong_to_wrong', 'joint_target_unavailable')}
    changes = {key: [] for key in ('same', 'changed', 'unparsed')}
    paired_count = 0
    axes = ('longitudinal', 'lateral')
    for before, after in zip(reference, align_predictions(reference, current), strict=True):
        target = before['target']
        actions = (before['parsed_action'], after['parsed_action'])
        transition = 'joint_target_unavailable'
        if all(target[f'{axis}_valid'] for axis in axes):
            correct = [action is not None and all(action[axis] == target[axis] for axis in axes)
                       for action in actions]
            transition = '_to_'.join('correct' if hit else 'wrong' for hit in correct)
        change = 'unparsed'
        if all(action is not None for action in actions):
            change = 'same' if actions[0] == actions[1] else 'changed'
        delta = None
        if all(row['prediction_valid'] and row['ade_3s_m'] is not None for row in (before, after)):
            delta = after['ade_3s_m'] - before['ade_3s_m']
            paired_count += 1
        transitions[transition].append(delta)
        changes[change].append(delta)

    def summarize(groups: dict) -> dict:
        result = {}
        for name, deltas in groups.items():
            valid = [d for d in deltas if d is not None]
            result[name] = {'observations': len(deltas), 'paired_ade_count': len(valid),
                            'mean_delta_m': sum(valid) / len(valid) if valid else None,
                            'contribution_to_overall_delta_m': sum(valid) / paired_count if paired_count else None}
        return result

    return {'paired_ade_count': paired_count, 'excluded_ade_count': len(reference) - paired_count,
            'joint_correctness': summarize(transitions), 'predicted_action_change': summarize(changes)}


def report(low512_validation: Path, control_root: Path, run_root: Path, repetitions: int = 10000) -> dict:
    low = json.loads((low512_validation / 'metadata.json').read_text())
    config = yaml.safe_load((ROOT / 'configs/drivema_low768_action_geometry.yaml').read_text())
    feedback_weight = config['action_geometry_weight']
    tokens = set(low['population_tokens'])
    if len(tokens) != 3594 or len(low['population_tokens']) != len(tokens):
        raise ValueError('expected 3594 unique Low-512 validation observations')
    rows, checkpoints = {}, {}
    feedback_signal = {}
    for arm, train_dir, full_dir, weight in (
        ('additive', control_root / 'additive', control_root / 'additive_full', 0.),
        ('feedback', run_root / 'train', run_root / 'full', feedback_weight),
    ):
        train = json.loads((train_dir / 'metadata.json').read_text())
        full = json.loads((full_dir / 'metadata.json').read_text())
        replay = reward_shape_replay(low, 'additive', action_geometry_weight=weight)
        expected = replay_identity(replay['baseline_identity'], replay)
        if train['identity'] != expected or full['identity'] != expected:
            raise ValueError(f'{arm}: training or full-validation identity differs from the matched arm')
        if train['resume'] != low['checkpoint']:
            raise ValueError(f'{arm}: training must start from the same Low-512 checkpoint')
        checkpoint = Path(full['checkpoint'])
        train_path = Path(train['config']['output_relative_dir']) / 'checkpoint_768'
        if checkpoint.parts[-len(train_path.parts):] != train_path.parts:
            raise ValueError(f'{arm}: full validation must use the corresponding trained checkpoint_768')
        history = read_rows(train_dir / 'training_history.jsonl')
        if ([(r['step'], r['sample_token']) for r in history]
                != [(s['optimizer_step'], s['sample_token']) for s in replay['steps'][512:]]):
            raise ValueError(f'{arm}: actual training steps must match the scheduled 513–768 tail')
        if weight:
            stats = [r['group_statistics'] for r in history]
            active = [s for s in stats if any(s['action_geometry_feedback'])]
            feedback_signal = {'training_groups': len(stats), 'nonzero_feedback_groups': len(active),
                               'nonzero_feedback_with_tied_label_rewards': sum(s['zero_variance'][0] for s in active)}
        rows[arm] = full_predictions(full_dir, full, 768, tokens)
        checkpoints[arm] = str(checkpoint)

    reference, current = rows['additive'], rows['feedback']
    diagnostics = validation_diagnostics(reference, current, low['identity']['rollout']['reward'],
                                         repetitions, 20261116, reference_label='additive', current_label='feedback')
    result = {'observations': len(tokens), 'scenes': 140, 'action_geometry_weight': feedback_weight,
              'parent_checkpoint': low['checkpoint'], 'checkpoints': checkpoints, 'training_signal': feedback_signal,
              'comparison': diagnostics['comparison'], 'paired': diagnostics['overall'],
              'bootstrap': diagnostics['bootstrap'], 'delta_definition': diagnostics['delta_definition'],
              'action_behavior_cluster_bootstrap': diagnostics['action_behavior_cluster_bootstrap'],
              'trajectory_change': paired_sensitivity(reference, current)['trajectory_change'],
              'action_contributions': action_contributions(reference, current)}
    write_json(run_root / 'comparison.json', result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare first-turn geometry feedback with the archived additive control.')
    parser.add_argument('--low512-validation', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions < 1 or args.run_root.resolve().is_relative_to(ROOT):
        parser.error('use positive bootstrap repetitions and a run directory outside the repository')
    if args.run_root.resolve() == args.control_root.resolve():
        parser.error('use a separate run directory for the new feedback arm')
    result = report(args.low512_validation, args.control_root, args.run_root, args.bootstrap_repetitions)
    print(json.dumps({key: result['paired'][key]['mean_delta_m'] for key in ('ade_3s_m', 'fde_3s_m')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

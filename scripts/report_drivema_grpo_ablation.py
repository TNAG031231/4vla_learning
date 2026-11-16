#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.drivema_grpo_diagnostics import analyze_archive, rollout_diagnostics, validation_diagnostics
from src.phase0.drivema_grpo_replay import load_observation_replay, load_observation_schedule, replay_identity


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def audit_ablation(replay: dict, directories: list[Path], expected_steps: int,
                   consistency_weight: float = 0.) -> dict:
    identity = deepcopy(replay['baseline_identity'])
    identity['rollout']['reward']['consistency_weight'] = consistency_weight
    identity = replay_identity(identity, replay)
    groups, history = [], []
    for index, directory in enumerate(directories):
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['identity'] != identity or metadata['config']['consistency_weight'] != consistency_weight:
            raise ValueError('ablation run identity differs from fixed baseline except weight')
        if index == 0 and metadata['resume'] is not None:
            raise ValueError('first ablation run must initialize from SFT, not resume GRPO')
        rows = read_rows(directory / 'training_history.jsonl')
        if index and Path(metadata['resume']).parts[-2:] != (directories[index - 1].name,
                                                             f'checkpoint_{history[-1]["step"]}'):
            raise ValueError('ablation continuation checkpoint lineage mismatch')
        for row in rows:
            group = json.loads((directory / f'rollout_{row["step"]}.json').read_text())
            expected = replay['steps'][row['step'] - 1]
            if any(group[k] != expected[k] for k in ('optimizer_step', 'sample_token', 'scene_token')):
                raise ValueError('ablation step does not match original observation sequence')
            if any(not math.isfinite(row[k]) for k in
                   ('loss', 'policy_loss', 'kl', 'gradient_norm', 'lora_parameter_delta_l2')):
                raise ValueError('nonfinite ablation training diagnostic')
            groups.append(group)
        history.extend(rows)
    if [r['step'] for r in history] != list(range(1, expected_steps + 1)):
        raise ValueError('ablation training history incomplete or duplicated')
    nonzero = sum(r['lora_parameter_delta_l2'] > 0 and r['gradient_norm'] > 0 for r in history)
    if nonzero == 0:
        raise ValueError('ablation has no observed nonzero gradient and LoRA update')
    result = rollout_diagnostics(groups, history, identity['train_tokens'], identity['rollout']['reward'])
    return {'identity': identity, 'observation_sequence_matches': True, 'verified_steps': expected_steps,
            'steps_with_nonzero_gradient_and_update': nonzero, 'training': result,
            'gate': 'offline checks passed; user reviews GPU Pilot before continuation'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='CPU audit of consistency ablation; optional full paired comparison.')
    parser.add_argument('--baseline-archive', type=Path, required=True)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True, help='ordered ablation Pilot then continuations')
    parser.add_argument('--steps', type=int, choices=(4, 32, 256, 512), required=True)
    parser.add_argument('--consistency-weight', type=float, choices=(0., .25, .5), default=0.)
    parser.add_argument('--observation-schedule', type=Path)
    parser.add_argument('--evaluation-dir', type=Path, help='completed ablation evaluate-full output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        parser.error('output must be a new file outside the repository')
    if args.bootstrap_repetitions < 1 or (args.evaluation_dir and args.steps not in (256, 512)):
        parser.error('full comparison requires 256/512 steps and positive bootstrap repetitions')
    replay = (load_observation_schedule(args.observation_schedule) if args.observation_schedule else
              load_observation_replay(args.baseline_archive))
    result = audit_ablation(replay, args.run_dirs, args.steps, args.consistency_weight)
    if args.evaluation_dir:
        directory = args.evaluation_dir
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['identity'] != result['identity'] or metadata['step'] != args.steps or metadata['mode'] != 'evaluate-full':
            raise ValueError('full evaluation is not the matched final ablation checkpoint')
        if Path(metadata['checkpoint']).parts[-2:] != (args.run_dirs[-1].name, f'checkpoint_{args.steps}'):
            raise ValueError('full evaluation checkpoint lineage mismatch')
        baseline = analyze_archive(args.baseline_archive, repetitions=args.bootstrap_repetitions, seed=args.seed)
        with ZipFile(args.baseline_archive) as archive:
            prefix = 'grpo_full_validation_reuse_sft_run01/'
            sft = [json.loads(line) for line in archive.read(prefix + 'sft_full_predictions.jsonl').splitlines()]
            original = [json.loads(line) for line in archive.read(prefix + 'evaluate-full_256_predictions.jsonl').splitlines()]
        current = read_rows(directory / f'evaluate-full_{args.steps}_predictions.jsonl')
        if (read_rows(directory / 'sft_full_predictions.jsonl') != sft
                or len(metadata['population_tokens']) != 3594
                or set(metadata['population_tokens']) != {r['sample_token'] for r in sft}):
            raise ValueError('ablation full population or historical SFT differs')
        reward = result['identity']['rollout']['reward']
        result['validation'] = {
            'original_grpo_vs_sft': baseline['validation'],
            'ablation_vs_sft': validation_diagnostics(deepcopy(sft), deepcopy(current), reward,
                args.bootstrap_repetitions, args.seed, current_label='ablation'),
            'ablation_vs_original_grpo': validation_diagnostics(original, current, reward,
                args.bootstrap_repetitions, args.seed, reference_label='original_grpo', current_label='ablation'),
        }
        result['original_training'] = baseline['training']
        result['interpretation'] = 'single matched observation-sequence experiment; not cross-seed stability'
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'verified_steps': args.steps,
                      'full_comparison': bool(args.evaluation_dir)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

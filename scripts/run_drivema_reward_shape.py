#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_reward_shape import main as report
from src.phase0.drivema_parallel import run_concurrent
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare additive and multiplicative GRPO rewards from Low-512 on one GPU.')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'),
                        help='original local Qwen model directory; otherwise use the offline snapshot cache')
    parser.add_argument('--low512-validation', type=Path, help='original Low-512 full validation directory')
    parser.add_argument('--run-relative-dir', type=Path, required=True, help='experiment directory under the derived root')
    parser.add_argument('--stage', choices=('all', 'train', 'evaluate-full'), default='all',
                        help='evaluate-full can also be run after a completed train stage')
    parser.add_argument('--dry-run', action='store_true', help='check both worker configurations without GPU execution')
    args = parser.parse_args(argv)
    if args.derived_root is None or args.dataset_root is None or args.run_relative_dir.is_absolute():
        parser.error('provide dataset and derived roots and a relative run directory')

    derived = args.derived_root.resolve(strict=True)
    run = resolve_derived_path(derived, str(args.run_relative_dir))
    if run == derived or run.is_relative_to(ROOT):
        parser.error('use a new experiment directory under the derived root and outside the repository')
    low = (args.low512_validation or derived / 'drivema/parallel_coverage_run01/worker_a_full_validation').resolve(strict=True)
    parent = low.parent / 'worker_a_continuation/checkpoint_512'
    sft_predictions = low / 'sft_full_predictions.jsonl'
    inputs = [low / 'metadata.json', sft_predictions, low / 'evaluate-full_512_predictions.jsonl']
    if args.stage != 'evaluate-full' and not args.dry_run:
        inputs.extend([parent / 'unified_state.pt', parent / 'training_state.pt'])
        for adapter in (parent / 'adapter', parent / 'adapter/reference'):
            inputs.extend([adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors'])
    for path in inputs:
        if not path.is_file():
            parser.error(f'missing Low-512 artifact: {path}')

    common = [sys.executable, '-u', str(ROOT / 'scripts/run_drivema_grpo.py'),
              '--config', str(ROOT / 'configs/drivema_low768_reward_shape.yaml'),
              '--dataset-root', str(args.dataset_root.resolve(strict=True)), '--derived-root', str(derived),
              '--low512-validation', str(low)]
    if args.local_model_dir:
        common += ['--local-model-dir', str(args.local_model_dir.resolve(strict=True))]
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    arms = ('additive', 'multiplicative')
    stages = ('train', 'evaluate-full') if args.stage == 'all' else (args.stage,)
    if 'train' in stages and not args.dry_run:
        run.mkdir(parents=True, exist_ok=False)
    for stage in stages:
        commands = {}
        for arm in arms:
            command = [*common, '--reward-shape', arm, '--mode', stage]
            output = args.run_relative_dir / arm
            if stage == 'train':
                command += ['--resume', str(parent)]
            else:
                command += ['--checkpoint', str(derived / output / 'checkpoint_768'),
                            '--baseline-predictions', str(sft_predictions)]
                output = args.run_relative_dir / f'{arm}_full'
            commands[arm] = [*command, '--output-relative-dir', str(output)]
        if args.dry_run:
            for command in commands.values():
                subprocess.run([*command, '--dry-run'], env=environment, check=True)
            continue
        logs = run / stage
        logs.mkdir()
        print(f'{stage} worker logs: {logs}', flush=True)
        run_concurrent(commands, {arm: ROOT for arm in arms}, environment, logs)
    if 'evaluate-full' in stages and not args.dry_run:
        return report(['--low512-validation', str(low), '--run-root', str(run)])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

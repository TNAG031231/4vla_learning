#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_drivema_low512_ce import main as report
from src.phase0.drivema_parallel import run_concurrent
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Train and fully evaluate both Low-512 meta-action CE arms.')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--low512-validation', type=Path, help='existing Low-512 full validation directory')
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'),
                        help='same local Qwen base used for Low-512; defaults to the offline cache')
    parser.add_argument('--run-relative-dir', required=True, type=Path, help='new output directory under derived root')
    args = parser.parse_args(argv)
    if args.derived_root is None or args.dataset_root is None or args.run_relative_dir.is_absolute():
        parser.error('provide dataset and derived roots and a relative run directory')
    derived = args.derived_root.resolve(strict=True)
    run = resolve_derived_path(derived, str(args.run_relative_dir))
    if run.is_relative_to(ROOT):
        parser.error('output must be outside the repository')
    low = args.low512_validation or derived / 'drivema/parallel_coverage_run01/worker_a_full_validation'
    common = [sys.executable, '-u', str(ROOT / 'scripts/run_drivema_grpo.py'),
              '--dataset-root', str(args.dataset_root.resolve(strict=True)), '--derived-root', str(derived),
              '--low512-validation', str(low.resolve(strict=True))]
    if args.local_model_dir is not None:
        common.extend(['--local-model-dir', str(args.local_model_dir.resolve(strict=True))])
    for name in ('metadata.json', 'sft_full_predictions.jsonl', 'evaluate-full_512_predictions.jsonl'):
        if not (low / name).is_file():
            parser.error(f'missing Low-512 artifact: {low / name}')
    run.mkdir(parents=True, exist_ok=False)
    arms = ('grpo_meta_ce', 'meta_ce_only')
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    for stage in ('train', 'evaluate-full'):
        commands = {}
        for arm in arms:
            output = args.run_relative_dir / arm
            command = [*common, '--config', str(ROOT / f'configs/drivema_low512_{arm}.yaml'), '--mode', stage]
            if stage == 'evaluate-full':
                command.extend(['--checkpoint', str(derived / output / 'checkpoint_512')])
                output = args.run_relative_dir / f'{arm}_full'
            commands[arm] = [*command, '--output-relative-dir', str(output)]
        logs = run / stage
        logs.mkdir()
        print(f'{stage}: {logs}', flush=True)
        run_concurrent(commands, {arm: Path.cwd() for arm in arms}, environment, logs)
    print(f'Comparing both arms with Low-512: {run / "comparison.json"}', flush=True)
    return report(['--low512', str(low), '--target', 'meta_action',
                   '--grpo-ce', str(run / 'grpo_meta_ce_full'), '--ce-only', str(run / 'meta_ce_only_full'),
                   '--output', str(run / 'comparison.json')])


if __name__ == '__main__':
    raise SystemExit(main())

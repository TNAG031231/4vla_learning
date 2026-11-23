#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path


def collect_results(derived: Path, low: Path, control: Path, run: Path, archive: Path, *,
                    feedback_start_step: int = 513) -> None:
    from scripts.report_drivema_action_geometry import main as report

    report(['--low512-validation', str(low), '--control-root', str(control), '--run-root', str(run),
            '--feedback-start-step', str(feedback_start_step)])
    final_step = 512 if feedback_start_step == 1 else 768
    prefix = f'evaluate-full_{final_step}'
    files = [low / 'metadata.json', run / 'comparison.json', run / 'train/summary.json',
             run / 'train/metadata.json', run / 'train/training_history.jsonl', run / 'full/metadata.json',
             run / 'full' / f'{prefix}_predictions.jsonl', run / 'full' / f'{prefix}_metrics.json',
             run / 'full/baseline_source.json']
    if feedback_start_step == 1:
        files.append(low / f'{prefix}_predictions.jsonl')
    else:
        files.extend([control / 'additive/metadata.json', control / 'additive/training_history.jsonl',
                      control / 'additive_full/metadata.json', control / 'additive_full' / f'{prefix}_predictions.jsonl'])
    with NamedTemporaryFile(dir=archive.parent) as temporary:
        with tarfile.open(fileobj=temporary, mode='w:gz') as bundle:
            for path in files:
                bundle.add(path, arcname=path.relative_to(derived), recursive=False)
        temporary.flush()
        os.link(temporary.name, archive)
    print(archive, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run one first-turn action-geometry ablation against its saved control.')
    parser.add_argument('stage', choices=('train', 'evaluate-full', 'collect'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--feedback-start-step', type=int, choices=(1, 513), default=513,
                        help='1 trains SFT through step 512; 513 continues Low-512 through step 768')
    parser.add_argument('--archive', type=Path, help='new .tar.gz output for collect')
    parser.add_argument('--dry-run', action='store_true', help='validate the train or evaluation config without a GPU')
    args = parser.parse_args(argv)
    if args.derived_root is None:
        parser.error('provide --derived-root or VLA_DERIVED_ROOT')
    if (args.stage == 'collect') != (args.archive is not None) or (args.stage == 'collect' and args.dry_run):
        parser.error('collect requires --archive; --dry-run applies to train and evaluate-full')
    derived = args.derived_root.resolve(strict=True)
    settings = yaml.safe_load((ROOT / 'configs/drivema_low768_action_geometry.yaml').read_text())
    low = resolve_derived_path(derived, settings['low512_validation'])
    early = args.feedback_start_step == 1
    run = resolve_derived_path(derived, settings['early_output_relative_dir' if early else 'output_relative_dir'])
    if args.stage == 'collect':
        control = low.parent if early else resolve_derived_path(derived, settings['control_root'])
        collect_results(derived, low, control, run, args.archive.resolve(), feedback_start_step=args.feedback_start_step)
        return 0

    training_config = ROOT / settings['training_config']
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_drivema_grpo.py'),
               '--config', str(training_config), '--mode', args.stage,
               '--derived-root', str(derived), '--low512-validation', str(low), '--reward-shape', 'additive',
               '--action-geometry-weight', str(settings['action_geometry_weight'])]
    final_step = 512 if early else 768
    if early:
        evaluation_steps = yaml.safe_load(training_config.read_text())['evaluation_steps']
        command.extend(['--feedback-start-step', '1', '--max-steps', str(final_step), '--evaluation-steps',
                        *(str(step) for step in evaluation_steps if step <= final_step)])
    if args.stage == 'train':
        if not early:
            command.extend(['--resume', str(low.parent / 'worker_a_continuation/checkpoint_512')])
        command.extend(['--output-relative-dir', str((run / 'train').relative_to(derived))])
    else:
        command.extend(['--checkpoint', str(run / 'train' / f'checkpoint_{final_step}'),
                        '--baseline-predictions', str(low / 'sft_full_predictions.jsonl'),
                        '--output-relative-dir', str((run / 'full').relative_to(derived))])
    if args.dry_run:
        command.append('--dry-run')
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    subprocess.run(command, env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

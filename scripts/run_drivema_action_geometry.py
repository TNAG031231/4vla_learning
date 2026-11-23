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


def collect_results(derived: Path, low: Path, control: Path, run: Path, archive: Path) -> None:
    from scripts.report_drivema_action_geometry import main as report

    report(['--low512-validation', str(low), '--control-root', str(control), '--run-root', str(run)])
    files = [low / 'metadata.json', run / 'comparison.json', run / 'train/summary.json']
    for train, full in ((control / 'additive', control / 'additive_full'), (run / 'train', run / 'full')):
        files.extend([train / 'metadata.json', train / 'training_history.jsonl', full / 'metadata.json',
                      full / 'evaluate-full_768_predictions.jsonl'])
    files.extend(run / 'full' / name for name in ('evaluate-full_768_metrics.json', 'baseline_source.json'))
    with NamedTemporaryFile(dir=archive.parent) as temporary:
        with tarfile.open(fileobj=temporary, mode='w:gz') as bundle:
            for path in files:
                bundle.add(path, arcname=path.relative_to(derived), recursive=False)
        temporary.flush()
        os.link(temporary.name, archive)
    print(archive, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the single Low-512 action-geometry continuation.')
    parser.add_argument('stage', choices=('train', 'evaluate-full', 'collect'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
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
    control = resolve_derived_path(derived, settings['control_root'])
    run = resolve_derived_path(derived, settings['output_relative_dir'])
    if args.stage == 'collect':
        collect_results(derived, low, control, run, args.archive.resolve())
        return 0

    command = [sys.executable, '-u', str(ROOT / 'scripts/run_drivema_grpo.py'),
               '--config', str(ROOT / settings['training_config']), '--mode', args.stage,
               '--derived-root', str(derived), '--low512-validation', str(low), '--reward-shape', 'additive',
               '--action-geometry-weight', str(settings['action_geometry_weight'])]
    if args.stage == 'train':
        command.extend(['--resume', str(low.parent / 'worker_a_continuation/checkpoint_512'),
                        '--output-relative-dir', str((run / 'train').relative_to(derived))])
    else:
        command.extend(['--checkpoint', str(run / 'train/checkpoint_768'),
                        '--baseline-predictions', str(low / 'sft_full_predictions.jsonl'),
                        '--output-relative-dir', str((run / 'full').relative_to(derived))])
    if args.dry_run:
        command.append('--dry-run')
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    subprocess.run(command, env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

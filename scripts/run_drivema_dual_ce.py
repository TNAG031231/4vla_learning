#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.report_drivema_dual_ce import historical_runs, report
from src.phase0.drivema_grpo_replay import low512_replay
from src.phase0.drivema_grpo_train import load_config
from src.phase0.drivema_parallel import run_concurrent
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path

CONFIG = ROOT / 'configs/drivema_low768_dual_ce.yaml'


def run_arm(arm: str, common: list[str], relative: Path, run: Path, parent: Path,
            compare_midpoint: Callable[[], None]) -> None:
    def stage(name: str, *options: str) -> None:
        command = [*common, '--dual-ce-first-step', '1' if arm == 'A' else '513',
                   '--output-relative-dir', str(relative / name), *options]
        print(f'Starting {name}: {run / name}.log', flush=True)
        with (run / f'{name}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    if arm == 'A':
        stage('A_to512', '--max-steps', '512')
        parent = run / 'A_to512/checkpoint_512'
        stage('A512_full', '--mode', 'evaluate-full', '--checkpoint', str(parent))
        compare_midpoint()
    tail = f'{arm}_to768'
    stage(tail, '--resume', str(parent), '--max-steps', '768')
    stage(f'{arm}768_full', '--mode', 'evaluate-full', '--checkpoint', str(run / tail / 'checkpoint_768'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Matched 768-step early vs delayed dual CE, one GPU and two workers.')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--run-relative-dir', type=Path, required=True)
    parser.add_argument('--local-model-dir', type=Path, default=os.environ.get('QWEN_LOCAL_DIR'))
    parser.add_argument('--low512-validation', type=Path)
    parser.add_argument('--low512-checkpoint', type=Path)
    parser.add_argument('--trajectory-run', type=Path, help='PR #66 experiment directory, containing both *_full outputs')
    parser.add_argument('--meta-run', type=Path, help='PR #67 experiment directory, containing available *_full outputs')
    parser.add_argument('--report-step', type=int, choices=(512, 768), help='refresh just this comparison from disk')
    parser.add_argument('--arm', choices=('A', 'B'), help=argparse.SUPPRESS)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if args.derived_root is None or args.run_relative_dir.is_absolute():
        parser.error('provide the derived root and a relative experiment directory')
    derived = args.derived_root.resolve(strict=True)
    run = resolve_derived_path(derived, str(args.run_relative_dir))
    if run.is_relative_to(ROOT) or run == derived:
        parser.error('use a new experiment directory outside the repository and below the derived root')
    low = (args.low512_validation or derived / 'drivema/parallel_coverage_run01/worker_a_full_validation').resolve()

    def compare(step: int) -> None:
        controls = historical_runs(derived, args.trajectory_run, args.meta_run) if step == 512 else (None, None)
        report(run, low, step, load_config(CONFIG), *controls)

    if args.report_step:
        compare(args.report_step)
        return 0
    if args.dataset_root is None:
        parser.error('provide the nuScenes dataset root')
    common = [sys.executable, '-u', str(ROOT / 'scripts/run_drivema_grpo.py'), '--config', str(CONFIG),
              '--dataset-root', str(args.dataset_root.resolve(strict=True)), '--derived-root', str(derived),
              '--low512-validation', str(low)]
    if args.local_model_dir:
        common.extend(['--local-model-dir', str(args.local_model_dir.resolve(strict=True))])
    parent = args.low512_checkpoint or low.parent / 'worker_a_continuation/checkpoint_512'
    if args.arm:
        run_arm(args.arm, common, args.run_relative_dir, run, parent, lambda: compare(512))
        return 0
    historical_runs(derived, args.trajectory_run, args.meta_run)
    control = json.loads((low / 'metadata.json').read_text())
    low512_replay(control)
    for path in (low / 'sft_full_predictions.jsonl', low / 'evaluate-full_512_predictions.jsonl',
                 parent / 'training_state.pt', parent / 'adapter/adapter_model.safetensors',
                 parent / 'adapter/reference/adapter_model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = torch.load(parent / 'unified_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != 512 or state['identity'] != control['identity']:
        raise ValueError('source checkpoint does not match the original Low-512 full validation')
    run.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    commands = {arm: [sys.executable, '-u', str(Path(__file__).resolve()), *arguments, '--arm', arm]
                for arm in ('A', 'B')}
    print(f'Running A and B in parallel: {run}', flush=True)
    run_concurrent(commands, {arm: Path.cwd() for arm in commands}, environment, run)
    compare(768)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

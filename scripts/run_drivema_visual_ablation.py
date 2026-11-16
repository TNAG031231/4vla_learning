#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.drivema_visual_ablation import load_config, run, summarize
from src.phase0.drivema_visual_interventions import CONDITIONS
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='P3-A validation-only visual dependency ablation; no training.')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/drivema_visual_ablation.yaml')
    parser.add_argument('--mode', choices=('run', 'summarize'), default='run')
    parser.add_argument('--dataset-root', type=Path, default=os.environ.get('NUSCENES_ROOT'))
    parser.add_argument('--derived-root', type=Path, default=os.environ.get('VLA_DERIVED_ROOT'))
    parser.add_argument('--samples', type=int, choices=(*range(4, 9), 128), default=128)
    parser.add_argument('--sanity-output', type=Path)
    parser.add_argument('--input-dir', type=Path, help='completed P3-A output; summarize only')
    parser.add_argument('--output-relative-dir')
    parser.add_argument('--dry-run', action='store_true', help='configuration only; no data, model or output writes')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_relative_dir is not None:
        config = replace(config, output_relative_dir=args.output_relative_dir)
    resolve_derived_path(args.derived_root or ROOT, config.output_relative_dir)
    if args.dry_run:
        print(json.dumps({'config': asdict(config), 'conditions': CONDITIONS, 'samples': args.samples,
                          'batch_size': 1, 'compute_log_probs': False, 'clean_inference': 'reuse saved files',
                          'mode': args.mode}, indent=2))
        return 0
    if args.mode == 'summarize':
        if args.input_dir is None:
            parser.error('summarize requires --input-dir')
        print(json.dumps(summarize(args.input_dir), indent=2, allow_nan=False))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error('provide dataset-root and derived-root')
    run(ROOT, args.dataset_root, args.derived_root, config, args.samples, args.sanity_output)
    print(json.dumps({'output': str(args.derived_root / config.output_relative_dir), 'status': 'completed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

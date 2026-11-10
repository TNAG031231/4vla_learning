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

from src.phase0.drivema_stage2_sft import load_config, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DriveMA-style Stage 2 two-turn trajectory token SFT smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/drivema_stage2_smoke.yaml")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dry-run", action="store_true", help="show config without reading data or model")
    parser.add_argument("--mode", choices=("gpu-smoke", "tiny-overfit"), default="tiny-overfit")
    parser.add_argument("--output-relative-dir", help="new output directory under derived root; never overwritten")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.mode == "gpu-smoke":
        config = replace(config, max_steps=2, output_relative_dir="drivema/stage2_gpu_smoke_v0_1")
    if args.output_relative_dir:
        config = replace(config, output_relative_dir=args.output_relative_dir)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_data_or_model_access", "config": asdict(config)}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    result = run(repository=ROOT, dataset_root=args.dataset_root, derived_root=args.derived_root, config=config)
    print(json.dumps(result, indent=2, allow_nan=False))
    passed = result["gpu_smoke_passed"]
    if args.mode == "tiny-overfit":
        passed = passed and result["tiny_overfit_passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

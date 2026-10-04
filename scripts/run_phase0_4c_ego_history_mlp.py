#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.ego_history_mlp import load_config, run
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4c-4B ego-history MLP baseline.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4c_ego_history_mlp.yaml")
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"),
                        help="standard handoff argument; raw dataset is not accessed")
    parser.add_argument("--train-split", choices=("train",), default="train")
    parser.add_argument("--validation-split", choices=("validation",), default="validation")
    parser.add_argument("--cv-result-dir", type=Path)
    parser.add_argument("--planner-result-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_data_or_model_access", "config": config}, indent=2))
        return 0
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or --derived-root")
    result = run(repository=ROOT, derived_root=args.derived_root, config=config,
                 git_provenance=collect_git_provenance(ROOT), train_split=args.train_split,
                 validation_split=args.validation_split, cv_result_dir=args.cv_result_dir,
                 planner_result_dir=args.planner_result_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4c_trajectory_stratification import load_config, run
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Phase 0.4c-5 validation trajectory stratification.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4c_trajectory_stratification.yaml")
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dry-run", action="store_true", help="validate config without artifact access")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_data_or_model_access", "config": config}, indent=2))
        return 0
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or provide --derived-root")
    result = run(repository=ROOT, derived_root=args.derived_root, config=config,
                 git_provenance=collect_git_provenance(ROOT), split=args.split)
    print(json.dumps({"status": "validation_analysis_completed", "overall": result["overall"],
                      "aggregate_reproduction": result["aggregate_reproduction"]}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

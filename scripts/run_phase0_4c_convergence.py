#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4c_convergence import diagnose_gt_action, load_config, run
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4c-6A action-conditioned planner convergence audit.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4c_convergence.yaml")
    parser.add_argument("--train-split", choices=("train",), default="train")
    parser.add_argument("--validation-split", choices=("validation",), default="validation")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dry-run", action="store_true", help="validate config without data/model access")
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--diagnose-gt-action-epoch2-3", action="store_true",
                              help="evaluate existing epoch-2/3 checkpoints with GT action on validation only")
    continuation.add_argument("--continue-after-epoch1-reproduction-fail", action="store_true",
                              help="resume failed historical reproduction from epoch_1 for epochs 2–3")
    continuation.add_argument("--extend-to-five", action="store_true",
                              help="resume epoch_3 with optimizer/RNG state after the extension gate passes")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_no_data_or_model_access", "config": asdict(config)}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    if args.diagnose_gt_action_epoch2_3:
        result = diagnose_gt_action(
            repository=ROOT, dataset_root=args.dataset_root, derived_root=args.derived_root,
            config=config, git_provenance=collect_git_provenance(ROOT),
            validation_split=args.validation_split)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    result = run(repository=ROOT, dataset_root=args.dataset_root, derived_root=args.derived_root,
                 config=config, git_provenance=collect_git_provenance(ROOT),
                 train_split=args.train_split, validation_split=args.validation_split,
                 extend_to_five=args.extend_to_five,
                 continue_after_epoch1_reproduction_fail=args.continue_after_epoch1_reproduction_fail)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

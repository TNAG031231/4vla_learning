from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4b_evaluation import majority_baselines
from src.phase0.phase0_4b_lora_full import checkpoint_steps, load_config, prepare_data, run_full
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4b-B full train SFT and validation")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4b_lora_full.yaml")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU full train/validation intake and majority statistics; no images/model/output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or provide --derived-root")
    if args.dry_run:
        train, validation, summary = prepare_data(ROOT, args.derived_root, config)
        total = (len(train) + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps
        summary.pop("train_sample_tokens")
        summary.update(evidence_kind="CPU_artifact_intake_only", optimizer_steps_planned=total,
                       checkpoint_steps=checkpoint_steps(total, config.checkpoint_fractions),
                       majority_baselines=majority_baselines(train, validation),
                       gpu_training_executed=False)
        print(json.dumps(summary, indent=2))
        return 0
    if args.dataset_root is None:
        parser.error("set NUSCENES_ROOT or provide --dataset-root")
    result = run_full(repository=ROOT, dataset_root=args.dataset_root, derived_root=args.derived_root,
                      config=config, git_provenance=collect_git_provenance(ROOT))
    print(json.dumps({key: result[key] for key in (
        "status", "optimizer_steps", "train_records_consumed", "validation_records_consumed",
        "selected_checkpoint", "generalization", "reload_status", "test_evaluation_performed",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

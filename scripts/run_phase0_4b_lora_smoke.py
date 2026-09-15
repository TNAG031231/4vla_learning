#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4b_lora_smoke import load_config, run_smoke
from src.phase0.phase0_4b_protocol import verify_processor_protocol
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0.4b-A structured action tiny train overfit.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4b_lora_smoke.yaml")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--verify-protocol", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.verify_protocol:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            FIXED_MODEL_ID, revision=FIXED_REVISION, local_files_only=True,
        )
        print(json.dumps({"model": FIXED_MODEL_ID, "revision": FIXED_REVISION,
                          "evidence_kind": "real_processor_with_synthetic_images",
                          "cases": verify_processor_protocol(processor)}, indent=2))
        return 0
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    result = run_smoke(repository=ROOT, dataset_root=args.dataset_root, derived_root=args.derived_root,
                       config=config, git_provenance=collect_git_provenance(ROOT), split=args.split)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("predictions_before", "predictions_after_reload")}, indent=2))
    return 0 if result["status"] == "smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

from data.inspect_nuscenes_sample import TrajectoryPoint  # noqa: E402
from src.phase0.phase0_4_factorized_targets import extract_motion_features  # noqa: E402
from src.phase0.phase0_4_source_projection import (  # noqa: E402
    SOURCE_SCHEMA,
    SOURCE_VERSION,
    load_config,
)


# User-confirmed AutoDL Phase 0.4a-1 audit; verify bytes before semantic access.
AUDITED_SOURCE_SHA256 = {
    "train": "8c1ad5cc2ad4fa01d7730b1458a7415061ed40b0198495ddb36fbb035d738352",
    "validation": "68ab5a06440447f31bbfcb8fa4678bf248b5ab3f718d978cba25d8b568e77246",
}
PROVENANCE_NOTICE = (
    "source_legacy_meta_action is provenance only, not factorized supervision."
)
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
PERCENTILE_NAMES = (
    "min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max",
)


def summarize(features: list[dict[str, float]]) -> dict[str, object]:
    if not features:
        raise ValueError("source split must not be empty")
    return {
        name: dict(zip(
            PERCENTILE_NAMES,
            np.percentile(
                [row[name] for row in features], PERCENTILES, method="linear",
            ).tolist(),
        ))
        for name in features[0]
    }


def analyze_sources(derived_root: Path, splits: list[str]) -> dict[str, object]:
    if not splits or any(split not in ("train", "validation") for split in splits):
        raise ValueError("analyzer only permits train and validation")
    config = load_config(
        REPOSITORY_ROOT / "configs/phase0_4_source_projection.yaml", REPOSITORY_ROOT,
    )
    payloads = {}
    for split in splits:
        path = derived_root / config.output_relative_dir / f"{split}.jsonl"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != AUDITED_SOURCE_SHA256[split]:
            raise ValueError(f"{split} source SHA-256 differs from audited artifact")
        payloads[split] = payload

    results = {}
    for split, payload in payloads.items():
        features = []
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        audit_count = 0
        for line in payload.splitlines():
            record = json.loads(line)
            if (
                record["split"] != split
                or record["source_projection_version"] != SOURCE_VERSION
                or record["source_projection_schema_version"] != SOURCE_SCHEMA
            ):
                raise ValueError("source split or projection version mismatch")
            points = [
                TrajectoryPoint(**point) for point in record["future_ego_trajectory"]
            ]
            motion = extract_motion_features(points)
            features.append(motion)
            # Legacy action selects only the provenance reporting group.
            groups[record["source_legacy_meta_action"]].append(motion)
            audit_count += record["source_audit_record"] is not None
        results[split] = {
            "sample_count": len(features),
            "source_sha256": AUDITED_SOURCE_SHA256[split],
            "source_audit_record_non_null_count": audit_count,
            "feature_distributions": summarize(features),
            "provenance_only_breakdown": {
                label: {
                    "sample_count": len(rows),
                    "feature_distributions": summarize(rows),
                }
                for label, rows in sorted(groups.items())
            },
        }
    return {
        "provenance_notice": PROVENANCE_NOTICE,
        "percentile_method": "linear",
        "splits": results,
        "access_evidence": {
            "source_records_parsed": {
                split: result["sample_count"] for split, result in results.items()
            },
            "scope": (
                "Only audited source bytes are parsed; no raw-data reader is called."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print train/validation trajectory calibration statistics only."
    )
    parser.add_argument(
        "--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"),
    )
    parser.add_argument(
        "--split", nargs="+", choices=("train", "validation"),
        default=["train", "validation"],
    )
    args = parser.parse_args(argv)
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or provide --derived-root")
    print(json.dumps(
        analyze_sources(args.derived_root, args.split),
        ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

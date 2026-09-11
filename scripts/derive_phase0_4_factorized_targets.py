#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

from data.inspect_nuscenes_sample import TrajectoryPoint  # noqa: E402
from scripts import analyze_phase0_4_factorized_features as analyzer  # noqa: E402
from src.phase0.manifest import (  # noqa: E402
    COORDINATE_METADATA, write_canonical_json, write_jsonl_records,
)
from src.phase0.phase0_4_factorized_targets import (  # noqa: E402
    FACTORIZED_ACTION_RULE_VERSION, LATERAL_ACTIONS, LONGITUDINAL_ACTIONS,
    LATERAL_DISPLACEMENT_M, SPEED_CHANGE_MPS, derive_trajectory_targets,
)
from src.phase0.phase0_4_source_projection import (  # noqa: E402
    SOURCE_SCHEMA, SOURCE_VERSION, SourceProjectionConfig, load_config,
)

OUTPUT_RELATIVE_DIR = "phase_0_4/factorized_targets_v0_1"


def derive_record(record: dict, split: str, config: SourceProjectionConfig) -> dict:
    if split not in ("train", "validation") or record["split"] != split:
        raise ValueError("derivation only permits matching train and validation")
    if (
        record["source_projection_version"] != SOURCE_VERSION
        or record["source_projection_schema_version"] != SOURCE_SCHEMA
    ):
        raise ValueError("source projection schema/version mismatch")
    coordinates = record["coordinate_metadata"]["future_ego_trajectory"]
    expected = COORDINATE_METADATA["future_ego_trajectory"]
    if any(coordinates[key] != expected[key] for key in (
        "source_frame", "target_frame", "x_axis", "y_axis", "unit", "transform",
    )):
        raise ValueError("source trajectory coordinate contract mismatch")
    raw = record.get("future_ego_trajectory")
    points = [] if raw is None else [TrajectoryPoint(**point) for point in raw]
    targets = derive_trajectory_targets(
        points,
        sample_interval_sec=config.sample_interval_sec,
        time_tolerance_sec=config.time_tolerance_sec,
        anchor_absolute_tolerance=config.anchor_absolute_tolerance,
    )
    return {
        **record,
        **targets,
        "source_future_trajectory_version": record["source_projection_version"],
    }


def summarize(records: list[dict]) -> dict:
    total = len(records)
    result = {"sample_count": total}
    for direction, actions in (
        ("longitudinal", LONGITUDINAL_ACTIONS), ("lateral", LATERAL_ACTIONS),
    ):
        counts = Counter(
            row[f"{direction}_action"] if row[f"{direction}_action_valid"]
            else "invalid" for row in records
        )
        result[direction] = {
            action: {"count": counts[action], "ratio": counts[action] / total}
            for action in (*actions, "invalid")
        }
    joint = sum(row["factorized_action_joint_valid"] for row in records)
    result["factorized_action_joint_valid"] = {"count": joint, "ratio": joint / total}
    result["joint_combinations"] = dict(Counter(
        f'{row["longitudinal_action"]} + {row["lateral_action"]}'
        for row in records if row["factorized_action_joint_valid"]
    ).most_common())
    return result


def select_review_records(records: list[dict]) -> list[dict]:
    selected = {}
    for split in ("train", "validation"):
        rows = [row for row in records if row["split"] == split]
        for longitudinal in LONGITUDINAL_ACTIONS:
            for lateral in LATERAL_ACTIONS:
                match = next((row for row in rows if (
                    row["longitudinal_action"] == longitudinal
                    and row["lateral_action"] == lateral
                )), None)
                if match is not None:
                    selected[match["sample_token"]] = match
    for field, threshold in (
        ("final_lateral_displacement_m", LATERAL_DISPLACEMENT_M),
        ("delta_speed_proxy_mps", SPEED_CHANGE_MPS),
    ):
        for boundary in (-threshold, threshold):
            for row in sorted(records, key=lambda row: abs(
                row["motion_features"][field] - boundary,
            ))[:2]:
                selected[row["sample_token"]] = row
    for row in records:
        if len(selected) >= 24:
            break
        selected[row["sample_token"]] = row
    return list(selected.values())[:30]


def write_review(records: list[dict], path: Path) -> None:
    cards = []
    for row in records:
        points = row["future_ego_trajectory"]
        xs = [point["x_m"] for point in points]
        ys = [point["y_m"] for point in points]
        scale = 240 / max(max(xs) - min(xs), max(ys) - min(ys), 2)
        coordinates = [
            (150 - (y - (max(ys) + min(ys)) / 2) * scale,
             150 - (x - (max(xs) + min(xs)) / 2) * scale)
            for x, y in zip(xs, ys)
        ]
        line = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
        labels = "".join(
            f'<circle cx="{x}" cy="{y}" r="3"/>'
            f'<text x="{x + 5}" y="{y}">{i}</text>'
            for i, (x, y) in enumerate(coordinates)
        )
        features = row["motion_features"]
        details = {key: round(features[key], 4) for key in (
            "path_length_m", "start_speed_proxy_mps", "end_speed_proxy_mps",
            "delta_speed_proxy_mps", "final_lateral_displacement_m",
        )}
        cards.append(
            f'<article><h3>{row["longitudinal_action"]} + {row["lateral_action"]}</h3>'
            f'<p>{html.escape(row["split"] + " / " + row["sample_token"])}</p>'
            f'<svg viewBox="0 0 300 300"><polyline points="{line}" '
            f'fill="none" stroke="blue" stroke-width="2"/>{labels}</svg>'
            f'<pre>{html.escape(json.dumps(details, indent=2))}</pre></article>'
        )
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Factorized target review</title>'
        '<style>body{font:14px sans-serif;margin:24px}.cards{display:grid;'
        'grid-template-columns:repeat(3,minmax(300px,1fr));gap:20px}'
        'article{border:1px solid #ccc;padding:12px;overflow-wrap:anywhere}'
        'svg{width:100%;max-height:300px}pre{font-size:12px}</style>'
        '<h1>REVIEWER-ONLY GT FUTURE INFORMATION</h1>'
        '<p>NOT AVAILABLE TO MODEL AT INFERENCE. Current ego frame: '
        'x forward (up), y left (left), meters. Points 0..6: current anchor '
        'through 3 seconds. Panels use independent equal-axis scales. '
        'Materials prepared; human review pending.</p><div class="cards">'
        + "".join(cards) + '</div>', encoding="utf-8",
    )


def derive_sources(derived_root: Path, splits: list[str]) -> dict:
    if not splits or any(split not in ("train", "validation") for split in splits):
        raise ValueError("derivation only permits train and validation")
    config = load_config(
        REPOSITORY_ROOT / "configs/phase0_4_source_projection.yaml", REPOSITORY_ROOT,
    )
    output = derived_root / OUTPUT_RELATIVE_DIR
    if output.exists():
        raise FileExistsError(f"derived artifact already exists: {output}")
    payloads = {}
    for split in splits:
        payload = (derived_root / config.output_relative_dir / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != analyzer.AUDITED_SOURCE_SHA256[split]:
            raise ValueError(f"{split} source SHA-256 differs from audited artifact")
        payloads[split] = payload
    records_by_split = {
        split: [derive_record(json.loads(line), split, config) for line in payload.splitlines()]
        for split, payload in payloads.items()
    }
    for split, records in records_by_split.items():
        if len(records) != config.source_contract.expected_sample_counts[split]:
            raise ValueError(f"{split} source count mismatch")
        if any(not row["factorized_action_joint_valid"] for row in records):
            raise ValueError(f"{split} unexpected invalid in audited frozen source")
        for direction in ("longitudinal", "lateral"):
            if len({row[f"{direction}_action"] for row in records}) == 1:
                raise ValueError(f"{split} {direction} collapsed to one class")
    summary = {
        "factorized_action_rule_version": FACTORIZED_ACTION_RULE_VERSION,
        "source_future_trajectory_version": SOURCE_VERSION,
        "source_sha256": {split: analyzer.AUDITED_SOURCE_SHA256[split] for split in splits},
        "splits": {split: summarize(rows) for split, rows in records_by_split.items()},
        "access_evidence": {
            "source_records_parsed": {split: len(rows) for split, rows in records_by_split.items()},
            "combined_manifest_records_parsed": 0,
            "test_sample_records_read": 0,
            "test_images_opened": 0,
            "test_labels_read": 0,
            "scope": "Only SHA-verified train/validation source files opened; no raw-data/image reader.",
        },
        "human_review_status": "materials_prepared_review_pending",
    }
    review = select_review_records([row for rows in records_by_split.values() for row in rows])
    output.mkdir(parents=True)
    for split, records in records_by_split.items():
        write_jsonl_records(records, output / f"{split}.jsonl")
    write_jsonl_records(review, output / "review_records.jsonl")
    write_review(review, output / "review.html")
    summary["review_record_count"] = len(review)
    write_canonical_json(summary, output / "derivation_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive factorized targets from audited train/validation trajectories.")
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--split", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    args = parser.parse_args(argv)
    if args.derived_root is None:
        parser.error("set VLA_DERIVED_ROOT or provide --derived-root")
    print(json.dumps(derive_sources(args.derived_root, args.split), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

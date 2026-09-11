#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from scripts.analyze_phase0_4_factorized_features import AUDITED_SOURCE_SHA256
from scripts.derive_phase0_4_factorized_targets import OUTPUT_RELATIVE_DIR as FACTORIZED_DIR
from src.phase0 import development_projection as development
from src.phase0.manifest import NuScenesReader, write_canonical_json, write_jsonl_records
from src.phase0.phase0_4_factorized_targets import (
    FACTORIZED_ACTION_RULE_VERSION, LATERAL_ACTIONS, LONGITUDINAL_ACTIONS,
)
from src.phase0.phase0_4_source_projection import (
    SOURCE_VERSION, SourceProjectionConfig, load_config,
)
from src.phase0.phase0_4_temporal_dataset import (
    FACTORIZED_FIELDS, INPUT_FIELDS, TARGET_FIELDS, SCHEMA_VERSION,
    build_record, collect_history, summarize, validate_record,
)
from src.phase0.scene_mapping import read_scene_mapping


def intake(
    derived_root: Path, config: SourceProjectionConfig, splits: list[str],
) -> tuple[dict[str, list[dict]], development.DevelopmentSceneSelection]:
    if not splits or any(split not in ("train", "validation") for split in splits):
        raise ValueError("only permits train and validation")
    mapping = read_scene_mapping(derived_root / config.source_contract.scene_mapping_relative_path)
    selection = development.select_development_scenes(mapping, config.source_contract)
    payloads = {}
    for split in splits:
        payload = (derived_root / config.output_relative_dir / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != AUDITED_SOURCE_SHA256[split]:
            raise ValueError(f"{split} source differs from audited artifact")
        payloads[split] = payload
    records_by_split = {}
    seen = set()
    for split, payload in payloads.items():
        sources = [json.loads(line) for line in payload.splitlines()]
        records = [json.loads(line) for line in
                   (derived_root / FACTORIZED_DIR / f"{split}.jsonl").read_bytes().splitlines()]
        if len(records) != len(sources) or len(records) != config.source_contract.expected_sample_counts[split]:
            raise ValueError(f"{split} frozen sample count mismatch")
        source_by_token = {row["sample_token"]: row for row in sources}
        for row in records:
            token = row["sample_token"]
            if token in seen or token not in source_by_token:
                raise ValueError("duplicate or unknown factorized sample token")
            seen.add(token)
            source = source_by_token[token]
            if any(row[key] != value for key, value in source.items()):
                raise ValueError("factorized artifact changed frozen source fields")
            if row["split"] != split or row["scene_token"] not in selection.scene_tokens_by_split[split]:
                raise ValueError("factorized sample outside frozen scene split")
            if row["split_mapping_sha256"] != mapping["scene_split_mapping_sha256"]:
                raise ValueError("source mapping mismatch")
            if (row["factorized_action_rule_version"] != FACTORIZED_ACTION_RULE_VERSION
                    or row["source_future_trajectory_version"] != SOURCE_VERSION):
                raise ValueError("frozen target version mismatch")
            for direction, actions in (("longitudinal", LONGITUDINAL_ACTIONS), ("lateral", LATERAL_ACTIONS)):
                if (row[f"{direction}_action_valid"] is not True
                        or row[f"{direction}_action"] not in actions
                        or not isinstance(row[f"{direction}_action_reason"], str)):
                    raise ValueError("unexpected invalid frozen factorized target")
            if row["factorized_action_joint_valid"] is not True:
                raise ValueError("unexpected invalid frozen joint target")
        records_by_split[split] = records
    return records_by_split, selection


def write_review(records: list[dict], output: Path, nuscenes_root: Path) -> None:
    cards = []
    for row in records:
        frames = []
        for token, path, time, motion, valid in zip(*(row[key] for key in (
            "historical_sample_tokens", "historical_cam_front_paths",
            "historical_relative_times_sec", "ego_motion_history", "history_valid_mask",
        ))):
            if valid:
                image = (nuscenes_root / path).resolve()
                if not image.is_relative_to(nuscenes_root.resolve()) or not image.is_file():
                    raise ValueError("review image missing or outside dataset root")
                image_path = os.path.relpath(image, output.parent)
                frames.append(f'<div><img src="{html.escape(image_path, quote=True)}">'
                              f'<p>{html.escape(token)} / {time:.3f}s / valid=true</p>'
                              f'<pre>{html.escape(json.dumps(motion, indent=2))}</pre></div>')
            else:
                frames.append('<div>Padding: null / valid=false</div>')
        points = [[0, 0], *row["future_waypoints"]]
        scale = 240 / max(2, *(abs(value) * 2 for point in points for value in point))
        line = ' '.join(f'{150-y*scale},{270-x*scale}' for x, y in points)
        cards.append(f'<article><h2>{html.escape(row["split"] + " / " + row["sample_token"])}</h2>'
                     f'<p>{row["longitudinal_action"]} + {row["lateral_action"]}</p>'
                     '<div class="frames">' + ''.join(frames) + '</div>'
                     f'<svg viewBox="0 0 300 540"><polyline points="{line}" fill="none" stroke="blue"/>'
                     '<circle cx="150" cy="270" r="3"/></svg>'
                     f'<pre>future_waypoints={html.escape(json.dumps(row["future_waypoints"]))}</pre></article>')
    output.write_text('<!doctype html><meta charset="utf-8"><title>Temporal dataset review</title>'
                      '<style>body{font:14px sans-serif;margin:24px}.frames{display:flex;gap:12px}'
                      '.frames>div{flex:1;min-width:0}img{width:100%}pre{white-space:pre-wrap}'
                      'svg{height:320px}article{border-bottom:1px solid #aaa}</style>'
                      '<h1>REVIEWER-ONLY GT FUTURE INFORMATION</h1>'
                      '<p>NOT AVAILABLE TO MODEL AT INFERENCE. Oldest to current. '
                      'BEV: x forward/up, y left/left, meters; dot = current anchor. '
                      'Human review pending.</p>' + ''.join(cards), encoding="utf-8")


def build_dataset(
    derived_root: Path, nuscenes_root: Path, settings: dict,
    splits: list[str], reader_factory: Callable[[], NuScenesReader], *,
    availability_only: bool = False,
) -> dict:
    if (settings["temporal_dataset_schema_version"] != SCHEMA_VERSION
            or settings["history_policy"] != "null_padding"
            or settings["history_order"] != "oldest_to_current"):
        raise ValueError("temporal schema/policy mismatch")
    length = settings["history_length"]
    candidates = settings["availability_lengths"]
    if any(type(value) is not int or value < 1 for value in [length, *candidates]):
        raise ValueError("history lengths must be positive integers")
    config = load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    output = development.resolve_derived_path(derived_root, settings["output_relative_dir"])
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("temporal dataset must be outside repository")
    if not availability_only and output.exists():
        raise FileExistsError(f"temporal dataset already exists: {output}")
    records, selection = intake(derived_root, config, splits)
    counters = development.IsolationCounters()
    reader = development.GuardedNuScenesReader(
        reader_factory(), selection.allowed_scene_tokens, selection.forbidden_test_scene_tokens, counters,
    )
    outputs = {}
    availability = {}
    review = []
    for split, rows in records.items():
        counts = {candidate: 0 for candidate in candidates}
        built = []
        review_groups = set()
        for row in rows:
            history = collect_history(reader, row, max(length, *candidates))
            for candidate in candidates:
                counts[candidate] += len(history) >= candidate
            record = build_record(row, history, length)
            validate_record(record)
            if any(record[key] != row[key] for key in FACTORIZED_FIELDS):
                raise ValueError("temporal output changed frozen factorized targets")
            built.append(record)
            group = (all(record["history_valid_mask"]), record["lateral_action"])
            if group not in review_groups:
                review_groups.add(group)
                review.append(record)
        outputs[split] = built
        availability[split] = {str(key): {"full_count": count, "total": len(rows),
                                       "ratio": count / len(rows)} for key, count in counts.items()}
    summary = {
        **settings, "history_availability": availability,
        "splits": {split: summarize(rows) for split, rows in outputs.items()},
        "factorized_target_preservation": True,
        "scene_overlap": len({r["scene_token"] for r in outputs.get("train", [])}
                             & {r["scene_token"] for r in outputs.get("validation", [])}),
        "access_evidence": asdict(counters),
        "input_fields": INPUT_FIELDS, "target_fields": TARGET_FIELDS,
        "human_review_status": "pending", "phase0_4a_gate": "NOT_YET_PASS",
    }
    if not availability_only:
        output.mkdir(parents=True)
        for split, rows in outputs.items():
            write_jsonl_records(rows, output / f"{split}.jsonl")
        write_jsonl_records(review, output / "review_records.jsonl")
        write_review(review, output / "review.html", nuscenes_root)
        write_canonical_json(summary, output / "dataset_summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/validation temporal waypoint dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_4_temporal_dataset.yaml")
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--nuscenes-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--split", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--availability-only", action="store_true")
    args = parser.parse_args(argv)
    if args.derived_root is None or args.nuscenes_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or provide root arguments")
    from nuscenes.nuscenes import NuScenes
    settings = yaml.safe_load(args.config.read_text())
    summary = build_dataset(args.derived_root, args.nuscenes_root, settings, args.split,
                            lambda: NuScenes(version="v1.0-trainval", dataroot=str(args.nuscenes_root), verbose=False),
                            availability_only=args.availability_only)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

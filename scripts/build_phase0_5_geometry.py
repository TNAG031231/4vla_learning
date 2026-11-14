#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from nuscenes.nuscenes import NuScenes
from scripts.build_phase0_4_temporal_dataset import intake
from src.phase0 import development_projection as development
from src.phase0.phase0_4_source_projection import load_config
from src.phase0.phase0_5_geometry import PROTOCOL, Grid, build_geometry, load_geometry, save_geometry
from src.phase0.phase0_5_visualization import render_geometry


def run(*, dataset_root: Path, derived_root: Path, settings: dict,
        splits: list[str], limit: int = 0) -> dict:
    if settings["protocol_version"] != PROTOCOL:
        raise ValueError("unsupported geometry protocol")
    if limit < 0 or settings["max_agents"] <= 0 or settings["review_samples_per_split"] < 0:
        raise ValueError("sample limit/review count must be nonnegative and max_agents positive")
    tolerance = settings["annotation_camera_tolerance_sec"]
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 0.075:
        raise ValueError("annotation/camera tolerance must be within the frozen 0.075s tolerance")
    grid = Grid(**settings["grid"])
    output = development.resolve_derived_path(derived_root, settings["output_relative_dir"])
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("geometry artifacts must be outside the repository")
    if output.exists():
        raise FileExistsError(f"geometry output already exists: {output}")
    source_config = load_config(ROOT / "configs/phase0_4_source_projection.yaml", ROOT)
    records, selection = intake(derived_root, source_config, splits)
    counters = development.IsolationCounters()
    reader = development.GuardedNuScenesReader(
        NuScenes(version="v1.0-trainval", dataroot=str(dataset_root), verbose=False),
        selection.allowed_scene_tokens, selection.forbidden_test_scene_tokens, counters,
    )
    output.mkdir(parents=True)
    summary = {"protocol_version": PROTOCOL, "settings": settings, "grid": grid.metadata(),
               "information_role": "evaluator_only_gt", "splits": {},
               "source_relative_dir": source_config.output_relative_dir,
               "sampling": "all records" if not limit else "evenly spaced source record indices",
               "metadata_loading": "nuScenes SDK tables; sample/annotation traversal restricted to development scenes"}
    for split, rows in records.items():
        source_count = len(rows)
        if limit and limit < len(rows):
            rows = [rows[i] for i in np.linspace(0, len(rows) - 1, limit, dtype=int)]
        directory = output / split
        directory.mkdir()
        counts = Counter()
        largest = 0
        max_offset = 0.0
        review_groups = set()
        with (directory / "index.jsonl").open("x") as index:
            for number, source in enumerate(rows, 1):
                arrays, metadata = build_geometry(
                    reader, source, grid, max_agents=settings["max_agents"],
                    annotation_camera_tolerance_sec=tolerance,
                )
                filename = f"{source['sample_token']}.npz"
                save_geometry(directory / filename, arrays, metadata)
                index.write(json.dumps({"sample_token": source["sample_token"],
                                        "scene_token": source["scene_token"], "file": filename}) + "\n")
                counts.update(metadata["object_counts"])
                largest = max(largest, metadata["instance_count"])
                max_offset = max(max_offset, *map(abs, metadata["annotation_camera_offsets_sec"]))
                group = (bool((arrays["agent_category_id"] == 2).any()),
                         source["longitudinal_action"], source["lateral_action"])
                if group not in review_groups and len(review_groups) < settings["review_samples_per_split"]:
                    restored, sidecar = load_geometry(directory / filename)
                    render_geometry(restored, sidecar, directory / f"review_{source['sample_token']}.png")
                    review_groups.add(group)
                if number % 250 == 0:
                    print(f"{split}: {number}/{len(rows)}", flush=True)
        summary["splits"][split] = {
            "source_count": source_count, "sample_count": len(rows),
            "scene_count": len({row["scene_token"] for row in rows}),
            "object_counts": dict(counts), "maximum_instances": largest,
            "maximum_annotation_camera_offset_sec": max_offset, "review_images": len(review_groups),
        }
    summary["access_evidence"] = asdict(counters)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluator-only temporal boxes and BEV occupancy.")
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("NUSCENES_ROOT"))
    parser.add_argument("--derived-root", type=Path, default=os.environ.get("VLA_DERIVED_ROOT"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase0_5_geometry.yaml")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--limit", type=int, default=0, help="Evenly spaced samples per split; 0 uses all.")
    parser.add_argument("--output-relative-dir", help="New directory under derived root; existing outputs are preserved.")
    args = parser.parse_args()
    if args.dataset_root is None or args.derived_root is None:
        parser.error("set NUSCENES_ROOT and VLA_DERIVED_ROOT or pass both root arguments")
    settings = yaml.safe_load(args.config.read_text())
    if args.output_relative_dir:
        settings["output_relative_dir"] = args.output_relative_dir
    result = run(dataset_root=args.dataset_root, derived_root=args.derived_root,
                 settings=settings, splits=args.splits, limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

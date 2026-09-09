from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from data.inspect_nuscenes_sample import TrajectoryPoint
from scripts import analyze_phase0_4_factorized_features as analyzer
from src.phase0.phase0_4_factorized_targets import extract_motion_features
from src.phase0 import phase0_4_source_projection as projection
from test_phase0_4_source_projection import setup_pipeline


def trajectory(xs, ys=None, headings=None):
    ys = [0.0] * 7 if ys is None else ys
    headings = [0.0] * 7 if headings is None else headings
    return tuple(
        TrajectoryPoint(f"sample-{i}", i * 0.5, x, y, heading)
        for i, (x, y, heading) in enumerate(zip(xs, ys, headings))
    )


def test_straight_forward_hand_calculation():
    result = extract_motion_features(trajectory(range(7)))
    assert result == {
        "forward_displacement_m": 6.0,
        "path_length_m": 6.0,
        "start_speed_proxy_mps": 2.0,
        "end_speed_proxy_mps": 2.0,
        "delta_speed_proxy_mps": 0.0,
        "final_lateral_displacement_m": 0.0,
        "max_left_displacement_m": 0.0,
        "max_right_displacement_m": 0.0,
        "max_abs_lateral_displacement_m": 0.0,
        "final_heading_delta_rad": 0.0,
        "max_abs_heading_delta_rad": 0.0,
    }


@pytest.mark.parametrize("xs,start,end,delta", [
    ([0, 1, 3, 6, 10, 15, 21], 2, 12, 10),
    ([0, 6, 11, 15, 18, 20, 21], 12, 2, -10),
])
def test_acceleration_and_deceleration(xs, start, end, delta):
    result = extract_motion_features(trajectory(xs))
    assert result["forward_displacement_m"] == 21
    assert result["path_length_m"] == 21
    assert result["start_speed_proxy_mps"] == start
    assert result["end_speed_proxy_mps"] == end
    assert result["delta_speed_proxy_mps"] == delta


@pytest.mark.parametrize("direction", [1, -1])
def test_lateral_motion_uses_planar_distance(direction):
    result = extract_motion_features(trajectory(
        [3 * i for i in range(7)], [direction * 4 * i for i in range(7)],
    ))
    assert result["path_length_m"] == 30
    assert result["start_speed_proxy_mps"] == 10
    assert result["end_speed_proxy_mps"] == 10
    assert result["final_lateral_displacement_m"] == direction * 24
    assert result["max_left_displacement_m"] == (24 if direction == 1 else 0)
    assert result["max_right_displacement_m"] == (-24 if direction == -1 else 0)
    assert result["max_abs_lateral_displacement_m"] == 24


def test_heading_and_lateral_extrema_are_not_final_values():
    result = extract_motion_features(trajectory(
        range(7), [0, 2, -3, 1, 0, 0, 1], [0, 0.2, -0.4, 0.3, 0.1, 0, -0.1],
    ))
    assert result["max_left_displacement_m"] == 2
    assert result["max_right_displacement_m"] == -3
    assert result["max_abs_lateral_displacement_m"] == 3
    assert result["final_lateral_displacement_m"] == 1
    assert result["final_heading_delta_rad"] == -0.1
    assert result["max_abs_heading_delta_rad"] == 0.4
    assert result["path_length_m"] == pytest.approx(
        math.sqrt(5) + math.sqrt(26) + math.sqrt(17) + 2 * math.sqrt(2) + 1,
    )


def test_anchor_and_actual_time_intervals_are_preserved():
    points = list(trajectory([0, 1, 3, 6, 10, 15, 21]))
    points[1] = replace(points[1], t_sec=0.575)
    points[6] = replace(points[6], t_sec=3.05)
    result = extract_motion_features(points)
    assert result["start_speed_proxy_mps"] == pytest.approx(1 / 0.575)
    assert result["end_speed_proxy_mps"] == pytest.approx(6 / 0.55)
    assert len(points) == 7
    with pytest.raises(ValueError, match="7 points including anchor"):
        extract_motion_features(points[1:])


@pytest.fixture
def source_artifacts(tmp_path, monkeypatch):
    config = projection.load_config(
        ROOT / "configs/phase0_4_source_projection.yaml", ROOT,
    )
    kwargs, _, _ = setup_pipeline(tmp_path, config)
    monkeypatch.setattr(
        projection.trainval_source, "load_audit_index", lambda *paths: {},
    )
    receipt = projection.build_source_projection(**kwargs)
    monkeypatch.setattr(analyzer, "AUDITED_SOURCE_SHA256", {
        split: output["sha256"] for split, output in receipt["outputs"].items()
    })
    return kwargs["derived_root"], config.output_relative_dir


def test_frozen_producer_to_analyzer_cli(source_artifacts, monkeypatch, capsys):
    root, _ = source_artifacts
    opened = []
    read_bytes = Path.read_bytes

    def tracked_read(path):
        opened.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    before = sorted(root.rglob("*"))
    assert analyzer.main(["--derived-root", str(root)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert sorted(root.rglob("*")) == before
    assert output["provenance_notice"] == analyzer.PROVENANCE_NOTICE
    assert {path.name for path in opened if path.suffix == ".jsonl"} == {
        "train.jsonl", "validation.jsonl",
    }
    for split in ("train", "validation"):
        result = output["splits"][split]
        assert result["sample_count"] == 1
        assert result["source_audit_record_non_null_count"] == 0
        for distribution in result["feature_distributions"].values():
            assert set(distribution) == set(analyzer.PERCENTILE_NAMES)
    assert output["access_evidence"]["source_records_parsed"] == {
        "train": 1, "validation": 1,
    }


def test_legacy_action_only_changes_provenance_group(source_artifacts, monkeypatch):
    root, relative = source_artifacts
    before = analyzer.analyze_sources(root, ["train"])["splits"]["train"]
    path = root / relative / "train.jsonl"
    record = json.loads(path.read_bytes())
    record["source_legacy_meta_action"] = "arbitrary-provenance"
    record["source_audit_record"] = {"synthetic": True}
    payload = json.dumps(record).encode() + b"\n"
    path.write_bytes(payload)
    monkeypatch.setitem(
        analyzer.AUDITED_SOURCE_SHA256, "train", hashlib.sha256(payload).hexdigest(),
    )
    after = analyzer.analyze_sources(root, ["train"])["splits"]["train"]
    assert after["feature_distributions"] == before["feature_distributions"]
    assert set(after["provenance_only_breakdown"]) == {"arbitrary-provenance"}
    assert after["source_audit_record_non_null_count"] == 1
    for forbidden in (
        "longitudinal_action", "lateral_action", "factorized_action_joint_valid",
        "factorized_action_rule_version",
    ):
        assert forbidden not in json.dumps(after)


def test_test_split_rejected_before_file_access(tmp_path, monkeypatch):
    def forbidden_read(*args, **kwargs):
        pytest.fail("test split must fail before file access")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="only permits"):
        analyzer.analyze_sources(tmp_path, ["train", "test"])
    with pytest.raises(SystemExit) as error:
        analyzer.main(["--derived-root", str(tmp_path), "--split", "test"])
    assert error.value.code == 2


def test_unapproved_bytes_rejected_before_any_semantic_parse(
    source_artifacts, monkeypatch,
):
    root, relative = source_artifacts
    (root / relative / "validation.jsonl").write_bytes(b'{"split":"test"}\n')

    def forbidden_parse(*args, **kwargs):
        pytest.fail("unapproved bytes must fail before JSON parsing")

    monkeypatch.setattr(analyzer.json, "loads", forbidden_parse)
    with pytest.raises(ValueError, match="validation source SHA-256"):
        analyzer.analyze_sources(root, ["train", "validation"])


def test_linear_percentiles():
    result = analyzer.summarize([{"distance": 0.0}, {"distance": 100.0}])
    assert result["distance"] == dict(zip(
        analyzer.PERCENTILE_NAMES, analyzer.PERCENTILES,
    ))

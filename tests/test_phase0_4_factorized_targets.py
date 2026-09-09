from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
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
from scripts import audit_phase0_4_candidate_factorized_rules as audit
from src.phase0.phase0_4_factorized_targets import (
    evaluate_candidate_rules,
    extract_motion_features,
    load_candidate_parameters,
)
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


@pytest.fixture
def candidate_parameters():
    return load_candidate_parameters(ROOT / "configs/phase0_4_candidate_rules.yaml")


def motion_features(**changes):
    features = extract_motion_features(trajectory(range(7)))
    features.update(changes)
    return features


@pytest.mark.parametrize("changes,action,reason", [
    ({"path_length_m": 0.4, "forward_displacement_m": 0.4}, "stop", "confident_stop"),
    ({"path_length_m": 0.5, "forward_displacement_m": 0.5}, None, "stop_boundary"),
    ({"path_length_m": 0.6, "forward_displacement_m": 0.6}, None, "stop_boundary"),
    ({}, "keep", "confident_keep"),
    ({"delta_speed_proxy_mps": 1.3}, "accelerate", "confident_accelerate"),
    ({"delta_speed_proxy_mps": -1.3}, "decelerate", "confident_decelerate"),
    ({"delta_speed_proxy_mps": 0.8}, None, "speed_boundary"),
    ({"delta_speed_proxy_mps": 1.2}, None, "speed_boundary"),
    ({"delta_speed_proxy_mps": -0.8}, None, "speed_boundary"),
    ({"delta_speed_proxy_mps": -1.2}, None, "speed_boundary"),
    ({"forward_displacement_m": 0.9}, None, "low_forward_displacement"),
    ({"forward_displacement_m": 1.0}, "keep", "confident_keep"),
    ({"forward_displacement_m": 0.0}, None, "low_forward_displacement"),
])
def test_candidate_longitudinal(changes, action, reason, candidate_parameters):
    result = evaluate_candidate_rules(
        motion_features(**changes), candidate_parameters,
    ).candidate_longitudinal
    assert result.candidate_action == action
    assert result.candidate_valid is (action is not None)
    assert result.candidate_reason == reason


@pytest.mark.parametrize("final_y,excursion,heading,action,reason", [
    (1.3, 1.3, 0.0, "left", "confident_lateral_displacement"),
    (-1.3, 1.3, 0.0, "right", "confident_lateral_displacement"),
    (1.3, 1.3, -0.12, "left", "confident_lateral_displacement"),
    (0.0, 0.0, 0.0, "straight", "confident_straight"),
    (0.8, 0.8, 0.0, None, "lateral_displacement_boundary"),
    (-0.8, 0.8, 0.0, None, "lateral_displacement_boundary"),
    (1.2, 1.2, 0.0, None, "lateral_displacement_boundary"),
    (-1.2, 1.2, 0.0, None, "lateral_displacement_boundary"),
    (1.3, 1.3, -0.121, None, "lateral_heading_sign_conflict"),
    (-1.3, 1.3, 0.121, None, "lateral_heading_sign_conflict"),
    (0.0, 1.3, 0.0, None, "lateral_excursion_without_final_displacement"),
    (0.0, 0.0, 0.13, None, "heading_without_lateral_displacement"),
    (0.0, 0.0, -0.13, None, "heading_without_lateral_displacement"),
    (0.0, 0.8, 0.0, None, "lateral_straight_boundary"),
    (0.0, 1.2, 0.0, None, "lateral_straight_boundary"),
    (0.0, 0.0, 0.08, None, "lateral_straight_boundary"),
    (0.0, 0.0, 0.12, None, "lateral_straight_boundary"),
])
def test_candidate_lateral(
    final_y, excursion, heading, action, reason, candidate_parameters,
):
    result = evaluate_candidate_rules(motion_features(
        final_lateral_displacement_m=final_y,
        max_abs_lateral_displacement_m=excursion,
        final_heading_delta_rad=heading,
    ), candidate_parameters).candidate_lateral
    assert result.candidate_action == action
    assert result.candidate_valid is (action is not None)
    assert result.candidate_reason == reason


@pytest.mark.parametrize("delta,final_y,longitudinal,lateral", [
    (0.0, 1.0, "keep", None),
    (1.0, 1.5, None, "left"),
    (0.0, 0.0, "keep", "straight"),
    (1.0, 1.0, None, None),
])
def test_candidate_independent_validity(
    delta, final_y, longitudinal, lateral, candidate_parameters,
):
    result = evaluate_candidate_rules(motion_features(
        delta_speed_proxy_mps=delta,
        final_lateral_displacement_m=final_y,
        max_abs_lateral_displacement_m=abs(final_y),
    ), candidate_parameters)
    assert result.candidate_longitudinal.candidate_action == longitudinal
    assert result.candidate_lateral.candidate_action == lateral
    assert result.candidate_joint_valid is (
        longitudinal is not None and lateral is not None
    )


def audit_record(template, token, points, legacy="keep", audited=False):
    record = deepcopy(template)
    record["sample_token"] = token
    record["future_ego_trajectory"] = [
        asdict(replace(point, future_sample_token=token if i == 0 else f"{token}-{i}"))
        for i, point in enumerate(points)
    ]
    record["source_legacy_meta_action"] = legacy
    record["source_audit_record"] = {"synthetic": True} if audited else None
    return record


def test_audit_joint_combinations_and_provenance(
    source_artifacts, candidate_parameters,
):
    root, relative = source_artifacts
    template = json.loads((root / relative / "train.jsonl").read_bytes())
    records = [
        audit_record(template, "keep", trajectory(range(7)), audited=True),
        audit_record(template, "stop", trajectory([0] * 7), "stop", True),
        audit_record(template, "decelerate-left", trajectory(
            [0, 6, 11, 15, 18, 20, 21], [i * 0.3 for i in range(7)],
        ), "left_lateral", True),
        audit_record(template, "accelerate-right", trajectory(
            [0, 1, 3, 6, 10, 15, 21], [-i * 0.3 for i in range(7)],
        ), "right_lateral"),
        audit_record(template, "longitudinal-only", trajectory(
            range(7), [i / 6 for i in range(7)],
        )),
        audit_record(template, "lateral-only", trajectory(
            [0] * 7, [i * 0.3 for i in range(7)],
        ), "left_lateral"),
        audit_record(template, "both-invalid", trajectory(
            [0] * 7, [i / 6 for i in range(7)],
        )),
    ]
    result = audit.audit_split(records, candidate_parameters)
    assert result["sample_count"] == 7
    assert result["candidate_longitudinal"]["valid_count"] == 5
    assert result["candidate_lateral"]["valid_count"] == 5
    assert result["independent_candidate_validity"] == {
        "longitudinal_valid_lateral_invalid": 1,
        "longitudinal_invalid_lateral_valid": 1,
        "both_valid": 4, "both_invalid": 1,
    }
    joint = result["candidate_joint_valid_combinations"]
    assert {key: count for key, count in joint.items() if count} == {
        "keep + straight": 1, "stop + straight": 1,
        "decelerate + left": 1, "accelerate + right": 1,
    }
    diagnostic = result["provenance_only_diagnostic"]
    assert diagnostic["audited_subset"]["sample_count"] == 3
    left = diagnostic["all_source"]["by_legacy_provenance"]["left_lateral"]
    assert left["compared_candidate_direction"] == "lateral"
    assert left["agreement_count"] == 2
    for group in diagnostic.values():
        for summary in group["by_legacy_provenance"].values():
            assert summary["sample_count"] == (
                summary["valid_count"] + summary["invalid_count"]
            )
    changed = deepcopy(records)
    for record in changed:
        record["source_legacy_meta_action"] = "accelerate"
    updated = audit.audit_split(changed, candidate_parameters)
    for field in (
        "candidate_longitudinal", "candidate_lateral", "independent_candidate_validity",
        "candidate_joint_valid_combinations",
    ):
        assert result[field] == updated[field]
    for category, rows in result["boundary_samples"].items():
        assert [row["candidate_result"] for row in rows] == [
            row["candidate_result"] for row in updated["boundary_samples"][category]
        ]


def test_boundary_samples_nearest_edge_then_token(
    source_artifacts, candidate_parameters,
):
    root, relative = source_artifacts
    template = json.loads((root / relative / "train.jsonl").read_bytes())
    records = [
        audit_record(template, f"sample-{i:02}", trajectory([0, 1, 2, 3, 4, 5, 6.5]))
        for i in range(12)
    ]
    records.append(audit_record(
        template, "nearest-edge", trajectory([0, 1, 2, 3, 4, 5, 6.40625]),
    ))
    result = audit.audit_split(records, candidate_parameters)
    assert result == audit.audit_split(list(reversed(records)), candidate_parameters)
    rows = result["boundary_samples"]["longitudinal_speed_boundary"]
    assert [row["sample_token"] for row in rows] == [
        "nearest-edge", *[f"sample-{i:02}" for i in range(9)],
    ]
    for row in rows:
        assert row["distance_unit"] == "m/s"
        lateral = row["candidate_result"]["candidate_lateral"]
        assert lateral["candidate_action"] == "straight"
        assert {
            "scene_token", "split", "cam_front_path", "motion_features",
            "source_legacy_meta_action", "source_audit_record_present",
        } <= row.keys()


@pytest.mark.parametrize("reason,changes,distance,unit", [
    ("speed_boundary", {"delta_speed_proxy_mps": 1.0}, 0.2, "m/s"),
    ("stop_boundary", {"path_length_m": 0.5, "forward_displacement_m": 0.5}, 0.1, "m"),
    ("low_forward_displacement", {"forward_displacement_m": 0.9}, 0.1, "m"),
    ("lateral_displacement_boundary", {"final_lateral_displacement_m": 1.0}, 0.2, "m"),
    ("lateral_heading_sign_conflict", {"final_heading_delta_rad": -0.2}, 0.08, "rad"),
    ("lateral_excursion_without_final_displacement",
     {"max_abs_lateral_displacement_m": 1.3}, 0.1, "m"),
    ("heading_without_lateral_displacement",
     {"final_heading_delta_rad": 0.2}, 0.08, "rad"),
])
def test_boundary_distance_definitions(
    reason, changes, distance, unit, candidate_parameters,
):
    actual_distance, actual_unit = audit.boundary_distance(
        reason, motion_features(**changes), candidate_parameters,
    )
    assert actual_distance == pytest.approx(distance)
    assert actual_unit == unit


def test_candidate_audit_cli_reads_only_sources(
    source_artifacts, candidate_parameters, monkeypatch, capsys,
):
    root, _ = source_artifacts
    reads = []
    read_bytes = Path.read_bytes

    def tracked_read(path):
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    before = {path: read_bytes(path) for path in root.rglob("*") if path.is_file()}
    assert audit.main(["--derived-root", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "provisional"
    assert result["candidate_parameters"] == asdict(candidate_parameters)
    assert result["provenance_notice"] == audit.PROVENANCE_NOTICE
    assert {path.name for path in reads if path.suffix == ".jsonl"} == {
        "train.jsonl", "validation.jsonl",
    }
    assert before == {
        path: read_bytes(path) for path in root.rglob("*") if path.is_file()
    }
    for split, summary in result["splits"].items():
        assert summary["sample_count"] == 1
        assert summary["source_sha256"] == analyzer.AUDITED_SOURCE_SHA256[split]


def test_candidate_audit_test_split_before_access(
    tmp_path, candidate_parameters, monkeypatch,
):
    def forbidden_read(*args, **kwargs):
        pytest.fail("test split must be rejected before reading any file")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    with pytest.raises(ValueError, match="only permits"):
        audit.audit_sources(tmp_path, ["train", "test"], candidate_parameters)
    with pytest.raises(SystemExit) as error:
        audit.main(["--split", "test"])
    assert error.value.code == 2


def test_candidate_audit_sha_before_parse(
    source_artifacts, candidate_parameters, monkeypatch,
):
    root, relative = source_artifacts
    (root / relative / "validation.jsonl").write_bytes(b'{"split":"test"}\n')

    def forbidden_parse(*args, **kwargs):
        pytest.fail("SHA mismatch must be rejected before semantic parsing")

    monkeypatch.setattr(analyzer.json, "loads", forbidden_parse)
    with pytest.raises(ValueError, match="SHA-256"):
        audit.audit_sources(root, ["train", "validation"], candidate_parameters)

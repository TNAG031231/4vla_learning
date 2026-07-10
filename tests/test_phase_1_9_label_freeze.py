import csv
import importlib.util
from collections import Counter
from pathlib import Path
import sys
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
FREEZE_SCRIPT = DATA_DIRECTORY / "validate_label_freeze.py"
BASE_AUDIT = DATA_DIRECTORY / "phase_1_7_manual_audit.csv"
SUPPLEMENT_AUDIT = DATA_DIRECTORY / "phase_1_7_lateral_supplement_audit.csv"
NUSCENES_ROOT = DATA_DIRECTORY / "nuscenes"
EXPECTED_DISTRIBUTION = Counter(
    {
        "accelerate": 6,
        "decelerate": 16,
        "keep": 55,
        "left_lateral": 5,
        "right_lateral": 5,
        "stop": 21,
    }
)


def load_freeze_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "validate_label_freeze",
        FREEZE_SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(DATA_DIRECTORY))
    try:
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(DATA_DIRECTORY))
    return module


def build_records(module: ModuleType):
    records = []
    for action, count in EXPECTED_DISTRIBUTION.items():
        for index in range(count):
            records.append(
                module.FreezeAuditRecord(
                    source_audit="base" if len(records) < 100 else "supplement",
                    sample_token=f"{action}-{index}",
                    scene_token="scene",
                    timestamp=1,
                    cam_front_path="samples/CAM_FRONT/image.jpg",
                    historical_derived_action=action,
                    reviewed_action=action,
                    frozen_action=action,
                    action_match="yes",
                    label_rule_version="phase-1.6-meta-action-v0.2",
                    action_confidence="high",
                    boundary_flags=("speed_threshold_boundary",)
                    if not records
                    else (),
                    uncertainty_reason="none",
                    trajectory_points=7,
                    trajectory_last_t_sec=3.0,
                    trajectory_complete=True,
                    nearby_agent_count=1,
                    nearby_vru_count=1 if not records else 0,
                    has_vru="yes" if not records else "no",
                    cam_front_exists=True,
                )
            )
    return tuple(records)


def write_audit(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "sample_token",
                "scene_token",
                "timestamp",
                "reviewed_action",
                "derived_action",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {"scene_token": "scene", "timestamp": "1", **row}
            for row in rows
        )


def test_merge_audits_preserves_base_then_supplement_order(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    base_path = tmp_path / "base.csv"
    supplement_path = tmp_path / "supplement.csv"
    write_audit(
        base_path,
        tuple(
            {
                "sample_token": f"base-{index}",
                "reviewed_action": "keep",
                "derived_action": "keep",
            }
            for index in range(100)
        ),
    )
    write_audit(
        supplement_path,
        tuple(
            {
                "sample_token": f"supplement-{index}",
                "reviewed_action": "left_lateral",
                "derived_action": "left_lateral",
            }
            for index in range(8)
        ),
    )

    rows = freeze.read_and_merge_audits(base_path, supplement_path)

    assert len(rows) == 108
    assert [(row.source_audit, row.sample_token) for row in rows[:2]] == [
        ("base", "base-0"),
        ("base", "base-1"),
    ]
    assert [(row.source_audit, row.sample_token) for row in rows[-2:]] == [
        ("supplement", "supplement-6"),
        ("supplement", "supplement-7"),
    ]


def test_duplicate_sample_token_is_rejected(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    base_path = tmp_path / "base.csv"
    supplement_path = tmp_path / "supplement.csv"
    duplicate_row = {"sample_token": "same", "reviewed_action": "keep", "derived_action": "keep"}
    write_audit(base_path, (duplicate_row,))
    write_audit(supplement_path, (duplicate_row,))

    with pytest.raises(ValueError, match="duplicate sample_token"):
        freeze.read_and_merge_audits(base_path, supplement_path)


def test_invalid_reviewed_action_is_rejected(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    base_path = tmp_path / "base.csv"
    supplement_path = tmp_path / "supplement.csv"
    write_audit(base_path, ({"sample_token": "a", "reviewed_action": "turn_left", "derived_action": "keep"},))
    write_audit(supplement_path, ())

    with pytest.raises(ValueError, match="unsupported reviewed_action"):
        freeze.read_and_merge_audits(base_path, supplement_path)


def test_summary_rejects_non_v0_2_rule_version() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[0] = records[0].__class__(
        **{**records[0].__dict__, "label_rule_version": "phase-1.6-meta-action-v0.1"}
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert "label_rule_version" in summary.failures[0]


def test_summary_rejects_action_match_shortfall() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[0] = records[0].__class__(
        **{**records[0].__dict__, "action_match": "no"}
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert "action_match=107/108" in summary.failures


def test_summary_rejects_inconsistent_action_match_value() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[0] = records[0].__class__(
        **{
            **records[0].__dict__,
            "reviewed_action": "stop",
            "action_match": "yes",
        }
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert "action_match value is inconsistent with frozen_action" in summary.failures


def test_summary_rejects_wrong_action_distribution() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[0] = records[0].__class__(
        **{**records[0].__dict__, "frozen_action": "keep"}
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert any("frozen action distribution" in failure for failure in summary.failures)


def test_summary_rejects_unavailable_or_single_class_vru_coverage() -> None:
    freeze = load_freeze_module()
    unavailable = list(build_records(freeze))
    unavailable[0] = unavailable[0].__class__(
        **{**unavailable[0].__dict__, "has_vru": "not_available"}
    )
    single_class = [
        record.__class__(**{**record.__dict__, "has_vru": "no"})
        for record in build_records(freeze)
    ]

    assert "has_vru contains unsupported values" in freeze.summarize_freeze_records(
        tuple(unavailable)
    ).failures
    assert "has_vru requires both yes and no coverage" in freeze.summarize_freeze_records(
        tuple(single_class)
    ).failures


def test_summary_rejects_missing_boundary_coverage() -> None:
    freeze = load_freeze_module()
    records = [
        record.__class__(
            **{
                **record.__dict__,
                "boundary_flags": (),
                "uncertainty_reason": "none",
            }
        )
        for record in build_records(freeze)
    ]

    summary = freeze.summarize_freeze_records(tuple(records))

    assert "boundary_case_count=0" in summary.failures


def test_summary_rejects_wrong_audit_source_counts() -> None:
    freeze = load_freeze_module()
    records = [
        record.__class__(**{**record.__dict__, "source_audit": "base"})
        for record in build_records(freeze)
    ]

    summary = freeze.summarize_freeze_records(tuple(records))

    assert "audit source counts must be base=100 and supplement=8" in summary.failures


def test_summary_rejects_incomplete_trajectory_or_missing_camera() -> None:
    freeze = load_freeze_module()
    incomplete = list(build_records(freeze))
    incomplete[0] = incomplete[0].__class__(
        **{**incomplete[0].__dict__, "trajectory_complete": False}
    )
    missing_camera = list(build_records(freeze))
    missing_camera[0] = missing_camera[0].__class__(
        **{**missing_camera[0].__dict__, "cam_front_exists": False}
    )

    assert "trajectory_complete=107/108" in freeze.summarize_freeze_records(
        tuple(incomplete)
    ).failures
    assert "cam_front_exists=107/108" in freeze.summarize_freeze_records(
        tuple(missing_camera)
    ).failures


def test_camera_path_must_be_relative_and_exist(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    image_path = tmp_path / "samples" / "CAM_FRONT" / "image.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    assert freeze.validate_cam_front_path("samples/CAM_FRONT/image.jpg", tmp_path)
    assert not freeze.validate_cam_front_path(str(image_path), tmp_path)
    assert not freeze.validate_cam_front_path("samples/CAM_FRONT/missing.jpg", tmp_path)


def test_correct_synthetic_summary_passes() -> None:
    freeze = load_freeze_module()

    summary = freeze.summarize_freeze_records(build_records(freeze))

    assert summary.failures == ()
    assert summary.action_distribution == EXPECTED_DISTRIBUTION
    assert summary.action_match_count == 108


def test_real_nuscenes_mini_108_sample_integration_gate(tmp_path: Path) -> None:
    if not (NUSCENES_ROOT / "v1.0-mini").is_dir():
        pytest.skip("nuScenes mini is unavailable under data/nuscenes")

    freeze = load_freeze_module()
    output_path = tmp_path / "label_freeze.csv"
    exit_code = freeze.main(
        (
            "--base-audit",
            str(BASE_AUDIT),
            "--supplement-audit",
            str(SUPPLEMENT_AUDIT),
            "--data-config",
            str(PROJECT_ROOT / "configs" / "data.yaml"),
            "--action-config",
            str(PROJECT_ROOT / "configs" / "action_rules.yaml"),
            "--dataroot",
            str(NUSCENES_ROOT),
            "--output",
            str(output_path),
        )
    )

    assert exit_code == 0
    with output_path.open(encoding="utf-8", newline="") as input_file:
        assert len(tuple(csv.DictReader(input_file))) == 108

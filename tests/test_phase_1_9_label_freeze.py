import csv
import importlib.util
from collections import Counter
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
FREEZE_SCRIPT = DATA_DIRECTORY / "validate_label_freeze.py"
BASE_AUDIT = DATA_DIRECTORY / "phase_1_7_manual_audit.csv"
SUPPLEMENT_AUDIT = DATA_DIRECTORY / "phase_1_7_lateral_supplement_audit.csv"
NUSCENES_ROOT = DATA_DIRECTORY / "nuscenes"
COMMITTED_FREEZE_AUDIT = DATA_DIRECTORY / "phase_1_9_label_freeze_audit.csv"
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
            historical_action = action
            if action == "accelerate" and index == 0:
                historical_action = "keep"
            elif action == "decelerate" and index < 3:
                historical_action = "keep"
            elif action == "stop" and index == 0:
                historical_action = "keep"
            records.append(
                module.FreezeAuditRecord(
                    source_audit="base" if len(records) < 100 else "supplement",
                    sample_token=f"{action}-{index}",
                    scene_token="scene",
                    timestamp=1,
                    cam_front_path="samples/CAM_FRONT/image.jpg",
                    historical_derived_action=historical_action,
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
                "cam_front_path",
                "reviewed_action",
                "derived_action",
                "label_correct",
                "trajectory_alignment_correct",
                "agent_alignment_correct",
                "label_rule_version",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "scene_token": "scene",
                "timestamp": "1",
                "cam_front_path": "samples/CAM_FRONT/image.jpg",
                "label_correct": "yes",
                "trajectory_alignment_correct": "yes",
                "agent_alignment_correct": "yes",
                "label_rule_version": "phase-1.6-meta-action-v0.1",
                **row,
            }
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


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("label_correct", "uncertain", "label_correct"),
        ("trajectory_alignment_correct", "no", "trajectory_alignment_correct"),
        ("agent_alignment_correct", "no", "agent_alignment_correct"),
        (
            "label_rule_version",
            "phase-1.6-meta-action-v0.2",
            "historical label_rule_version",
        ),
    ),
)
def test_historical_audit_integrity_is_required(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    freeze = load_freeze_module()
    base_path = tmp_path / "base.csv"
    supplement_path = tmp_path / "supplement.csv"
    write_audit(
        base_path,
        (
            {
                "sample_token": "sample",
                "reviewed_action": "keep",
                "derived_action": "keep",
                field: value,
            },
        ),
    )
    write_audit(supplement_path, ())

    with pytest.raises(ValueError, match=error):
        freeze.read_and_merge_audits(base_path, supplement_path)


def test_historical_label_correct_no_is_preserved(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    base_path = tmp_path / "base.csv"
    supplement_path = tmp_path / "supplement.csv"
    write_audit(
        base_path,
        (
            {
                "sample_token": "historical-no",
                "reviewed_action": "keep",
                "derived_action": "keep",
                "label_correct": "no",
            },
        ),
    )
    write_audit(supplement_path, ())

    rows = freeze.read_and_merge_audits(base_path, supplement_path)

    assert rows[0].label_correct == "no"


def test_historical_label_correct_relation_and_distribution_are_frozen() -> None:
    freeze = load_freeze_module()
    rows = []
    for index in range(108):
        label_correct = "yes" if index < 103 else "no"
        reviewed_action = "keep" if label_correct == "yes" else "stop"
        rows.append(
            freeze.HistoricalAuditRow(
                source_audit="base" if index < 100 else "supplement",
                sample_token=f"sample-{index}",
                scene_token="scene",
                timestamp=str(index),
                cam_front_path=f"samples/CAM_FRONT/{index}.jpg",
                historical_derived_action="keep",
                reviewed_action=reviewed_action,
                label_correct=label_correct,
                trajectory_alignment_correct="yes",
                agent_alignment_correct="yes",
                label_rule_version="phase-1.6-meta-action-v0.1",
            )
        )

    freeze.validate_historical_audit_integrity(tuple(rows))

    invalid_rows = list(rows)
    invalid_rows[0] = invalid_rows[0].__class__(
        **{**invalid_rows[0].__dict__, "label_correct": "no"}
    )
    invalid_rows[103] = invalid_rows[103].__class__(
        **{**invalid_rows[103].__dict__, "label_correct": "yes"}
    )
    with pytest.raises(ValueError, match="label_correct does not match"):
        freeze.validate_historical_audit_integrity(tuple(invalid_rows))


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


def test_summary_rejects_historical_action_transition_change() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[1] = records[1].__class__(
        **{**records[1].__dict__, "historical_derived_action": "keep"}
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert any("transition distribution" in failure for failure in summary.failures)


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

    assert "boundary_flag_case_count=0" in summary.failures


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


def test_camera_path_must_stay_inside_dataroot(tmp_path: Path) -> None:
    freeze = load_freeze_module()
    dataroot = tmp_path / "dataroot"
    dataroot.mkdir()
    outside_path = tmp_path / "outside.jpg"
    outside_path.touch()

    assert not freeze.validate_cam_front_path("../outside.jpg", dataroot)


def test_build_rejects_historical_cam_front_path_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    freeze = load_freeze_module()
    current_path = "samples/CAM_FRONT/current.jpg"
    image_path = tmp_path / current_path
    image_path.parent.mkdir(parents=True)
    image_path.touch()
    historical_row = freeze.HistoricalAuditRow(
        source_audit="base",
        sample_token="sample",
        scene_token="scene",
        timestamp="1",
        cam_front_path="samples/CAM_FRONT/historical.jpg",
        historical_derived_action="keep",
        reviewed_action="keep",
        label_correct="yes",
        trajectory_alignment_correct="yes",
        agent_alignment_correct="yes",
        label_rule_version="phase-1.6-meta-action-v0.1",
    )
    derived_record = SimpleNamespace(
        scene_token="scene",
        timestamp_us=1,
        cam_front_path=current_path,
        derived_action="keep",
        label_rule_version="phase-1.6-meta-action-v0.2",
        action_confidence="high",
        boundary_flags=(),
        uncertainty_reason="none",
    )
    trajectory = SimpleNamespace(
        points=tuple(SimpleNamespace(t_sec=index * 0.5) for index in range(7)),
        is_truncated=False,
    )
    monkeypatch.setattr(
        freeze,
        "derive_sample_record",
        lambda **_: derived_record,
    )
    monkeypatch.setattr(
        freeze,
        "extract_future_ego_trajectory",
        lambda **_: trajectory,
    )
    monkeypatch.setattr(
        freeze,
        "get_nearby_agents",
        lambda **_: SimpleNamespace(agents=()),
    )

    _, failures = freeze.build_freeze_records(
        nuscenes=SimpleNamespace(
            get=lambda table_name, token: {"scene_token": "scene", "timestamp": 1}
        ),
        historical_rows=(historical_row,),
        rules=SimpleNamespace(horizon_sec=3.0, sample_interval_sec=0.5),
        dataroot=tmp_path,
        agent_radius_m=50.0,
        time_tolerance_sec=0.075,
    )

    assert any(
        "sample: historical CAM_FRONT path mismatch" in failure
        and "historical.jpg" in failure
        and "current.jpg" in failure
        for failure in failures
    )


def test_summary_separates_strict_boundary_flags_from_diagnostics() -> None:
    freeze = load_freeze_module()
    records = list(build_records(freeze))
    records[1] = records[1].__class__(
        **{
            **records[1].__dict__,
            "uncertainty_reason": "trajectory_speed_proxy_only",
        }
    )

    summary = freeze.summarize_freeze_records(tuple(records))

    assert summary.boundary_flag_case_count == 1
    assert summary.diagnostic_case_count == 2
    assert summary.uncertainty_reason_distribution == Counter(
        {"none": 107, "trajectory_speed_proxy_only": 1}
    )


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
        regenerated_reader = csv.DictReader(input_file)
        regenerated_rows = tuple(regenerated_reader)
        regenerated_fieldnames = regenerated_reader.fieldnames
    with COMMITTED_FREEZE_AUDIT.open(encoding="utf-8", newline="") as input_file:
        committed_reader = csv.DictReader(input_file)
        committed_rows = tuple(committed_reader)
        committed_fieldnames = committed_reader.fieldnames

    assert regenerated_fieldnames == committed_fieldnames
    assert len(regenerated_rows) == 108
    assert len(committed_rows) == 108
    assert regenerated_rows == committed_rows

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_ego_motion_inputs import require_manifest_sha256
from src.actions.schema import ACTION_SCHEMA
from src.baselines.ego_motion import (
    EgoMotionFeatures,
    audit_manifest_rows,
    build_test_label_access_evidence,
    parse_ego_motion_audit_sample,
)
from src.phase0.protocol import validate_manifest


def manifest_row(
    *,
    sample_token: str = "sample",
    scene_token: str = "scene",
    split: str = "train",
    availability: str = "full",
    meta_action: str = "keep",
) -> dict[str, object]:
    motion_by_availability = {
        "full": {
            "speed_mps": 4.0,
            "longitudinal_acceleration_mps2": 0.5,
            "yaw_rate_radps": 0.1,
            "availability": "full",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": 0.5,
        },
        "partial": {
            "speed_mps": 3.0,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": 0.2,
            "availability": "partial",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": None,
        },
        "unavailable": {
            "speed_mps": None,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": None,
            "availability": "unavailable",
            "history_interval_sec": None,
            "acceleration_interval_sec": None,
        },
    }
    return {
        "sample_token": sample_token,
        "scene_token": scene_token,
        "split": split,
        "current_ego_motion": motion_by_availability[availability],
        "label_rule_version": "phase-1.6-meta-action-v0.2",
        "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
        "split_mapping_sha256": "a" * 64,
        "meta_action": meta_action,
        "future_ego_trajectory": [[999.0, 999.0]],
        "nearby_agents": [{"translation_m": [999.0, 999.0, 999.0]}],
        "current_ego_pose": {"translation_m": [999.0, 999.0, 999.0]},
        "cam_front_path": "samples/CAM_FRONT/image.jpg",
    }


@pytest.mark.parametrize("availability", ("full", "partial", "unavailable"))
def test_parser_supports_all_motion_availability_values(availability: str) -> None:
    sample = parse_ego_motion_audit_sample(
        manifest_row(availability=availability)
    )

    assert sample.features.availability == availability


def test_partial_motion_keeps_acceleration_null() -> None:
    features = parse_ego_motion_audit_sample(
        manifest_row(availability="partial")
    ).features

    assert features.longitudinal_acceleration_mps2 is None
    assert features.acceleration_interval_sec is None


def test_unavailable_motion_keeps_all_motion_values_null() -> None:
    features = parse_ego_motion_audit_sample(
        manifest_row(availability="unavailable")
    ).features

    assert features == EgoMotionFeatures(
        speed_mps=None,
        longitudinal_acceleration_mps2=None,
        yaw_rate_radps=None,
        availability="unavailable",
        history_interval_sec=None,
        acceleration_interval_sec=None,
    )


@pytest.mark.parametrize("invalid_value", (float("nan"), float("inf"), -float("inf")))
def test_parser_rejects_non_finite_motion_values(invalid_value: float) -> None:
    row = manifest_row()
    row["current_ego_motion"]["speed_mps"] = invalid_value

    with pytest.raises(ValueError, match="must be finite"):
        parse_ego_motion_audit_sample(row)


@pytest.mark.parametrize("invalid_value", ("4.0", True, [], {}))
def test_parser_rejects_wrong_motion_types(invalid_value: object) -> None:
    row = manifest_row()
    row["current_ego_motion"]["speed_mps"] = invalid_value

    with pytest.raises(ValueError, match="number or null"):
        parse_ego_motion_audit_sample(row)


def test_parser_rejects_missing_motion_mapping() -> None:
    row = manifest_row()
    del row["current_ego_motion"]

    with pytest.raises(ValueError, match="must be a mapping"):
        parse_ego_motion_audit_sample(row)


def test_parser_rejects_missing_full_motion_field() -> None:
    row = manifest_row()
    del row["current_ego_motion"]["speed_mps"]

    with pytest.raises(ValueError, match="full motion must provide"):
        parse_ego_motion_audit_sample(row)


def test_forbidden_future_and_scene_fields_do_not_affect_features() -> None:
    original = manifest_row()
    changed = copy.deepcopy(original)
    changed["future_ego_trajectory"] = [[-123.0, 456.0]]
    changed["nearby_agents"] = [{"category": "sentinel"}]
    changed["current_ego_pose"] = {"translation_m": [-1e9, 1e9, 7.0]}
    changed["cam_front_path"] = "sentinel-image-path"

    assert (
        parse_ego_motion_audit_sample(original).features
        == parse_ego_motion_audit_sample(changed).features
    )


class NonTrainLabelGuard(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key == "meta_action":
            raise AssertionError("test meta_action was accessed")
        return super().get(key, default)


def test_test_meta_action_does_not_enter_statistics_path() -> None:
    row = NonTrainLabelGuard(
        manifest_row(split="test", meta_action="TEST_LABEL_SENTINEL")
    )

    result = audit_manifest_rows((row,))

    assert result["sample_count_by_split"]["test"] == 1
    assert "test_label_accessed" not in result


def test_validation_meta_action_does_not_enter_statistics_path() -> None:
    row = NonTrainLabelGuard(
        manifest_row(split="validation", meta_action="VALIDATION_LABEL_SENTINEL")
    )

    result = audit_manifest_rows((row,))

    assert result["sample_count_by_split"]["validation"] == 1


def test_manifest_validator_reads_test_meta_action_for_contract(
    tmp_path: Path,
) -> None:
    row = manifest_row(split="test", meta_action="TEST_LABEL_SENTINEL")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported action"):
        validate_manifest(manifest_path)


def test_audit_evidence_distinguishes_schema_validation_from_label_use() -> None:
    evidence = build_test_label_access_evidence()

    assert evidence["test_label_access_policy"] == {
        "schema_validation": "allowed_and_performed",
        "input_statistics": "forbidden_and_not_performed",
        "class_conditional_statistics": "forbidden_and_not_performed",
        "threshold_selection": "forbidden_and_not_performed",
        "model_selection": "forbidden_and_not_performed",
        "classification_metrics": "forbidden_and_not_performed",
        "failure_case_analysis": "forbidden_and_not_performed",
    }
    assert evidence["test_label_used_for_statistics"] is False
    assert evidence["test_label_used_for_threshold_selection"] is False
    assert evidence["test_label_used_for_model_selection"] is False
    assert "test_label_accessed" not in evidence


def test_only_train_produces_class_conditional_statistics() -> None:
    rows = (
        manifest_row(sample_token="train", scene_token="train-scene"),
        manifest_row(
            sample_token="validation",
            scene_token="validation-scene",
            split="validation",
            meta_action="TEST_LABEL_SENTINEL",
        ),
        manifest_row(
            sample_token="test",
            scene_token="test-scene",
            split="test",
            availability="partial",
            meta_action="TEST_LABEL_SENTINEL",
        ),
    )

    result = audit_manifest_rows(rows)

    assert tuple(result["train_class_conditional_motion_statistics"]) == ACTION_SCHEMA
    assert result["train_class_conditional_motion_statistics"]["keep"][
        "speed_mps"
    ]["count"] == 1
    assert "validation_class_conditional_motion_statistics" not in result
    assert "test_motion_statistics" not in result
    assert "test_class_conditional_motion_statistics" not in result
    assert "test_classification_metrics" not in result
    assert "test_failure_cases" not in result
    assert result["motion_availability_by_split"]["test"] == {
        "full": 0,
        "partial": 1,
        "unavailable": 0,
    }


def test_audit_is_deterministic() -> None:
    rows = (
        manifest_row(sample_token="a", scene_token="scene-a"),
        manifest_row(
            sample_token="b",
            scene_token="scene-b",
            split="validation",
            availability="partial",
        ),
    )

    assert audit_manifest_rows(rows) == audit_manifest_rows(rows)


def test_manifest_sha_mismatch_fails_before_audit(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_manifest_sha256(manifest_path, "0" * 64)


def test_audit_rejects_scene_split_reassignment() -> None:
    rows = (
        manifest_row(sample_token="train", scene_token="shared", split="train"),
        manifest_row(
            sample_token="test",
            scene_token="shared",
            split="test",
            meta_action="TEST_LABEL_SENTINEL",
        ),
    )

    with pytest.raises(ValueError, match="scene_token spans splits"):
        audit_manifest_rows(rows)

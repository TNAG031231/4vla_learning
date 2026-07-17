from collections import Counter
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.manifest import write_jsonl_records
from src.phase0.protocol import (
    ManifestSample,
    assign_scene_splits,
    complete_action_distribution,
    evaluate_classification,
    iter_manifest_rows,
    read_manifest_samples,
    select_pilot_scene_tokens,
    validate_manifest,
    validate_scene_split_isolation,
)


def test_scene_level_split_is_reproducible_and_has_six_two_two_scenes() -> None:
    scene_tokens = tuple(f"scene-{index}" for index in range(10))

    first = assign_scene_splits(
        scene_tokens=scene_tokens,
        seed=20260710,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    second = assign_scene_splits(
        scene_tokens=scene_tokens,
        seed=20260710,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert first == second
    assert Counter(first.values()) == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_scene_level_split_rejects_same_scene_across_splits() -> None:
    samples = (
        ManifestSample("sample-a", "scene-a", "keep", "train"),
        ManifestSample("sample-b", "scene-a", "keep", "test"),
    )

    with pytest.raises(ValueError, match="scene_token spans splits"):
        validate_scene_split_isolation(samples)


def test_pilot_scene_selection_is_reproducible_and_covers_all_splits() -> None:
    scene_splits = {
        **{f"train-{index}": "train" for index in range(560)},
        **{f"validation-{index}": "validation" for index in range(140)},
        **{f"test-{index}": "test" for index in range(150)},
    }
    frozen_mapping = dict(scene_splits)

    first = select_pilot_scene_tokens(scene_splits, scene_count=20, seed=20260715)
    second = select_pilot_scene_tokens(scene_splits, scene_count=20, seed=20260715)

    assert first == second
    assert scene_splits == frozen_mapping
    assert len(first) == len(set(first)) == 20
    assert Counter(scene_splits[token] for token in first) == {
        "train": 13,
        "validation": 3,
        "test": 4,
    }


def test_metrics_keep_all_actions_when_split_has_missing_classes() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "stop"),
        predictions=("keep", "keep"),
    )

    assert metrics.sample_count == 2
    assert metrics.class_distribution == {
        "keep": 1,
        "accelerate": 0,
        "decelerate": 0,
        "stop": 1,
        "left_lateral": 0,
        "right_lateral": 0,
    }
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(1 / 9)
    assert tuple(metrics.per_class_f1) == ACTION_SCHEMA
    assert all(len(row) == 6 for row in metrics.confusion_matrix)
    assert len(metrics.confusion_matrix) == 6
    assert metrics.invalid_prediction_count == 0


def test_complete_action_distribution_retains_zero_count_classes() -> None:
    distribution = complete_action_distribution(("stop", "stop"))

    assert distribution == {
        "keep": 0,
        "accelerate": 0,
        "decelerate": 0,
        "stop": 2,
        "left_lateral": 0,
        "right_lateral": 0,
    }


def test_invalid_prediction_counts_as_ground_truth_false_negative() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "keep"),
        predictions=("keep", "INVALID"),
    )

    assert metrics.sample_count == 2
    assert metrics.correct_count == 1
    assert metrics.valid_prediction_count == 1
    assert metrics.invalid_prediction_count == 1
    assert metrics.invalid_output_rate == pytest.approx(0.5)
    assert metrics.action_parsing_success_rate == pytest.approx(0.5)
    assert (
        metrics.invalid_output_rate + metrics.action_parsing_success_rate
    ) == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.per_class_precision["keep"] == pytest.approx(1.0)
    assert metrics.per_class_recall["keep"] == pytest.approx(0.5)
    assert metrics.per_class_f1["keep"] == pytest.approx(2 / 3)
    assert sum(sum(row) for row in metrics.confusion_matrix) == 1


def test_all_invalid_predictions_keep_ground_truth_support() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "accelerate"),
        predictions=("INVALID", "INVALID"),
    )

    assert metrics.accuracy == 0.0
    assert metrics.valid_prediction_count == 0
    assert metrics.invalid_prediction_count == 2
    assert metrics.invalid_output_rate == 1.0
    assert metrics.action_parsing_success_rate == 0.0
    assert all(sum(row) == 0 for row in metrics.confusion_matrix)
    assert metrics.per_class_recall["keep"] == 0.0
    assert metrics.per_class_recall["accelerate"] == 0.0


def test_ground_truth_invalid_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported action"):
        evaluate_classification(
            ground_truth=("INVALID",),
            predictions=("keep",),
        )


def test_protocol_owns_manifest_reader_without_baseline_dependency(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"sample_token": "sample", "scene_token": "scene", "meta_action": "keep", "split": "train", "label_rule_version": "v0"}\n',
        encoding="utf-8",
    )

    samples = read_manifest_samples(manifest_path)

    assert samples == (ManifestSample("sample", "scene", "keep", "train", "v0"),)
    assert "src.baselines" not in (PROJECT_ROOT / "src/phase0/protocol.py").read_text()


def manifest_row(sample_token: str = "sample") -> dict[str, object]:
    return {
        "manifest_schema_version": "phase0_audited_seed_subset_v1",
        "sample_token": sample_token,
        "scene_token": "scene",
        "timestamp": 1,
        "cam_front_path": "samples/CAM_FRONT/image.jpg",
        "current_ego_pose": {
            "frame": "nuScenes_global",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "timestamp_us": 1,
            "timestamp_source": "CAM_FRONT_sample_data",
        },
        "current_ego_motion": {
            "speed_mps": None,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": None,
            "source": "ego_pose_past_difference",
            "timestamp_source": "CAM_FRONT_sample_data",
            "availability": "unavailable",
            "history_interval_sec": None,
            "acceleration_interval_sec": None,
            "unavailable_reason": "insufficient_past_history",
        },
        "coordinate_metadata": {
            "current_ego_pose": {
                "translation_unit": "meter",
                "rotation_order": "wxyz",
                "timestamp_unit": "microsecond",
                "timestamp_source": "CAM_FRONT_sample_data",
            },
            "current_ego_motion": {
                "speed_unit": "meter_per_second",
                "longitudinal_acceleration_unit": "meter_per_second_squared",
                "yaw_rate_unit": "radian_per_second",
                "timestamp_source": "CAM_FRONT_sample_data",
            },
        },
        "future_ego_trajectory": [],
        "nearby_agents": [],
        "meta_action": "keep",
        "label_rule_version": "phase-1.6-meta-action-v0.2",
        "safety_rule_version": "not_available",
        "split": "train",
    }


def trainval_manifest_row(sample_token: str = "sample") -> dict[str, object]:
    row = manifest_row(sample_token)
    row.update(
        {
            "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
            "audit_status": "unaudited",
            "source_audit_record": None,
            "official_split": "train",
            "split_seed": 20260710,
            "split_strategy_version": (
                "official_train_scene_label_stratified_v1"
            ),
            "split_mapping_sha256": "a" * 64,
        }
    )
    return row


def test_complete_manifest_validator_accepts_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(manifest_row()) + "\n",
        encoding="utf-8",
    )

    summary = validate_manifest(manifest_path)

    assert summary.sample_count == 1
    assert summary.scene_count == 1
    assert summary.manifest_schema_version == "phase0_audited_seed_subset_v1"
    assert summary.label_rule_version == "phase-1.6-meta-action-v0.2"


def test_trainval_manifest_validator_accepts_unaudited_contract(
    tmp_path: Path,
) -> None:
    row = trainval_manifest_row()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = validate_manifest(manifest_path)

    assert summary.manifest_schema_version == "phase0_trainval_dataset_manifest_v1"
    assert summary.split_mapping_sha256 == "a" * 64


def test_trainval_manifest_validator_accepts_audited_provenance(
    tmp_path: Path,
) -> None:
    row = trainval_manifest_row()
    row["audit_status"] = "audited"
    row["source_audit_record"] = {
        "source_audit": "base",
        "sample_token": "sample",
        "historical_derived_action": "keep",
        "reviewed_action": "keep",
        "label_correct": "yes",
        "historical_label_rule_version": "phase-1.6-meta-action-v0.1",
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = validate_manifest(manifest_path)

    assert summary.sample_count == 1


def test_trainval_manifest_validator_rejects_missing_audited_provenance(
    tmp_path: Path,
) -> None:
    row = trainval_manifest_row()
    row["audit_status"] = "audited"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        validate_manifest(manifest_path)


def test_trainval_manifest_validator_rejects_audit_record(
    tmp_path: Path,
) -> None:
    row = trainval_manifest_row()
    row["source_audit_record"] = {"source_audit": "unexpected"}
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_audit_record must be null"):
        validate_manifest(manifest_path)


@pytest.mark.parametrize(
    "cam_front_path",
    (
        "/private/data/image.jpg",
        "C:/data/image.jpg",
        "C:\\data\\image.jpg",
        "\\\\server\\share\\image.jpg",
        "samples/CAM_FRONT/../image.jpg",
        "samples/CAM_BACK/image.jpg",
        "other/samples/CAM_FRONT/image.jpg",
    ),
)
def test_manifest_validator_rejects_nonportable_cam_front_path(
    tmp_path: Path,
    cam_front_path: str,
) -> None:
    row = manifest_row()
    row["cam_front_path"] = cam_front_path
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative to NUSCENES_ROOT"):
        validate_manifest(manifest_path)


def test_trainval_manifest_validator_requires_one_mapping_hash(
    tmp_path: Path,
) -> None:
    first = trainval_manifest_row("sample-a")
    second = trainval_manifest_row("sample-b")
    second["split_mapping_sha256"] = "b" * 64
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be singular"):
        validate_manifest(manifest_path)


@pytest.mark.parametrize(
    "invalid_hash",
    (
        "a" * 63,
        "g" * 64,
        "A" * 64,
        123,
    ),
)
def test_trainval_manifest_validator_rejects_invalid_mapping_hash(
    tmp_path: Path,
    invalid_hash: object,
) -> None:
    row = trainval_manifest_row()
    row["split_mapping_sha256"] = invalid_hash
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase 64-character SHA-256"):
        validate_manifest(manifest_path)


def test_streaming_manifest_validator_does_not_call_path_read_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(trainval_manifest_row()) + "\n",
        encoding="utf-8",
    )

    def reject_read_text(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("streaming validator must not call Path.read_text")

    monkeypatch.setattr(Path, "read_text", reject_read_text)

    assert tuple(iter_manifest_rows(manifest_path))[0]["sample_token"] == "sample"
    assert validate_manifest(manifest_path).sample_count == 1


def test_large_generator_is_atomically_written_and_summary_is_returned(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"

    def records():
        for index in range(2_000):
            yield trainval_manifest_row(f"sample-{index:04d}")

    summary = write_jsonl_records(
        records(),
        manifest_path,
        validator=validate_manifest,
    )

    assert summary.sample_count == 2_000
    assert summary.scene_count == 1
    assert summary.split_mapping_sha256 == "a" * 64


def test_streaming_validator_detects_distant_duplicate_token(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("old\n", encoding="utf-8")

    def records():
        for index in range(1_000):
            yield trainval_manifest_row(f"sample-{index:04d}")
        yield trainval_manifest_row("sample-0000")

    with pytest.raises(ValueError, match="duplicate sample_token"):
        write_jsonl_records(
            records(),
            manifest_path,
            validator=validate_manifest,
        )

    assert manifest_path.read_text(encoding="utf-8") == "old\n"
    assert tuple(tmp_path.glob(".manifest.jsonl.*.tmp")) == ()


def test_streaming_validator_detects_distant_scene_split_overlap(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"

    def records():
        first = trainval_manifest_row("sample-first")
        first["scene_token"] = "shared-scene"
        yield first
        for index in range(1_000):
            row = trainval_manifest_row(f"sample-{index:04d}")
            row["scene_token"] = f"scene-{index:04d}"
            yield row
        last = trainval_manifest_row("sample-last")
        last["scene_token"] = "shared-scene"
        last["split"] = "validation"
        yield last

    with pytest.raises(ValueError, match="scene_token spans splits"):
        write_jsonl_records(
            records(),
            manifest_path,
            validator=validate_manifest,
        )

    assert not manifest_path.exists()
    assert tuple(tmp_path.glob(".manifest.jsonl.*.tmp")) == ()


def test_complete_manifest_validator_rejects_duplicate_sample_token(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(manifest_row()) for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_token"):
        validate_manifest(manifest_path)


def test_complete_manifest_validator_requires_shared_timestamp_source(
    tmp_path: Path,
) -> None:
    row = manifest_row()
    row["current_ego_motion"]["timestamp_source"] = "sample_keyframe"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp_source"):
        validate_manifest(manifest_path)


def test_complete_manifest_validator_requires_pose_timestamp_metadata(
    tmp_path: Path,
) -> None:
    row = manifest_row()
    del row["coordinate_metadata"]["current_ego_pose"]["timestamp_unit"]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp_unit"):
        validate_manifest(manifest_path)


def test_complete_manifest_validator_rejects_timestamp_metadata_mismatch(
    tmp_path: Path,
) -> None:
    row = manifest_row()
    row["coordinate_metadata"]["current_ego_pose"][
        "timestamp_source"
    ] = "sample_keyframe"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp_source"):
        validate_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("schema", "manifest_schema_version"),
        ("timestamp_source", "timestamp_source"),
        ("timestamp_unit", "timestamp_unit"),
        ("timestamp_type", "timestamp_us"),
        ("speed_unit", "speed_unit"),
    ),
)
def test_complete_manifest_validator_requires_frozen_protocol_values(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    row = manifest_row()
    if mutation == "schema":
        row["manifest_schema_version"] = "other_schema"
    elif mutation == "timestamp_source":
        row["current_ego_pose"]["timestamp_source"] = "sample_keyframe"
        row["current_ego_motion"]["timestamp_source"] = "sample_keyframe"
        row["coordinate_metadata"]["current_ego_pose"][
            "timestamp_source"
        ] = "sample_keyframe"
        row["coordinate_metadata"]["current_ego_motion"][
            "timestamp_source"
        ] = "sample_keyframe"
    elif mutation == "timestamp_unit":
        row["coordinate_metadata"]["current_ego_pose"][
            "timestamp_unit"
        ] = "second"
    elif mutation == "timestamp_type":
        row["current_ego_pose"]["timestamp_us"] = 1.5
    else:
        row["coordinate_metadata"]["current_ego_motion"][
            "speed_unit"
        ] = "meter"
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        validate_manifest(manifest_path)

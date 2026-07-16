from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
from typing import Final, Sequence

from src.actions.schema import ACTION_SCHEMA, is_valid_action, normalize_action
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


SPLITS: Final = ("train", "validation", "test")
AUDITED_MANIFEST_SCHEMA_VERSION: Final = "phase0_audited_seed_subset_v1"
TRAINVAL_MANIFEST_SCHEMA_VERSION: Final = "phase0_trainval_dataset_manifest_v1"
MANIFEST_SCHEMA_VERSION: Final = AUDITED_MANIFEST_SCHEMA_VERSION
SUPPORTED_MANIFEST_SCHEMA_VERSIONS: Final = frozenset(
    (AUDITED_MANIFEST_SCHEMA_VERSION, TRAINVAL_MANIFEST_SCHEMA_VERSION)
)
PHASE0_SPLIT_SEED: Final = 20260710
OFFICIAL_TRAIN_SCENE_COUNT: Final = 700
OFFICIAL_VAL_SCENE_COUNT: Final = 150
PROJECT_TRAIN_SCENE_COUNT: Final = 560
PROJECT_VALIDATION_SCENE_COUNT: Final = 140
POSE_TIMESTAMP_SOURCE: Final = "CAM_FRONT_sample_data"
MOTION_SOURCE: Final = "ego_pose_past_difference"
POSE_TIMESTAMP_UNIT: Final = "microsecond"
SPEED_UNIT: Final = "meter_per_second"
ACCELERATION_UNIT: Final = "meter_per_second_squared"
YAW_RATE_UNIT: Final = "radian_per_second"


@dataclass(frozen=True)
class ManifestSample:
    sample_token: str
    scene_token: str
    meta_action: str
    split: str
    label_rule_version: str = ""


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    correct_count: int
    valid_prediction_count: int
    invalid_prediction_count: int
    invalid_output_rate: float
    action_parsing_success_rate: float
    class_distribution: dict[str, int]
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ManifestValidationSummary:
    sample_count: int
    scene_count: int
    manifest_schema_version: str
    label_rule_version: str
    motion_availability_distribution: dict[str, int]
    split_mapping_sha256: str | None


def assign_scene_splits(
    scene_tokens: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    unique_scenes = sorted(set(scene_tokens))
    if len(unique_scenes) != len(scene_tokens):
        raise ValueError("scene_tokens must be unique")

    raw_counts = tuple(len(unique_scenes) * ratio for ratio in ratios)
    split_counts = [int(count) for count in raw_counts]
    remaining = len(unique_scenes) - sum(split_counts)
    remainders = sorted(
        range(len(SPLITS)),
        key=lambda index: (raw_counts[index] - split_counts[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        split_counts[index] += 1

    shuffled_scenes = list(unique_scenes)
    random.Random(seed).shuffle(shuffled_scenes)
    assignments = {}
    start = 0
    for split, count in zip(SPLITS, split_counts, strict=True):
        for scene_token in shuffled_scenes[start : start + count]:
            assignments[scene_token] = split
        start += count
    return assignments


def _pilot_split_counts(
    scene_splits: Mapping[str, str],
    scene_count: int,
) -> dict[str, int]:
    available = Counter(scene_splits.values())
    if set(available) != set(SPLITS):
        raise ValueError(
            "pilot selection requires train, validation, and test scenes"
        )
    if scene_count < len(SPLITS) or scene_count > len(scene_splits):
        raise ValueError("pilot scene_count is outside the available scene range")

    raw_counts = {
        split: scene_count * available[split] / len(scene_splits)
        for split in SPLITS
    }
    selected_counts = {
        split: max(1, int(raw_counts[split])) for split in SPLITS
    }
    while sum(selected_counts.values()) < scene_count:
        split = max(
            SPLITS,
            key=lambda name: (
                raw_counts[name] - selected_counts[name],
                -SPLITS.index(name),
            ),
        )
        selected_counts[split] += 1
    while sum(selected_counts.values()) > scene_count:
        candidates = tuple(
            split for split in SPLITS if selected_counts[split] > 1
        )
        split = max(
            candidates,
            key=lambda name: (
                selected_counts[name] - raw_counts[name],
                SPLITS.index(name),
            ),
        )
        selected_counts[split] -= 1
    if any(selected_counts[split] > available[split] for split in SPLITS):
        raise ValueError("pilot scene allocation exceeds available scenes")
    return selected_counts


def select_pilot_scene_tokens(
    scene_splits: Mapping[str, str],
    scene_count: int,
    seed: int,
) -> tuple[str, ...]:
    selected_counts = _pilot_split_counts(scene_splits, scene_count)
    rng = random.Random(seed)
    selected = []
    for split in SPLITS:
        candidates = sorted(
            token for token, assigned_split in scene_splits.items()
            if assigned_split == split
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: selected_counts[split]])
    return tuple(selected)


def validate_scene_split_isolation(samples: Sequence[ManifestSample]) -> None:
    scene_splits: dict[str, str] = {}
    for sample in samples:
        if sample.split not in SPLITS:
            raise ValueError(f"Unsupported split: {sample.split!r}")
        existing_split = scene_splits.setdefault(
            sample.scene_token,
            sample.split,
        )
        if existing_split != sample.split:
            raise ValueError(
                "scene_token spans splits: "
                f"{sample.scene_token} ({existing_split}, {sample.split})"
            )


def complete_action_distribution(actions: Sequence[str]) -> dict[str, int]:
    counts = Counter(normalize_action(action) for action in actions)
    return {action: counts[action] for action in ACTION_SCHEMA}


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row missing {key}")
    return value


def read_manifest_samples(path: Path) -> tuple[ManifestSample, ...]:
    rows = _read_manifest_rows(path)
    samples = []
    for line_number, row in enumerate(rows, 1):
        samples.append(
            ManifestSample(
                sample_token=_required_string(row, "sample_token"),
                scene_token=_required_string(row, "scene_token"),
                meta_action=normalize_action(_required_string(row, "meta_action")),
                split=_required_string(row, "split"),
                label_rule_version=_required_string(row, "label_rule_version"),
            )
        )
    result = tuple(samples)
    validate_scene_split_isolation(result)
    return result


def _read_manifest_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number}: manifest row must be an object")
        rows.append(row)
    return tuple(rows)


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest row missing mapping {key}")
    return value


def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"manifest row missing numeric {key}")
    return float(value)


def is_valid_cam_front_path(path_value: str) -> bool:
    posix_path = PurePosixPath(path_value)
    windows_path = PureWindowsPath(path_value)
    return (
        "\\" not in path_value
        and not posix_path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and ".." not in posix_path.parts
        and len(posix_path.parts) == 3
        and posix_path.parts[:2] == ("samples", "CAM_FRONT")
        and bool(posix_path.name)
    )


def _validate_cam_front_path(row: Mapping[str, object]) -> None:
    path_value = _required_string(row, "cam_front_path")
    if not is_valid_cam_front_path(path_value):
        raise ValueError(
            "cam_front_path must be under samples/CAM_FRONT/ and relative "
            "to NUSCENES_ROOT"
        )


def _validate_source_audit_record(
    row: Mapping[str, object],
    source_audit_record: Mapping[str, object],
) -> None:
    _required_string(source_audit_record, "source_audit")
    if _required_string(source_audit_record, "sample_token") != _required_string(
        row,
        "sample_token",
    ):
        raise ValueError("source audit sample_token must match manifest row")
    normalize_action(
        _required_string(source_audit_record, "historical_derived_action")
    )
    reviewed_action = normalize_action(
        _required_string(source_audit_record, "reviewed_action")
    )
    if reviewed_action != normalize_action(_required_string(row, "meta_action")):
        raise ValueError("source audit reviewed_action must match frozen action")
    if source_audit_record.get("label_correct") not in {"yes", "no"}:
        raise ValueError("source audit label_correct must be yes or no")
    _required_string(source_audit_record, "historical_label_rule_version")


def _validate_audit_fields(
    row: Mapping[str, object],
    schema_version: str,
) -> None:
    audit_status = row.get("audit_status")
    source_audit_record = row.get("source_audit_record")
    if schema_version == TRAINVAL_MANIFEST_SCHEMA_VERSION:
        if audit_status == "unaudited":
            if source_audit_record is not None:
                raise ValueError("unaudited source_audit_record must be null")
            return
        if audit_status == "audited":
            if not isinstance(source_audit_record, Mapping):
                raise ValueError(
                    "audited trainval source_audit_record must be a mapping"
                )
            _validate_source_audit_record(row, source_audit_record)
            return
        raise ValueError(
            "trainval manifest audit_status must be audited or unaudited"
        )
    if audit_status is not None and audit_status != "audited":
        raise ValueError("audited manifest audit_status must be audited")
    if "source_audit_record" in row and not isinstance(
        source_audit_record,
        Mapping,
    ):
        raise ValueError("audited source_audit_record must be a mapping")


def _validate_trainval_split_fields(row: Mapping[str, object]) -> None:
    split = _required_string(row, "split")
    official_split = _required_string(row, "official_split")
    split_seed = row.get("split_seed")
    split_strategy_version = _required_string(row, "split_strategy_version")
    if official_split not in {"train", "val"}:
        raise ValueError("trainval official_split must be train or val")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("trainval split_seed must be an integer")
    if split_seed != PHASE0_SPLIT_SEED:
        raise ValueError("trainval split_seed does not match the frozen seed")
    if split_strategy_version != SPLIT_STRATEGY_VERSION:
        raise ValueError("unsupported trainval split_strategy_version")
    if official_split == "val" and split != "test":
        raise ValueError("official val scenes must map to project test")
    if official_split == "train" and split not in {"train", "validation"}:
        raise ValueError(
            "official train scenes must map to project train or validation"
        )


def _validate_pose(row: Mapping[str, object]) -> str:
    pose = _required_mapping(row, "current_ego_pose")
    _required_string(pose, "frame")
    translation = pose.get("translation_m")
    rotation = pose.get("rotation_wxyz")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("current_ego_pose.translation_m must have three values")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError("current_ego_pose.rotation_wxyz must have four values")
    timestamp_us = pose.get("timestamp_us")
    if not isinstance(timestamp_us, int) or isinstance(timestamp_us, bool):
        raise ValueError("current_ego_pose.timestamp_us must be an integer")
    timestamp_source = _required_string(pose, "timestamp_source")
    if timestamp_source != POSE_TIMESTAMP_SOURCE:
        raise ValueError("unsupported current_ego_pose.timestamp_source")
    return timestamp_source


def _validate_motion(row: Mapping[str, object]) -> tuple[str, str]:
    motion = _required_mapping(row, "current_ego_motion")
    availability = _required_string(motion, "availability")
    if availability not in {"full", "partial", "unavailable"}:
        raise ValueError(f"unsupported motion availability: {availability!r}")
    if _required_string(motion, "source") != MOTION_SOURCE:
        raise ValueError("unsupported current_ego_motion.source")
    timestamp_source = _required_string(motion, "timestamp_source")
    if timestamp_source != POSE_TIMESTAMP_SOURCE:
        raise ValueError("unsupported current_ego_motion.timestamp_source")
    for field in ("speed_mps", "yaw_rate_radps"):
        value = motion.get(field)
        if availability == "unavailable":
            if value is not None:
                raise ValueError(f"unavailable motion must set {field} to null")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"available motion must provide {field}")
    acceleration = motion.get("longitudinal_acceleration_mps2")
    acceleration_interval = motion.get("acceleration_interval_sec")
    history_interval = motion.get("history_interval_sec")
    if availability == "full":
        if not isinstance(acceleration, (int, float)):
            raise ValueError("full motion must provide acceleration")
        if not isinstance(acceleration_interval, (int, float)):
            raise ValueError("full motion must provide acceleration interval")
        if not isinstance(history_interval, (int, float)):
            raise ValueError("full motion must provide history interval")
        if motion.get("unavailable_reason") is not None:
            raise ValueError("full motion cannot have unavailable_reason")
    elif availability == "partial":
        if acceleration is not None or acceleration_interval is not None:
            raise ValueError("partial motion must not provide acceleration")
        if not isinstance(history_interval, (int, float)):
            raise ValueError("partial motion must provide history interval")
        if not isinstance(motion.get("unavailable_reason"), str):
            raise ValueError("partial motion must explain unavailable acceleration")
    else:
        if any(
            value is not None
            for value in (acceleration, acceleration_interval, history_interval)
        ):
            raise ValueError("unavailable motion must not provide intervals")
        if not isinstance(motion.get("unavailable_reason"), str):
            raise ValueError("unavailable motion must provide unavailable_reason")
    return availability, timestamp_source


def _validate_coordinate_metadata(
    row: Mapping[str, object],
) -> tuple[str, str]:
    metadata = _required_mapping(row, "coordinate_metadata")
    pose_metadata = _required_mapping(metadata, "current_ego_pose")
    motion_metadata = _required_mapping(metadata, "current_ego_motion")
    _required_string(pose_metadata, "translation_unit")
    _required_string(pose_metadata, "rotation_order")
    if _required_string(pose_metadata, "timestamp_unit") != POSE_TIMESTAMP_UNIT:
        raise ValueError("unsupported current_ego_pose.timestamp_unit")
    pose_timestamp_source = _required_string(
        pose_metadata,
        "timestamp_source",
    )
    if _required_string(motion_metadata, "speed_unit") != SPEED_UNIT:
        raise ValueError("unsupported current_ego_motion.speed_unit")
    if (
        _required_string(motion_metadata, "longitudinal_acceleration_unit")
        != ACCELERATION_UNIT
    ):
        raise ValueError("unsupported current_ego_motion.longitudinal_acceleration_unit")
    if _required_string(motion_metadata, "yaw_rate_unit") != YAW_RATE_UNIT:
        raise ValueError("unsupported current_ego_motion.yaw_rate_unit")
    return pose_timestamp_source, _required_string(
        motion_metadata,
        "timestamp_source",
    )


def validate_manifest(path: Path) -> ManifestValidationSummary:
    rows = _read_manifest_rows(path)
    if not rows:
        raise ValueError("manifest must contain at least one row")
    sample_tokens = set()
    schema_versions = set()
    label_rule_versions = set()
    manifest_samples = []
    motion_availability = Counter[str]()
    split_mapping_hashes = set()
    for row in rows:
        sample_token = _required_string(row, "sample_token")
        if sample_token in sample_tokens:
            raise ValueError(f"duplicate sample_token: {sample_token}")
        sample_tokens.add(sample_token)
        schema_version = _required_string(row, "manifest_schema_version")
        if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError("unsupported manifest_schema_version")
        schema_versions.add(schema_version)
        label_rule_versions.add(_required_string(row, "label_rule_version"))
        normalize_action(_required_string(row, "meta_action"))
        _required_number(row, "timestamp")
        _validate_cam_front_path(row)
        _validate_audit_fields(row, schema_version)
        if schema_version == TRAINVAL_MANIFEST_SCHEMA_VERSION:
            _validate_trainval_split_fields(row)
            split_mapping_hashes.add(
                _required_string(row, "split_mapping_sha256")
            )
        pose_timestamp_source = _validate_pose(row)
        availability, motion_timestamp_source = _validate_motion(row)
        if motion_timestamp_source != pose_timestamp_source:
            raise ValueError("pose and motion timestamp_source must match")
        motion_availability[availability] += 1
        pose_metadata_source, motion_metadata_source = (
            _validate_coordinate_metadata(row)
        )
        if pose_metadata_source != pose_timestamp_source:
            raise ValueError("pose timestamp_source must match metadata")
        if motion_metadata_source != motion_timestamp_source:
            raise ValueError("motion timestamp_source must match metadata")
        if not isinstance(row.get("future_ego_trajectory"), list):
            raise ValueError("manifest row missing future_ego_trajectory")
        if not isinstance(row.get("nearby_agents"), list):
            raise ValueError("manifest row missing nearby_agents")
        manifest_samples.append(
            ManifestSample(
                sample_token=sample_token,
                scene_token=_required_string(row, "scene_token"),
                meta_action=_required_string(row, "meta_action"),
                split=_required_string(row, "split"),
                label_rule_version=_required_string(row, "label_rule_version"),
            )
        )
    validate_scene_split_isolation(manifest_samples)
    if len(schema_versions) != 1:
        raise ValueError("manifest_schema_version must be singular")
    if len(label_rule_versions) != 1:
        raise ValueError("label_rule_version must be singular")
    schema_version = schema_versions.pop()
    if (
        schema_version == TRAINVAL_MANIFEST_SCHEMA_VERSION
        and len(split_mapping_hashes) != 1
    ):
        raise ValueError("trainval split_mapping_sha256 must be singular")
    return ManifestValidationSummary(
        sample_count=len(rows),
        scene_count=len({sample.scene_token for sample in manifest_samples}),
        manifest_schema_version=schema_version,
        label_rule_version=label_rule_versions.pop(),
        motion_availability_distribution=dict(motion_availability),
        split_mapping_sha256=(
            split_mapping_hashes.pop() if split_mapping_hashes else None
        ),
    )


def evaluate_classification(
    ground_truth: Sequence[str],
    predictions: Sequence[str],
) -> ClassificationMetrics:
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have equal length")

    confusion = [[0 for _ in ACTION_SCHEMA] for _ in ACTION_SCHEMA]
    normalized_ground_truth = []
    invalid_prediction_count = 0
    for expected, predicted in zip(ground_truth, predictions, strict=True):
        normalized_expected = normalize_action(expected)
        normalized_ground_truth.append(normalized_expected)
        if not is_valid_action(predicted):
            invalid_prediction_count += 1
            continue
        expected_index = ACTION_SCHEMA.index(normalized_expected)
        predicted_index = ACTION_SCHEMA.index(predicted)
        confusion[expected_index][predicted_index] += 1

    class_distribution = complete_action_distribution(normalized_ground_truth)
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    for index, action in enumerate(ACTION_SCHEMA):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = class_distribution[action] - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_denominator = precision + recall
        per_class_precision[action] = precision
        per_class_recall[action] = recall
        per_class_f1[action] = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        )

    correct_count = sum(
        confusion[index][index] for index in range(len(ACTION_SCHEMA))
    )
    sample_count = len(ground_truth)
    valid_prediction_count = sample_count - invalid_prediction_count
    return ClassificationMetrics(
        sample_count=sample_count,
        correct_count=correct_count,
        valid_prediction_count=valid_prediction_count,
        invalid_prediction_count=invalid_prediction_count,
        invalid_output_rate=(
            invalid_prediction_count / sample_count if sample_count else 0.0
        ),
        action_parsing_success_rate=(
            valid_prediction_count / sample_count if sample_count else 0.0
        ),
        class_distribution=class_distribution,
        accuracy=correct_count / sample_count if sample_count else 0.0,
        macro_f1=sum(per_class_f1.values()) / len(ACTION_SCHEMA),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=tuple(tuple(row) for row in confusion),
    )

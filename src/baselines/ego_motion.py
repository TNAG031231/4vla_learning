from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from statistics import fmean, pstdev
from typing import Final

from src.actions.schema import (
    ACCELERATE,
    ACTION_SCHEMA,
    DECELERATE,
    KEEP,
    LEFT_LATERAL,
    RIGHT_LATERAL,
    STOP,
    normalize_action,
)
from src.phase0.protocol import (
    SPLITS,
    ClassificationMetrics,
    complete_action_distribution,
    evaluate_classification,
    validate_sha256,
)


AUDIT_FIELDS: Final = (
    "speed_mps",
    "longitudinal_acceleration_mps2",
    "yaw_rate_radps",
    "history_interval_sec",
    "acceleration_interval_sec",
)
AVAILABILITY_VALUES: Final = ("full", "partial", "unavailable")
STATISTIC_NAMES: Final = (
    "count",
    "min",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
    "mean",
    "std",
)


@dataclass(frozen=True)
class EgoMotionFeatures:
    speed_mps: float | None
    longitudinal_acceleration_mps2: float | None
    yaw_rate_radps: float | None
    availability: str
    history_interval_sec: float | None
    acceleration_interval_sec: float | None


@dataclass(frozen=True)
class EgoMotionAuditSample:
    sample_token: str
    scene_token: str
    split: str
    features: EgoMotionFeatures
    label_rule_version: str
    manifest_schema_version: str
    split_mapping_sha256: str
    train_meta_action: str | None


@dataclass(frozen=True)
class EgoMotionRuleThresholds:
    stop_speed_threshold_mps: float
    lateral_yaw_rate_threshold_radps: float
    accelerate_threshold_mps2: float
    decelerate_threshold_mps2: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")

    def canonical_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.stop_speed_threshold_mps,
            self.lateral_yaw_rate_threshold_radps,
            self.accelerate_threshold_mps2,
            self.decelerate_threshold_mps2,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "stop_speed_threshold_mps": self.stop_speed_threshold_mps,
            "lateral_yaw_rate_threshold_radps": (
                self.lateral_yaw_rate_threshold_radps
            ),
            "accelerate_threshold_mps2": self.accelerate_threshold_mps2,
            "decelerate_threshold_mps2": self.decelerate_threshold_mps2,
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EgoMotionPredictionSample:
    sample_token: str
    scene_token: str
    split: str
    features: EgoMotionFeatures
    ground_truth_action: str
    label_rule_version: str
    manifest_schema_version: str
    split_mapping_sha256: str


@dataclass(frozen=True)
class EgoMotionDecision:
    predicted_action: str
    decision_reason: str


@dataclass(frozen=True)
class EgoMotionPredictionRecord:
    sample_token: str
    scene_token: str
    split: str
    ground_truth_action: str
    predicted_action: str
    is_correct: bool
    baseline_name: str
    rule_version: str
    candidate_id: str
    thresholds_sha256: str
    motion_availability: str
    speed_mps: float | None
    longitudinal_acceleration_mps2: float | None
    yaw_rate_radps: float | None
    decision_reason: str
    label_rule_version: str
    manifest_schema_version: str
    split_mapping_sha256: str


@dataclass(frozen=True)
class EgoMotionCandidateEvaluation:
    candidate_id: str
    thresholds: EgoMotionRuleThresholds
    thresholds_sha256: str
    metrics: ClassificationMetrics
    minimum_per_class_f1: float
    predicted_class_distribution: dict[str, int]


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _optional_finite_number(
    mapping: Mapping[str, object],
    key: str,
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"current_ego_motion.{key} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"current_ego_motion.{key} must be finite")
    return number


def parse_ego_motion_features(row: Mapping[str, object]) -> EgoMotionFeatures:
    motion = _required_mapping(row, "current_ego_motion")
    availability = _required_string(motion, "availability")
    if availability not in AVAILABILITY_VALUES:
        raise ValueError(f"unsupported motion availability: {availability!r}")

    values = {
        field: _optional_finite_number(motion, field) for field in AUDIT_FIELDS
    }
    speed = values["speed_mps"]
    acceleration = values["longitudinal_acceleration_mps2"]
    yaw_rate = values["yaw_rate_radps"]
    history_interval = values["history_interval_sec"]
    acceleration_interval = values["acceleration_interval_sec"]

    if availability == "full" and any(value is None for value in values.values()):
        raise ValueError("full motion must provide every audited motion field")
    if availability == "partial":
        if any(value is None for value in (speed, yaw_rate, history_interval)):
            raise ValueError(
                "partial motion must provide speed, yaw rate, and history interval"
            )
        if acceleration is not None or acceleration_interval is not None:
            raise ValueError("partial motion must keep acceleration fields null")
    if availability == "unavailable" and any(
        value is not None for value in values.values()
    ):
        raise ValueError("unavailable motion must keep every audited field null")

    return EgoMotionFeatures(
        speed_mps=speed,
        longitudinal_acceleration_mps2=acceleration,
        yaw_rate_radps=yaw_rate,
        availability=availability,
        history_interval_sec=history_interval,
        acceleration_interval_sec=acceleration_interval,
    )


def parse_ego_motion_audit_sample(
    row: Mapping[str, object],
) -> EgoMotionAuditSample:
    split = _required_string(row, "split")
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    train_meta_action = None
    if split == "train":
        train_meta_action = normalize_action(_required_string(row, "meta_action"))
    return EgoMotionAuditSample(
        sample_token=_required_string(row, "sample_token"),
        scene_token=_required_string(row, "scene_token"),
        split=split,
        features=parse_ego_motion_features(row),
        label_rule_version=_required_string(row, "label_rule_version"),
        manifest_schema_version=_required_string(row, "manifest_schema_version"),
        split_mapping_sha256=validate_sha256(
            row.get("split_mapping_sha256"),
            "split_mapping_sha256",
        ),
        train_meta_action=train_meta_action,
    )


def parse_rule_evaluation_sample(
    row: Mapping[str, object],
) -> EgoMotionPredictionSample | None:
    split = _required_string(row, "split")
    if split != "validation":
        if split not in SPLITS:
            raise ValueError(f"unsupported split: {split!r}")
        return None
    return EgoMotionPredictionSample(
        sample_token=_required_string(row, "sample_token"),
        scene_token=_required_string(row, "scene_token"),
        split=split,
        features=parse_ego_motion_features(row),
        ground_truth_action=normalize_action(_required_string(row, "meta_action")),
        label_rule_version=_required_string(row, "label_rule_version"),
        manifest_schema_version=_required_string(row, "manifest_schema_version"),
        split_mapping_sha256=validate_sha256(
            row.get("split_mapping_sha256"),
            "split_mapping_sha256",
        ),
    )


def predict_ego_motion_action(
    features: EgoMotionFeatures,
    thresholds: EgoMotionRuleThresholds,
) -> EgoMotionDecision:
    if features.availability == "unavailable":
        return EgoMotionDecision(KEEP, "unavailable_motion_fallback_keep")
    if features.availability not in {"full", "partial"}:
        raise ValueError(f"unsupported motion availability: {features.availability!r}")
    if (
        features.availability == "partial"
        and features.longitudinal_acceleration_mps2 is not None
    ):
        raise ValueError("partial motion must keep acceleration null")
    if features.speed_mps is None or features.yaw_rate_radps is None:
        raise ValueError("available motion requires speed and yaw rate")

    if features.speed_mps <= thresholds.stop_speed_threshold_mps:
        return EgoMotionDecision(STOP, "speed_below_stop_threshold")
    if features.yaw_rate_radps >= thresholds.lateral_yaw_rate_threshold_radps:
        return EgoMotionDecision(LEFT_LATERAL, "positive_yaw_rate_lateral")
    if features.yaw_rate_radps <= -thresholds.lateral_yaw_rate_threshold_radps:
        return EgoMotionDecision(RIGHT_LATERAL, "negative_yaw_rate_lateral")
    if features.availability == "full":
        if features.longitudinal_acceleration_mps2 is None:
            raise ValueError("full motion requires longitudinal acceleration")
        if (
            features.longitudinal_acceleration_mps2
            >= thresholds.accelerate_threshold_mps2
        ):
            return EgoMotionDecision(
                ACCELERATE,
                "positive_longitudinal_acceleration",
            )
        if (
            features.longitudinal_acceleration_mps2
            <= -thresholds.decelerate_threshold_mps2
        ):
            return EgoMotionDecision(
                DECELERATE,
                "negative_longitudinal_acceleration",
            )
    return EgoMotionDecision(KEEP, "default_keep")


def build_rule_candidates(
    candidate_grid: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, EgoMotionRuleThresholds], ...]:
    field_names = (
        "stop_speed_threshold_mps",
        "lateral_yaw_rate_threshold_radps",
        "accelerate_threshold_mps2",
        "decelerate_threshold_mps2",
    )
    values_by_field: list[tuple[float, ...]] = []
    for field_name in field_names:
        raw_values = candidate_grid.get(field_name)
        if raw_values is None or len(raw_values) == 0:
            raise ValueError(f"candidate grid field {field_name} must be non-empty")
        values = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"candidate grid field {field_name} must contain numbers"
                )
            values.append(float(value))
        canonical_values = tuple(values)
        if len(canonical_values) != len(set(canonical_values)):
            raise ValueError(f"candidate grid field {field_name} has duplicates")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in canonical_values
        ):
            raise ValueError(
                f"candidate grid field {field_name} must contain finite positives"
            )
        values_by_field.append(tuple(sorted(canonical_values)))

    candidates = []
    for index, values in enumerate(product(*values_by_field), 1):
        thresholds = EgoMotionRuleThresholds(*values)
        candidates.append((f"candidate-{index:04d}", thresholds))
    return tuple(candidates)


def evaluate_rule_candidate(
    samples: Sequence[EgoMotionPredictionSample],
    candidate_id: str,
    thresholds: EgoMotionRuleThresholds,
) -> EgoMotionCandidateEvaluation:
    if not samples:
        raise ValueError("rule candidate evaluation requires validation samples")
    if any(sample.split != "validation" for sample in samples):
        raise ValueError("rule candidate evaluation only accepts validation samples")
    ground_truth = tuple(sample.ground_truth_action for sample in samples)
    predictions = tuple(
        predict_ego_motion_action(sample.features, thresholds).predicted_action
        for sample in samples
    )
    metrics = evaluate_classification(ground_truth, predictions)
    return EgoMotionCandidateEvaluation(
        candidate_id=candidate_id,
        thresholds=thresholds,
        thresholds_sha256=thresholds.sha256(),
        metrics=metrics,
        minimum_per_class_f1=min(metrics.per_class_f1.values()),
        predicted_class_distribution=complete_action_distribution(predictions),
    )


def select_best_rule_candidate(
    candidates: Sequence[EgoMotionCandidateEvaluation],
) -> EgoMotionCandidateEvaluation:
    if not candidates:
        raise ValueError("candidate selection requires at least one evaluation")
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.metrics.macro_f1,
            -candidate.minimum_per_class_f1,
            -candidate.metrics.accuracy,
            candidate.thresholds.canonical_tuple(),
        ),
    )


def build_prediction_records(
    samples: Sequence[EgoMotionPredictionSample],
    candidate: EgoMotionCandidateEvaluation,
    rule_version: str,
) -> tuple[EgoMotionPredictionRecord, ...]:
    records = []
    for sample in samples:
        if sample.split != "validation":
            raise ValueError("prediction records only support validation samples")
        decision = predict_ego_motion_action(sample.features, candidate.thresholds)
        records.append(
            EgoMotionPredictionRecord(
                sample_token=sample.sample_token,
                scene_token=sample.scene_token,
                split=sample.split,
                ground_truth_action=sample.ground_truth_action,
                predicted_action=decision.predicted_action,
                is_correct=decision.predicted_action == sample.ground_truth_action,
                baseline_name="ego_motion_rule",
                rule_version=rule_version,
                candidate_id=candidate.candidate_id,
                thresholds_sha256=candidate.thresholds_sha256,
                motion_availability=sample.features.availability,
                speed_mps=sample.features.speed_mps,
                longitudinal_acceleration_mps2=(
                    sample.features.longitudinal_acceleration_mps2
                ),
                yaw_rate_radps=sample.features.yaw_rate_radps,
                decision_reason=decision.decision_reason,
                label_rule_version=sample.label_rule_version,
                manifest_schema_version=sample.manifest_schema_version,
                split_mapping_sha256=sample.split_mapping_sha256,
            )
        )
    return tuple(records)


def _empty_field_values() -> dict[str, list[float]]:
    return {field: [] for field in AUDIT_FIELDS}


def _feature_values(features: EgoMotionFeatures) -> dict[str, float | None]:
    return {
        field: getattr(features, field)
        for field in AUDIT_FIELDS
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def summarize_values(values: Iterable[float]) -> dict[str, int | float | None]:
    sorted_values = sorted(values)
    if not sorted_values:
        return {name: 0 if name == "count" else None for name in STATISTIC_NAMES}
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p01": _percentile(sorted_values, 0.01),
        "p05": _percentile(sorted_values, 0.05),
        "p25": _percentile(sorted_values, 0.25),
        "median": _percentile(sorted_values, 0.50),
        "p75": _percentile(sorted_values, 0.75),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": fmean(sorted_values),
        "std": pstdev(sorted_values),
    }


def _summarize_fields(
    field_values: Mapping[str, Iterable[float]],
) -> dict[str, dict[str, int | float | None]]:
    return {
        field: summarize_values(field_values[field]) for field in AUDIT_FIELDS
    }


def build_test_label_access_evidence() -> dict[str, object]:
    return {
        "test_label_access_policy": {
            "schema_validation": "allowed_and_performed",
            "input_statistics": "forbidden_and_not_performed",
            "class_conditional_statistics": "forbidden_and_not_performed",
            "threshold_selection": "forbidden_and_not_performed",
            "model_selection": "forbidden_and_not_performed",
            "classification_metrics": "forbidden_and_not_performed",
            "failure_case_analysis": "forbidden_and_not_performed",
        },
        "test_label_used_for_statistics": False,
        "test_label_used_for_threshold_selection": False,
        "test_label_used_for_model_selection": False,
    }


def audit_manifest_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    sample_counts = Counter({split: 0 for split in SPLITS})
    availability_counts = {
        split: Counter({value: 0 for value in AVAILABILITY_VALUES})
        for split in SPLITS
    }
    null_counts = {
        split: Counter({field: 0 for field in AUDIT_FIELDS}) for split in SPLITS
    }
    non_finite_counts = {
        split: Counter({field: 0 for field in AUDIT_FIELDS}) for split in SPLITS
    }
    split_values = {
        "train": _empty_field_values(),
        "validation": _empty_field_values(),
    }
    class_values = {
        action: _empty_field_values() for action in ACTION_SCHEMA
    }
    scene_splits: dict[str, str] = {}
    schema_versions: set[str] = set()
    label_rule_versions: set[str] = set()
    split_mapping_hashes: set[str] = set()

    for row in rows:
        sample = parse_ego_motion_audit_sample(row)
        existing_split = scene_splits.setdefault(sample.scene_token, sample.split)
        if existing_split != sample.split:
            raise ValueError(
                f"scene_token spans splits: {sample.scene_token} "
                f"({existing_split}, {sample.split})"
            )
        sample_counts[sample.split] += 1
        availability_counts[sample.split][sample.features.availability] += 1
        schema_versions.add(sample.manifest_schema_version)
        label_rule_versions.add(sample.label_rule_version)
        split_mapping_hashes.add(sample.split_mapping_sha256)
        values = _feature_values(sample.features)
        for field, value in values.items():
            if value is None:
                null_counts[sample.split][field] += 1
            elif sample.split in split_values:
                split_values[sample.split][field].append(value)
            if sample.train_meta_action is not None and value is not None:
                class_values[sample.train_meta_action][field].append(value)

    if sum(sample_counts.values()) == 0:
        raise ValueError("ego-motion audit requires at least one manifest row")
    if len(schema_versions) != 1:
        raise ValueError("manifest_schema_version must be singular")
    if len(label_rule_versions) != 1:
        raise ValueError("label_rule_version must be singular")
    if len(split_mapping_hashes) != 1:
        raise ValueError("split_mapping_sha256 must be singular")

    return {
        "manifest_schema_version": next(iter(schema_versions)),
        "label_rule_version": next(iter(label_rule_versions)),
        "split_mapping_sha256": next(iter(split_mapping_hashes)),
        "sample_count_by_split": {
            split: sample_counts[split] for split in SPLITS
        },
        "motion_availability_by_split": {
            split: {
                value: availability_counts[split][value]
                for value in AVAILABILITY_VALUES
            }
            for split in SPLITS
        },
        "null_count_by_field_and_split": {
            split: {field: null_counts[split][field] for field in AUDIT_FIELDS}
            for split in SPLITS
        },
        "non_finite_count_by_field_and_split": {
            split: {
                field: non_finite_counts[split][field] for field in AUDIT_FIELDS
            }
            for split in SPLITS
        },
        "train_motion_statistics": _summarize_fields(split_values["train"]),
        "validation_motion_statistics": _summarize_fields(
            split_values["validation"]
        ),
        "train_class_conditional_motion_statistics": {
            action: _summarize_fields(class_values[action])
            for action in ACTION_SCHEMA
        },
        "forbidden_field_policy": {
            "allowed_predictor_fields": list(AUDIT_FIELDS[:3])
            + ["availability"]
            + list(AUDIT_FIELDS[3:]),
            "forbidden_predictor_fields": [
                "future_ego_trajectory",
                "nearby_agents",
                "current_ego_pose.translation_m",
                "cam_front_path",
                "image_content",
                "GT_boxes",
                "GT_occupancy",
                "future_agents",
                "test_labels",
            ],
            "statistical_label_access": {
                "train": "class_conditional_statistics_only",
                "validation": "forbidden",
                "test": "forbidden",
            },
        },
        "contract_violation_count": 0,
    }

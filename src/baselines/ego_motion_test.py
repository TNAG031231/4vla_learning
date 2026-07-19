from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from src.actions.schema import ACTION_SCHEMA, is_valid_action
from src.baselines.ego_motion import (
    EgoMotionFeatures,
    EgoMotionRuleThresholds,
    predict_ego_motion_action,
)
from src.baselines.ego_motion_analysis import (
    DiagnosticMargins,
    SourcePrediction,
    confusion_pairs,
    threshold_boundary_analysis,
    trigger_overlap_analysis,
)
from src.baselines.majority import fit_majority_action
from src.phase0.protocol import (
    ClassificationMetrics,
    ManifestSample,
    complete_action_distribution,
    evaluate_classification,
    validate_sha256,
)


BASELINE_NAME: Final = "ego_motion_rule"
TEST_SAMPLE_FIELDS: Final = frozenset(
    {
        "sample_token",
        "scene_token",
        "split",
        "features",
        "ground_truth_action",
        "label_rule_version",
        "manifest_schema_version",
        "split_mapping_sha256",
    }
)
FORBIDDEN_TEST_FIELDS: Final = frozenset(
    {
        "future_ego_trajectory",
        "nearby_agents",
        "current_ego_pose",
        "cam_front_path",
        "image",
        "image_content",
        "GT_boxes",
        "GT_occupancy",
        "occupancy",
        "future_agents",
        "validation_failure_annotations",
    }
)
TEST_PREDICTION_FIELDS: Final = (
    "sample_token",
    "scene_token",
    "split",
    "ground_truth_action",
    "predicted_action",
    "is_correct",
    "baseline_name",
    "frozen_rule_version",
    "source_rule_version",
    "candidate_id",
    "thresholds_sha256",
    "motion_availability",
    "speed_mps",
    "longitudinal_acceleration_mps2",
    "yaw_rate_radps",
    "decision_reason",
    "label_rule_version",
    "manifest_schema_version",
    "split_mapping_sha256",
)
TEST_METRIC_FIELDS: Final = (
    "sample_count",
    "correct_count",
    "accuracy",
    "macro_f1",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "confusion_matrix",
    "ground_truth_class_distribution",
    "prediction_class_distribution",
    "invalid_prediction_count",
    "action_parsing_success_rate",
)


@dataclass(frozen=True)
class FrozenRuleTestProtocol:
    frozen_rule_version: str
    source_rule_version: str
    candidate_id: str
    thresholds: EgoMotionRuleThresholds
    thresholds_sha256: str
    label_rule_version: str
    manifest_schema_version: str
    split_mapping_sha256: str


@dataclass(frozen=True)
class FrozenRuleTestSample:
    sample_token: str
    scene_token: str
    split: str
    features: EgoMotionFeatures
    ground_truth_action: str
    label_rule_version: str
    manifest_schema_version: str
    split_mapping_sha256: str


@dataclass(frozen=True)
class FrozenRuleTestPredictionRecord:
    sample_token: str
    scene_token: str
    split: str
    ground_truth_action: str
    predicted_action: str
    is_correct: bool
    baseline_name: str
    frozen_rule_version: str
    source_rule_version: str
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


def validate_frozen_rule_contract(protocol: FrozenRuleTestProtocol) -> None:
    for field_name in (
        "frozen_rule_version",
        "source_rule_version",
        "candidate_id",
        "label_rule_version",
        "manifest_schema_version",
    ):
        value = getattr(protocol, field_name)
        if not value:
            raise ValueError(f"{field_name} must be a non-empty string")
    validate_sha256(protocol.thresholds_sha256, "thresholds_sha256")
    validate_sha256(protocol.split_mapping_sha256, "split_mapping_sha256")
    if protocol.thresholds.sha256() != protocol.thresholds_sha256:
        raise ValueError("threshold values do not match thresholds_sha256")


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"test sample missing {key}")
    return value


def _coerce_sample(
    sample: FrozenRuleTestSample | Mapping[str, object],
) -> FrozenRuleTestSample:
    if isinstance(sample, FrozenRuleTestSample):
        return sample
    forbidden = FORBIDDEN_TEST_FIELDS.intersection(sample)
    if forbidden:
        raise ValueError(f"test sample contains forbidden fields: {sorted(forbidden)}")
    unknown = set(sample).difference(TEST_SAMPLE_FIELDS)
    if unknown:
        raise ValueError(f"test sample contains unsupported fields: {sorted(unknown)}")
    features = sample.get("features")
    if not isinstance(features, EgoMotionFeatures):
        raise ValueError("test sample features must be EgoMotionFeatures")
    return FrozenRuleTestSample(
        sample_token=_required_string(sample, "sample_token"),
        scene_token=_required_string(sample, "scene_token"),
        split=_required_string(sample, "split"),
        features=features,
        ground_truth_action=_required_string(sample, "ground_truth_action"),
        label_rule_version=_required_string(sample, "label_rule_version"),
        manifest_schema_version=_required_string(
            sample, "manifest_schema_version"
        ),
        split_mapping_sha256=validate_sha256(
            sample.get("split_mapping_sha256"), "split_mapping_sha256"
        ),
    )


def _validated_samples(
    samples: Sequence[FrozenRuleTestSample | Mapping[str, object]],
    protocol: FrozenRuleTestProtocol,
) -> tuple[FrozenRuleTestSample, ...]:
    validate_frozen_rule_contract(protocol)
    parsed = tuple(_coerce_sample(sample) for sample in samples)
    if not parsed:
        raise ValueError("test evaluation requires at least one sample")
    sample_tokens: set[str] = set()
    for sample in parsed:
        if sample.split != "test":
            raise ValueError("frozen rule test evaluation only accepts test samples")
        if not is_valid_action(sample.ground_truth_action):
            raise ValueError("test sample contains an illegal ground-truth action")
        if sample.sample_token in sample_tokens:
            raise ValueError(f"duplicate sample_token: {sample.sample_token}")
        sample_tokens.add(sample.sample_token)
        trace = (
            (
                "label_rule_version",
                sample.label_rule_version,
                protocol.label_rule_version,
            ),
            (
                "manifest_schema_version",
                sample.manifest_schema_version,
                protocol.manifest_schema_version,
            ),
            (
                "split_mapping_sha256",
                sample.split_mapping_sha256,
                protocol.split_mapping_sha256,
            ),
        )
        for field_name, actual, expected in trace:
            if actual != expected:
                raise ValueError(f"test sample {field_name} is inconsistent")
    return tuple(sorted(parsed, key=lambda sample: sample.sample_token))


def _metrics_payload(
    metrics: ClassificationMetrics,
    predictions: Sequence[str],
) -> dict[str, object]:
    payload = {
        "sample_count": metrics.sample_count,
        "correct_count": metrics.correct_count,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class_precision": dict(metrics.per_class_precision),
        "per_class_recall": dict(metrics.per_class_recall),
        "per_class_f1": dict(metrics.per_class_f1),
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
        "ground_truth_class_distribution": dict(metrics.class_distribution),
        "prediction_class_distribution": complete_action_distribution(predictions),
        "invalid_prediction_count": metrics.invalid_prediction_count,
        "action_parsing_success_rate": metrics.action_parsing_success_rate,
    }
    if tuple(payload) != TEST_METRIC_FIELDS:
        raise ValueError("test metric field contract changed")
    return payload


def evaluate_frozen_rule_test_samples(
    samples: Sequence[FrozenRuleTestSample | Mapping[str, object]],
    protocol: FrozenRuleTestProtocol,
) -> tuple[tuple[FrozenRuleTestPredictionRecord, ...], dict[str, object]]:
    validated = _validated_samples(samples, protocol)
    records = []
    predictions = []
    ground_truth = []
    for sample in validated:
        decision = predict_ego_motion_action(sample.features, protocol.thresholds)
        predictions.append(decision.predicted_action)
        ground_truth.append(sample.ground_truth_action)
        record = FrozenRuleTestPredictionRecord(
            sample_token=sample.sample_token,
            scene_token=sample.scene_token,
            split=sample.split,
            ground_truth_action=sample.ground_truth_action,
            predicted_action=decision.predicted_action,
            is_correct=decision.predicted_action == sample.ground_truth_action,
            baseline_name=BASELINE_NAME,
            frozen_rule_version=protocol.frozen_rule_version,
            source_rule_version=protocol.source_rule_version,
            candidate_id=protocol.candidate_id,
            thresholds_sha256=protocol.thresholds_sha256,
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
        if tuple(asdict(record)) != TEST_PREDICTION_FIELDS:
            raise ValueError("test prediction field contract changed")
        records.append(record)
    metrics = evaluate_classification(tuple(ground_truth), tuple(predictions))
    return tuple(records), _metrics_payload(metrics, predictions)


def evaluate_majority_on_test_samples(
    train_samples: Sequence[ManifestSample],
    test_samples: Sequence[FrozenRuleTestSample | Mapping[str, object]],
    frozen_rule_metrics: Mapping[str, object],
    protocol: FrozenRuleTestProtocol,
) -> dict[str, object]:
    if not train_samples or any(sample.split != "train" for sample in train_samples):
        raise ValueError("majority action must be fitted from train samples only")
    validated = _validated_samples(test_samples, protocol)
    majority_action = fit_majority_action(train_samples)
    ground_truth = tuple(sample.ground_truth_action for sample in validated)
    predictions = tuple(majority_action for _ in validated)
    metrics = evaluate_classification(ground_truth, predictions)
    frozen_per_class = frozen_rule_metrics.get("per_class_f1")
    if not isinstance(frozen_per_class, Mapping):
        raise ValueError("frozen rule metrics missing per_class_f1")
    return {
        "majority_action": majority_action,
        "sample_count": metrics.sample_count,
        "test_accuracy": metrics.accuracy,
        "test_macro_f1": metrics.macro_f1,
        "test_per_class_f1": dict(metrics.per_class_f1),
        "frozen_rule_minus_majority_accuracy": (
            float(frozen_rule_metrics["accuracy"]) - metrics.accuracy
        ),
        "frozen_rule_minus_majority_macro_f1": (
            float(frozen_rule_metrics["macro_f1"]) - metrics.macro_f1
        ),
        "frozen_rule_minus_majority_per_class_f1": {
            action: float(frozen_per_class[action]) - metrics.per_class_f1[action]
            for action in ACTION_SCHEMA
        },
    }


def build_validation_to_test_comparison(
    validation_metrics: Mapping[str, object],
    test_metrics: Mapping[str, object],
) -> dict[str, object]:
    validation_f1 = validation_metrics.get("per_class_f1")
    test_f1 = test_metrics.get("per_class_f1")
    validation_distribution = validation_metrics.get(
        "prediction_class_distribution"
    )
    test_distribution = test_metrics.get("prediction_class_distribution")
    if not all(
        isinstance(value, Mapping)
        for value in (
            validation_f1,
            test_f1,
            validation_distribution,
            test_distribution,
        )
    ):
        raise ValueError("comparison metrics are incomplete")
    return {
        "comparison_schema_version": "phase0.2_validation_to_test_v0.1",
        "test_minus_validation_accuracy": (
            float(test_metrics["accuracy"]) - float(validation_metrics["accuracy"])
        ),
        "test_minus_validation_macro_f1": (
            float(test_metrics["macro_f1"])
            - float(validation_metrics["macro_f1"])
        ),
        "test_minus_validation_per_class_f1": {
            action: float(test_f1[action]) - float(validation_f1[action])
            for action in ACTION_SCHEMA
        },
        "prediction_distribution_count_difference": {
            action: int(test_distribution[action])
            - int(validation_distribution[action])
            for action in ACTION_SCHEMA
        },
        "rule_modified_from_test_results": False,
    }


def _source_predictions(
    records: Sequence[FrozenRuleTestPredictionRecord],
) -> tuple[SourcePrediction, ...]:
    return tuple(
        SourcePrediction(
            sample_token=record.sample_token,
            scene_token=record.scene_token,
            split=record.split,
            ground_truth_action=record.ground_truth_action,
            predicted_action=record.predicted_action,
            is_correct=record.is_correct,
            baseline_name=record.baseline_name,
            rule_version=record.source_rule_version,
            candidate_id=record.candidate_id,
            thresholds_sha256=record.thresholds_sha256,
            motion_availability=record.motion_availability,
            speed_mps=record.speed_mps,
            longitudinal_acceleration_mps2=(
                record.longitudinal_acceleration_mps2
            ),
            yaw_rate_radps=record.yaw_rate_radps,
            decision_reason=record.decision_reason,
            label_rule_version=record.label_rule_version,
            manifest_schema_version=record.manifest_schema_version,
            split_mapping_sha256=record.split_mapping_sha256,
        )
        for record in records
    )


def build_test_diagnostics(
    records: Sequence[FrozenRuleTestPredictionRecord],
    protocol: FrozenRuleTestProtocol,
    diagnostic_margins: DiagnosticMargins,
) -> dict[str, object]:
    if not records:
        raise ValueError("test diagnostics require prediction records")
    predictions = _source_predictions(records)
    availability = {}
    for value in ("full", "partial", "unavailable"):
        group = tuple(item for item in records if item.motion_availability == value)
        correct = sum(item.is_correct for item in group)
        availability[value] = {
            "sample_count": len(group),
            "accuracy": correct / len(group) if group else 0.0,
        }
    reasons = {}
    for reason in sorted({item.decision_reason for item in records}):
        group = tuple(item for item in records if item.decision_reason == reason)
        correct = sum(item.is_correct for item in group)
        reasons[reason] = {
            "sample_count": len(group),
            "accuracy": correct / len(group),
            "error_count": len(group) - correct,
            "error_rate": (len(group) - correct) / len(group),
        }
    boundary = threshold_boundary_analysis(
        predictions, protocol.thresholds, diagnostic_margins
    )
    boundary_groups = {
        name: {
            "sample_count": values["sample_count"],
            "error_count": values["error_count"],
            "error_rate": values["error_rate"],
        }
        for name, values in boundary["groups"].items()
    }
    overlap = trigger_overlap_analysis(predictions, protocol.thresholds)
    return {
        "diagnostics_schema_version": "phase0.2_test_diagnostics_v0.1",
        "availability": availability,
        "decision_reason": reasons,
        "confusion_pairs": confusion_pairs(predictions),
        "threshold_boundary": {
            "diagnostic_margins": boundary["diagnostic_margins"],
            "definitions": boundary["definitions"],
            "groups": boundary_groups,
        },
        "trigger_overlap": {
            "trigger_count_distribution": overlap["trigger_count_distribution"],
            "trigger_combinations": overlap["trigger_combinations"],
            "priority_conflict_definition": overlap[
                "priority_conflict_definition"
            ],
            "priority_conflict_sample_count": overlap[
                "priority_conflict_sample_count"
            ],
            "priority_conflict_error_count": overlap[
                "priority_conflict_error_count"
            ],
            "priority_conflict_error_rate": overlap[
                "priority_conflict_error_rate"
            ],
        },
        "post_hoc_subgroups_added": False,
    }


def build_one_shot_receipt(
    protocol: FrozenRuleTestProtocol,
    *,
    manifest_sha256: str,
    freeze_sha256: str,
    output_sha256: Mapping[str, str],
    test_sample_count: int,
) -> dict[str, object]:
    validate_frozen_rule_contract(protocol)
    validate_sha256(manifest_sha256, "manifest_sha256")
    validate_sha256(freeze_sha256, "freeze_sha256")
    for name, digest in output_sha256.items():
        validate_sha256(digest, f"output_sha256.{name}")
    return {
        "one_shot_schema_version": "phase0.2_one_shot_test_v0.1",
        "protocol_status": "executed_once",
        "manifest_sha256": manifest_sha256,
        "freeze_sha256": freeze_sha256,
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "test_sample_count": test_sample_count,
        "output_sha256": dict(output_sha256),
        "one_shot_execution_performed": True,
        "rule_modified_from_test_results": False,
    }

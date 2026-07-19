from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
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
EVALUATOR_SOURCE_PATHS: Final = (
    "configs/phase0_2_one_shot_test_v0_1.yaml",
    "src/actions/schema.py",
    "src/baselines/ego_motion.py",
    "src/baselines/ego_motion_analysis.py",
    "src/baselines/ego_motion_test.py",
    "src/baselines/majority.py",
    "src/phase0/manifest.py",
    "src/phase0/protocol.py",
    "scripts/prepare_ego_motion_one_shot_test.py",
)
EXPECTED_TEST_SAMPLE_COUNT: Final = 3799
EXPECTED_TEST_SCENE_COUNT: Final = 150
DECLARED_TEST_OUTPUTS: Final = (
    "test_predictions.jsonl",
    "test_metrics.json",
    "majority_test_metrics.json",
    "validation_to_test_comparison.json",
    "test_diagnostics.json",
    "one_shot_test_receipt.json",
)
FORMAL_RESULT_SCHEMA: Final = {
    "test_predictions": "phase0.2_test_predictions_v0.1",
    "test_metrics": "phase0.2_test_metrics_v0.1",
    "majority_test_metrics": "phase0.2_majority_test_metrics_v0.1",
    "validation_to_test_comparison": "phase0.2_validation_to_test_v0.1",
    "test_diagnostics": "phase0.2_test_diagnostics_v0.1",
    "one_shot_test_receipt": "phase0.2_one_shot_test_receipt_v0.1",
}
FORMAL_RESULT_SHA_FILENAMES: Final = (
    "test_predictions.jsonl",
    "test_metrics.json",
    "majority_test_metrics.json",
    "validation_to_test_comparison.json",
    "test_diagnostics.json",
)
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
    if not isinstance(validation_f1, Mapping) or not isinstance(test_f1, Mapping):
        raise ValueError("comparison metrics are incomplete")
    validation_count, validation_distribution, validation_rates = (
        _validated_prediction_distribution(validation_metrics, "validation")
    )
    test_count, test_distribution, test_rates = (
        _validated_prediction_distribution(test_metrics, "test")
    )
    return {
        "comparison_schema_version": "phase0.2_validation_to_test_v0.1",
        "validation_sample_count": validation_count,
        "test_sample_count": test_count,
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
        "validation_prediction_distribution_rate": validation_rates,
        "test_prediction_distribution_rate": test_rates,
        "prediction_distribution_count_difference": {
            action: test_distribution[action] - validation_distribution[action]
            for action in ACTION_SCHEMA
        },
        "prediction_distribution_rate_difference": {
            action: test_rates[action] - validation_rates[action]
            for action in ACTION_SCHEMA
        },
        "rule_modified_from_test_results": False,
    }


def _validated_prediction_distribution(
    metrics: Mapping[str, object],
    split_name: str,
) -> tuple[int, dict[str, int], dict[str, float]]:
    sample_count = metrics.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError(f"{split_name} sample_count must be a positive integer")
    raw_distribution = metrics.get("prediction_class_distribution")
    if not isinstance(raw_distribution, Mapping):
        raise ValueError(f"{split_name} prediction distribution must be a mapping")
    if set(raw_distribution) != set(ACTION_SCHEMA):
        raise ValueError(
            f"{split_name} prediction distribution must contain all action classes"
        )
    distribution = {}
    for action in ACTION_SCHEMA:
        value = raw_distribution[action]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                f"{split_name} prediction distribution counts "
                "must be non-negative integers"
            )
        distribution[action] = value
    if sum(distribution.values()) != sample_count:
        raise ValueError(
            f"{split_name} prediction distribution count does not match sample_count"
        )
    rates = {
        action: distribution[action] / sample_count for action in ACTION_SCHEMA
    }
    if not math.isclose(
        math.fsum(rates.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{split_name} prediction distribution rates must sum to 1")
    return sample_count, distribution, rates


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


def _validated_sha_mapping(
    payload: object,
    *,
    expected_keys: Sequence[str],
    field_name: str,
) -> dict[str, str]:
    if not isinstance(payload, Mapping) or set(payload) != set(expected_keys):
        raise ValueError(f"{field_name} must contain the complete frozen file set")
    return {
        key: validate_sha256(payload.get(key), f"{field_name}.{key}")
        for key in expected_keys
    }


def validate_preflight_receipt_for_execution(
    preflight_receipt_bytes: bytes,
    preflight_receipt: Mapping[str, object],
    *,
    actual_preflight_receipt_sha256: str,
    actual_evaluator_source_sha256: Mapping[str, str],
    protocol: FrozenRuleTestProtocol,
    manifest_sha256: str,
    freeze_sha256: str,
) -> dict[str, bool]:
    validate_frozen_rule_contract(protocol)
    if not isinstance(preflight_receipt_bytes, bytes):
        raise ValueError("preflight receipt bytes must be bytes")
    if not isinstance(preflight_receipt, Mapping):
        raise ValueError("preflight receipt must be a mapping")
    actual_preflight_receipt_sha256 = validate_sha256(
        actual_preflight_receipt_sha256, "actual_preflight_receipt_sha256"
    )
    calculated_preflight_sha256 = hashlib.sha256(
        preflight_receipt_bytes
    ).hexdigest()
    preflight_receipt_sha_matches = (
        calculated_preflight_sha256 == actual_preflight_receipt_sha256
    )
    if not preflight_receipt_sha_matches:
        raise ValueError("preflight receipt bytes do not match the supplied SHA-256")
    try:
        parsed_receipt = json.loads(preflight_receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("preflight receipt bytes must contain valid JSON") from error
    if parsed_receipt != preflight_receipt:
        raise ValueError("preflight receipt mapping does not match receipt bytes")
    manifest_sha256 = validate_sha256(manifest_sha256, "manifest_sha256")
    freeze_sha256 = validate_sha256(freeze_sha256, "freeze_sha256")

    expected_fields = {
        "preflight_schema_version": "phase0.2_one_shot_test_preflight_v0.1",
        "protocol_status": "preflight_passed",
        "ready_for_execution": True,
        "one_shot_execution_performed": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "manifest_sha256": manifest_sha256,
        "freeze_sha256": freeze_sha256,
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "test_sample_count": EXPECTED_TEST_SAMPLE_COUNT,
        "test_scene_count": EXPECTED_TEST_SCENE_COUNT,
        "test_manifest_rows_parsed": True,
        "test_label_value_accessed_by_application_logic": False,
        "test_motion_value_accessed_by_application_logic": False,
        "declared_outputs": list(DECLARED_TEST_OUTPUTS),
        "formal_result_schema": FORMAL_RESULT_SCHEMA,
    }
    for field_name, expected in expected_fields.items():
        if preflight_receipt.get(field_name) != expected:
            raise ValueError(f"preflight receipt {field_name} does not match")

    preflight_sources = _validated_sha_mapping(
        preflight_receipt.get("evaluator_source_sha256"),
        expected_keys=EVALUATOR_SOURCE_PATHS,
        field_name="preflight evaluator_source_sha256",
    )
    actual_sources = _validated_sha_mapping(
        actual_evaluator_source_sha256,
        expected_keys=EVALUATOR_SOURCE_PATHS,
        field_name="actual evaluator_source_sha256",
    )
    if actual_sources != preflight_sources:
        raise ValueError("evaluator source SHA-256 values differ from preflight")

    return {
        "preflight_receipt_sha_matches": preflight_receipt_sha_matches,
        "preflight_status_matches": True,
        "ready_for_execution": True,
        "execution_not_previously_performed": True,
        "formal_outputs_not_previously_generated": True,
        "manifest_sha_matches": True,
        "freeze_sha_matches": True,
        "frozen_rule_version_matches": True,
        "source_rule_version_matches": True,
        "candidate_id_matches": True,
        "thresholds_match": True,
        "threshold_sha_matches": True,
        "evaluator_source_hashes_match": True,
    }


def build_one_shot_receipt(
    protocol: FrozenRuleTestProtocol,
    *,
    preflight_receipt_bytes: bytes,
    preflight_receipt: Mapping[str, object],
    preflight_receipt_sha256: str,
    actual_evaluator_source_sha256: Mapping[str, str],
    execution_source_sha256: str,
    manifest_sha256: str,
    freeze_sha256: str,
    output_sha256: Mapping[str, str],
    test_sample_count: int,
) -> dict[str, object]:
    provenance = validate_preflight_receipt_for_execution(
        preflight_receipt_bytes,
        preflight_receipt,
        actual_preflight_receipt_sha256=preflight_receipt_sha256,
        actual_evaluator_source_sha256=actual_evaluator_source_sha256,
        protocol=protocol,
        manifest_sha256=manifest_sha256,
        freeze_sha256=freeze_sha256,
    )
    execution_source_sha256 = validate_sha256(
        execution_source_sha256, "execution_source_sha256"
    )
    output_hashes = _validated_sha_mapping(
        output_sha256,
        expected_keys=FORMAL_RESULT_SHA_FILENAMES,
        field_name="output_sha256",
    )
    if test_sample_count != EXPECTED_TEST_SAMPLE_COUNT or isinstance(
        test_sample_count, bool
    ):
        raise ValueError("test_sample_count must match preflight count 3799")
    if test_sample_count != preflight_receipt.get("test_sample_count"):
        raise ValueError("test_sample_count does not match preflight receipt")
    preflight_sources = preflight_receipt["evaluator_source_sha256"]
    if not isinstance(preflight_sources, Mapping):
        raise ValueError("preflight evaluator_source_sha256 must be a mapping")
    return {
        "one_shot_schema_version": FORMAL_RESULT_SCHEMA[
            "one_shot_test_receipt"
        ],
        "protocol_status": "executed_once",
        "manifest_sha256": manifest_sha256,
        "freeze_sha256": freeze_sha256,
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "test_sample_count": test_sample_count,
        "test_scene_count": EXPECTED_TEST_SCENE_COUNT,
        "formal_result_schema": dict(FORMAL_RESULT_SCHEMA),
        "output_sha256": output_hashes,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "preflight_schema_version": preflight_receipt[
            "preflight_schema_version"
        ],
        "preflight_protocol_status": preflight_receipt["protocol_status"],
        "preflight_ready_for_execution": preflight_receipt[
            "ready_for_execution"
        ],
        "preflight_evaluator_source_sha256": dict(preflight_sources),
        "execution_source_sha256": execution_source_sha256,
        "evaluator_sources_match_preflight": provenance[
            "evaluator_source_hashes_match"
        ],
        "execution_count": 1,
        "rerun_permitted": False,
        "test_result_used_for_rule_modification": False,
        "one_shot_execution_performed": True,
        "rule_modified_from_test_results": False,
    }

from __future__ import annotations

from collections.abc import Mapping
import math

from src.actions.schema import ACTION_SCHEMA
from src.phase0.protocol import validate_sha256


PHASE0_2B_VALIDATION_ARTIFACT_FIELDS = (
    "rule_version",
    "selected_candidate_id",
    "thresholds_sha256",
    "metrics",
    "predicted_class_distribution",
    "decision_reason_distribution",
    "test_evaluation_performed",
)
NORMALIZED_VALIDATION_METRICS_FIELDS = (
    "sample_count",
    "accuracy",
    "macro_f1",
    "per_class_f1",
    "prediction_class_distribution",
)


def _required_string(artifact: Mapping[str, object], key: str) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"validation artifact {key} must be a non-empty string")
    return value


def _required_mapping(
    artifact: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = artifact.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"validation artifact {key} must be a mapping")
    return value


def _classification_score(
    mapping: Mapping[str, object],
    key: str,
    description: str,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number")
    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{description} must be between 0.0 and 1.0 inclusive"
        )
    return number


def _per_class_f1(metrics: Mapping[str, object]) -> dict[str, float]:
    values = metrics.get("per_class_f1")
    if not isinstance(values, Mapping):
        raise ValueError("validation metrics per_class_f1 must be a mapping")
    if set(values) != set(ACTION_SCHEMA):
        raise ValueError(
            "validation metrics per_class_f1 must contain exactly ACTION_SCHEMA"
        )
    return {
        action: _classification_score(
            values,
            action,
            f"validation metrics per_class_f1[{action}]",
        )
        for action in ACTION_SCHEMA
    }


def _prediction_distribution(
    artifact: Mapping[str, object],
    sample_count: int,
) -> dict[str, int]:
    values = artifact.get("predicted_class_distribution")
    if not isinstance(values, Mapping):
        raise ValueError(
            "validation artifact predicted_class_distribution must be a mapping"
        )
    if set(values) != set(ACTION_SCHEMA):
        raise ValueError(
            "validation artifact predicted_class_distribution must contain "
            "exactly ACTION_SCHEMA"
        )
    distribution = {}
    for action in ACTION_SCHEMA:
        value = values[action]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                "validation artifact predicted_class_distribution values "
                "must be non-negative integers"
            )
        distribution[action] = value
    if sum(distribution.values()) != sample_count:
        raise ValueError(
            "validation artifact predicted_class_distribution must sum to "
            "sample_count"
        )
    return distribution


def _decision_reason_distribution(
    artifact: Mapping[str, object],
    sample_count: int,
) -> dict[str, int]:
    values = artifact.get("decision_reason_distribution")
    if not isinstance(values, Mapping):
        raise ValueError(
            "validation artifact decision_reason_distribution must be a mapping"
        )
    distribution = {}
    for reason, count in values.items():
        if not isinstance(reason, str) or not reason:
            raise ValueError(
                "validation artifact decision_reason_distribution reason keys "
                "must be non-empty strings"
            )
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                "validation artifact decision_reason_distribution counts "
                "must be non-negative integers"
            )
        distribution[reason] = count
    if sum(distribution.values()) != sample_count:
        raise ValueError(
            "validation artifact decision_reason_distribution must sum to "
            "sample_count"
        )
    return distribution


def normalize_phase0_2b_validation_metrics_artifact(
    artifact: Mapping[str, object],
    *,
    expected_rule_version: str,
    expected_candidate_id: str,
    expected_thresholds_sha256: str,
    expected_sample_count: int,
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise ValueError("validation artifact must be a mapping")
    if set(artifact) != set(PHASE0_2B_VALIDATION_ARTIFACT_FIELDS):
        raise ValueError(
            "validation artifact Phase 0.2b producer fields must match exactly"
        )
    if _required_string(artifact, "rule_version") != expected_rule_version:
        raise ValueError("validation artifact rule_version does not match")
    if (
        _required_string(artifact, "selected_candidate_id")
        != expected_candidate_id
    ):
        raise ValueError("validation artifact selected_candidate_id does not match")
    expected_thresholds_sha256 = validate_sha256(
        expected_thresholds_sha256,
        "expected_thresholds_sha256",
    )
    actual_thresholds_sha256 = validate_sha256(
        artifact.get("thresholds_sha256"),
        "validation artifact thresholds_sha256",
    )
    if actual_thresholds_sha256 != expected_thresholds_sha256:
        raise ValueError("validation artifact thresholds_sha256 does not match")
    if (
        not isinstance(expected_sample_count, int)
        or isinstance(expected_sample_count, bool)
        or expected_sample_count <= 0
    ):
        raise ValueError("expected_sample_count must be a positive integer")
    metrics = _required_mapping(artifact, "metrics")
    sample_count = metrics.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
    ):
        raise ValueError("validation metrics sample_count must be a positive integer")
    if sample_count != expected_sample_count:
        raise ValueError("validation metrics sample_count does not match")
    if artifact.get("test_evaluation_performed") is not False:
        raise ValueError(
            "validation artifact test_evaluation_performed must be false"
        )
    _decision_reason_distribution(artifact, sample_count)
    normalized = {
        "sample_count": sample_count,
        "accuracy": _classification_score(
            metrics,
            "accuracy",
            "validation metrics accuracy",
        ),
        "macro_f1": _classification_score(
            metrics,
            "macro_f1",
            "validation metrics macro_f1",
        ),
        "per_class_f1": _per_class_f1(metrics),
        "prediction_class_distribution": _prediction_distribution(
            artifact,
            sample_count,
        ),
    }
    if tuple(normalized) != NORMALIZED_VALIDATION_METRICS_FIELDS:
        raise ValueError("normalized validation metrics field contract changed")
    return normalized

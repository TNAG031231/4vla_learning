from __future__ import annotations

from collections.abc import Mapping
import math

from src.actions.schema import ACTION_SCHEMA
from src.phase0.protocol import validate_sha256


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


def _finite_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"validation metrics {key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"validation metrics {key} must be a finite number")
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
        action: _finite_number(values, action) for action in ACTION_SCHEMA
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
    if set(NORMALIZED_VALIDATION_METRICS_FIELDS).intersection(artifact):
        raise ValueError(
            "validation artifact must use the Phase 0.2b producer schema"
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
    normalized = {
        "sample_count": sample_count,
        "accuracy": _finite_number(metrics, "accuracy"),
        "macro_f1": _finite_number(metrics, "macro_f1"),
        "per_class_f1": _per_class_f1(metrics),
        "prediction_class_distribution": _prediction_distribution(
            artifact,
            sample_count,
        ),
    }
    if tuple(normalized) != NORMALIZED_VALIDATION_METRICS_FIELDS:
        raise ValueError("normalized validation metrics field contract changed")
    return normalized

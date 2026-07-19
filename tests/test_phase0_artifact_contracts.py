from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ego_motion_rule_baseline import (
    build_validation_metrics_artifact_payload,
)
from src.actions.schema import ACTION_SCHEMA
from src.baselines.ego_motion import (
    EgoMotionCandidateEvaluation,
    EgoMotionRuleThresholds,
)
from src.baselines.ego_motion_test import build_validation_to_test_comparison
from src.phase0.artifact_contracts import (
    NORMALIZED_VALIDATION_METRICS_FIELDS,
    normalize_phase0_2b_validation_metrics_artifact,
)
from src.phase0.protocol import evaluate_classification


RULE_VERSION = "phase0.2b-ego-motion-rule-v0.1"
CANDIDATE_ID = "candidate-0293"
SAMPLE_COUNT = 3594
THRESHOLDS = EgoMotionRuleThresholds(0.2, 0.05, 0.5, 0.3)
THRESHOLDS_SHA256 = THRESHOLDS.sha256()


def producer_payload() -> dict[str, object]:
    actions = tuple(
        ACTION_SCHEMA[index % len(ACTION_SCHEMA)]
        for index in range(SAMPLE_COUNT)
    )
    metrics = evaluate_classification(actions, actions)
    selected = EgoMotionCandidateEvaluation(
        candidate_id=CANDIDATE_ID,
        thresholds=THRESHOLDS,
        thresholds_sha256=THRESHOLDS_SHA256,
        metrics=metrics,
        minimum_per_class_f1=min(metrics.per_class_f1.values()),
        predicted_class_distribution={
            action: SAMPLE_COUNT // len(ACTION_SCHEMA)
            for action in ACTION_SCHEMA
        },
    )
    return build_validation_metrics_artifact_payload(
        RULE_VERSION,
        selected,
        {"default_keep": SAMPLE_COUNT},
    )


def synthetic_test_metrics() -> dict[str, object]:
    metrics = evaluate_classification(ACTION_SCHEMA, ACTION_SCHEMA)
    return {
        "sample_count": metrics.sample_count,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class_f1": metrics.per_class_f1,
        "prediction_class_distribution": {
            action: 1 for action in ACTION_SCHEMA
        },
    }


def normalize(artifact: object) -> dict[str, object]:
    return normalize_phase0_2b_validation_metrics_artifact(
        artifact,
        expected_rule_version=RULE_VERSION,
        expected_candidate_id=CANDIDATE_ID,
        expected_thresholds_sha256=THRESHOLDS_SHA256,
        expected_sample_count=SAMPLE_COUNT,
    )


def test_adapter_rejects_non_mapping_artifact() -> None:
    with pytest.raises(ValueError, match="validation artifact must be a mapping"):
        normalize([])


def nested_mapping(
    payload: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = payload[key]
    assert isinstance(value, dict)
    return value


def test_producer_payload_reproduces_historical_comparison_failure() -> None:
    with pytest.raises(ValueError, match="comparison metrics are incomplete"):
        build_validation_to_test_comparison(
            producer_payload(),
            synthetic_test_metrics(),
        )


def test_adapter_normalizes_producer_payload_for_comparison() -> None:
    payload = producer_payload()
    normalized = normalize(payload)

    comparison = build_validation_to_test_comparison(
        normalized,
        synthetic_test_metrics(),
    )

    assert "metrics" in payload
    assert "predicted_class_distribution" in payload
    assert "per_class_f1" not in payload
    assert "prediction_class_distribution" not in payload
    assert "per_class_f1" in normalized
    assert "prediction_class_distribution" in normalized
    assert comparison["comparison_schema_version"] == (
        "phase0.2_validation_to_test_v0.1"
    )


@pytest.mark.parametrize("value", (None, [], "metrics"))
def test_adapter_rejects_missing_or_non_mapping_metrics(value: object) -> None:
    payload = producer_payload()
    if value is None:
        payload.pop("metrics")
    else:
        payload["metrics"] = value

    with pytest.raises(ValueError, match="metrics must be a mapping"):
        normalize(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (None, "positive integer"),
        (0, "positive integer"),
        (-1, "positive integer"),
        (True, "positive integer"),
        (3594.0, "positive integer"),
        (3593, "does not match"),
    ),
)
def test_adapter_rejects_invalid_sample_count(
    value: object,
    message: str,
) -> None:
    payload = producer_payload()
    metrics = nested_mapping(payload, "metrics")
    if value is None:
        metrics.pop("sample_count")
    else:
        metrics["sample_count"] = value

    with pytest.raises(ValueError, match=message):
        normalize(payload)


@pytest.mark.parametrize("field", ("accuracy", "macro_f1"))
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, True, "1.0"))
def test_adapter_rejects_non_finite_metrics(
    field: str,
    value: object,
) -> None:
    payload = producer_payload()
    nested_mapping(payload, "metrics")[field] = value

    with pytest.raises(ValueError, match=f"{field} must be a finite number"):
        normalize(payload)


@pytest.mark.parametrize("mutation", ("missing", "extra", "non_numeric"))
def test_adapter_rejects_invalid_per_class_f1(mutation: str) -> None:
    payload = producer_payload()
    per_class = nested_mapping(nested_mapping(payload, "metrics"), "per_class_f1")
    if mutation == "missing":
        per_class.pop(ACTION_SCHEMA[0])
        message = "exactly ACTION_SCHEMA"
    elif mutation == "extra":
        per_class["unknown"] = 0.0
        message = "exactly ACTION_SCHEMA"
    else:
        per_class[ACTION_SCHEMA[0]] = "invalid"
        message = "finite number"

    with pytest.raises(ValueError, match=message):
        normalize(payload)


def test_adapter_rejects_missing_prediction_distribution() -> None:
    payload = producer_payload()
    payload.pop("predicted_class_distribution")

    with pytest.raises(ValueError, match="must be a mapping"):
        normalize(payload)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "non_integer", "negative", "wrong_sum"),
)
def test_adapter_rejects_invalid_prediction_distribution(
    mutation: str,
) -> None:
    payload = producer_payload()
    distribution = nested_mapping(payload, "predicted_class_distribution")
    if mutation == "missing":
        distribution.pop(ACTION_SCHEMA[0])
        message = "exactly ACTION_SCHEMA"
    elif mutation == "extra":
        distribution["unknown"] = 0
        message = "exactly ACTION_SCHEMA"
    elif mutation == "non_integer":
        distribution[ACTION_SCHEMA[0]] = 1.0
        message = "non-negative integers"
    elif mutation == "negative":
        distribution[ACTION_SCHEMA[0]] = -1
        message = "non-negative integers"
    else:
        distribution[ACTION_SCHEMA[0]] += 1
        message = "sum to sample_count"

    with pytest.raises(ValueError, match=message):
        normalize(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("rule_version", "wrong", "rule_version does not match"),
        (
            "selected_candidate_id",
            "candidate-0001",
            "selected_candidate_id does not match",
        ),
        ("thresholds_sha256", "e" * 64, "thresholds_sha256 does not match"),
    ),
)
def test_adapter_rejects_provenance_mismatch(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = producer_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        normalize(payload)


def test_adapter_rejects_test_evaluation_performed() -> None:
    payload = producer_payload()
    payload["test_evaluation_performed"] = True

    with pytest.raises(ValueError, match="must be false"):
        normalize(payload)


def test_adapter_rejects_flattened_synthetic_schema() -> None:
    with pytest.raises(ValueError, match="producer schema"):
        normalize(synthetic_test_metrics())


def test_adapter_output_is_exact_and_deterministic() -> None:
    payload = producer_payload()

    first = normalize(deepcopy(payload))
    second = normalize(deepcopy(payload))

    assert tuple(first) == NORMALIZED_VALIDATION_METRICS_FIELDS
    assert first == second

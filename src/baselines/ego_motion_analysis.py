from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Final

from src.actions.schema import ACTION_SCHEMA, is_valid_action
from src.baselines.ego_motion import (
    EgoMotionFeatures,
    EgoMotionPredictionSample,
    EgoMotionRuleThresholds,
    predict_ego_motion_action,
)
from src.phase0.manifest import json_compatible
from src.phase0.protocol import (
    ClassificationMetrics,
    complete_action_distribution,
    evaluate_classification,
    validate_sha256,
)


EXPECTED_VALIDATION_SAMPLE_COUNT: Final = 3594
TRIGGER_NAMES: Final = (
    "stop_trigger",
    "left_lateral_trigger",
    "right_lateral_trigger",
    "accelerate_trigger",
    "decelerate_trigger",
)
BOUNDARY_NAMES: Final = (
    "stop_boundary",
    "lateral_boundary",
    "accelerate_boundary",
    "decelerate_boundary",
)
PREDICTION_FIELDS: Final = frozenset(
    {
        "sample_token",
        "scene_token",
        "split",
        "ground_truth_action",
        "predicted_action",
        "is_correct",
        "baseline_name",
        "rule_version",
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
    }
)
FAILURE_FIELDS: Final = (
    "sample_token",
    "scene_token",
    "split",
    "ground_truth_action",
    "predicted_action",
    "decision_reason",
    "motion_availability",
    "speed_mps",
    "longitudinal_acceleration_mps2",
    "yaw_rate_radps",
    "stop_margin_mps",
    "lateral_margin_radps",
    "accelerate_margin_mps2",
    "decelerate_margin_mps2",
    "active_triggers",
    "threshold_boundary_flags",
    "candidate_id",
    "thresholds_sha256",
    "label_rule_version",
    "manifest_schema_version",
    "split_mapping_sha256",
)
FORBIDDEN_FIELDS: Final = frozenset(
    {
        "test_label",
        "test_prediction",
        "test_metrics",
        "test_confusion_matrix",
        "test_failure_cases",
        "future_ego_trajectory",
        "nearby_agents",
        "current_ego_pose",
        "current_ego_pose.translation_m",
        "cam_front_path",
        "image",
        "image_content",
        "GT_boxes",
        "GT_occupancy",
        "future_agents",
        "scene_name",
    }
)


@dataclass(frozen=True)
class DiagnosticMargins:
    stop_speed_mps: float
    lateral_yaw_rate_radps: float
    longitudinal_acceleration_mps2: float

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("diagnostic margins must be finite and non-negative")


@dataclass(frozen=True)
class SourcePrediction:
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


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"prediction missing {key}")
    return value


def validate_source_hashes(
    source_dir: Path,
    expected_hashes: Mapping[str, str],
    sha256_file: Callable[[Path], str],
) -> dict[str, str]:
    actual_hashes = {}
    for filename, expected in expected_hashes.items():
        validate_sha256(expected, f"expected_source_sha256.{filename}")
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"source artifact not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"source artifact SHA-256 mismatch: {filename}: "
                f"expected {expected}, got {actual}"
            )
        actual_hashes[filename] = actual
    return actual_hashes


def _optional_number(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prediction {key} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"prediction {key} must be finite")
    return number


def parse_source_prediction(payload: Mapping[str, object]) -> SourcePrediction:
    forbidden = FORBIDDEN_FIELDS.intersection(payload)
    if forbidden:
        raise ValueError(f"prediction contains forbidden fields: {sorted(forbidden)}")
    unknown = set(payload).difference(PREDICTION_FIELDS)
    if unknown:
        raise ValueError(f"prediction contains unsupported fields: {sorted(unknown)}")
    split = _required_string(payload, "split")
    if split != "validation":
        raise ValueError("source predictions must contain validation only")
    ground_truth = _required_string(payload, "ground_truth_action")
    predicted = _required_string(payload, "predicted_action")
    if not is_valid_action(ground_truth) or not is_valid_action(predicted):
        raise ValueError("source prediction contains an illegal action")
    is_correct = payload.get("is_correct")
    if not isinstance(is_correct, bool):
        raise ValueError("prediction is_correct must be boolean")
    if is_correct != (ground_truth == predicted):
        raise ValueError("prediction is_correct is inconsistent")
    return SourcePrediction(
        sample_token=_required_string(payload, "sample_token"),
        scene_token=_required_string(payload, "scene_token"),
        split=split,
        ground_truth_action=ground_truth,
        predicted_action=predicted,
        is_correct=is_correct,
        baseline_name=_required_string(payload, "baseline_name"),
        rule_version=_required_string(payload, "rule_version"),
        candidate_id=_required_string(payload, "candidate_id"),
        thresholds_sha256=validate_sha256(
            payload.get("thresholds_sha256"), "thresholds_sha256"
        ),
        motion_availability=_required_string(payload, "motion_availability"),
        speed_mps=_optional_number(payload, "speed_mps"),
        longitudinal_acceleration_mps2=_optional_number(
            payload, "longitudinal_acceleration_mps2"
        ),
        yaw_rate_radps=_optional_number(payload, "yaw_rate_radps"),
        decision_reason=_required_string(payload, "decision_reason"),
        label_rule_version=_required_string(payload, "label_rule_version"),
        manifest_schema_version=_required_string(
            payload, "manifest_schema_version"
        ),
        split_mapping_sha256=validate_sha256(
            payload.get("split_mapping_sha256"), "split_mapping_sha256"
        ),
    )


def read_source_predictions(
    path: Path,
    expected_count: int = EXPECTED_VALIDATION_SAMPLE_COUNT,
) -> tuple[SourcePrediction, ...]:
    predictions = []
    sample_tokens: set[str] = set()
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: prediction must be an object")
            prediction = parse_source_prediction(payload)
            if prediction.sample_token in sample_tokens:
                raise ValueError(
                    f"duplicate sample_token: {prediction.sample_token}"
                )
            sample_tokens.add(prediction.sample_token)
            predictions.append(prediction)
    if len(predictions) != expected_count:
        raise ValueError(
            f"validation prediction count must be {expected_count}, "
            f"got {len(predictions)}"
        )
    return tuple(predictions)


def validate_selected_rule(
    selected_rule: Mapping[str, object],
    candidate_id: str,
    thresholds: EgoMotionRuleThresholds,
    thresholds_sha256: str,
    source_rule_version: str,
) -> None:
    if selected_rule.get("selected_candidate_id") != candidate_id:
        raise ValueError("selected candidate does not match freeze config")
    if selected_rule.get("rule_version") != source_rule_version:
        raise ValueError("source rule version does not match freeze config")
    if selected_rule.get("thresholds") != thresholds.as_dict():
        raise ValueError("selected threshold values do not match freeze config")
    if thresholds.sha256() != thresholds_sha256:
        raise ValueError("configured threshold SHA-256 does not match thresholds")
    if selected_rule.get("thresholds_sha256") != thresholds_sha256:
        raise ValueError("selected threshold SHA-256 does not match freeze config")


def reproduce_predictions(
    samples: Sequence[EgoMotionPredictionSample],
    predictions: Sequence[SourcePrediction],
    thresholds: EgoMotionRuleThresholds,
) -> dict[str, object]:
    if len(samples) != len(predictions):
        raise ValueError("manifest and source prediction counts differ")
    match_count = 0
    for sample, source in zip(samples, predictions, strict=True):
        if sample.sample_token != source.sample_token:
            raise ValueError("manifest and source prediction order differs")
        if sample.scene_token != source.scene_token:
            raise ValueError("manifest and source scene_token differs")
        if sample.ground_truth_action != source.ground_truth_action:
            raise ValueError("manifest and source ground truth differs")
        decision = predict_ego_motion_action(sample.features, thresholds)
        if decision.predicted_action != source.predicted_action:
            raise ValueError(
                f"reproduced prediction differs: {sample.sample_token}"
            )
        if decision.decision_reason != source.decision_reason:
            raise ValueError(
                f"reproduced decision reason differs: {sample.sample_token}"
            )
        source_features = (
            source.speed_mps,
            source.longitudinal_acceleration_mps2,
            source.yaw_rate_radps,
            source.motion_availability,
        )
        manifest_features = (
            sample.features.speed_mps,
            sample.features.longitudinal_acceleration_mps2,
            sample.features.yaw_rate_radps,
            sample.features.availability,
        )
        if source_features != manifest_features:
            raise ValueError(f"source motion differs: {sample.sample_token}")
        match_count += 1
    return {
        "source_prediction_count": len(predictions),
        "manifest_validation_sample_count": len(samples),
        "match_count": match_count,
        "all_predictions_match": match_count == len(predictions),
    }


def metrics_for_predictions(
    predictions: Sequence[SourcePrediction],
) -> ClassificationMetrics:
    return evaluate_classification(
        tuple(item.ground_truth_action for item in predictions),
        tuple(item.predicted_action for item in predictions),
    )


def metrics_payload(metrics: ClassificationMetrics) -> dict[str, object]:
    payload = json_compatible(metrics)
    if not isinstance(payload, dict):
        raise TypeError("classification metrics must serialize to an object")
    return payload


def confusion_pairs(
    predictions: Sequence[SourcePrediction],
) -> list[dict[str, object]]:
    errors = tuple(item for item in predictions if not item.is_correct)
    ground_truth_counts = Counter(
        item.ground_truth_action for item in predictions
    )
    pair_counts = Counter(
        (item.ground_truth_action, item.predicted_action) for item in errors
    )
    ordered = sorted(
        pair_counts,
        key=lambda pair: (
            -pair_counts[pair],
            ACTION_SCHEMA.index(pair[0]),
            ACTION_SCHEMA.index(pair[1]),
        ),
    )
    return [
        {
            "ground_truth_action": expected,
            "predicted_action": predicted,
            "count": pair_counts[(expected, predicted)],
            "fraction_of_all_errors": (
                pair_counts[(expected, predicted)] / len(errors)
                if errors
                else 0.0
            ),
            "fraction_of_ground_truth_class": (
                pair_counts[(expected, predicted)] / ground_truth_counts[expected]
            ),
        }
        for expected, predicted in ordered
    ]


def _group_analysis(
    predictions: Sequence[SourcePrediction],
) -> dict[str, object]:
    metrics = metrics_for_predictions(predictions)
    return {
        "sample_count": len(predictions),
        "correct_count": metrics.correct_count,
        "incorrect_count": len(predictions) - metrics.correct_count,
        "accuracy": metrics.accuracy,
        "ground_truth_distribution": metrics.class_distribution,
        "predicted_distribution": complete_action_distribution(
            tuple(item.predicted_action for item in predictions)
        ),
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
    }


def availability_analysis(
    predictions: Sequence[SourcePrediction],
) -> dict[str, dict[str, object]]:
    return {
        availability: _group_analysis(
            tuple(
                item
                for item in predictions
                if item.motion_availability == availability
            )
        )
        for availability in ("full", "partial", "unavailable")
    }


def decision_reason_analysis(
    predictions: Sequence[SourcePrediction],
) -> dict[str, dict[str, object]]:
    reasons = sorted({item.decision_reason for item in predictions})
    result = {}
    for reason in reasons:
        group = tuple(item for item in predictions if item.decision_reason == reason)
        analysis = _group_analysis(group)
        analysis["error_count"] = analysis.pop("incorrect_count")
        analysis["top_confusion_pairs"] = confusion_pairs(group)
        analysis.pop("confusion_matrix")
        result[reason] = analysis
    return result


def motion_margins(
    prediction: SourcePrediction,
    thresholds: EgoMotionRuleThresholds,
) -> dict[str, float | None]:
    acceleration = prediction.longitudinal_acceleration_mps2
    return {
        "stop_margin_mps": (
            None
            if prediction.speed_mps is None
            else prediction.speed_mps - thresholds.stop_speed_threshold_mps
        ),
        "lateral_margin_radps": (
            None
            if prediction.yaw_rate_radps is None
            else abs(prediction.yaw_rate_radps)
            - thresholds.lateral_yaw_rate_threshold_radps
        ),
        "accelerate_margin_mps2": (
            None
            if acceleration is None
            else acceleration - thresholds.accelerate_threshold_mps2
        ),
        "decelerate_margin_mps2": (
            None
            if acceleration is None
            else acceleration + thresholds.decelerate_threshold_mps2
        ),
    }


def threshold_boundary_flags(
    prediction: SourcePrediction,
    thresholds: EgoMotionRuleThresholds,
    margins: DiagnosticMargins,
) -> tuple[str, ...]:
    values = motion_margins(prediction, thresholds)
    limits = {
        "stop_boundary": margins.stop_speed_mps,
        "lateral_boundary": margins.lateral_yaw_rate_radps,
        "accelerate_boundary": margins.longitudinal_acceleration_mps2,
        "decelerate_boundary": margins.longitudinal_acceleration_mps2,
    }
    value_names = {
        "stop_boundary": "stop_margin_mps",
        "lateral_boundary": "lateral_margin_radps",
        "accelerate_boundary": "accelerate_margin_mps2",
        "decelerate_boundary": "decelerate_margin_mps2",
    }
    return tuple(
        boundary
        for boundary in BOUNDARY_NAMES
        if values[value_names[boundary]] is not None
        and (
            abs(values[value_names[boundary]]) <= limits[boundary]
            or math.isclose(
                abs(values[value_names[boundary]]),
                limits[boundary],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    )


def threshold_boundary_analysis(
    predictions: Sequence[SourcePrediction],
    thresholds: EgoMotionRuleThresholds,
    margins: DiagnosticMargins,
) -> dict[str, object]:
    definitions = {
        "stop_boundary": (
            "abs(speed_mps - stop_speed_threshold_mps) <= stop_speed_mps"
        ),
        "lateral_boundary": (
            "abs(abs(yaw_rate_radps) - lateral_yaw_rate_threshold_radps) "
            "<= lateral_yaw_rate_radps"
        ),
        "accelerate_boundary": (
            "abs(longitudinal_acceleration_mps2 - accelerate_threshold_mps2) "
            "<= longitudinal_acceleration_mps2"
        ),
        "decelerate_boundary": (
            "abs(longitudinal_acceleration_mps2 + decelerate_threshold_mps2) "
            "<= longitudinal_acceleration_mps2"
        ),
    }
    groups: dict[str, list[SourcePrediction]] = {
        boundary: [] for boundary in BOUNDARY_NAMES
    }
    for prediction in predictions:
        for boundary in threshold_boundary_flags(prediction, thresholds, margins):
            groups[boundary].append(prediction)
    return {
        "diagnostic_margins": asdict(margins),
        "prediction_effect": "none",
        "definitions": definitions,
        "groups": {
            boundary: {
                "sample_count": len(groups[boundary]),
                "error_count": sum(not item.is_correct for item in groups[boundary]),
                "error_rate": (
                    sum(not item.is_correct for item in groups[boundary])
                    / len(groups[boundary])
                    if groups[boundary]
                    else 0.0
                ),
                "confusion_pair_distribution": confusion_pairs(groups[boundary]),
            }
            for boundary in BOUNDARY_NAMES
        },
    }


def active_triggers(
    prediction: SourcePrediction,
    thresholds: EgoMotionRuleThresholds,
) -> tuple[str, ...]:
    speed = prediction.speed_mps
    yaw_rate = prediction.yaw_rate_radps
    acceleration = prediction.longitudinal_acceleration_mps2
    flags = {
        "stop_trigger": (
            speed is not None and speed <= thresholds.stop_speed_threshold_mps
        ),
        "left_lateral_trigger": (
            yaw_rate is not None
            and yaw_rate >= thresholds.lateral_yaw_rate_threshold_radps
        ),
        "right_lateral_trigger": (
            yaw_rate is not None
            and yaw_rate <= -thresholds.lateral_yaw_rate_threshold_radps
        ),
        "accelerate_trigger": (
            prediction.motion_availability == "full"
            and acceleration is not None
            and acceleration >= thresholds.accelerate_threshold_mps2
        ),
        "decelerate_trigger": (
            prediction.motion_availability == "full"
            and acceleration is not None
            and acceleration <= -thresholds.decelerate_threshold_mps2
        ),
    }
    return tuple(name for name in TRIGGER_NAMES if flags[name])


def trigger_overlap_analysis(
    predictions: Sequence[SourcePrediction],
    thresholds: EgoMotionRuleThresholds,
) -> dict[str, object]:
    combination_groups: dict[tuple[str, ...], list[SourcePrediction]] = {}
    priority_conflicts = []
    trigger_counts = Counter({"0": 0, "1": 0, "2_or_more": 0})
    for prediction in predictions:
        triggers = active_triggers(prediction, thresholds)
        combination_groups.setdefault(triggers, []).append(prediction)
        count_key = (
            "0" if not triggers else "1" if len(triggers) == 1 else "2_or_more"
        )
        trigger_counts[count_key] += 1
        longitudinal = {
            "accelerate_trigger",
            "decelerate_trigger",
        }.intersection(triggers)
        higher_priority = {
            "stop_trigger",
            "left_lateral_trigger",
            "right_lateral_trigger",
        }.intersection(triggers)
        if longitudinal and higher_priority:
            priority_conflicts.append(prediction)
    combinations = []
    for triggers, group in sorted(
        combination_groups.items(), key=lambda item: (len(item[0]), item[0])
    ):
        correct = sum(item.is_correct for item in group)
        combinations.append(
            {
                "active_triggers": list(triggers),
                "sample_count": len(group),
                "accuracy": correct / len(group),
            }
        )
    conflict_errors = sum(not item.is_correct for item in priority_conflicts)
    return {
        "trigger_count_distribution": dict(trigger_counts),
        "trigger_combinations": combinations,
        "priority_conflict_definition": (
            "stop or lateral trigger co-occurs with a longitudinal trigger"
        ),
        "priority_conflict_sample_count": len(priority_conflicts),
        "priority_conflict_error_count": conflict_errors,
        "priority_conflict_error_rate": (
            conflict_errors / len(priority_conflicts) if priority_conflicts else 0.0
        ),
        "rule_priority_unchanged": True,
    }


def scene_error_concentration(
    predictions: Sequence[SourcePrediction],
) -> dict[str, object]:
    scenes: dict[str, list[SourcePrediction]] = {}
    for prediction in predictions:
        scenes.setdefault(prediction.scene_token, []).append(prediction)
    rows = []
    for scene_token, group in scenes.items():
        errors = sum(not item.is_correct for item in group)
        rows.append(
            {
                "scene_token": scene_token,
                "sample_count": len(group),
                "error_count": errors,
                "error_rate": errors / len(group),
            }
        )
    return {
        "scene_count": len(rows),
        "scene_with_errors_count": sum(row["error_count"] > 0 for row in rows),
        "top_20_by_error_count": sorted(
            rows,
            key=lambda row: (
                -row["error_count"],
                -row["error_rate"],
                row["scene_token"],
            ),
        )[:20],
        "top_20_by_error_rate_min_5_samples": sorted(
            (row for row in rows if row["sample_count"] >= 5),
            key=lambda row: (
                -row["error_rate"],
                -row["error_count"],
                row["scene_token"],
            ),
        )[:20],
        "scene_semantics_used": False,
    }


def _candidate_sort_key(candidate: Mapping[str, object]) -> tuple[object, ...]:
    thresholds = candidate.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("leaderboard candidate missing thresholds")
    return (
        -float(candidate["validation_macro_f1"]),
        -float(candidate["minimum_per_class_f1"]),
        -float(candidate["validation_accuracy"]),
        float(thresholds["stop_speed_threshold_mps"]),
        float(thresholds["lateral_yaw_rate_threshold_radps"]),
        float(thresholds["accelerate_threshold_mps2"]),
        float(thresholds["decelerate_threshold_mps2"]),
    )


def candidate_stability(
    leaderboard: Mapping[str, object],
    selected_candidate_id: str,
) -> dict[str, object]:
    raw_candidates = leaderboard.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("leaderboard candidates must be a non-empty list")
    if not all(isinstance(item, Mapping) for item in raw_candidates):
        raise ValueError("leaderboard candidate must be an object")
    candidates = tuple(raw_candidates)
    ranked = sorted(candidates, key=_candidate_sort_key)
    selected = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == selected_candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("selected candidate is absent from leaderboard")
    rank = ranked.index(selected) + 1
    if rank != 1:
        raise ValueError("selected candidate is not rank one")
    second = ranked[1]
    selected_thresholds = selected.get("thresholds")
    if not isinstance(selected_thresholds, Mapping):
        raise ValueError("selected candidate missing thresholds")
    threshold_fields = (
        "stop_speed_threshold_mps",
        "lateral_yaw_rate_threshold_radps",
        "accelerate_threshold_mps2",
        "decelerate_threshold_mps2",
    )
    grid = {
        field: sorted(
            {
                float(item["thresholds"][field])
                for item in candidates
                if isinstance(item.get("thresholds"), Mapping)
            }
        )
        for field in threshold_fields
    }
    neighbours = []
    for item in candidates:
        item_thresholds = item.get("thresholds")
        if not isinstance(item_thresholds, Mapping) or item is selected:
            continue
        changed = [
            field
            for field in threshold_fields
            if float(item_thresholds[field]) != float(selected_thresholds[field])
        ]
        if len(changed) != 1:
            continue
        field = changed[0]
        selected_index = grid[field].index(float(selected_thresholds[field]))
        item_index = grid[field].index(float(item_thresholds[field]))
        if abs(selected_index - item_index) != 1:
            continue
        neighbours.append(
            {
                "candidate_id": item["candidate_id"],
                "changed_threshold": field,
                "thresholds": dict(item_thresholds),
                "validation_macro_f1": item["validation_macro_f1"],
                "minimum_per_class_f1": item["minimum_per_class_f1"],
                "validation_accuracy": item["validation_accuracy"],
                "macro_f1_delta_from_selected": (
                    float(item["validation_macro_f1"])
                    - float(selected["validation_macro_f1"])
                ),
                "minimum_per_class_f1_delta_from_selected": (
                    float(item["minimum_per_class_f1"])
                    - float(selected["minimum_per_class_f1"])
                ),
                "accuracy_delta_from_selected": (
                    float(item["validation_accuracy"])
                    - float(selected["validation_accuracy"])
                ),
            }
        )
    selected_macro_f1 = float(selected["validation_macro_f1"])
    return {
        "source": "existing_candidate_leaderboard_only",
        "candidate_count": len(candidates),
        "selected_candidate_rank": rank,
        "second_ranked_candidate": dict(second),
        "macro_f1_gap_to_second": (
            selected_macro_f1 - float(second["validation_macro_f1"])
        ),
        "accuracy_gap_to_second": (
            float(selected["validation_accuracy"])
            - float(second["validation_accuracy"])
        ),
        "candidate_count_within_selected_macro_f1": {
            "0.001": sum(
                selected_macro_f1 - float(item["validation_macro_f1"]) <= 0.001
                for item in candidates
            ),
            "0.005": sum(
                selected_macro_f1 - float(item["validation_macro_f1"]) <= 0.005
                for item in candidates
            ),
            "0.010": sum(
                selected_macro_f1 - float(item["validation_macro_f1"]) <= 0.010
                for item in candidates
            ),
        },
        "local_grid_neighbours": sorted(
            neighbours, key=lambda item: item["candidate_id"]
        ),
        "candidate_reselection_performed": False,
    }


def build_failure_records(
    predictions: Sequence[SourcePrediction],
    thresholds: EgoMotionRuleThresholds,
    diagnostic_margins: DiagnosticMargins,
) -> tuple[dict[str, object], ...]:
    records = []
    for prediction in predictions:
        if prediction.is_correct:
            continue
        margins = motion_margins(prediction, thresholds)
        payload: dict[str, object] = {
            "sample_token": prediction.sample_token,
            "scene_token": prediction.scene_token,
            "split": prediction.split,
            "ground_truth_action": prediction.ground_truth_action,
            "predicted_action": prediction.predicted_action,
            "decision_reason": prediction.decision_reason,
            "motion_availability": prediction.motion_availability,
            "speed_mps": prediction.speed_mps,
            "longitudinal_acceleration_mps2": (
                prediction.longitudinal_acceleration_mps2
            ),
            "yaw_rate_radps": prediction.yaw_rate_radps,
            **margins,
            "active_triggers": list(active_triggers(prediction, thresholds)),
            "threshold_boundary_flags": list(
                threshold_boundary_flags(
                    prediction, thresholds, diagnostic_margins
                )
            ),
            "candidate_id": prediction.candidate_id,
            "thresholds_sha256": prediction.thresholds_sha256,
            "label_rule_version": prediction.label_rule_version,
            "manifest_schema_version": prediction.manifest_schema_version,
            "split_mapping_sha256": prediction.split_mapping_sha256,
        }
        if tuple(payload) != FAILURE_FIELDS:
            raise ValueError("failure record field contract changed")
        records.append(payload)
    return tuple(records)


def build_overall_analysis(
    predictions: Sequence[SourcePrediction],
) -> dict[str, object]:
    metrics = metrics_for_predictions(predictions)
    return {
        "total_samples": metrics.sample_count,
        "correct_count": metrics.correct_count,
        "incorrect_count": metrics.sample_count - metrics.correct_count,
        "error_rate": 1.0 - metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "accuracy": metrics.accuracy,
        "per_class_precision": metrics.per_class_precision,
        "per_class_recall": metrics.per_class_recall,
        "per_class_f1": metrics.per_class_f1,
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
    }


def assert_payload_equal(
    actual: object,
    expected: object,
    description: str,
) -> None:
    if actual != expected:
        raise ValueError(f"reproduced {description} does not match source output")


def build_freeze_record(
    *,
    gates: Mapping[str, bool],
    payload: Mapping[str, object],
) -> dict[str, object]:
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise ValueError(f"freeze gates failed: {failed}")
    record = dict(payload)
    record["freeze_status"] = "frozen"
    record["freeze_gates"] = dict(gates)
    return record

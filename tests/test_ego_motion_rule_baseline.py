from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ego_motion_rule_baseline import (
    build_selected_rule_payload,
    build_selection_objective,
    run_baseline,
)
from src.actions.schema import ACTION_SCHEMA
from src.baselines.ego_motion import (
    EgoMotionFeatures,
    EgoMotionPredictionSample,
    EgoMotionRuleThresholds,
    build_prediction_records,
    build_rule_candidates,
    evaluate_rule_candidate,
    parse_rule_evaluation_sample,
    predict_ego_motion_action,
    select_best_rule_candidate,
)
from src.baselines.majority import fit_majority_action
from src.phase0.protocol import ManifestSample


def thresholds(
    *,
    stop: float = 0.2,
    lateral: float = 0.1,
    accelerate: float = 0.3,
    decelerate: float = 0.3,
) -> EgoMotionRuleThresholds:
    return EgoMotionRuleThresholds(stop, lateral, accelerate, decelerate)


def features(
    *,
    speed: float | None = 4.0,
    acceleration: float | None = 0.0,
    yaw_rate: float | None = 0.0,
    availability: str = "full",
) -> EgoMotionFeatures:
    return EgoMotionFeatures(
        speed_mps=speed,
        longitudinal_acceleration_mps2=acceleration,
        yaw_rate_radps=yaw_rate,
        availability=availability,
        history_interval_sec=None if availability == "unavailable" else 0.5,
        acceleration_interval_sec=0.5 if availability == "full" else None,
    )


def prediction_sample(
    *,
    sample_token: str = "sample",
    scene_token: str = "scene",
    split: str = "validation",
    ground_truth_action: str = "keep",
    motion: EgoMotionFeatures | None = None,
) -> EgoMotionPredictionSample:
    return EgoMotionPredictionSample(
        sample_token=sample_token,
        scene_token=scene_token,
        split=split,
        features=motion or features(),
        ground_truth_action=ground_truth_action,
        label_rule_version="phase-1.6-meta-action-v0.2",
        manifest_schema_version="phase0_trainval_dataset_manifest_v1",
        split_mapping_sha256="a" * 64,
    )


def manifest_row(
    *,
    split: str = "validation",
    meta_action: str = "keep",
) -> dict[str, object]:
    return {
        "sample_token": "sample",
        "scene_token": "scene",
        "split": split,
        "meta_action": meta_action,
        "label_rule_version": "phase-1.6-meta-action-v0.2",
        "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
        "split_mapping_sha256": "a" * 64,
        "current_ego_motion": {
            "speed_mps": 4.0,
            "longitudinal_acceleration_mps2": 0.4,
            "yaw_rate_radps": 0.0,
            "availability": "full",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": 0.5,
        },
        "future_ego_trajectory": [[999.0, 999.0]],
        "nearby_agents": [{"category": "sentinel"}],
        "current_ego_pose": {"translation_m": [999.0, 999.0, 999.0]},
        "cam_front_path": "samples/CAM_FRONT/sentinel.jpg",
    }


def candidate_grid() -> dict[str, tuple[float, ...]]:
    return {
        "stop_speed_threshold_mps": (0.05, 0.10, 0.20, 0.50, 1.00),
        "lateral_yaw_rate_threshold_radps": (0.03, 0.05, 0.07, 0.10, 0.15),
        "accelerate_threshold_mps2": (0.10, 0.20, 0.30, 0.50, 0.75),
        "decelerate_threshold_mps2": (0.10, 0.20, 0.30, 0.50, 0.75),
    }


def test_unavailable_motion_falls_back_to_keep_without_values() -> None:
    decision = predict_ego_motion_action(
        features(
            speed=None,
            acceleration=None,
            yaw_rate=None,
            availability="unavailable",
        ),
        thresholds(),
    )

    assert decision.predicted_action == "keep"
    assert decision.decision_reason == "unavailable_motion_fallback_keep"


def test_stop_has_inclusive_priority_over_yaw_and_acceleration() -> None:
    decision = predict_ego_motion_action(
        features(speed=0.2, acceleration=9.0, yaw_rate=9.0),
        thresholds(),
    )

    assert decision.predicted_action == "stop"
    assert decision.decision_reason == "speed_below_stop_threshold"


@pytest.mark.parametrize(
    ("yaw_rate", "expected_action"),
    ((0.1, "left_lateral"), (-0.1, "right_lateral")),
)
def test_lateral_threshold_is_inclusive(
    yaw_rate: float,
    expected_action: str,
) -> None:
    decision = predict_ego_motion_action(
        features(acceleration=9.0, yaw_rate=yaw_rate),
        thresholds(),
    )

    assert decision.predicted_action == expected_action


@pytest.mark.parametrize(
    ("acceleration", "expected_action"),
    ((0.3, "accelerate"), (-0.3, "decelerate")),
)
def test_longitudinal_threshold_is_inclusive(
    acceleration: float,
    expected_action: str,
) -> None:
    decision = predict_ego_motion_action(
        features(acceleration=acceleration),
        thresholds(),
    )

    assert decision.predicted_action == expected_action


def test_default_prediction_is_keep() -> None:
    decision = predict_ego_motion_action(features(), thresholds())

    assert decision.predicted_action == "keep"
    assert decision.decision_reason == "default_keep"


@pytest.mark.parametrize(
    ("speed", "yaw_rate", "expected_action"),
    ((4.0, 0.0, "keep"), (0.1, 0.0, "stop"), (4.0, 0.2, "left_lateral")),
)
def test_partial_motion_never_uses_acceleration(
    speed: float,
    yaw_rate: float,
    expected_action: str,
) -> None:
    decision = predict_ego_motion_action(
        features(
            speed=speed,
            acceleration=None,
            yaw_rate=yaw_rate,
            availability="partial",
        ),
        thresholds(),
    )

    assert decision.predicted_action == expected_action


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, 0.0, -0.1))
def test_thresholds_reject_non_positive_or_non_finite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        thresholds(stop=invalid)


def test_forbidden_manifest_fields_do_not_affect_prediction() -> None:
    original = manifest_row()
    changed = manifest_row()
    changed["future_ego_trajectory"] = [[-1e9, 1e9]]
    changed["nearby_agents"] = [{"category": "changed"}]
    changed["current_ego_pose"] = {"translation_m": [-1e9, 1e9, 0.0]}
    changed["cam_front_path"] = "changed-image-path"

    original_sample = parse_rule_evaluation_sample(original)
    changed_sample = parse_rule_evaluation_sample(changed)

    assert original_sample is not None
    assert changed_sample is not None
    assert original_sample.features == changed_sample.features
    assert (
        predict_ego_motion_action(original_sample.features, thresholds())
        == predict_ego_motion_action(changed_sample.features, thresholds())
    )


class GuardedTestRow(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key == "meta_action":
            raise AssertionError("test meta_action entered predictor/evaluation")
        return super().get(key, default)


def test_test_row_does_not_enter_predictor_or_evaluation() -> None:
    row = GuardedTestRow(manifest_row(split="test", meta_action="sentinel"))

    assert parse_rule_evaluation_sample(row) is None


def test_validation_label_changes_metrics_but_not_features_or_predictions() -> None:
    correct_row = manifest_row(meta_action="accelerate")
    stop_row = manifest_row(meta_action="stop")
    correct_sample = parse_rule_evaluation_sample(correct_row)
    stop_sample = parse_rule_evaluation_sample(stop_row)

    assert correct_sample is not None
    assert stop_sample is not None
    assert correct_sample.features == stop_sample.features
    correct_result = evaluate_rule_candidate(
        (correct_sample,),
        "candidate",
        thresholds(),
    )
    stop_result = evaluate_rule_candidate((stop_sample,), "candidate", thresholds())
    assert correct_result.predicted_class_distribution == (
        stop_result.predicted_class_distribution
    )
    assert correct_result.metrics.accuracy != stop_result.metrics.accuracy


def test_candidate_grid_contains_625_unique_stable_candidates() -> None:
    first = build_rule_candidates(candidate_grid())
    reversed_grid = {
        field: tuple(reversed(values)) for field, values in candidate_grid().items()
    }
    second = build_rule_candidates(reversed_grid)

    assert len(first) == 625
    assert len({threshold.sha256() for _, threshold in first}) == 625
    assert first == second
    assert first[0][0] == "candidate-0001"
    assert first[-1][0] == "candidate-0625"


def test_candidate_grid_rejects_duplicate_values() -> None:
    grid = candidate_grid()
    grid["stop_speed_threshold_mps"] = (0.1, 0.1)

    with pytest.raises(ValueError, match="duplicates"):
        build_rule_candidates(grid)


def test_candidate_selection_uses_canonical_threshold_tie_break() -> None:
    sample = prediction_sample()
    higher_tuple = evaluate_rule_candidate(
        (sample,),
        "higher",
        thresholds(stop=0.2),
    )
    lower_tuple = evaluate_rule_candidate(
        (sample,),
        "lower",
        thresholds(stop=0.1),
    )

    selected = select_best_rule_candidate((higher_tuple, lower_tuple))

    assert selected.candidate_id == "lower"


def test_majority_fit_uses_only_train_labels() -> None:
    samples = (
        ManifestSample("train-1", "train-scene", "keep", "train"),
        ManifestSample("train-2", "train-scene", "keep", "train"),
        ManifestSample("validation", "validation-scene", "stop", "validation"),
    )

    assert fit_majority_action(samples) == "keep"


def test_prediction_records_are_legal_and_exclude_forbidden_fields() -> None:
    sample = prediction_sample()
    evaluation = evaluate_rule_candidate((sample,), "candidate", thresholds())
    records = build_prediction_records((sample,), evaluation, "rule-v0")
    payload = asdict(records[0])

    assert records[0].predicted_action in ACTION_SCHEMA
    assert not {
        "future_ego_trajectory",
        "nearby_agents",
        "current_ego_pose",
        "cam_front_path",
        "test_information",
    } & payload.keys()


def test_rule_evaluation_and_records_are_deterministic() -> None:
    sample = prediction_sample()
    evaluation = evaluate_rule_candidate((sample,), "candidate", thresholds())

    assert evaluation == evaluate_rule_candidate(
        (sample,),
        "candidate",
        thresholds(),
    )
    assert build_prediction_records((sample,), evaluation, "rule-v0") == (
        build_prediction_records((sample,), evaluation, "rule-v0")
    )


def test_rule_evaluation_rejects_non_validation_split() -> None:
    sample = prediction_sample(split="train")

    with pytest.raises(ValueError, match="only accepts validation"):
        evaluate_rule_candidate((sample,), "candidate", thresholds())


def test_selected_rule_records_no_test_evaluation() -> None:
    sample = prediction_sample()
    evaluation = evaluate_rule_candidate((sample,), "candidate", thresholds())
    objective = build_selection_objective((evaluation,), evaluation)
    payload = build_selected_rule_payload(
        rule_version="rule-v0",
        selected=evaluation,
        selection_objective=objective,
        manifest_sha256="b" * 64,
        manifest_schema_version="phase0_trainval_dataset_manifest_v1",
        label_rule_version="phase-1.6-meta-action-v0.2",
        split_mapping_sha256="a" * 64,
        decision_reason_distribution={"default_keep": 1},
        motion_availability_distribution={"full": 1},
    )

    assert payload["test_evaluation_performed"] is False
    assert payload["selection_objective"]["test_used_for_selection"] is False


def test_manifest_sha_mismatch_fails_before_output(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "manifest_relative_path": "manifest.jsonl",
                "output_relative_dir": "output",
                "expected_manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_baseline(config_path, tmp_path)

    assert not (tmp_path / "output").exists()

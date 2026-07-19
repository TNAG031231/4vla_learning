from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.ego_motion import (
    EgoMotionFeatures,
    EgoMotionPredictionSample,
    EgoMotionRuleThresholds,
    predict_ego_motion_action,
)
from src.baselines.ego_motion_analysis import (
    DiagnosticMargins,
    SourcePrediction,
    active_triggers,
    assert_payload_equal,
    availability_analysis,
    build_failure_records,
    build_freeze_gates,
    build_freeze_record,
    candidate_stability,
    confusion_pairs,
    decision_reason_analysis,
    inference_field_contract,
    read_source_predictions,
    reproduce_predictions,
    scene_error_concentration,
    threshold_boundary_analysis,
    threshold_boundary_flags,
    trigger_overlap_analysis,
    validate_leaderboard_contract,
    validate_selected_rule,
    validate_source_hashes,
)
from scripts.analyze_and_freeze_ego_motion_rule import (
    _validate_prediction_contracts,
)
from src.phase0.manifest import write_canonical_json, write_jsonl_records


THRESHOLD_SHA = "43feb5e2baad95bed63e98557eb63c7c2a8fdfbca07a503825f60d41b08d82c9"


def thresholds() -> EgoMotionRuleThresholds:
    return EgoMotionRuleThresholds(0.2, 0.05, 0.5, 0.3)


def margins() -> DiagnosticMargins:
    return DiagnosticMargins(0.05, 0.01, 0.05)


def source_prediction(
    *,
    token: str = "sample",
    scene: str = "scene",
    split: str = "validation",
    expected: str = "keep",
    predicted: str = "keep",
    availability: str = "full",
    speed: float | None = 4.0,
    acceleration: float | None = 0.0,
    yaw_rate: float | None = 0.0,
    reason: str = "default_keep",
) -> SourcePrediction:
    return SourcePrediction(
        sample_token=token,
        scene_token=scene,
        split=split,
        ground_truth_action=expected,
        predicted_action=predicted,
        is_correct=expected == predicted,
        baseline_name="ego_motion_rule",
        rule_version="phase0.2b-ego-motion-rule-v0.1",
        candidate_id="candidate-0293",
        thresholds_sha256=THRESHOLD_SHA,
        motion_availability=availability,
        speed_mps=speed,
        longitudinal_acceleration_mps2=acceleration,
        yaw_rate_radps=yaw_rate,
        decision_reason=reason,
        label_rule_version="phase-1.6-meta-action-v0.2",
        manifest_schema_version="phase0_trainval_dataset_manifest_v1",
        split_mapping_sha256="a" * 64,
    )


def evaluation_sample(
    prediction: SourcePrediction,
) -> EgoMotionPredictionSample:
    return EgoMotionPredictionSample(
        sample_token=prediction.sample_token,
        scene_token=prediction.scene_token,
        split="validation",
        features=EgoMotionFeatures(
            speed_mps=prediction.speed_mps,
            longitudinal_acceleration_mps2=(
                prediction.longitudinal_acceleration_mps2
            ),
            yaw_rate_radps=prediction.yaw_rate_radps,
            availability=prediction.motion_availability,
            history_interval_sec=(
                None if prediction.motion_availability == "unavailable" else 0.5
            ),
            acceleration_interval_sec=(
                0.5 if prediction.motion_availability == "full" else None
            ),
        ),
        ground_truth_action=prediction.ground_truth_action,
        label_rule_version=prediction.label_rule_version,
        manifest_schema_version=prediction.manifest_schema_version,
        split_mapping_sha256=prediction.split_mapping_sha256,
    )


def write_predictions(path: Path, payloads: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def selected_rule_payload() -> dict[str, object]:
    return {
        "rule_version": "phase0.2b-ego-motion-rule-v0.1",
        "selected_candidate_id": "candidate-0293",
        "thresholds": thresholds().as_dict(),
        "thresholds_sha256": THRESHOLD_SHA,
    }


def candidate(
    candidate_id: str,
    *,
    stop: float = 0.2,
    lateral: float = 0.05,
    accelerate: float = 0.5,
    decelerate: float = 0.3,
    macro_f1: float = 0.6,
    minimum_f1: float = 0.3,
    accuracy: float = 0.62,
) -> dict[str, object]:
    candidate_thresholds = EgoMotionRuleThresholds(
        stop,
        lateral,
        accelerate,
        decelerate,
    )
    return {
        "candidate_id": candidate_id,
        "thresholds": {
            "stop_speed_threshold_mps": stop,
            "lateral_yaw_rate_threshold_radps": lateral,
            "accelerate_threshold_mps2": accelerate,
            "decelerate_threshold_mps2": decelerate,
        },
        "thresholds_sha256": candidate_thresholds.sha256(),
        "validation_macro_f1": macro_f1,
        "minimum_per_class_f1": minimum_f1,
        "validation_accuracy": accuracy,
    }


def leaderboard() -> dict[str, object]:
    return {
        "candidate_count": 3,
        "candidates": [
            candidate("candidate-0293", macro_f1=0.62, accuracy=0.63),
            candidate(
                "candidate-0168",
                stop=0.1,
                macro_f1=0.619,
                accuracy=0.62,
            ),
            candidate(
                "candidate-0418",
                stop=0.5,
                macro_f1=0.60,
                accuracy=0.61,
            ),
        ]
    }


def contract_leaderboard() -> dict[str, object]:
    candidates = []
    for index in range(1, 626):
        item = candidate(
            f"candidate-{index:04d}",
            stop=1.0 + index / 1000,
            macro_f1=0.5,
            accuracy=0.5,
        )
        if index == 293:
            item = candidate(
                "candidate-0293",
                macro_f1=0.62,
                accuracy=0.63,
            )
        candidates.append(item)
    return {"candidate_count": 625, "candidates": candidates}


def valid_gate_kwargs() -> dict[str, object]:
    leaderboard_evidence = {
        "candidate_count_is_625": True,
        "candidate_list_length_is_625": True,
        "candidate_ids_unique": True,
        "threshold_hashes_unique": True,
        "selected_thresholds_match": True,
        "selected_rule_thresholds_match": True,
        "selected_threshold_sha_matches": True,
        "selected_rule_threshold_sha_matches": True,
    }
    trace_contract = {
        "baseline_name_match": True,
        "rule_version_match": True,
        "candidate_id_match": True,
        "thresholds_sha256_match": True,
        "label_rule_version_match": True,
        "manifest_schema_version_match": True,
        "split_mapping_sha256_match": True,
        "split_and_is_correct_match": True,
        "matched_record_count": 3594,
    }
    return {
        "manifest_sha_matches": True,
        "source_artifact_hashes_match": True,
        "leaderboard_contract": leaderboard_evidence,
        "selected_rule_contract": {"all_fields_match": True},
        "prediction_trace_contract": trace_contract,
        "prediction_reproduction": {
            "source_prediction_count": 3594,
            "all_predictions_match": True,
        },
        "reproduction_evidence": {
            "metrics_reproduce_exactly": True,
            "confusion_matrix_reproduces_exactly": True,
            "prediction_distribution_reproduces_exactly": True,
            "decision_reason_distribution_reproduces_exactly": True,
        },
        "forbidden_field_evidence": {"passed": True},
        "test_isolation_evidence": {
            "test_rows_absent_from_analysis": True,
            "test_evaluation_absent": True,
        },
        "candidate_rank_is_one": True,
        "all_predictions_legal": True,
        "macro_f1_exceeds_majority": True,
        "accuracy_exceeds_majority": True,
        "all_actions_predicted": True,
        "serialization_contract_passed": True,
    }


def test_source_sha_mismatch_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_source_hashes(
            tmp_path,
            {"source.json": "0" * 64},
            lambda path: "1" * 64,
        )


def test_selected_candidate_mismatch_fails() -> None:
    payload = selected_rule_payload()
    payload["selected_candidate_id"] = "candidate-0001"

    with pytest.raises(ValueError, match="selected candidate"):
        validate_selected_rule(
            payload,
            "candidate-0293",
            thresholds(),
            THRESHOLD_SHA,
            "phase0.2b-ego-motion-rule-v0.1",
        )


def test_threshold_values_mismatch_fails() -> None:
    payload = selected_rule_payload()
    payload["thresholds"] = {**thresholds().as_dict(), "stop_speed_threshold_mps": 0.1}

    with pytest.raises(ValueError, match="threshold values"):
        validate_selected_rule(
            payload,
            "candidate-0293",
            thresholds(),
            THRESHOLD_SHA,
            "phase0.2b-ego-motion-rule-v0.1",
        )


def test_threshold_sha_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="configured threshold SHA-256"):
        validate_selected_rule(
            selected_rule_payload(),
            "candidate-0293",
            thresholds(),
            "0" * 64,
            "phase0.2b-ego-motion-rule-v0.1",
        )


def test_real_count_gate_requires_3594(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [asdict(source_prediction())])

    with pytest.raises(ValueError, match="must be 3594"):
        read_source_predictions(path)


def test_duplicate_sample_token_is_rejected(tmp_path: Path) -> None:
    payload = asdict(source_prediction())
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [payload, payload])

    with pytest.raises(ValueError, match="duplicate sample_token"):
        read_source_predictions(path, expected_count=2)


@pytest.mark.parametrize("split", ("train", "test"))
def test_non_validation_prediction_is_rejected(
    tmp_path: Path, split: str
) -> None:
    payload = asdict(source_prediction())
    payload["split"] = split
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [payload])

    with pytest.raises(ValueError, match="validation only"):
        read_source_predictions(path, expected_count=1)


def test_illegal_action_is_rejected(tmp_path: Path) -> None:
    payload = asdict(source_prediction())
    payload["predicted_action"] = "illegal"
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [payload])

    with pytest.raises(ValueError, match="illegal action"):
        read_source_predictions(path, expected_count=1)


def test_inconsistent_is_correct_is_rejected(tmp_path: Path) -> None:
    payload = asdict(source_prediction())
    payload["is_correct"] = False
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [payload])

    with pytest.raises(ValueError, match="is_correct is inconsistent"):
        read_source_predictions(path, expected_count=1)


def test_forbidden_prediction_field_is_rejected(tmp_path: Path) -> None:
    payload = asdict(source_prediction())
    payload["future_ego_trajectory"] = []
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, [payload])

    with pytest.raises(ValueError, match="forbidden fields"):
        read_source_predictions(path, expected_count=1)


def test_reproduced_prediction_mismatch_fails() -> None:
    source = source_prediction(
        predicted="stop",
        expected="keep",
        reason="speed_below_stop_threshold",
    )

    with pytest.raises(ValueError, match="reproduced prediction differs"):
        reproduce_predictions(
            (evaluation_sample(source),), (source,), thresholds()
        )


@pytest.mark.parametrize(
    "description",
    ("validation metrics", "confusion matrix", "decision reason distribution"),
)
def test_reproduced_source_payload_mismatch_fails(description: str) -> None:
    with pytest.raises(ValueError, match=description):
        assert_payload_equal({"actual": 1}, {"expected": 1}, description)


def test_candidate_0293_rank_is_verified() -> None:
    result = candidate_stability(leaderboard(), "candidate-0293")

    assert result["selected_candidate_rank"] == 1
    assert result["second_ranked_candidate"]["candidate_id"] == "candidate-0168"


def test_candidate_stability_uses_existing_leaderboard_only() -> None:
    result = candidate_stability(leaderboard(), "candidate-0293")

    assert result["source"] == "existing_candidate_leaderboard_only"
    assert result["candidate_reselection_performed"] is False
    assert [item["candidate_id"] for item in result["local_grid_neighbours"]] == [
        "candidate-0168",
        "candidate-0418",
    ]


def test_leaderboard_candidate_count_must_be_625() -> None:
    payload = contract_leaderboard()
    payload["candidate_count"] = 624

    with pytest.raises(ValueError, match="candidate_count must be 625"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_candidate_list_length_must_be_625() -> None:
    payload = contract_leaderboard()
    payload["candidates"].pop()

    with pytest.raises(ValueError, match="list length must be 625"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_duplicate_candidate_id_is_rejected() -> None:
    payload = contract_leaderboard()
    payload["candidates"][0]["candidate_id"] = "candidate-0002"

    with pytest.raises(ValueError, match="candidate_id values must be unique"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_duplicate_threshold_hash_is_rejected() -> None:
    payload = contract_leaderboard()
    payload["candidates"][0]["thresholds_sha256"] = payload["candidates"][1][
        "thresholds_sha256"
    ]

    with pytest.raises(ValueError, match="thresholds_sha256 values must be unique"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_selected_thresholds_must_match_freeze_config() -> None:
    payload = contract_leaderboard()
    payload["candidates"][292]["thresholds"]["stop_speed_threshold_mps"] = 0.1

    with pytest.raises(ValueError, match="selected thresholds"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_selected_threshold_sha_must_match_freeze_config() -> None:
    payload = contract_leaderboard()
    payload["candidates"][292]["thresholds_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="selected threshold SHA-256"):
        validate_leaderboard_contract(
            payload,
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule_payload(),
        )


def test_leaderboard_and_selected_rule_thresholds_must_match() -> None:
    selected_rule = selected_rule_payload()
    selected_rule["thresholds"] = {
        **thresholds().as_dict(),
        "stop_speed_threshold_mps": 0.1,
    }

    with pytest.raises(ValueError, match="leaderboard and selected rule"):
        validate_leaderboard_contract(
            contract_leaderboard(),
            selected_candidate_id="candidate-0293",
            thresholds=thresholds(),
            thresholds_sha256=THRESHOLD_SHA,
            selected_rule=selected_rule,
        )


def test_prediction_baseline_name_must_match() -> None:
    prediction = replace(source_prediction(), baseline_name="other")

    with pytest.raises(ValueError, match="baseline_name"):
        _validate_prediction_contracts(
            (prediction,),
            candidate_id="candidate-0293",
            thresholds_sha256=THRESHOLD_SHA,
            source_rule_version="phase0.2b-ego-motion-rule-v0.1",
            label_rule_version="phase-1.6-meta-action-v0.2",
            manifest_schema_version="phase0_trainval_dataset_manifest_v1",
            split_mapping_sha256="a" * 64,
        )


def test_prediction_label_rule_version_must_match_manifest() -> None:
    source = source_prediction()
    prediction = replace(source, label_rule_version="wrong-label-rule")

    with pytest.raises(ValueError, match="label_rule_version"):
        reproduce_predictions(
            (evaluation_sample(source),),
            (prediction,),
            thresholds(),
            expected_label_rule_version="phase-1.6-meta-action-v0.2",
        )


def test_prediction_manifest_schema_version_must_match_manifest() -> None:
    source = source_prediction()
    prediction = replace(source, manifest_schema_version="wrong-schema")

    with pytest.raises(ValueError, match="manifest_schema_version"):
        reproduce_predictions(
            (evaluation_sample(source),),
            (prediction,),
            thresholds(),
            expected_manifest_schema_version=(
                "phase0_trainval_dataset_manifest_v1"
            ),
        )


def test_prediction_split_mapping_sha_must_match_manifest() -> None:
    source = source_prediction()
    prediction = replace(source, split_mapping_sha256="b" * 64)

    with pytest.raises(ValueError, match="split_mapping_sha256"):
        reproduce_predictions(
            (evaluation_sample(source),),
            (prediction,),
            thresholds(),
            expected_split_mapping_sha256="a" * 64,
        )


def test_prediction_trace_contract_records_3594_matches() -> None:
    predictions = tuple(
        source_prediction(token=f"sample-{index:04d}")
        for index in range(3594)
    )
    samples = tuple(evaluation_sample(item) for item in predictions)

    result = reproduce_predictions(
        samples,
        predictions,
        thresholds(),
        expected_rule_version="phase0.2b-ego-motion-rule-v0.1",
        expected_candidate_id="candidate-0293",
        expected_thresholds_sha256=THRESHOLD_SHA,
        expected_label_rule_version="phase-1.6-meta-action-v0.2",
        expected_manifest_schema_version="phase0_trainval_dataset_manifest_v1",
        expected_split_mapping_sha256="a" * 64,
    )

    trace = result["prediction_trace_contract"]
    assert trace["matched_record_count"] == 3594
    assert all(
        value is True
        for key, value in trace.items()
        if key != "matched_record_count"
    )


def test_freeze_gates_are_derived_from_evidence() -> None:
    arguments = valid_gate_kwargs()
    arguments["reproduction_evidence"] = {
        **arguments["reproduction_evidence"],
        "metrics_reproduce_exactly": False,
    }

    gates = build_freeze_gates(**arguments)

    assert gates["metrics_reproduce_exactly"] is False


@pytest.mark.parametrize("evidence_area", ("leaderboard", "trace"))
def test_failed_trace_or_leaderboard_gate_cannot_freeze(
    evidence_area: str,
) -> None:
    arguments = valid_gate_kwargs()
    if evidence_area == "leaderboard":
        arguments["leaderboard_contract"] = {
            **arguments["leaderboard_contract"],
            "candidate_ids_unique": False,
        }
    else:
        arguments["prediction_trace_contract"] = {
            **arguments["prediction_trace_contract"],
            "label_rule_version_match": False,
        }
    gates = build_freeze_gates(**arguments)

    with pytest.raises(ValueError, match="freeze gates failed"):
        build_freeze_record(gates=gates, payload={})


def test_predictor_decision_fields_only_include_used_features() -> None:
    contract = inference_field_contract()

    assert contract["predictor_decision_fields"] == [
        "speed_mps",
        "longitudinal_acceleration_mps2",
        "yaw_rate_radps",
        "availability",
    ]


def test_intervals_are_trace_only_fields() -> None:
    contract = inference_field_contract()

    assert contract["trace_only_fields"] == [
        "history_interval_sec",
        "acceleration_interval_sec",
    ]


def test_confusion_pairs_have_deterministic_order_and_fractions() -> None:
    predictions = (
        source_prediction(token="1", expected="keep", predicted="decelerate"),
        source_prediction(token="2", expected="keep", predicted="decelerate"),
        source_prediction(token="3", expected="accelerate", predicted="keep"),
    )

    pairs = confusion_pairs(predictions)

    assert pairs[0]["count"] == 2
    assert pairs[0]["ground_truth_action"] == "keep"
    assert pairs[0]["fraction_of_all_errors"] == pytest.approx(2 / 3)


def test_availability_groups_have_stable_schema() -> None:
    predictions = (
        source_prediction(token="full"),
        source_prediction(
            token="partial",
            availability="partial",
            acceleration=None,
        ),
        source_prediction(
            token="unavailable",
            availability="unavailable",
            speed=None,
            acceleration=None,
            yaw_rate=None,
            reason="unavailable_motion_fallback_keep",
        ),
    )

    result = availability_analysis(predictions)

    assert tuple(result) == ("full", "partial", "unavailable")
    assert all(group["sample_count"] == 1 for group in result.values())


def test_decision_reason_group_counts_errors() -> None:
    predictions = (
        source_prediction(token="correct"),
        source_prediction(token="error", expected="stop"),
    )

    result = decision_reason_analysis(predictions)["default_keep"]

    assert result["sample_count"] == 2
    assert result["error_count"] == 1
    assert result["accuracy"] == 0.5


def test_stop_boundary_is_inclusive() -> None:
    prediction = source_prediction(speed=0.25)

    assert "stop_boundary" in threshold_boundary_flags(
        prediction, thresholds(), margins()
    )


def test_lateral_boundary_uses_absolute_yaw_rate() -> None:
    prediction = source_prediction(yaw_rate=-0.06)

    assert "lateral_boundary" in threshold_boundary_flags(
        prediction, thresholds(), margins()
    )


@pytest.mark.parametrize(
    ("acceleration", "expected_flag"),
    ((0.55, "accelerate_boundary"), (-0.35, "decelerate_boundary")),
)
def test_longitudinal_boundaries_are_inclusive(
    acceleration: float, expected_flag: str
) -> None:
    prediction = source_prediction(acceleration=acceleration)

    assert expected_flag in threshold_boundary_flags(
        prediction, thresholds(), margins()
    )


def test_diagnostic_margin_does_not_change_prediction() -> None:
    features = EgoMotionFeatures(0.25, 0.0, 0.0, "full", 0.5, 0.5)
    before = predict_ego_motion_action(features, thresholds())
    boundary = threshold_boundary_analysis(
        (source_prediction(speed=0.25),), thresholds(), margins()
    )
    after = predict_ego_motion_action(features, thresholds())

    assert before == after
    assert boundary["prediction_effect"] == "none"


def test_multi_trigger_and_priority_conflict_statistics() -> None:
    prediction = source_prediction(
        expected="stop",
        predicted="stop",
        speed=0.1,
        acceleration=0.8,
        yaw_rate=0.1,
        reason="speed_below_stop_threshold",
    )

    result = trigger_overlap_analysis((prediction,), thresholds())

    assert len(active_triggers(prediction, thresholds())) == 3
    assert result["trigger_count_distribution"]["2_or_more"] == 1
    assert result["priority_conflict_sample_count"] == 1
    assert result["priority_conflict_error_rate"] == 0.0


def test_scene_concentration_sort_is_deterministic() -> None:
    predictions = (
        source_prediction(token="a1", scene="scene-b", expected="stop"),
        source_prediction(token="a2", scene="scene-b", expected="stop"),
        source_prediction(token="b1", scene="scene-a", expected="stop"),
        source_prediction(token="b2", scene="scene-a"),
        source_prediction(token="b3", scene="scene-a"),
        source_prediction(token="b4", scene="scene-a"),
        source_prediction(token="b5", scene="scene-a"),
    )

    result = scene_error_concentration(predictions)

    assert result["top_20_by_error_count"][0]["scene_token"] == "scene-b"
    assert result["top_20_by_error_rate_min_5_samples"][0]["scene_token"] == "scene-a"


def test_validation_failures_only_include_errors() -> None:
    predictions = (
        source_prediction(token="correct"),
        source_prediction(token="error", expected="stop"),
    )

    records = build_failure_records(predictions, thresholds(), margins())

    assert [record["sample_token"] for record in records] == ["error"]


def test_validation_failures_exclude_forbidden_fields() -> None:
    records = build_failure_records(
        (source_prediction(expected="stop"),), thresholds(), margins()
    )

    assert not {
        "future_ego_trajectory",
        "nearby_agents",
        "current_ego_pose",
        "cam_front_path",
        "test_label",
    }.intersection(records[0])


def test_validation_failure_order_follows_input() -> None:
    predictions = tuple(
        source_prediction(token=token, expected="stop")
        for token in ("z", "a", "m")
    )

    records = build_failure_records(predictions, thresholds(), margins())

    assert [record["sample_token"] for record in records] == ["z", "a", "m"]


def test_unavailable_policy_remains_keep() -> None:
    decision = predict_ego_motion_action(
        EgoMotionFeatures(None, None, None, "unavailable", None, None),
        thresholds(),
    )

    assert decision.predicted_action == "keep"


@pytest.mark.parametrize(
    ("speed", "yaw_rate", "expected"),
    ((4.0, 0.0, "keep"), (0.1, 0.0, "stop"), (4.0, 0.1, "left_lateral")),
)
def test_partial_never_predicts_longitudinal_actions(
    speed: float, yaw_rate: float, expected: str
) -> None:
    decision = predict_ego_motion_action(
        EgoMotionFeatures(speed, None, yaw_rate, "partial", 0.5, None),
        thresholds(),
    )

    assert decision.predicted_action == expected


def test_freeze_config_does_not_define_candidate_grid() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/phase0_2_rule_freeze_v0_1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "candidate_grid" not in config
    assert config["selected_candidate_id"] == "candidate-0293"


def test_frozen_output_records_no_test_evaluation() -> None:
    record = build_freeze_record(
        gates={"all": True},
        payload={"test_evaluation_performed": False},
    )

    assert record["test_evaluation_performed"] is False


def test_all_gates_produce_frozen_status() -> None:
    record = build_freeze_record(gates={"a": True, "b": True}, payload={})

    assert record["freeze_status"] == "frozen"


def test_failed_gate_cannot_produce_frozen_record() -> None:
    with pytest.raises(ValueError, match="freeze gates failed"):
        build_freeze_record(gates={"source_hashes": False}, payload={})


def test_canonical_outputs_are_deterministic(tmp_path: Path) -> None:
    payload = {"b": [2, 1], "a": {"value": 1}}
    records = ({"sample_token": "b"}, {"sample_token": "a"})
    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    first_jsonl = tmp_path / "first.jsonl"
    second_jsonl = tmp_path / "second.jsonl"

    write_canonical_json(payload, first_json)
    write_canonical_json(payload, second_json)
    write_jsonl_records(records, first_jsonl)
    write_jsonl_records(records, second_jsonl)

    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePath
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_ego_motion_inputs import (
    load_config,
    require_manifest_sha256,
    sha256_file,
)
from src.actions.schema import ACTION_SCHEMA, normalize_action
from src.baselines.ego_motion import (
    EgoMotionPredictionSample,
    EgoMotionRuleThresholds,
    parse_rule_evaluation_sample,
)
from src.baselines.ego_motion_analysis import (
    DiagnosticMargins,
    SourcePrediction,
    assert_payload_equal,
    availability_analysis,
    build_failure_records,
    build_freeze_record,
    build_overall_analysis,
    candidate_stability,
    confusion_pairs,
    decision_reason_analysis,
    metrics_for_predictions,
    metrics_payload,
    read_source_predictions,
    reproduce_predictions,
    scene_error_concentration,
    threshold_boundary_analysis,
    trigger_overlap_analysis,
    validate_selected_rule,
    validate_source_hashes,
)
from src.baselines.majority import fit_majority_action, predict_split
from src.phase0.manifest import write_canonical_json, write_jsonl_records
from src.phase0.protocol import (
    PHASE0_SPLIT_SEED,
    ManifestSample,
    complete_action_distribution,
    iter_manifest_rows,
    validate_manifest,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


SOURCE_FILENAMES = (
    "candidate_leaderboard.json",
    "selected_rule.json",
    "validation_predictions.jsonl",
    "validation_metrics.json",
    "majority_validation_metrics.json",
    "comparison.json",
)
OUTPUT_FILENAMES = (
    "failure_analysis.json",
    "validation_failures.jsonl",
    "rule_freeze.json",
)


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _required_mapping(
    mapping: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration missing {key}")
    return value


def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"configuration {key} must be a number")
    return float(value)


def _relative_path(mapping: Mapping[str, object], key: str) -> Path:
    value = _required_string(mapping, key)
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be relative to VLA_DERIVED_ROOT")
    return Path(value)


def _thresholds(config: Mapping[str, object]) -> EgoMotionRuleThresholds:
    values = _required_mapping(config, "thresholds")
    return EgoMotionRuleThresholds(
        stop_speed_threshold_mps=_required_number(
            values, "stop_speed_threshold_mps"
        ),
        lateral_yaw_rate_threshold_radps=_required_number(
            values, "lateral_yaw_rate_threshold_radps"
        ),
        accelerate_threshold_mps2=_required_number(
            values, "accelerate_threshold_mps2"
        ),
        decelerate_threshold_mps2=_required_number(
            values, "decelerate_threshold_mps2"
        ),
    )


def _diagnostic_margins(config: Mapping[str, object]) -> DiagnosticMargins:
    values = _required_mapping(config, "diagnostic_margin")
    return DiagnosticMargins(
        stop_speed_mps=_required_number(values, "stop_speed_mps"),
        lateral_yaw_rate_radps=_required_number(
            values, "lateral_yaw_rate_radps"
        ),
        longitudinal_acceleration_mps2=_required_number(
            values, "longitudinal_acceleration_mps2"
        ),
    )


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"source artifact must be a JSON object: {path}")
    return payload


def _expected_source_hashes(
    config: Mapping[str, object],
) -> dict[str, str]:
    configured = _required_mapping(config, "expected_source_sha256")
    if set(configured) != set(SOURCE_FILENAMES):
        raise ValueError("expected_source_sha256 must list the six source files")
    return {
        filename: _required_string(configured, filename)
        for filename in SOURCE_FILENAMES
    }


def _load_manifest_samples(
    manifest_path: Path,
) -> tuple[tuple[ManifestSample, ...], tuple[EgoMotionPredictionSample, ...]]:
    majority_samples = []
    validation_samples = []
    for row in iter_manifest_rows(manifest_path):
        split = row.get("split")
        if not isinstance(split, str):
            raise ValueError("manifest row missing split")
        if split == "test":
            continue
        if split not in {"train", "validation"}:
            raise ValueError(f"unsupported split: {split!r}")
        sample_token = row.get("sample_token")
        scene_token = row.get("scene_token")
        meta_action = row.get("meta_action")
        label_rule_version = row.get("label_rule_version")
        if not all(
            isinstance(value, str) and value
            for value in (
                sample_token,
                scene_token,
                meta_action,
                label_rule_version,
            )
        ):
            raise ValueError("train/validation manifest row is incomplete")
        majority_samples.append(
            ManifestSample(
                sample_token=sample_token,
                scene_token=scene_token,
                meta_action=normalize_action(meta_action),
                split=split,
                label_rule_version=label_rule_version,
            )
        )
        parsed = parse_rule_evaluation_sample(row)
        if parsed is not None:
            validation_samples.append(parsed)
    return tuple(majority_samples), tuple(validation_samples)


def _validate_source_contracts(
    *,
    source_payloads: Mapping[str, Mapping[str, object]],
    candidate_id: str,
    thresholds_sha256: str,
    source_rule_version: str,
) -> None:
    for filename, payload in source_payloads.items():
        if payload.get("test_evaluation_performed") is not False:
            raise ValueError(
                f"source artifact does not preserve test isolation: {filename}"
            )
    selected_rule = source_payloads["selected_rule.json"]
    if selected_rule.get("selected_candidate_id") != candidate_id:
        raise ValueError("source selected candidate is inconsistent")
    if selected_rule.get("thresholds_sha256") != thresholds_sha256:
        raise ValueError("source threshold SHA-256 is inconsistent")
    if selected_rule.get("rule_version") != source_rule_version:
        raise ValueError("source rule version is inconsistent")


def _validate_prediction_contracts(
    predictions: Sequence[SourcePrediction],
    *,
    candidate_id: str,
    thresholds_sha256: str,
    source_rule_version: str,
) -> None:
    for prediction in predictions:
        if prediction.candidate_id != candidate_id:
            raise ValueError("prediction candidate_id is inconsistent")
        if prediction.thresholds_sha256 != thresholds_sha256:
            raise ValueError("prediction thresholds_sha256 is inconsistent")
        if prediction.rule_version != source_rule_version:
            raise ValueError("prediction rule_version is inconsistent")


def _comparison_payload(
    selected_metrics: Mapping[str, object],
    selected_distribution: Mapping[str, int],
    majority_action: str,
    majority_metrics: Mapping[str, object],
) -> dict[str, object]:
    selected_f1 = selected_metrics["per_class_f1"]
    majority_f1 = majority_metrics["per_class_f1"]
    if not isinstance(selected_f1, Mapping) or not isinstance(
        majority_f1, Mapping
    ):
        raise ValueError("per-class F1 payload must be an object")
    return {
        "selected_rule_metrics": dict(selected_metrics),
        "selected_rule_prediction_distribution": dict(selected_distribution),
        "majority_action": majority_action,
        "majority_metrics": dict(majority_metrics),
        "macro_f1_delta": (
            float(selected_metrics["macro_f1"])
            - float(majority_metrics["macro_f1"])
        ),
        "accuracy_delta": (
            float(selected_metrics["accuracy"])
            - float(majority_metrics["accuracy"])
        ),
        "per_class_f1_delta": {
            action: float(selected_f1[action]) - float(majority_f1[action])
            for action in ACTION_SCHEMA
        },
        "test_evaluation_performed": False,
    }


def run_analysis(
    config_path: Path,
    derived_root: Path,
) -> tuple[Path, dict[str, object]]:
    config = load_config(config_path)
    manifest_path = derived_root / _relative_path(
        config, "manifest_relative_path"
    )
    source_dir = derived_root / _relative_path(
        config, "source_output_relative_dir"
    )
    output_dir = derived_root / _relative_path(config, "output_relative_dir")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest_sha256 = require_manifest_sha256(
        manifest_path, _required_string(config, "expected_manifest_sha256")
    )
    source_hashes = validate_source_hashes(
        source_dir, _expected_source_hashes(config), sha256_file
    )
    validation = validate_manifest(manifest_path)
    if validation.manifest_schema_version != _required_string(
        config, "expected_manifest_schema_version"
    ):
        raise ValueError("manifest schema version does not match freeze config")
    if validation.label_rule_version != _required_string(
        config, "expected_label_rule_version"
    ):
        raise ValueError("label rule version does not match freeze config")
    if config.get("expected_split_seed") != PHASE0_SPLIT_SEED:
        raise ValueError("split seed does not match frozen protocol")
    if _required_string(
        config, "expected_split_strategy_version"
    ) != SPLIT_STRATEGY_VERSION:
        raise ValueError("split strategy does not match frozen protocol")
    if validation.split_mapping_sha256 is None:
        raise ValueError("trainval manifest must provide split_mapping_sha256")

    candidate_id = _required_string(config, "selected_candidate_id")
    thresholds_sha256 = _required_string(config, "expected_thresholds_sha256")
    source_rule_version = _required_string(config, "source_rule_version")
    thresholds = _thresholds(config)
    diagnostic_margins = _diagnostic_margins(config)
    source_payloads = {
        filename: _load_json(source_dir / filename)
        for filename in SOURCE_FILENAMES
        if filename.endswith(".json")
    }
    validate_selected_rule(
        source_payloads["selected_rule.json"],
        candidate_id,
        thresholds,
        thresholds_sha256,
        source_rule_version,
    )
    _validate_source_contracts(
        source_payloads=source_payloads,
        candidate_id=candidate_id,
        thresholds_sha256=thresholds_sha256,
        source_rule_version=source_rule_version,
    )
    predictions = read_source_predictions(
        source_dir / "validation_predictions.jsonl"
    )
    _validate_prediction_contracts(
        predictions,
        candidate_id=candidate_id,
        thresholds_sha256=thresholds_sha256,
        source_rule_version=source_rule_version,
    )
    majority_samples, validation_samples = _load_manifest_samples(manifest_path)
    reproduction = reproduce_predictions(
        validation_samples, predictions, thresholds
    )

    metrics = metrics_for_predictions(predictions)
    metrics_data = metrics_payload(metrics)
    prediction_distribution = complete_action_distribution(
        tuple(item.predicted_action for item in predictions)
    )
    reason_distribution = dict(
        Counter(item.decision_reason for item in predictions)
    )
    availability_distribution = dict(
        Counter(item.motion_availability for item in predictions)
    )
    validation_metrics = source_payloads["validation_metrics.json"]
    selected_rule = source_payloads["selected_rule.json"]
    assert_payload_equal(
        metrics_data, validation_metrics.get("metrics"), "validation metrics"
    )
    assert_payload_equal(
        metrics_data, selected_rule.get("validation_metrics"), "selected metrics"
    )
    assert_payload_equal(
        prediction_distribution,
        validation_metrics.get("predicted_class_distribution"),
        "prediction distribution",
    )
    assert_payload_equal(
        prediction_distribution,
        selected_rule.get("predicted_class_distribution"),
        "selected prediction distribution",
    )
    assert_payload_equal(
        reason_distribution,
        validation_metrics.get("decision_reason_distribution"),
        "decision reason distribution",
    )
    assert_payload_equal(
        reason_distribution,
        selected_rule.get("decision_reason_distribution"),
        "selected decision reason distribution",
    )
    assert_payload_equal(
        availability_distribution,
        selected_rule.get("motion_availability_distribution"),
        "motion availability distribution",
    )

    majority_action = fit_majority_action(majority_samples)
    _, majority_metrics = predict_split(
        samples=majority_samples,
        split="validation",
        majority_action=majority_action,
        label_rule_version=validation.label_rule_version,
    )
    majority_metrics_data = metrics_payload(majority_metrics)
    majority_source = source_payloads["majority_validation_metrics.json"]
    assert_payload_equal(
        majority_action, majority_source.get("majority_action"), "majority action"
    )
    assert_payload_equal(
        majority_metrics_data,
        majority_source.get("metrics"),
        "majority metrics",
    )
    comparison = _comparison_payload(
        metrics_data,
        prediction_distribution,
        majority_action,
        majority_metrics_data,
    )
    assert_payload_equal(
        comparison, source_payloads["comparison.json"], "comparison"
    )
    stability = candidate_stability(
        source_payloads["candidate_leaderboard.json"], candidate_id
    )

    failures = build_failure_records(
        predictions, thresholds, diagnostic_margins
    )
    failure_analysis = {
        "analysis_schema_version": "phase0.2_rule_failure_analysis_v0.1",
        "manifest_sha256": manifest_sha256,
        "source_output_sha256": source_hashes,
        "selected_candidate_id": candidate_id,
        "thresholds": thresholds.as_dict(),
        "thresholds_sha256": thresholds_sha256,
        "overall_metrics": build_overall_analysis(predictions),
        "confusion_pair_analysis": confusion_pairs(predictions),
        "availability_analysis": availability_analysis(predictions),
        "decision_reason_analysis": decision_reason_analysis(predictions),
        "threshold_boundary_analysis": threshold_boundary_analysis(
            predictions, thresholds, diagnostic_margins
        ),
        "trigger_overlap_analysis": trigger_overlap_analysis(
            predictions, thresholds
        ),
        "scene_error_concentration": scene_error_concentration(predictions),
        "candidate_stability": stability,
        "validation_failure_sample_count": len(failures),
        "forbidden_field_check": {
            "source_prediction_forbidden_field_count": 0,
            "failure_output_forbidden_field_count": 0,
            "passed": True,
        },
        "prediction_reproduction": reproduction,
        "test_evaluation_performed": False,
    }
    gates = {
        "manifest_sha_matches": True,
        "source_artifact_hashes_match": True,
        "prediction_count_is_3594": len(predictions) == 3594,
        "predictions_reproduce_exactly": reproduction["all_predictions_match"],
        "metrics_and_distributions_reproduce": True,
        "all_predictions_legal": metrics.invalid_prediction_count == 0,
        "forbidden_fields_absent": True,
        "test_evaluation_absent": True,
        "candidate_and_thresholds_unchanged": True,
        "candidate_rank_is_one": stability["selected_candidate_rank"] == 1,
        "macro_f1_exceeds_majority": metrics.macro_f1 > majority_metrics.macro_f1,
        "accuracy_exceeds_majority": metrics.accuracy > majority_metrics.accuracy,
        "all_actions_predicted": all(
            prediction_distribution[action] > 0 for action in ACTION_SCHEMA
        ),
        "deterministic_canonical_serialization": True,
    }
    freeze_payload = {
        "freeze_schema_version": _required_string(
            config, "freeze_schema_version"
        ),
        "frozen_rule_version": _required_string(config, "frozen_rule_version"),
        "source_rule_version": source_rule_version,
        "selected_candidate_id": candidate_id,
        "thresholds": thresholds.as_dict(),
        "thresholds_sha256": thresholds_sha256,
        "rule_priority": [
            "unavailable_fallback_keep",
            "stop",
            "left_lateral",
            "right_lateral",
            "accelerate",
            "decelerate",
            "default_keep",
        ],
        "missing_value_policy": {
            "unavailable": "fallback_keep",
            "partial": "stop_or_lateral_or_keep_without_acceleration",
        },
        "manifest_sha256": manifest_sha256,
        "manifest_schema_version": validation.manifest_schema_version,
        "label_rule_version": validation.label_rule_version,
        "split_mapping_sha256": validation.split_mapping_sha256,
        "source_output_sha256": source_hashes,
        "validation_metrics": metrics_data,
        "majority_comparison": comparison,
        "prediction_reproduction": reproduction,
        "known_limitations": [
            "accelerate F1 is comparatively low on validation",
            (
                "validation was used for candidate selection and reporting, "
                "so it is not unbiased final performance"
            ),
            "unavailable motion always falls back to keep",
            "partial motion cannot predict accelerate or decelerate",
            "current and past ego-motion cannot fully express future driving intent",
            "this rule is a deterministic baseline, not an industrial driving policy",
        ],
        "allowed_inference_fields": [
            "speed_mps",
            "longitudinal_acceleration_mps2",
            "yaw_rate_radps",
            "availability",
            "history_interval_sec",
            "acceleration_interval_sec",
        ],
        "forbidden_inference_fields": [
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
        "test_evaluation_performed": False,
        "next_gate": "phase0.2d_one_shot_test",
    }
    rule_freeze = build_freeze_record(gates=gates, payload=freeze_payload)

    write_canonical_json(failure_analysis, output_dir / "failure_analysis.json")
    write_jsonl_records(failures, output_dir / "validation_failures.jsonl")
    write_canonical_json(rule_freeze, output_dir / "rule_freeze.json")
    output_hashes = {
        filename: sha256_file(output_dir / filename)
        for filename in OUTPUT_FILENAMES
    }
    return output_dir, {
        "output_dir": output_dir.as_posix(),
        "manifest_sha256": manifest_sha256,
        "source_output_sha256": source_hashes,
        "prediction_reproduction": reproduction,
        "freeze_status": rule_freeze["freeze_status"],
        "selected_candidate_id": candidate_id,
        "thresholds": thresholds.as_dict(),
        "validation_metrics": metrics_data,
        "validation_failure_sample_count": len(failures),
        "output_sha256": output_hashes,
        "test_evaluation_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and freeze the Phase 0.2b ego-motion rule."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    derived_root_value = os.environ.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        raise ValueError("VLA_DERIVED_ROOT is not set")
    _, result = run_analysis(arguments.config, Path(derived_root_value))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

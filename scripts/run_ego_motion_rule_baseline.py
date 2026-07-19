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
from src.actions.schema import ACTION_SCHEMA, KEEP, normalize_action
from src.baselines.ego_motion import (
    EgoMotionCandidateEvaluation,
    EgoMotionPredictionRecord,
    EgoMotionPredictionSample,
    build_prediction_records,
    build_rule_candidates,
    evaluate_rule_candidate,
    parse_rule_evaluation_sample,
    select_best_rule_candidate,
)
from src.baselines.majority import fit_majority_action, predict_split
from src.phase0.manifest import write_canonical_json, write_jsonl_records
from src.phase0.protocol import (
    PHASE0_SPLIT_SEED,
    ClassificationMetrics,
    ManifestSample,
    iter_manifest_rows,
    validate_manifest,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


CANDIDATE_GRID_FIELDS = (
    "stop_speed_threshold_mps",
    "lateral_yaw_rate_threshold_radps",
    "accelerate_threshold_mps2",
    "decelerate_threshold_mps2",
)
OUTPUT_FILENAMES = (
    "candidate_leaderboard.json",
    "selected_rule.json",
    "validation_predictions.jsonl",
    "validation_metrics.json",
    "majority_validation_metrics.json",
    "comparison.json",
)


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _relative_path(mapping: Mapping[str, object], key: str) -> Path:
    value = _required_string(mapping, key)
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be relative to VLA_DERIVED_ROOT")
    return Path(value)


def _candidate_grid(config: Mapping[str, object]) -> dict[str, tuple[float, ...]]:
    raw_grid = config.get("candidate_grid")
    if not isinstance(raw_grid, Mapping):
        raise ValueError("configuration missing candidate_grid")
    grid = {}
    for field in CANDIDATE_GRID_FIELDS:
        raw_values = raw_grid.get(field)
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
            raise ValueError(f"candidate_grid.{field} must be a sequence")
        values = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"candidate_grid.{field} must contain numbers")
            values.append(float(value))
        grid[field] = tuple(values)
    return grid


def _metrics_payload(metrics: ClassificationMetrics) -> dict[str, object]:
    return asdict(metrics)


def _candidate_payload(
    evaluation: EgoMotionCandidateEvaluation,
) -> dict[str, object]:
    return {
        "candidate_id": evaluation.candidate_id,
        "thresholds": evaluation.thresholds.as_dict(),
        "thresholds_sha256": evaluation.thresholds_sha256,
        "validation_macro_f1": evaluation.metrics.macro_f1,
        "minimum_per_class_f1": evaluation.minimum_per_class_f1,
        "validation_accuracy": evaluation.metrics.accuracy,
        "validation_per_class_f1": evaluation.metrics.per_class_f1,
        "predicted_class_distribution": (
            evaluation.predicted_class_distribution
        ),
        "invalid_output_rate": evaluation.metrics.invalid_output_rate,
    }


def build_selection_objective(
    evaluations: Sequence[EgoMotionCandidateEvaluation],
    selected: EgoMotionCandidateEvaluation,
) -> dict[str, object]:
    best_macro_f1 = max(item.metrics.macro_f1 for item in evaluations)
    macro_tied = tuple(
        item for item in evaluations if item.metrics.macro_f1 == best_macro_f1
    )
    best_minimum_f1 = max(item.minimum_per_class_f1 for item in macro_tied)
    minimum_f1_tied = tuple(
        item for item in macro_tied
        if item.minimum_per_class_f1 == best_minimum_f1
    )
    best_accuracy = max(item.metrics.accuracy for item in minimum_f1_tied)
    accuracy_tied = tuple(
        item for item in minimum_f1_tied
        if item.metrics.accuracy == best_accuracy
    )
    return {
        "ranking": [
            "maximum_validation_macro_f1",
            "maximum_minimum_per_class_f1",
            "maximum_validation_accuracy",
            "minimum_canonical_threshold_tuple",
        ],
        "tie_break_trace": {
            "candidate_count": len(evaluations),
            "maximum_macro_f1_candidate_count": len(macro_tied),
            "maximum_minimum_f1_candidate_count": len(minimum_f1_tied),
            "maximum_accuracy_candidate_count": len(accuracy_tied),
            "selected_canonical_threshold_tuple": list(
                selected.thresholds.canonical_tuple()
            ),
        },
        "selected_objective_values": {
            "validation_macro_f1": selected.metrics.macro_f1,
            "minimum_per_class_f1": selected.minimum_per_class_f1,
            "validation_accuracy": selected.metrics.accuracy,
        },
        "test_used_for_selection": False,
    }


def build_selected_rule_payload(
    rule_version: str,
    selected: EgoMotionCandidateEvaluation,
    selection_objective: Mapping[str, object],
    manifest_sha256: str,
    manifest_schema_version: str,
    label_rule_version: str,
    split_mapping_sha256: str,
    decision_reason_distribution: Mapping[str, int],
    motion_availability_distribution: Mapping[str, int],
) -> dict[str, object]:
    return {
        "rule_version": rule_version,
        "selected_candidate_id": selected.candidate_id,
        "thresholds": selected.thresholds.as_dict(),
        "thresholds_sha256": selected.thresholds_sha256,
        "selection_objective": dict(selection_objective),
        "manifest_sha256": manifest_sha256,
        "manifest_schema_version": manifest_schema_version,
        "label_rule_version": label_rule_version,
        "split_mapping_sha256": split_mapping_sha256,
        "validation_metrics": _metrics_payload(selected.metrics),
        "predicted_class_distribution": selected.predicted_class_distribution,
        "decision_reason_distribution": dict(decision_reason_distribution),
        "motion_availability_distribution": dict(
            motion_availability_distribution
        ),
        "missing_value_policy": {
            "unavailable": "fallback_keep",
            "partial": "stop_or_lateral_or_keep_without_acceleration",
        },
        "test_evaluation_performed": False,
    }


def build_validation_metrics_artifact_payload(
    rule_version: str,
    selected: EgoMotionCandidateEvaluation,
    decision_reason_distribution: Mapping[str, int],
) -> dict[str, object]:
    return {
        "rule_version": rule_version,
        "selected_candidate_id": selected.candidate_id,
        "thresholds_sha256": selected.thresholds_sha256,
        "metrics": _metrics_payload(selected.metrics),
        "predicted_class_distribution": (
            selected.predicted_class_distribution
        ),
        "decision_reason_distribution": dict(decision_reason_distribution),
        "test_evaluation_performed": False,
    }


def _load_samples(
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
        label_rule_version = row.get("label_rule_version")
        meta_action = row.get("meta_action")
        if not all(
            isinstance(value, str) and value
            for value in (
                sample_token,
                scene_token,
                label_rule_version,
                meta_action,
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


def _comparison_payload(
    selected: EgoMotionCandidateEvaluation,
    majority_action: str,
    majority_metrics: ClassificationMetrics,
) -> dict[str, object]:
    return {
        "selected_rule_metrics": _metrics_payload(selected.metrics),
        "selected_rule_prediction_distribution": (
            selected.predicted_class_distribution
        ),
        "majority_action": majority_action,
        "majority_metrics": _metrics_payload(majority_metrics),
        "macro_f1_delta": selected.metrics.macro_f1 - majority_metrics.macro_f1,
        "accuracy_delta": selected.metrics.accuracy - majority_metrics.accuracy,
        "per_class_f1_delta": {
            action: (
                selected.metrics.per_class_f1[action]
                - majority_metrics.per_class_f1[action]
            )
            for action in ACTION_SCHEMA
        },
        "test_evaluation_performed": False,
    }


def _output_sha256(output_dir: Path) -> dict[str, str]:
    return {
        filename: sha256_file(output_dir / filename)
        for filename in OUTPUT_FILENAMES
    }


def run_baseline(
    config_path: Path,
    derived_root: Path,
) -> tuple[Path, dict[str, object]]:
    config = load_config(config_path)
    manifest_relative_path = _relative_path(config, "manifest_relative_path")
    output_relative_dir = _relative_path(config, "output_relative_dir")
    manifest_path = derived_root / manifest_relative_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest_sha256 = require_manifest_sha256(
        manifest_path,
        _required_string(config, "expected_manifest_sha256"),
    )
    validation = validate_manifest(manifest_path)
    expected_schema = _required_string(
        config,
        "expected_manifest_schema_version",
    )
    if validation.manifest_schema_version != expected_schema:
        raise ValueError("manifest schema version does not match rule config")
    expected_label_rule = _required_string(config, "expected_label_rule_version")
    if validation.label_rule_version != expected_label_rule:
        raise ValueError("label rule version does not match rule config")
    if config.get("expected_split_seed") != PHASE0_SPLIT_SEED:
        raise ValueError("split seed does not match the frozen protocol")
    if (
        _required_string(config, "expected_split_strategy_version")
        != SPLIT_STRATEGY_VERSION
    ):
        raise ValueError("split strategy does not match the frozen protocol")
    if validation.split_mapping_sha256 is None:
        raise ValueError("trainval manifest must provide split_mapping_sha256")
    if normalize_action(_required_string(config, "fallback_action")) != KEEP:
        raise ValueError("Phase 0.2b fallback_action must be keep")

    candidates = build_rule_candidates(_candidate_grid(config))
    if len(candidates) != 625:
        raise ValueError("Phase 0.2b candidate grid must contain 625 candidates")
    majority_samples, validation_samples = _load_samples(manifest_path)
    evaluations = tuple(
        evaluate_rule_candidate(validation_samples, candidate_id, thresholds)
        for candidate_id, thresholds in candidates
    )
    selected = select_best_rule_candidate(evaluations)
    rule_version = _required_string(config, "version")
    records = build_prediction_records(validation_samples, selected, rule_version)
    decision_reasons = Counter(record.decision_reason for record in records)
    motion_availability = Counter(
        record.motion_availability for record in records
    )

    majority_action = fit_majority_action(majority_samples)
    _, majority_metrics = predict_split(
        samples=majority_samples,
        split="validation",
        majority_action=majority_action,
        label_rule_version=expected_label_rule,
    )
    selection_objective = build_selection_objective(evaluations, selected)
    selected_rule = build_selected_rule_payload(
        rule_version=rule_version,
        selected=selected,
        selection_objective=selection_objective,
        manifest_sha256=manifest_sha256,
        manifest_schema_version=validation.manifest_schema_version,
        label_rule_version=validation.label_rule_version,
        split_mapping_sha256=validation.split_mapping_sha256,
        decision_reason_distribution=decision_reasons,
        motion_availability_distribution=motion_availability,
    )
    comparison = _comparison_payload(selected, majority_action, majority_metrics)

    output_dir = derived_root / output_relative_dir
    write_canonical_json(
        {
            "rule_version": rule_version,
            "candidate_count": len(evaluations),
            "candidates": [_candidate_payload(item) for item in evaluations],
            "test_evaluation_performed": False,
        },
        output_dir / "candidate_leaderboard.json",
    )
    write_canonical_json(selected_rule, output_dir / "selected_rule.json")
    write_jsonl_records(records, output_dir / "validation_predictions.jsonl")
    validation_metrics = build_validation_metrics_artifact_payload(
        rule_version,
        selected,
        decision_reasons,
    )
    write_canonical_json(
        validation_metrics,
        output_dir / "validation_metrics.json",
    )
    write_canonical_json(
        {
            "baseline_name": "majority",
            "majority_action": majority_action,
            "metrics": _metrics_payload(majority_metrics),
            "test_evaluation_performed": False,
        },
        output_dir / "majority_validation_metrics.json",
    )
    write_canonical_json(comparison, output_dir / "comparison.json")
    result = {
        "output_dir": output_dir.as_posix(),
        "manifest_sha256": manifest_sha256,
        "candidate_count": len(evaluations),
        "selected_rule": selected_rule,
        "comparison": comparison,
        "output_sha256": _output_sha256(output_dir),
    }
    return output_dir, result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 0.2b ego-motion rule baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    derived_root_value = os.environ.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        raise ValueError("VLA_DERIVED_ROOT is not set")
    _, result = run_baseline(arguments.config, Path(derived_root_value))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

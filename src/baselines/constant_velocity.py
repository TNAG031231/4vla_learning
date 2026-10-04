from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
import yaml

from src.phase0.development_projection import IsolationCounters
from src.phase0.manifest import COORDINATE_METADATA
from src.phase0.phase0_4_temporal_dataset import SCHEMA_VERSION
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4c_evaluation import aggregate_metrics, prediction_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance, resolve_derived_path, validate_git_provenance,
)

TRACK = "constant_velocity"


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    expected = {
        "protocol_version": "phase0.4c-constant-velocity-v0.1",
        "waypoint_times_sec": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "speed_source": "ego_motion_history[-1].speed_mps",
        "speed_semantics": "latest_available_past_interval_average_speed_magnitude",
        "direction_assumption": "current_ego_frame_positive_x",
        "motion_model": "straight_line_constant_velocity",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"{key} differs from frozen CV protocol")
    return config


def predict(history: list[dict | None], history_mask: list[bool],
            times: list[float]) -> tuple[torch.Tensor | None, str | None]:
    if not history or len(history) < len(history_mask) or history[-1] is None:
        return None, "missing_current_history_entry"
    if not history_mask or len(history_mask) != len(history) or history_mask[-1] is not True:
        return None, "invalid_current_history_mask"
    speed = history[-1].get("speed_mps")
    if speed is None:
        return None, "missing_current_speed"
    if type(speed) not in (int, float):
        raise ValueError("speed_mps must be numeric or null")
    if not math.isfinite(speed):
        return None, "nonfinite_current_speed"
    return torch.tensor([[speed * t, 0.0] for t in times], dtype=torch.float32), None


def evaluate(records: list[dict], times: list[float]) -> tuple[dict, list[dict]]:
    if any(row["split"] != "validation" for row in records):
        raise ValueError("CV evaluation requires validation only")
    rows = []
    for record in records:
        prediction, reason = predict(record["ego_motion_history"], record["history_valid_mask"], times)
        target = torch.tensor(record["future_waypoints"], dtype=torch.float32)
        mask = torch.tensor(record["trajectory_valid_mask"], dtype=torch.bool)
        metrics = prediction_metrics(prediction, target, mask)
        if reason is not None:
            metrics["invalid_reason"] = reason
        rows.append({
            "sample_token": record["sample_token"], "scene_token": record["scene_token"],
            "split": "validation", "conditioning_type": TRACK,
            "target_waypoints": target.tolist(), "trajectory_valid_mask": mask.tolist(), **metrics,
        })
    return aggregate_metrics(rows, TRACK), rows


def prepare_data(repository: Path, derived_root: Path) -> tuple[list[dict], dict]:
    records = load_temporal_records(repository, derived_root, split="validation")
    for row in records:
        if (row["temporal_dataset_schema_version"] != SCHEMA_VERSION
                or row["history_order"] != "oldest_to_current"
                or row["history_policy"] != "null_padding"
                or row["historical_sample_tokens"][-1] != row["sample_token"]
                or row["historical_relative_times_sec"][-1] != 0
                or row["coordinate_metadata"]["future_ego_trajectory"]
                != COORDINATE_METADATA["future_ego_trajectory"]):
            raise ValueError("CV temporal coordinate/history contract mismatch")
        current = row["ego_motion_history"][-1] if row["ego_motion_history"] else None
        if current is not None and current["timestamp_source"] != "CAM_FRONT_sample_data":
            raise ValueError("CV current motion timestamp source mismatch")
        # The frozen v0.1 planner intake requires all six targets; motion availability
        # remains a prediction outcome so missing speed never removes a denominator.
        if len(row["trajectory_valid_mask"]) != 6 or any(v is not True for v in row["trajectory_valid_mask"]):
            raise ValueError("frozen temporal v0.1 requires six valid future points")
    return records, {
        "validation": {"total_records": len(records), "eligible_records": len(records),
                       "excluded_records": 0, "exclusion_reason_counts": {}},
        "temporal_dataset_schema_version": SCHEMA_VERSION,
        "source_provenance": {key: sorted({r[key] for r in records})
                              for key in ("source_combined_manifest_sha256", "split_mapping_sha256")},
        "train_records_used_for_fitting": 0, "test_records_read": 0,
        **asdict(IsolationCounters()), "test_evaluation_performed": False,
        "images_opened": 0, "model_loads": 0, "validation_parameter_selection_performed": False,
    }


def compare_planner(rows: list[dict], result_dir: Path) -> dict:
    path = result_dir / "validation_predicted_action_predictions.jsonl"
    reference = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "validation" or row["conditioning_type"] != "predicted_action":
                raise ValueError("comparison requires formal predicted-action validation")
            token = row["sample_token"]
            if token in reference:
                raise ValueError("duplicate planner sample_token")
            reference[token] = row
    common_cv, common_planner = [], []
    matched = 0
    for row in rows:
        token = row["sample_token"]
        if token not in reference:
            continue
        other = reference[token]
        matched += 1
        if any(row[key] != other[key] for key in ("scene_token", "target_waypoints", "trajectory_valid_mask")):
            raise ValueError("paired validation target/mask/scene mismatch")
        if type(other["prediction_valid"]) is not bool:
            raise ValueError("planner prediction_valid must be boolean")
        if not other["prediction_valid"]:
            continue
        recomputed = prediction_metrics(
            torch.tensor(other["predicted_waypoints"], dtype=torch.float32),
            torch.tensor(row["target_waypoints"], dtype=torch.float32),
            torch.tensor(row["trajectory_valid_mask"], dtype=torch.bool))
        if not recomputed["prediction_valid"]:
            raise ValueError("planner validity contradicts saved trajectory")
        if row["prediction_valid"]:
            common_cv.append(row)
            common_planner.append({"conditioning_type": "predicted_action", **recomputed})
    cv = aggregate_metrics(common_cv, TRACK)
    planner = aggregate_metrics(common_planner, "predicted_action")
    deltas = {}
    for key in ("ade_1s_m", "ade_2s_m", "ade_3s_m", "fde_1s_m", "fde_2s_m", "fde_3s_m"):
        a, b = cv[key], planner[key]
        delta = a - b if a is not None and b is not None else None
        deltas[key] = {"cv": a, "planner": b, "absolute_delta_m": delta,
                       "relative_delta": delta / b if delta is not None and b != 0 else None,
                       "sample_count": cv[f"{key}_sample_count"]}
    return {
        "reference_filename": path.name, "definition": "cv_minus_planner_on_jointly_valid_matching_tokens",
        "relative_delta_definition": "(cv-planner)/planner; null_when_planner_zero_or_no_pairs",
        "cv_sample_count": len(rows), "planner_sample_count": len(reference),
        "matching_sample_count": matched, "cv_unmatched_count": len(rows) - matched,
        "planner_unmatched_count": len(reference) - matched,
        "jointly_valid_sample_count": len(common_cv),
        "paired_sample_tokens": [r["sample_token"] for r in common_cv], "metrics": deltas,
    }


def run(*, repository: Path, derived_root: Path, config: dict,
        git_provenance: GitProvenance, split: str = "validation",
        planner_result_dir: Path | None = None) -> dict:
    if split != "validation":
        raise ValueError("CV execution permits validation only")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config["output_relative_dir"])
    if output.is_relative_to(repository.resolve()):
        raise ValueError("CV artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"CV output already exists: {output}")
    records, summary = prepare_data(repository, derived_root)
    metrics, rows = evaluate(records, config["waypoint_times_sec"])
    summary.update({key: metrics[key] for key in (
        "sample_count", "valid_prediction_count", "invalid_prediction_count", "trajectory_valid_rate",
        "invalid_prediction_rate", "invalid_reason_counts")})
    comparison = compare_planner(rows, planner_result_dir) if planner_result_dir is not None else None
    metadata = {
        **config, "execution_git_commit": git.commit, "split": split,
        "coordinates": "current_ego_frame_x_forward_y_left_meters", "time_unit": "seconds",
        "trainable_parameters": 0, "data": summary,
        "comparison_performed": comparison is not None,
    }
    output.mkdir(parents=True)
    write_predictions(output / "predictions.jsonl", rows)
    for name, payload in (("metrics", metrics), ("data_summary", summary),
                          ("run_metadata", metadata), ("resolved_config", config)):
        write_json(output / f"{name}.json", payload)
    if comparison is not None:
        write_json(output / "comparison_to_action_conditioned_planner.json", comparison)
    return {"status": "validation_completed", "metrics": metrics, "comparison": comparison}

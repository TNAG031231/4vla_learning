from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
import yaml

from src.baselines.ego_history_mlp import METRICS, index_predictions, read_predictions
from src.phase0.development_projection import IsolationCounters
from src.phase0.phase0_4_factorized_targets import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from src.phase0.phase0_4_temporal_dataset import validate_record
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import ActionTarget, parse_action
from src.phase0.phase0_4c_evaluation import aggregate_metrics, prediction_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance, resolve_derived_path, validate_git_provenance,
)

TRACKS = {"mlp": "ego_history_mlp", "planner": "predicted_action",
          "direct": "direct_qwen_waypoint", "cv": "constant_velocity"}
SOURCES = {
    "mlp": ("phase0_4c_ego_history_mlp.yaml", "validation_predictions.jsonl"),
    "planner": ("phase0_4c_full_train.yaml", "validation_predicted_action_predictions.jsonl"),
    "direct": ("phase0_4c_direct_waypoint.yaml", "validation_predictions.jsonl"),
    "cv": ("phase0_4c_constant_velocity.yaml", "predictions.jsonl"),
}
PAIRS = (("planner", "mlp"), ("direct", "mlp"), ("planner", "direct"),
         ("mlp", "cv"), ("planner", "cv"), ("direct", "cv"))
ACTION_GROUPS = ("joint_action_correct", "longitudinal_only_correct",
                 "lateral_only_correct", "both_wrong", "invalid_gt_action")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config["protocol_version"] != "phase0.4c-trajectory-stratification-v0.1":
        raise ValueError("stratification protocol mismatch")
    if type(config["top_k"]) is not int or config["top_k"] < 1:
        raise ValueError("top_k must be a positive integer")
    return config


def recompute(row: dict) -> dict:
    if type(row["prediction_valid"]) is not bool:
        raise ValueError("prediction_valid must be boolean")
    prediction = row["predicted_waypoints"]
    result = prediction_metrics(
        None if prediction is None else torch.tensor(prediction, dtype=torch.float32),
        torch.tensor(row["target_waypoints"], dtype=torch.float32),
        torch.tensor(row["trajectory_valid_mask"], dtype=torch.bool))
    if result["prediction_valid"] != row["prediction_valid"]:
        raise ValueError("saved validity contradicts trajectory")
    result["invalid_reason"] = row["invalid_reason"]
    result.pop("predicted_waypoints")
    return {"conditioning_type": row["conditioning_type"], **result}


def summarize(samples: list[dict]) -> dict:
    models = {name: aggregate_metrics([r["models"][name] for r in samples], track)
              for name, track in TRACKS.items()}
    pairs = {}
    for left, right in PAIRS:
        direction = f"{left}_minus_{right}"
        metrics = {}
        for key in METRICS[2:]:
            values = [r[direction][key] for r in samples if r[direction][key] is not None]
            left_wins = sum(v < 0 for v in values)
            metrics[key] = {"sample_count": len(values),
                            f"mean_{direction}": sum(values) / len(values) if values else None,
                            f"{left}_win_count": left_wins,
                            f"{right}_win_count": sum(v > 0 for v in values),
                            "tie_count": sum(v == 0 for v in values),
                            f"{left}_win_rate": left_wins / len(values) if values else None}
        pairs[direction] = metrics
    return {"sample_count": len(samples), "models": models, "pairwise": pairs}


def analyze(records: list[dict], predictions: dict[str, list[dict]], config: dict) -> tuple[dict, list[dict]]:
    if any(r["split"] != "validation" for r in records):
        raise ValueError("stratification requires validation only")
    temporal = {}
    for record in records:
        token = record["sample_token"]
        if token in temporal:
            raise ValueError("duplicate temporal sample_token")
        validate_record(record)
        temporal[token] = record
    indexed = {name: index_predictions(predictions[name], track) for name, track in TRACKS.items()}
    expected = config["expected_sample_count"]
    if len(temporal) != expected:
        raise ValueError(f"validation population mismatch: actual={len(temporal)}, expected={expected}")
    for name, rows in indexed.items():
        if rows.keys() != temporal.keys():
            raise ValueError(f"{name} exact sample-token alignment failed: "
                             f"missing={sorted(temporal.keys() - rows.keys())}, "
                             f"extra={sorted(rows.keys() - temporal.keys())}")
    action_present = ["action_context" in row for row in indexed["planner"].values()]
    if any(action_present) and not all(action_present):
        raise ValueError("incomplete planner action_context coverage")
    actions_available = all(action_present)
    samples = []
    for token in sorted(temporal):
        record = temporal[token]
        target = torch.tensor(record["future_waypoints"], dtype=torch.float32).tolist()
        models = {}
        for name, rows in indexed.items():
            row = rows[token]
            if (row["scene_token"] != record["scene_token"] or row["target_waypoints"] != target
                    or any(type(v) is not bool for v in row["trajectory_valid_mask"])
                    or row["trajectory_valid_mask"] != record["trajectory_valid_mask"]):
                raise ValueError(f"{name} scene/target/mask mismatch: {token}")
            models[name] = recompute(row)
            if name != "cv" and not models[name]["prediction_valid"]:
                raise ValueError(f"{name} primary prediction invalid: {token}")
        motion = record["ego_motion_history"][-1]
        speed = motion["speed_mps"]
        speed_available = speed is not None and math.isfinite(speed)
        if models["cv"]["prediction_valid"] != speed_available:
            raise ValueError(f"CV validity differs from current-speed availability: {token}")
        if not speed_available and models["cv"]["invalid_reason"] != "missing_current_speed":
            raise ValueError(f"CV invalid reason differs from confirmed missing-speed population: {token}")
        action = ActionTarget(record["longitudinal_action"], record["lateral_action"],
                              record["longitudinal_action_valid"], record["lateral_action_valid"])
        if record["factorized_action_joint_valid"] != (action.longitudinal_valid and action.lateral_valid):
            raise ValueError("joint action validity mismatch")
        strata = {"longitudinal": action.longitudinal if action.longitudinal_valid else "invalid",
                  "lateral": action.lateral if action.lateral_valid else "invalid",
                  "motion_availability": motion["availability"],
                  "valid_history_count": sum(record["history_valid_mask"]),
                  "cv_valid": models["cv"]["prediction_valid"]}
        sample = {"sample_token": token, "scene_token": record["scene_token"], "split": "validation",
                  "gt_longitudinal_action": action.longitudinal, "gt_lateral_action": action.lateral,
                  "cam_front_path": record["historical_cam_front_paths"][-1],
                  "strata": strata, "models": models}
        if actions_available:
            predicted = parse_action(indexed["planner"][token]["action_context"])
            if predicted is None:
                raise ValueError(f"valid planner prediction has invalid action_context: {token}")
            sample["predicted_action"] = predicted
            if not record["factorized_action_joint_valid"]:
                strata["action_correctness"] = "invalid_gt_action"
            else:
                correct = (predicted["longitudinal"] == action.longitudinal,
                           predicted["lateral"] == action.lateral)
                strata["action_correctness"] = {
                    (True, True): ACTION_GROUPS[0], (True, False): ACTION_GROUPS[1],
                    (False, True): ACTION_GROUPS[2], (False, False): ACTION_GROUPS[3],
                }[correct]
        for left, right in PAIRS:
            a, b = models[left], models[right]
            sample[f"{left}_minus_{right}"] = {
                key: a[key] - b[key] if a["prediction_valid"] and b["prediction_valid"]
                and a[key] is not None and b[key] is not None else None for key in METRICS[2:]}
        samples.append(sample)
    overall = summarize(samples)
    cv = overall["models"]["cv"]
    actual = (cv["valid_prediction_count"], cv["invalid_prediction_count"])
    expected_cv = (config["expected_cv_valid_count"], config["expected_cv_invalid_count"])
    if actual != expected_cv:
        raise ValueError(f"CV coverage mismatch: actual={actual}, expected={expected_cv}")
    for name in ("mlp", "planner", "direct"):
        for key in METRICS[2:]:
            actual = overall["models"][name][key]
            expected_metric = config["expected_metrics"][name][key]
            if actual is None or not math.isclose(actual, expected_metric,
                                                 abs_tol=config["metric_atol"], rel_tol=config["metric_rtol"]):
                raise ValueError(f"aggregate reproduction failed: {name}.{key}: "
                                 f"actual={actual}, expected={expected_metric}")
    groups = {"longitudinal": (*LONGITUDINAL_ACTIONS, "invalid"),
              "lateral": (*LATERAL_ACTIONS, "invalid"),
              "motion_availability": ("full", "partial", "unavailable"),
              "valid_history_count": tuple(range(max(r["history_length"] for r in records) + 1)),
              "cv_valid": (True, False)}
    if actions_available:
        groups["action_correctness"] = ACTION_GROUPS
    stratified = {}
    for axis, values in groups.items():
        summaries = [{"value": value, **summarize([s for s in samples if s["strata"][axis] == value])}
                     for value in values]
        if sum(group["sample_count"] for group in summaries) != len(samples):
            raise ValueError(f"stratum count conservation failed: {axis}")
        stratified[axis] = summaries
    ranked = sorted(samples, key=lambda s: (s["planner_minus_mlp"]["ade_3s_m"], s["sample_token"]))
    return {
        "overall": overall, "stratification": stratified,
        "action_correctness_status": "available" if actions_available else "not available from existing artifacts",
        "ranking": {
            "planner_strongest_relative_wins": [
                s for s in ranked if s["planner_minus_mlp"]["ade_3s_m"] < 0][:config["top_k"]],
            "mlp_strongest_relative_wins": [
                s for s in reversed(ranked) if s["planner_minus_mlp"]["ade_3s_m"] > 0][:config["top_k"]],
        },
        "aggregate_reproduction": {"passed": True, "atol": config["metric_atol"], "rtol": config["metric_rtol"]},
        "semantics": {"motion_availability": "ego_motion_history[-1].availability",
                      "valid_history_count": "sum(history_valid_mask); includes current observation",
                      "pairwise": "left_minus_right on jointly valid samples; negative means left wins",
                      "ties": "exact zero per-sample metric difference; win rate includes ties in denominator",
                      "ranking": "strict wins only, ordered by planner_minus_mlp ADE@3s; up to top_k",
                      "invalid_gt_action": "excluded from four correctness groups; retained separately"},
    }, samples


def run(*, repository: Path, derived_root: Path, config: dict,
        git_provenance: GitProvenance, split: str = "validation") -> dict:
    if split != "validation":
        raise ValueError("stratification requires validation only")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config["output_relative_dir"])
    if output.is_relative_to(repository.resolve()):
        raise ValueError("analysis artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    records = load_temporal_records(repository, derived_root, split="validation")
    predictions, sources = {}, {}
    for name, (config_file, filename) in SOURCES.items():
        producer = yaml.safe_load((repository / "configs" / config_file).read_text())
        directory = resolve_derived_path(derived_root, producer["output_relative_dir"])
        path = directory / filename
        predictions[name] = read_predictions(path, TRACKS[name])
        sources[name] = {"prediction_relative_path": path.relative_to(derived_root.resolve()).as_posix(),
                         "producer_config": config_file,
                         "run_metadata": json.loads((directory / "run_metadata.json").read_text())}
    summary, samples = analyze(records, predictions, config)
    summary.update(
        protocol_version=config["protocol_version"], split=split, config=config,
        artifact_provenance={"execution_git_commit": git.commit, "sources": sources,
                             "temporal_source_provenance": {k: sorted({r[k] for r in records}) for k in
                                 ("source_combined_manifest_sha256", "split_mapping_sha256",
                                  "temporal_dataset_schema_version", "factorized_action_rule_version")}},
        test_isolation={**asdict(IsolationCounters()), "test_records_read": 0,
                        "test_evaluation_performed": False},
        execution={"model_loads": 0, "model_forward_calls": 0, "training_steps": 0, "images_opened": 0})
    output.mkdir(parents=True)
    write_predictions(output / "sample_analysis.jsonl", samples)
    write_json(output / "summary.json", summary)
    return summary

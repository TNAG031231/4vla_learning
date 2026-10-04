from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import yaml

from src.phase0.development_projection import IsolationCounters
from src.phase0.phase0_4_temporal_dataset import validate_record
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4c_evaluation import aggregate_metrics, prediction_metrics
from src.phase0.phase0_4c_full_train import write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.phase0_4c_two_turn_planner import masked_waypoint_loss
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance, resolve_derived_path, validate_git_provenance,
)

TRACK = "ego_history_mlp"
FIELDS = ("speed_mps", "longitudinal_acceleration_mps2", "yaw_rate_radps")
METRICS = ("ade_m", "fde_m", "ade_1s_m", "ade_2s_m", "ade_3s_m",
           "fde_1s_m", "fde_2s_m", "fde_3s_m")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    frozen = {
        "protocol_version": "phase0.4c-ego-history-mlp-v0.1", "history_length": 3,
        "hidden_dim": 128, "seed": 20260812, "batch_size": 256, "optimizer": "AdamW",
        "learning_rate": .001, "weight_decay": .0001, "smooth_l1_beta": 1., "max_epochs": 20,
        "normalization_epsilon": 1e-6, "device": "cpu", "num_threads": 1,
        "reload_atol": 1e-6, "reload_rtol": 1e-6,
    }
    for key, value in frozen.items():
        if config[key] != value:
            raise ValueError(f"{key} differs from frozen MLP protocol")
    return config


def motion_values(history: list[dict | None], mask: list[bool]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(len(history), 3, dtype=torch.float64)
    available = torch.zeros(len(history), 3, dtype=torch.bool)
    for i, (motion, valid) in enumerate(zip(history, mask, strict=True)):
        if not valid:
            continue
        for j, field in enumerate(FIELDS):
            value = motion.get(field)
            if value is None:
                continue
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite numeric or null")
            values[i, j], available[i, j] = value, True
    return values, available


def normalization_stats(records: list[dict], epsilon: float) -> dict:
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("normalization requires TRAIN only")
    pairs = [motion_values(r["ego_motion_history"], r["history_valid_mask"]) for r in records]
    values, available = (torch.cat([p[i] for p in pairs]) for i in (0, 1))
    channels = {}
    for j, field in enumerate(FIELDS):
        observed = values[:, j][available[:, j]]
        if not len(observed):
            raise ValueError(f"no observed TRAIN values for {field}")
        mean, std = float(observed.mean()), float(observed.std(correction=0))
        channels[field] = {"count": len(observed), "mean": mean, "std": std, "scale": max(std, epsilon)}
    return {"fit_split": "train", "train_sample_count": len(records),
            "train_sample_tokens": [r["sample_token"] for r in records],
            "validation_records_used": 0, "test_records_used": 0,
            "reduction": "population_std_float64_over_available_history_occurrences",
            "epsilon": epsilon, "channels": channels}


def encode(history: list[dict | None], mask: list[bool], stats: dict) -> torch.Tensor:
    values, available = motion_values(history, mask)
    features = torch.zeros(len(history), 7, dtype=torch.float32)
    for j, field in enumerate(FIELDS):
        channel = stats["channels"][field]
        features[:, 2*j] = torch.where(
            available[:, j], (values[:, j] - channel["mean"]) / channel["scale"], 0).float()
        features[:, 2*j + 1] = available[:, j].float()
    features[:, 6] = torch.tensor(mask, dtype=torch.float32)
    return features.flatten()


class EgoHistoryMLP(nn.Module):
    def __init__(self, history_length: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7 * history_length, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 12))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).reshape(-1, 6, 2)


def prepare_data(repository: Path, derived_root: Path, history_length: int,
                 train_split: str, validation_split: str) -> tuple[list[dict], list[dict], dict]:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("MLP only permits train fitting and validation evaluation")
    splits = {}
    for split in (train_split, validation_split):
        rows = load_temporal_records(repository, derived_root, split=split)
        for row in rows:
            if row["split"] != split or row["history_length"] != history_length:
                raise ValueError("MLP split/history length mismatch")
            validate_record(row)
            motion_values(row["ego_motion_history"], row["history_valid_mask"])
        splits[split] = rows
    train, validation = splits["train"], splits["validation"]
    for key in ("sample_token", "scene_token"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError("train/validation overlap")
    return train, validation, {
        "sample_counts": {s: len(rows) for s, rows in splits.items()},
        "source_provenance": {s: {k: sorted({r[k] for r in rows}) for k in
                                 ("source_combined_manifest_sha256", "split_mapping_sha256")}
                              for s, rows in splits.items()},
        "train_usage": "model_fitting_and_normalization",
        "validation_usage": "checkpoint_selection_and_final_evaluation",
        **asdict(IsolationCounters()), "test_records_read": 0, "test_evaluation_performed": False,
        "images_opened": 0, "qwen_model_loads": 0, "lora_model_loads": 0,
        "structured_action_inputs": 0, "future_information_inputs": 0,
    }


def evaluate(model: EgoHistoryMLP, records: list[dict], stats: dict, batch_size: int) -> tuple[dict, list[dict]]:
    if any(r["split"] != "validation" for r in records):
        raise ValueError("MLP evaluation requires validation")
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            features = torch.stack([encode(r["ego_motion_history"], r["history_valid_mask"], stats) for r in batch])
            predictions = model(features)
            for record, prediction in zip(batch, predictions, strict=True):
                target = torch.tensor(record["future_waypoints"], dtype=torch.float32)
                mask = torch.tensor(record["trajectory_valid_mask"], dtype=torch.bool)
                rows.append({"sample_token": record["sample_token"], "scene_token": record["scene_token"],
                             "split": "validation", "conditioning_type": TRACK,
                             "target_waypoints": target.tolist(), "trajectory_valid_mask": mask.tolist(),
                             **prediction_metrics(prediction, target, mask)})
    return aggregate_metrics(rows, TRACK), rows


def selection_key(metrics: dict, epoch: int) -> tuple[float, float, int]:
    if metrics["invalid_prediction_count"] or not metrics["valid_prediction_count"]:
        raise ValueError("MLP validation must have finite predictions before selection")
    return metrics["ade_m"], metrics["fde_m"], epoch


def read_predictions(path: Path, track: str) -> list[dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    index_predictions(rows, track)
    return rows


def index_predictions(rows: list[dict], track: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if row["split"] != "validation" or row["conditioning_type"] != track:
            raise ValueError("comparison track/split mismatch")
        token = row["sample_token"]
        if token in indexed:
            raise ValueError("duplicate comparison sample_token")
        indexed[token] = row
    return indexed


def paired_comparison(rows: list[dict], reference: list[dict], reference_track: str) -> dict:
    left, right = index_predictions(rows, TRACK), index_predictions(reference, reference_track)
    common = sorted(left.keys() & right.keys())
    pairs = [], []
    tokens = []
    for token in common:
        a, b = left[token], right[token]
        if any(a[k] != b[k] for k in ("scene_token", "target_waypoints", "trajectory_valid_mask")):
            raise ValueError("paired scene/target/mask mismatch")
        recomputed = []
        for row in (a, b):
            if type(row["prediction_valid"]) is not bool:
                raise ValueError("prediction_valid must be boolean")
            if row["prediction_valid"]:
                result = prediction_metrics(torch.tensor(row["predicted_waypoints"], dtype=torch.float32),
                                            torch.tensor(row["target_waypoints"], dtype=torch.float32),
                                            torch.tensor(row["trajectory_valid_mask"], dtype=torch.bool))
                if not result["prediction_valid"]:
                    raise ValueError("saved validity contradicts trajectory")
                recomputed.append({"conditioning_type": row["conditioning_type"], **result})
        if len(recomputed) == 2:
            for group, result in zip(pairs, recomputed, strict=True):
                group.append(result)
            tokens.append(token)
    metrics = [aggregate_metrics(pairs[0], TRACK), aggregate_metrics(pairs[1], reference_track)]
    deltas = {}
    for key in METRICS[2:]:
        a, b = metrics[0][key], metrics[1][key]
        delta = a - b if a is not None and b is not None else None
        deltas[key] = {"mlp": a, "reference": b, "absolute_delta_m": delta,
                       "relative_delta": delta / b if delta is not None and b != 0 else None,
                       "sample_count": metrics[0][f"{key}_sample_count"]}
    return {"reference_track": reference_track, "delta_definition": "mlp_minus_reference; negative_is_better",
            "relative_delta_definition": "(mlp-reference)/reference; null_if_zero_or_unavailable",
            "mlp_sample_count": len(left), "reference_sample_count": len(right),
            "matching_sample_count": len(common), "jointly_valid_sample_count": len(tokens),
            "paired_sample_tokens": tokens, "mlp_unmatched_tokens": sorted(left.keys() - right.keys()),
            "reference_unmatched_tokens": sorted(right.keys() - left.keys()), "metrics": deltas}


def reload_consistency(before: list[dict], after: list[dict], atol: float, rtol: float) -> dict:
    matched = [r["sample_token"] for r in before] == [r["sample_token"] for r in after]
    matched &= all(a["prediction_valid"] and b["prediction_valid"] and torch.allclose(
        torch.tensor(a["predicted_waypoints"]), torch.tensor(b["predicted_waypoints"]), atol=atol, rtol=rtol)
        for a, b in zip(before, after))
    first, second = aggregate_metrics(before, TRACK), aggregate_metrics(after, TRACK)
    metrics_match = all(first[k] is not None and second[k] is not None
                        and math.isclose(first[k], second[k], abs_tol=atol, rel_tol=rtol) for k in METRICS)
    return {"matched": bool(matched and metrics_match), "predictions_match": bool(matched),
            "metrics_match": metrics_match, "sample_count": len(before), "atol": atol, "rtol": rtol}


def fit(train: list[dict], validation: list[dict], config: dict, output: Path) -> tuple[dict, list[dict]]:
    if not train or any(r["split"] != "train" for r in train):
        raise ValueError("fitting requires train only")
    if not validation or any(r["split"] != "validation" for r in validation):
        raise ValueError("checkpoint selection requires validation only")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    stats = normalization_stats(train, config["normalization_epsilon"])
    write_json(output / "normalization_stats.json", stats)
    validation_stats = json.loads((output / "normalization_stats.json").read_text())
    model = EgoHistoryMLP(config["history_length"], config["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    features = torch.stack([encode(r["ego_motion_history"], r["history_valid_mask"], stats) for r in train])
    targets = torch.tensor([r["future_waypoints"] for r in train], dtype=torch.float32)
    masks = torch.tensor([r["trajectory_valid_mask"] for r in train], dtype=torch.bool)
    generator = torch.Generator().manual_seed(config["seed"])
    best_key, best, best_rows = None, None, None
    steps = 0
    with (output / "training_history.jsonl").open("w") as history:
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            order = torch.randperm(len(train), generator=generator)
            loss_sum = 0.
            for indices in order.split(config["batch_size"]):
                optimizer.zero_grad()
                loss = masked_waypoint_loss(model(features[indices]), targets[indices], masks[indices],
                                            beta=config["smooth_l1_beta"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite MLP training loss")
                loss.backward()
                optimizer.step()
                steps += 1
                loss_sum += float(loss.detach()) * len(indices)
            metrics, rows = evaluate(model, validation, validation_stats, config["batch_size"])
            key = selection_key(metrics, epoch)
            if best_key is None or key < best_key:
                best_key, best_rows = key, rows
                best = {"epoch": epoch, "checkpoint": "best_model.pt", "metrics": metrics,
                        "selection_policy": "lowest_validation_ade_then_fde_then_earlier_epoch"}
                torch.save({"state_dict": model.state_dict(), "history_length": config["history_length"],
                            "hidden_dim": config["hidden_dim"], "epoch": epoch}, output / "best_model.pt")
                write_json(output / "best_checkpoint.json", best)
            entry = {"epoch": epoch, "optimizer_steps": steps, "train_loss": loss_sum / len(train),
                     "validation_metrics": metrics}
            history.write(json.dumps(entry, allow_nan=False) + "\n")
            history.flush()
            print(json.dumps(entry, allow_nan=False), flush=True)
    saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    reloaded = EgoHistoryMLP(saved["history_length"], saved["hidden_dim"])
    reloaded.load_state_dict(saved["state_dict"])
    reloaded_stats = json.loads((output / "normalization_stats.json").read_text())
    metrics, rows = evaluate(reloaded, validation, reloaded_stats, config["batch_size"])
    consistency = reload_consistency(best_rows, rows, config["reload_atol"], config["reload_rtol"])
    write_json(output / "reload_consistency.json", consistency)
    if not consistency["matched"]:
        raise ValueError("MLP checkpoint reload mismatch")
    write_predictions(output / "validation_predictions.jsonl", rows)
    write_json(output / "validation_metrics.json", metrics)
    return {"status": "training_completed", "epochs": config["max_epochs"], "optimizer_steps": steps,
            "best_checkpoint": best, "validation_metrics": metrics, "reload_consistency": consistency}, rows


def run(*, repository: Path, derived_root: Path, config: dict, git_provenance: GitProvenance,
        train_split: str = "train", validation_split: str = "validation",
        cv_result_dir: Path | None = None, planner_result_dir: Path | None = None) -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("MLP only permits train fitting and validation evaluation")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config["output_relative_dir"])
    if output.is_relative_to(repository.resolve()):
        raise ValueError("MLP artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"MLP output already exists: {output}")
    cv_dir = cv_result_dir if cv_result_dir is not None else derived_root / "phase_0_4/constant_velocity_baseline_v0_1"
    planner_dir = (planner_result_dir if planner_result_dir is not None
                   else derived_root / "phase_0_4/two_turn_planner_full_v0_1")
    cv_rows = read_predictions(cv_dir / "predictions.jsonl", "constant_velocity")
    planner_rows = read_predictions(planner_dir / "validation_predicted_action_predictions.jsonl", "predicted_action")
    train, validation, data = prepare_data(repository, derived_root, config["history_length"], train_split, validation_split)
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "data_summary.json", data)
    write_json(output / "run_metadata.json", {
        "execution_git_commit": git.commit, "config": config, "data": data,
        "feature_order": [item for field in FIELDS for item in (field, f"{field}_available")] + ["history_valid_mask"],
        "coordinates": "current_ego_frame_x_forward_y_left_meters", "waypoint_times_sec": [.5, 1., 1.5, 2., 2.5, 3.],
        "device": "cpu", "deterministic_algorithms": True, "torch_version": str(torch.__version__),
        "normalization_policy": "train_only; population_std; scale=max(std,epsilon); missing_normalized_value=0",
    })
    result, rows = fit(train, validation, config, output)
    write_json(output / "comparison_to_constant_velocity.json", paired_comparison(rows, cv_rows, "constant_velocity"))
    write_json(output / "comparison_to_action_conditioned_planner.json", paired_comparison(rows, planner_rows, "predicted_action"))
    write_json(output / "training_summary.json", result)
    return result

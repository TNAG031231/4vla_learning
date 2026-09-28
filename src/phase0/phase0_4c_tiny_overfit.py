from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import yaml

from src.phase0.phase0_4b_lora_smoke import select_tiny
from src.phase0.phase0_4b_protocol import SFTSample, adapt_record, serialize_action
from src.phase0.phase0_4c_two_turn_planner import (
    PlannerConfig, WaypointDecoder, contextual_hidden_states, load_config as load_planner_config,
    masked_waypoint_loss, planning_inputs, planning_messages,
)
from src.phase0.qwen3vl_dataset_adapter import AdapterConfig
from src.phase0.qwen3vl_lora_smoke import RuntimeDependencies
from src.phase0.qwen3vl_smoke import resolve_image_path


@dataclass(frozen=True)
class TinyOverfitConfig(PlannerConfig):
    optimizer: str
    learning_rate: float
    weight_decay: float
    max_optimizer_steps: int
    log_every_steps: int


def load_config(path: Path) -> TinyOverfitConfig:
    values = yaml.safe_load(path.read_text())
    planner = load_planner_config(path.parent / values.pop("planner_config"))
    config = TinyOverfitConfig(**(asdict(planner) | values))
    if config.train_subset_size != 8 or config.optimizer != "AdamW":
        raise ValueError("tiny overfit requires eight train samples and AdamW")
    if (not 1 <= config.max_optimizer_steps <= 1500 or config.log_every_steps < 1
            or not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not math.isfinite(config.weight_decay) or config.weight_decay < 0):
        raise ValueError("invalid tiny-overfit optimizer settings")
    return config


def select_training_samples(records: list[dict], ego_config: AdapterConfig,
                            config: TinyOverfitConfig) -> list[SFTSample]:
    samples = [adapt_record(row, ego_config) for row in records]
    eligible = [sample for sample, row in zip(samples, records, strict=True)
                if row["factorized_action_joint_valid"] and any(row["trajectory_valid_mask"])]
    return select_tiny(eligible, config.train_subset_size, config.seed)


@dataclass(frozen=True)
class CachedContext:
    sample_token: str
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    target: torch.Tensor
    valid_mask: torch.Tensor


def cache_contexts(*, model: nn.Module, processor: object, samples: list[SFTSample],
                   records: dict[str, dict], dataset_root: Path, device: str,
                   runtime: RuntimeDependencies) -> tuple[list[CachedContext], list[dict]]:
    contexts, evidence = [], []
    for sample in samples:
        images = [runtime.image_loader(resolve_image_path(dataset_root, path))
                  for path in sample.observation.image_paths]
        action = serialize_action(sample.target.longitudinal, sample.target.lateral)
        inputs, token_evidence = planning_inputs(
            processor, planning_messages(sample.observation, images, action), device,
        )
        hidden = contextual_hidden_states(model, inputs)
        row = records[sample.sample_token]
        contexts.append(CachedContext(
            sample.sample_token, hidden.to(device="cpu", dtype=torch.bfloat16),
            inputs["attention_mask"].cpu(),
            torch.tensor([row["future_waypoints"]], dtype=torch.float32),
            torch.tensor([row["trajectory_valid_mask"]], dtype=torch.bool),
        ))
        evidence.append({
            "sample_token": sample.sample_token, "scene_token": sample.scene_token,
            "split": sample.split, "conditioning_type": "gt_action_teacher_forced_train",
            "assistant_action_text": action, "action_context_matches": token_evidence["action_context_matches"],
            "action_token_ids": token_evidence["action_token_ids"],
            "hidden_state_shape": list(hidden.shape), "cache_dtype": "torch.bfloat16",
            "cache_device": "cpu",
        })
        print(json.dumps({"event": "context_cached", **evidence[-1]}), flush=True)
        del inputs, hidden
    return contexts, evidence


def trajectory_metrics(prediction: torch.Tensor, target: torch.Tensor,
                       valid_mask: torch.Tensor) -> dict[str, float]:
    # One sample in raw current-ego-frame meters; the final valid point need not be index 5.
    distances = torch.linalg.vector_norm(prediction[valid_mask] - target[valid_mask], dim=-1)
    return {"ade_m": float(distances.mean()), "fde_m": float(distances[-1])}


def evaluate(planner: WaypointDecoder, contexts: list[CachedContext], *,
             beta: float, device: str) -> tuple[dict, list[dict]]:
    planner.eval()
    predictions = []
    with torch.no_grad():
        for context in contexts:
            prediction = planner(context.hidden_states.to(device), context.attention_mask.to(device)).cpu()
            loss = masked_waypoint_loss(prediction, context.target, context.valid_mask, beta=beta)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise ValueError("non-finite planner evaluation")
            predictions.append({
                "sample_token": context.sample_token, "predicted_waypoints": prediction[0].tolist(),
                "target_waypoints": context.target[0].tolist(),
                "trajectory_valid_mask": context.valid_mask[0].tolist(),
                "loss": float(loss), **trajectory_metrics(prediction, context.target, context.valid_mask),
            })
    metrics = {key: sum(row[key] for row in predictions) / len(predictions)
               for key in ("loss", "ade_m", "fde_m")}
    return {"sample_count": len(predictions), **metrics}, predictions


def train_planner(planner: WaypointDecoder, contexts: list[CachedContext], *,
                  config: TinyOverfitConfig, device: str, history_path: Path) -> None:
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    planner.train()
    started, total_loss = time.monotonic(), 0.0
    with history_path.open("w") as stream:
        for index in range(config.max_optimizer_steps):
            context = contexts[index % len(contexts)]
            optimizer.zero_grad(set_to_none=True)
            prediction = planner(context.hidden_states.to(device), context.attention_mask.to(device))
            loss = masked_waypoint_loss(prediction, context.target.to(device),
                                        context.valid_mask.to(device), beta=config.smooth_l1_beta)
            if not torch.isfinite(loss):
                raise ValueError("non-finite planner training loss")
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            total_loss += value
            entry = {"step": index + 1, "sample_token": context.sample_token, "loss": value,
                     "running_mean_loss": total_loss / (index + 1),
                     "elapsed_seconds": time.monotonic() - started}
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if (index + 1) % config.log_every_steps == 0:
                stream.flush()
                print(json.dumps(entry), flush=True)


def compare_reload(before: tuple[dict, list[dict]], after: tuple[dict, list[dict]], *,
                   atol: float = 1e-5, rtol: float = 1e-5) -> dict:
    metrics, predictions = before
    reloaded_metrics, reloaded = after
    tokens_match = [r["sample_token"] for r in predictions] == [r["sample_token"] for r in reloaded]
    predictions_match = tokens_match and all(
        torch.allclose(torch.tensor(a["predicted_waypoints"]), torch.tensor(b["predicted_waypoints"]),
                       atol=atol, rtol=rtol)
        for a, b in zip(predictions, reloaded, strict=True)
    )
    metrics_match = tokens_match and all(
        math.isclose(a[key], b[key], abs_tol=atol, rel_tol=rtol)
        for a, b in [(metrics, reloaded_metrics), *zip(predictions, reloaded, strict=True)]
        for key in ("loss", "ade_m", "fde_m")
    )
    return {"reload_consistency": tokens_match and predictions_match and metrics_match,
            "sample_tokens_identical": tokens_match, "predictions_match": predictions_match,
            "metrics_match": metrics_match, "atol": atol, "rtol": rtol,
            "reloaded_metrics": reloaded_metrics}


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def fit_cached_contexts(*, contexts: list[CachedContext], config: TinyOverfitConfig,
                        hidden_size: int, device: str, output: Path, provenance: dict) -> dict:
    torch.manual_seed(config.seed)
    planner = WaypointDecoder(hidden_size, config).to(device)
    parameter_count = sum(p.numel() for p in planner.parameters())
    initial_weights = {name: value.detach().cpu().clone() for name, value in planner.state_dict().items()}
    before = evaluate(planner, contexts, beta=config.smooth_l1_beta, device=device)
    write_json(output / "metrics_before.json", before[0])
    (output / "predictions_before.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in before[1]))
    train_planner(planner, contexts, config=config, device=device, history_path=output / "training_history.jsonl")
    after = evaluate(planner, contexts, beta=config.smooth_l1_beta, device=device)
    updated = any(not torch.equal(initial_weights[name], value.detach().cpu())
                  for name, value in planner.state_dict().items())
    write_json(output / "metrics_after.json", after[0])
    (output / "predictions_after.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in after[1]))
    checkpoint = {
        "planner_state_dict": {name: value.detach().cpu() for name, value in planner.state_dict().items()},
        "planner_config": {name: getattr(config, name) for name in PlannerConfig.__dataclass_fields__},
        "training_config": asdict(config), "hidden_size": hidden_size,
        "tiny_sample_tokens": [c.sample_token for c in contexts], "provenance": provenance,
    }
    torch.save(checkpoint, output / "planner_state.pt")
    del planner, checkpoint, initial_weights
    saved = torch.load(output / "planner_state.pt", map_location="cpu", weights_only=True)
    fresh = WaypointDecoder(saved["hidden_size"], PlannerConfig(**saved["planner_config"])).to(device)
    fresh.load_state_dict(saved["planner_state_dict"])
    reloaded = evaluate(fresh, contexts, beta=config.smooth_l1_beta, device=device)
    consistency = compare_reload(after, reloaded)
    write_json(output / "reload_consistency.json", consistency)
    initial, final = before[0], after[0]
    learning_passed = (final["loss"] < initial["loss"] and final["loss"] <= 0.30 * initial["loss"]
                       and final["ade_m"] < initial["ade_m"] and final["fde_m"] < initial["fde_m"])
    passed = learning_passed and updated and consistency["reload_consistency"]
    result = {
        "status": "tiny_overfit_passed" if passed else "tiny_overfit_learning_gate_failed",
        "learning_gate_passed": learning_passed, "planner_parameters_updated": updated,
        "planner_trainable_parameters": parameter_count, "reload_consistency": consistency["reload_consistency"],
        "initial_metrics": initial, "final_metrics": final, "optimizer_steps": config.max_optimizer_steps,
        "initial_loss": initial["loss"], "final_loss": final["loss"],
        "loss_ratio": final["loss"] / initial["loss"] if initial["loss"] > 0 else None,
        "initial_ADE": initial["ade_m"], "final_ADE": final["ade_m"],
        "initial_FDE": initial["fde_m"], "final_FDE": final["fde_m"],
        "train_sample_tokens": saved["tiny_sample_tokens"], "train_sample_count": len(contexts),
        "conditioning_type": "gt_action_teacher_forced_train", "generalization_evaluated": False,
        "coordinate_units": "meter", "metric_aggregation": "mean_per_sample_on_exact_tiny_train_subset",
        "checkpoint": "planner_state.pt", "full_model_saved": False, "hidden_state_cache_saved": False,
        **provenance,
    }
    write_json(output / "training_summary.json", result)
    return result

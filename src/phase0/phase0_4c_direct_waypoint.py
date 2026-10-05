from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
import yaml

from src.baselines.ego_history_mlp import index_predictions, read_predictions
from src.phase0.phase0_4b_lora_full import epoch_groups, load_config as load_semantic_config
from src.phase0.phase0_4b_protocol import Observation
from src.phase0.phase0_4c_evaluation import aggregate_metrics, prediction_metrics
from src.phase0.phase0_4c_full_train import (
    FullConfig, load_config as load_formal_config, prepare_data as prepare_formal_data,
    reload_planner, save_checkpoint, verify_freeze, write_predictions,
)
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.phase0_4c_two_turn_planner import (
    WaypointDecoder, contextual_hidden_states, freeze_backbone, masked_waypoint_loss,
)
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, resolve_derived_path, validate_git_provenance
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import RuntimeDependencies, _move_batch, default_runtime_dependencies
from src.phase0.qwen3vl_smoke import resolve_image_path

TRACK = "direct_qwen_waypoint"
PROMPT_VERSION = "phase0.4c-direct-waypoint-prompt-v0.1"
DIRECT_PROMPT = (
    "Based only on the provided past/current CAM_FRONT observations and ego states, "
    "prepare a representation for predicting the ego trajectory over the next 3.0 seconds "
    "at 0.5-second intervals. Observations are ordered oldest to current; "
    "unavailable history slots have no image."
)


@dataclass(frozen=True)
class DirectSample:
    sample_token: str
    scene_token: str
    split: str
    observation: Observation


def load_config(path: Path, repository: Path) -> FullConfig:
    config = FullConfig(train_subset_size=0, **yaml.safe_load(path.read_text()))
    expected = asdict(load_formal_config(repository / "configs/phase0_4c_full_train.yaml"))
    expected.update(protocol_version="phase0.4c-direct-waypoint-v0.1",
                    output_relative_dir="phase_0_4/direct_qwen_waypoint_v0_1")
    if asdict(config) != expected:
        raise ValueError("Direct config must match formal planner except protocol/output identity")
    return config


def prepare_data(repository: Path, derived_root: Path) -> tuple[list[DirectSample], list[DirectSample], dict, dict]:
    train, validation, records, summary = prepare_formal_data(repository, derived_root)
    summary.pop("gt_action_diagnostic")
    summary.update(train_sample_tokens=[s.sample_token for s in train],
                   eligibility_policy="formal_planner_train_joint_action_valid; validation_all_eligible",
                   action_validity_usage="controlled_sample_selection_only")
    groups = [[DirectSample(s.sample_token, s.scene_token, s.split, s.observation) for s in group]
              for group in (train, validation)]
    return groups[0], groups[1], records, summary


def direct_messages(observation: Observation, images: Sequence[object]) -> list[dict]:
    content = [{"type": "text", "text": DIRECT_PROMPT + "\nHistory availability: "
                + ", ".join("available" if valid else "unavailable" for valid in observation.history_valid_mask)}]
    for image, text in zip(images, observation.frame_texts, strict=True):
        content.extend(({"type": "text", "text": text}, {"type": "image", "image": image}))
    return [{"role": "user", "content": content}]


class DirectRunner:
    def __init__(self, model: nn.Module, processor: object, runtime: RuntimeDependencies,
                 dataset_root: Path, device: str) -> None:
        self.model, self.processor, self.runtime = model, processor, runtime
        self.dataset_root, self.device = dataset_root, device

    def predict(self, planner: WaypointDecoder, sample: DirectSample) -> tuple[torch.Tensor, dict]:
        if sample.split not in ("train", "validation"):
            raise ValueError("Direct prediction requires train/validation")
        images = [self.runtime.image_loader(resolve_image_path(self.dataset_root, path))
                  for path in sample.observation.image_paths]
        messages = direct_messages(sample.observation, images)
        inputs = _move_batch(self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"), self.device)
        self.model.eval()
        hidden = contextual_hidden_states(self.model, inputs)
        prediction = planner(hidden, inputs["attention_mask"])
        return prediction, {"hidden_state_field": "hidden_states[-1]", "hidden_state_shape": list(hidden.shape),
                            "hidden_state_dtype": str(hidden.dtype),
                            "attention_mask_shape": list(inputs["attention_mask"].shape),
                            "waypoint_output_shape": list(prediction.shape)}


def evaluate(planner: WaypointDecoder, runner: DirectRunner, samples: list[DirectSample],
             records: dict) -> tuple[dict, list[dict]]:
    if not samples or any(s.split != "validation" for s in samples):
        raise ValueError("Direct evaluation requires validation only")
    planner.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            prediction, evidence = runner.predict(planner, sample)
            record = records[sample.sample_token]
            target = torch.tensor(record["future_waypoints"], dtype=torch.float32)
            mask = torch.tensor(record["trajectory_valid_mask"], dtype=torch.bool)
            rows.append({"sample_token": sample.sample_token, "scene_token": sample.scene_token,
                         "split": sample.split, "conditioning_type": TRACK, **evidence,
                         "target_waypoints": target.tolist(), "trajectory_valid_mask": mask.tolist(),
                         **prediction_metrics(prediction.detach().cpu().squeeze(0), target, mask)})
    return aggregate_metrics(rows, TRACK), rows


def selection_key(metrics: dict, step: int) -> tuple[int, float, float, int]:
    if not metrics["valid_prediction_count"]:
        raise ValueError("Direct validation produced no valid trajectories")
    return metrics["invalid_prediction_count"], metrics["ade_m"], metrics["fde_m"], step


def compare_reload(before: list[dict], after: list[dict]) -> dict:
    matched = [r["sample_token"] for r in before] == [r["sample_token"] for r in after]
    for a, b in zip(before, after):
        matched &= all(a[k] == b[k] for k in ("conditioning_type", "prediction_valid", "invalid_reason"))
        if a["prediction_valid"] and b["prediction_valid"]:
            matched &= torch.allclose(torch.tensor(a["predicted_waypoints"]), torch.tensor(b["predicted_waypoints"]),
                                      atol=1e-5, rtol=1e-5)
    first, second = aggregate_metrics(before, TRACK), aggregate_metrics(after, TRACK)
    metrics_match = all(math.isclose(v, second[k], abs_tol=1e-5, rel_tol=1e-5)
                        if isinstance(v, float) and isinstance(second[k], float) else v == second[k]
                        for k, v in first.items())
    return {"reload_consistency": bool(matched and metrics_match), "predictions_match": bool(matched),
            "metrics_match": metrics_match, "sample_count": len(before), "scope": "full_validation",
            "atol": 1e-5, "rtol": 1e-5, "reference_metrics": first, "reloaded_metrics": second}


def paired_comparison(rows: list[dict], reference: list[dict], track: str) -> dict:
    left, right = index_predictions(rows, TRACK), index_predictions(reference, track)
    common = sorted(left.keys() & right.keys())
    direct_rows, reference_rows, tokens = [], [], []
    for token in common:
        a, b = left[token], right[token]
        if any(a[k] != b[k] for k in ("scene_token", "target_waypoints", "trajectory_valid_mask")):
            raise ValueError("paired scene/target/mask mismatch")
        recomputed = []
        for row in (a, b):
            if type(row["prediction_valid"]) is not bool:
                raise ValueError("prediction_valid must be boolean")
            if row["prediction_valid"]:
                metrics = prediction_metrics(torch.tensor(row["predicted_waypoints"], dtype=torch.float32),
                                             torch.tensor(row["target_waypoints"], dtype=torch.float32),
                                             torch.tensor(row["trajectory_valid_mask"], dtype=torch.bool))
                if not metrics["prediction_valid"]:
                    raise ValueError("saved validity contradicts trajectory")
                recomputed.append({"conditioning_type": row["conditioning_type"], **metrics})
        if len(recomputed) == 2:
            direct_rows.append(recomputed[0])
            reference_rows.append(recomputed[1])
            tokens.append(token)
    direct, other = aggregate_metrics(direct_rows, TRACK), aggregate_metrics(reference_rows, track)
    deltas = {}
    for key in ("ade_1s_m", "ade_2s_m", "ade_3s_m", "fde_1s_m", "fde_2s_m", "fde_3s_m"):
        a, b = direct[key], other[key]
        delta = a - b if a is not None and b is not None else None
        deltas[key] = {"direct": a, "reference": b, "absolute_delta_m": delta,
                       "relative_delta": delta / b if delta is not None and b != 0 else None,
                       "sample_count": direct[f"{key}_sample_count"]}
    return {"reference_track": track, "delta_definition": "direct_minus_reference; negative_is_better",
            "relative_delta_definition": "(direct-reference)/reference; null_if_zero_or_unavailable",
            "direct_sample_count": len(left), "reference_sample_count": len(right),
            "matching_sample_count": len(common), "jointly_valid_sample_count": len(tokens),
            "paired_sample_tokens": tokens, "direct_unmatched_tokens": sorted(left.keys() - right.keys()),
            "reference_unmatched_tokens": sorted(right.keys() - left.keys()), "metrics": deltas}


def fit(*, model: nn.Module, planner: WaypointDecoder, runner: DirectRunner,
        train: list[DirectSample], validation: list[DirectSample], records: dict,
        config: FullConfig, output: Path, provenance: dict) -> dict:
    if not train or any(s.split != "train" for s in train):
        raise ValueError("Direct fitting requires train only")
    if not validation or any(s.split != "validation" for s in validation):
        raise ValueError("Direct selection requires validation only")
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    provenance = {**provenance, **verify_freeze(model, planner, optimizer)}
    write_json(output / "run_metadata.json", provenance)
    total = math.ceil(len(train) / config.gradient_accumulation_steps) * config.num_train_epochs
    step, consumed = 0, 0
    best, best_score, checkpoints = None, None, []
    with (output / "training_history.jsonl").open("w") as history:
        for epoch in range(config.num_train_epochs):
            for group in epoch_groups(train, config.gradient_accumulation_steps, config.seed + epoch):
                planner.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                for sample in group:
                    prediction, evidence = runner.predict(planner, sample)
                    if step == 0 and sample is group[0]:
                        provenance["first_training_forward"] = evidence
                        write_json(output / "run_metadata.json", provenance)
                    row = records[sample.sample_token]
                    target = torch.tensor([row["future_waypoints"]], device=runner.device, dtype=torch.float32)
                    mask = torch.tensor([row["trajectory_valid_mask"]], device=runner.device, dtype=torch.bool)
                    loss = masked_waypoint_loss(prediction, target, mask, beta=config.smooth_l1_beta)
                    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                        raise ValueError("nonfinite Direct training output/loss")
                    (loss / len(group)).backward()
                    loss_sum += float(loss.detach())
                    del prediction, loss
                optimizer.step()
                step += 1
                consumed += len(group)
                entry = {"optimizer_step": step, "epoch": epoch + 1, "samples_consumed": consumed,
                         "sample_tokens": [s.sample_token for s in group], "conditioning_type": TRACK,
                         "train_loss": loss_sum / len(group), "learning_rate": config.learning_rate}
                validate = step % config.validation_interval == 0 or step == total
                checkpoint = output / f"planner_step_{step:04d}.pt"
                if validate or step % config.checkpoint_interval == 0:
                    save_checkpoint(checkpoint, planner, config, provenance, step)
                if validate:
                    metrics, rows = evaluate(planner, runner, validation, records)
                    write_json(output / f"validation_step_{step:04d}_metrics.json", metrics)
                    write_predictions(output / f"validation_step_{step:04d}_predictions.jsonl", rows)
                    score = selection_key(metrics, step)
                    metadata = {"optimizer_step": step, "checkpoint": checkpoint.name,
                                "conditioning_type": TRACK, "metrics": metrics}
                    checkpoints.append(metadata)
                    if best_score is None or score < best_score:
                        best, best_score = metadata, score
                        write_predictions(output / "validation_predictions.jsonl", rows)
                        write_json(output / "validation_metrics.json", metrics)
                    write_json(output / "checkpoint_selection.json", checkpoints)
                    entry["validation_metrics"] = metrics
                history.write(json.dumps(entry, allow_nan=False) + "\n")
                history.flush()
                if step % config.log_every_steps == 0 or validate:
                    print(json.dumps(entry, allow_nan=False), flush=True)
    verify_freeze(model, planner, optimizer)
    del optimizer, planner
    selected = reload_planner(output / best["checkpoint"], runner.device)
    before = read_predictions(output / "validation_predictions.jsonl", TRACK)
    metrics, rows = evaluate(selected, runner, validation, records)
    consistency = compare_reload(before, rows)
    write_json(output / "reload_consistency.json", consistency)
    if not consistency["reload_consistency"]:
        raise ValueError("Direct checkpoint reload mismatch")
    write_predictions(output / "validation_predictions.jsonl", rows)
    write_json(output / "validation_metrics.json", metrics)
    write_json(output / "best_checkpoint.json", best)
    return {"status": "training_completed", "optimizer_steps": step, "samples_consumed": consumed,
            "best_checkpoint": best, "validation_metrics": metrics, "reload_consistency": consistency,
            "full_model_saved": False}


def run(*, repository: Path, dataset_root: Path, derived_root: Path, config: FullConfig,
        git_provenance: GitProvenance, train_split: str = "train", validation_split: str = "validation") -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("Direct only permits train fitting and validation evaluation")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("Direct artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"Direct output already exists: {output}")
    references = {}
    for name, directory, filename, track in (
        ("action_conditioned_planner", "two_turn_planner_full_v0_1", "validation_predicted_action_predictions.jsonl", "predicted_action"),
        ("ego_history_mlp", "ego_history_mlp_baseline_v0_1", "validation_predictions.jsonl", "ego_history_mlp"),
        ("constant_velocity", "constant_velocity_baseline_v0_1", "predictions.jsonl", "constant_velocity"),
    ):
        references[name] = (read_predictions(derived_root / "phase_0_4" / directory / filename, track), track)
    train, validation, records, data = prepare_data(repository, derived_root)
    semantic = load_semantic_config(repository / "configs/phase0_4b_lora_full.yaml")
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector(semantic.precision)
    if dtype != torch.bfloat16:
        raise ValueError("selected Phase 0.4b model requires BF16 support")
    torch.manual_seed(config.seed)
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, semantic.local_files_only)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                semantic.attention_implementation, semantic.local_files_only)
    model = runtime.adapter_loader(base, resolve_derived_path(derived_root, config.selected_adapter_relative_path)).to(device)
    freeze_backbone(model)
    if model.config.text_config.hidden_size != 2560:
        raise ValueError("Qwen planner interface must have hidden size 2560")
    runner = DirectRunner(model, processor, runtime, dataset_root, device)
    provenance = {
        "execution_git_commit": git.commit, "config": asdict(config), "data": data,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION, "processor_revision": FIXED_REVISION,
        "selected_adapter": config.selected_adapter_relative_path, "selected_adapter_loaded": True,
        "prompt_version": PROMPT_VERSION, "direct_prompt": DIRECT_PROMPT, "conditioning_type": TRACK,
        "action_generation_calls": 0, "action_tokens_inserted": 0,
        "action_values_used_for_prediction": 0, "action_values_used_for_loss": 0,
        "future_information_inputs": 0, "reload_scope": "full_validation",
        "reload_subset_size_usage": "inherited_config_field_unused_for_full_validation_reload",
        "waypoint_times_sec": [.5, 1., 1.5, 2., 2.5, 3.],
        "coordinates": "current_ego_frame_x_forward_y_left_meters",
        "transformers_version": runtime.package_version("transformers"), "peft_version": runtime.package_version("peft"),
    }
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", asdict(config))
    write_json(output / "data_summary.json", data)
    result = fit(model=model, planner=WaypointDecoder(2560, config).to(device), runner=runner,
                 train=train, validation=validation, records=records, config=config, output=output, provenance=provenance)
    rows = read_predictions(output / "validation_predictions.jsonl", TRACK)
    for name, (reference, track) in references.items():
        write_json(output / f"comparison_to_{name}.json", paired_comparison(rows, reference, track))
    write_json(output / "training_summary.json", result)
    return result

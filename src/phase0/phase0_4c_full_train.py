from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import yaml

from src.phase0.development_projection import IsolationCounters
from src.phase0.phase0_4_factorized_targets import FACTORIZED_ACTION_RULE_VERSION
from src.phase0.phase0_4_temporal_dataset import SCHEMA_VERSION
from src.phase0.phase0_4b_lora_full import epoch_groups, load_config as load_semantic_config
from src.phase0.phase0_4b_lora_smoke import generate_action, load_temporal_records
from src.phase0.phase0_4b_protocol import (
    PARSER_VERSION, PROMPT_VERSION, SERIALIZATION_VERSION, SFTSample, adapt_record,
    inference_messages, parse_action, serialize_action,
)
from src.phase0.phase0_4c_evaluation import aggregate_metrics, compare_reload, paired_gap, prediction_metrics
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.phase0_4c_two_turn_planner import (
    PlannerConfig, WaypointDecoder, contextual_hidden_states, freeze_backbone,
    masked_waypoint_loss, planning_inputs, planning_messages,
)
from src.phase0.qwen3vl_dataset_adapter import (
    AdapterConfig, GitProvenance, load_config as load_ego_config, resolve_derived_path,
    validate_git_provenance,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import RuntimeDependencies, default_runtime_dependencies
from src.phase0.qwen3vl_smoke import resolve_image_path


@dataclass(frozen=True)
class FullConfig(PlannerConfig):
    protocol_version: str
    num_train_epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    lr_scheduler: str
    checkpoint_interval: int
    validation_interval: int
    log_every_steps: int
    early_stopping: bool
    selection_policy: str
    reload_subset_size: int


def load_config(path: Path) -> FullConfig:
    config = FullConfig(train_subset_size=0, **yaml.safe_load(path.read_text()))
    fixed = {
        "protocol_version": "phase0.4c-full-planner-v0.1", "planner_dimension": 256,
        "num_decoder_layers": 2, "num_heads": 4, "dropout": 0.1,
        "num_waypoint_queries": 6, "memory_normalization": True, "smooth_l1_beta": 1.0,
        "micro_batch_size": 1, "optimizer": "AdamW", "lr_scheduler": "constant",
        "early_stopping": False,
        "selection_policy": "invalid_count_then_ade_then_fde_then_earliest",
        "selected_adapter_relative_path": "phase_0_4/structured_action_lora_full_v0_1/adapter_step_3564",
        "output_relative_dir": "phase_0_4/two_turn_planner_full_v0_1",
    }
    for key, value in fixed.items():
        if getattr(config, key) != value:
            raise ValueError(f"{key} differs from full-planner protocol")
    for key in ("num_train_epochs", "gradient_accumulation_steps", "checkpoint_interval",
                "validation_interval", "log_every_steps", "reload_subset_size"):
        if type(getattr(config, key)) is not int or getattr(config, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (type(config.seed) is not int or not math.isfinite(config.learning_rate)
            or config.learning_rate <= 0 or not math.isfinite(config.weight_decay)
            or config.weight_decay < 0):
        raise ValueError("invalid seed or optimizer settings")
    return config


def eligible_samples(records: list[dict], ego_config: AdapterConfig, split: str) -> tuple[list[SFTSample], dict]:
    if split not in ("train", "validation"):
        raise ValueError("planner intake only permits train and validation")
    selected, reasons = [], Counter()
    for row in records:
        sample = adapt_record(row, ego_config, expected_split=split)
        excluded = []
        if not any(row["trajectory_valid_mask"]):
            excluded.append("no_valid_waypoints")
        if split == "train" and not row["factorized_action_joint_valid"]:
            excluded.append("invalid_teacher_forcing_action")
        if excluded:
            reasons.update(excluded)
        else:
            selected.append(sample)
    if not selected:
        raise ValueError(f"no eligible planner {split} samples")
    return selected, {"total_records": len(records), "eligible_records": len(selected),
                      "excluded_records": len(records) - len(selected),
                      "exclusion_reason_counts": dict(reasons)}


def prepare_data(repository: Path, derived_root: Path) -> tuple[list[SFTSample], list[SFTSample], dict, dict]:
    ego = load_ego_config(repository / "configs/phase0_3_dataset_adapter.yaml")
    train_rows = load_temporal_records(repository, derived_root, split="train")
    validation_rows = load_temporal_records(repository, derived_root, split="validation")
    if ({r["scene_token"] for r in train_rows} & {r["scene_token"] for r in validation_rows}
            or {r["sample_token"] for r in train_rows} & {r["sample_token"] for r in validation_rows}):
        raise ValueError("train/validation contamination")
    train, train_counts = eligible_samples(train_rows, ego, "train")
    validation, validation_counts = eligible_samples(validation_rows, ego, "validation")
    diagnostic_count = sum(s.target.longitudinal_valid and s.target.lateral_valid for s in validation)
    summary = {"train": train_counts, "validation": validation_counts,
               "gt_action_diagnostic": {"eligible_records": diagnostic_count,
                                        "excluded_invalid_action_records": len(validation) - diagnostic_count},
               "temporal_dataset_schema_version": SCHEMA_VERSION,
               "factorized_action_rule_version": FACTORIZED_ACTION_RULE_VERSION,
               "source_provenance": {
                   split: {key: sorted({r[key] for r in rows})
                           for key in ("source_combined_manifest_sha256", "split_mapping_sha256")}
                   for split, rows in (("train", train_rows), ("validation", validation_rows))},
               "test_records_read": 0, **asdict(IsolationCounters()),
               "test_evaluation_performed": False}
    return train, validation, {r["sample_token"]: r for r in train_rows + validation_rows}, summary


def verify_freeze(model: nn.Module, planner: WaypointDecoder, optimizer: torch.optim.Optimizer) -> dict:
    counts = {
        "qwen_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                          if "lora_" not in n and p.requires_grad),
        "lora_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                          if "lora_" in n and p.requires_grad),
        "lora_parameter_tensors": sum("lora_" in n for n, _ in model.named_parameters()),
        "planner_trainable_parameters": sum(p.numel() for p in planner.parameters() if p.requires_grad),
        "memory_norm_trainable": isinstance(planner.memory_norm, nn.LayerNorm)
        and all(p.requires_grad for p in planner.memory_norm.parameters()),
    }
    expected = {id(p) for p in planner.parameters()}
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if (counts["qwen_trainable_parameters"] or counts["lora_trainable_parameters"]
            or not counts["lora_parameter_tensors"] or not counts["planner_trainable_parameters"]
            or not counts["memory_norm_trainable"] or not all(p.requires_grad for p in planner.parameters())
            or len(actual) != len(expected) or {id(p) for p in actual} != expected
            or expected & {id(p) for p in model.parameters()}):
        raise ValueError("frozen Qwen/LoRA or planner-only optimizer contract mismatch")
    return counts


class PlannerRunner:
    def __init__(self, model: nn.Module, processor: object, runtime: RuntimeDependencies,
                 dataset_root: Path, generation_kwargs: dict, device: str) -> None:
        self.model, self.processor, self.runtime = model, processor, runtime
        self.dataset_root, self.generation_kwargs, self.device = dataset_root, generation_kwargs, device

    def predict(self, planner: WaypointDecoder, sample: SFTSample,
                conditioning_type: str) -> tuple[torch.Tensor | None, dict]:
        expected = "train" if conditioning_type == "gt_action_teacher_forced_train" else "validation"
        if (conditioning_type not in ("gt_action_teacher_forced_train", "predicted_action", "gt_action_diagnostic")
                or sample.split != expected):
            raise ValueError("conditioning path split mismatch")
        images = [self.runtime.image_loader(resolve_image_path(self.dataset_root, path))
                  for path in sample.observation.image_paths]
        self.model.eval()
        raw = None
        if conditioning_type == "predicted_action":
            with torch.no_grad():
                generated = generate_action(self.model, inference_messages(sample.observation, images),
                                            self.processor, self.generation_kwargs, self.device)
            raw = generated["raw_output"]
            action = parse_action(raw)
            if action is None:
                return None, {"raw_output": raw, "action_context": None}
            text = serialize_action(action["longitudinal"], action["lateral"])
        else:
            text = serialize_action(sample.target.longitudinal, sample.target.lateral)
        inputs, evidence = planning_inputs(
            self.processor, planning_messages(sample.observation, images, text), self.device)
        hidden = contextual_hidden_states(self.model, inputs)
        prediction = planner(hidden, inputs["attention_mask"])
        return prediction, {"raw_output": raw, "action_context": text,
                            "action_context_matches": evidence["action_context_matches"]}


def evaluate(planner: WaypointDecoder, runner: PlannerRunner, samples: list[SFTSample],
             records: dict, conditioning_type: str) -> tuple[dict, list[dict]]:
    if conditioning_type not in ("predicted_action", "gt_action_diagnostic"):
        raise ValueError("evaluation requires a named validation track")
    if any(sample.split != "validation" for sample in samples):
        raise ValueError("evaluation requires validation samples only")
    planner.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            prediction, evidence = runner.predict(planner, sample, conditioning_type)
            record = records[sample.sample_token]
            target = torch.tensor(record["future_waypoints"], dtype=torch.float32)
            mask = torch.tensor(record["trajectory_valid_mask"], dtype=torch.bool)
            point_prediction = None if prediction is None else prediction.detach().cpu().squeeze(0)
            rows.append({"sample_token": sample.sample_token, "scene_token": sample.scene_token,
                         "split": sample.split, "conditioning_type": conditioning_type, **evidence,
                         "target_waypoints": target.tolist(), "trajectory_valid_mask": mask.tolist(),
                         **prediction_metrics(point_prediction, target, mask)})
    return aggregate_metrics(rows, conditioning_type), rows


def write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def save_checkpoint(path: Path, planner: WaypointDecoder, config: FullConfig,
                    provenance: dict, step: int) -> None:
    torch.save({"planner_state_dict": {n: p.detach().cpu() for n, p in planner.state_dict().items()},
                "planner_config": {key: getattr(config, key) for key in PlannerConfig.__dataclass_fields__},
                "hidden_size": planner.context_projection.in_features, "training_config": asdict(config),
                "optimizer_step": step, "provenance": provenance}, path)


def reload_planner(path: Path, device: str) -> WaypointDecoder:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    planner = WaypointDecoder(saved["hidden_size"], PlannerConfig(**saved["planner_config"])).to(device)
    planner.load_state_dict(saved["planner_state_dict"])
    return planner


def fit(*, model: nn.Module, planner: WaypointDecoder, runner: PlannerRunner,
        train: list[SFTSample], validation: list[SFTSample], records: dict,
        config: FullConfig, output: Path, provenance: dict) -> dict:
    if not train or any(s.split != "train" for s in train):
        raise ValueError("optimization requires train samples only")
    if not validation or any(s.split != "validation" for s in validation):
        raise ValueError("model selection requires validation samples only")
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    freeze = verify_freeze(model, planner, optimizer)
    provenance = {**provenance, "freeze_policy": freeze}
    write_json(output / "run_metadata.json", provenance)
    total = math.ceil(len(train) / config.gradient_accumulation_steps) * config.num_train_epochs
    started, step, consumed = time.monotonic(), 0, 0
    best, best_score, checkpoints = None, None, []
    with (output / "training_history.jsonl").open("w") as history:
        for epoch in range(config.num_train_epochs):
            for group in epoch_groups(train, config.gradient_accumulation_steps, config.seed + epoch):
                planner.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for sample in group:
                    prediction, _ = runner.predict(planner, sample, "gt_action_teacher_forced_train")
                    row = records[sample.sample_token]
                    target = torch.tensor([row["future_waypoints"]], device=runner.device, dtype=torch.float32)
                    mask = torch.tensor([row["trajectory_valid_mask"]], device=runner.device, dtype=torch.bool)
                    loss = masked_waypoint_loss(prediction, target, mask, beta=config.smooth_l1_beta)
                    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                        raise ValueError("nonfinite planner training output/loss")
                    (loss / len(group)).backward()
                    loss_sum += float(loss.detach())
                    del prediction, loss
                optimizer.step()
                step += 1
                consumed += len(group)
                entry = {"optimizer_step": step, "epoch": epoch + 1, "samples_consumed": consumed,
                         "sample_tokens": [s.sample_token for s in group],
                         "conditioning_type": "gt_action_teacher_forced_train",
                         "train_loss": loss_sum / len(group), "learning_rate": config.learning_rate}
                validate = step % config.validation_interval == 0 or step == total
                checkpoint = output / f"planner_step_{step:04d}.pt"
                if validate or step % config.checkpoint_interval == 0:
                    save_checkpoint(checkpoint, planner, config, provenance, step)
                if validate:
                    metrics, rows = evaluate(planner, runner, validation, records, "predicted_action")
                    write_json(output / f"validation_step_{step:04d}_metrics.json", metrics)
                    write_predictions(output / f"validation_step_{step:04d}_predictions.jsonl", rows)
                    if not metrics["valid_prediction_count"]:
                        raise ValueError("formal validation produced no valid predicted-action trajectories")
                    score = (metrics["invalid_prediction_count"], metrics["ade_m"], metrics["fde_m"], step)
                    metadata = {"optimizer_step": step, "checkpoint": checkpoint.name,
                                "conditioning_type": "predicted_action", "metrics": metrics}
                    checkpoints.append(metadata)
                    if best_score is None or score < best_score:
                        best, best_score = metadata, score
                        write_predictions(output / "validation_predicted_action_predictions.jsonl", rows)
                        write_json(output / "validation_predicted_action_metrics.json", metrics)
                    entry["validation_metrics"] = metrics
                    write_json(output / "checkpoint_selection.json", checkpoints)
                entry["elapsed_seconds"] = time.monotonic() - started
                history.write(json.dumps(entry, allow_nan=False) + "\n")
                history.flush()
                if step % config.log_every_steps == 0 or validate:
                    print(json.dumps(entry, allow_nan=False), flush=True)
    verify_freeze(model, planner, optimizer)
    del optimizer, planner
    selected = reload_planner(output / best["checkpoint"], runner.device)
    saved_rows = [json.loads(line) for line in
                  (output / "validation_predicted_action_predictions.jsonl").read_text().splitlines()]
    by_token = {r["sample_token"]: r for r in saved_rows}
    subset = sorted((s for s in validation if by_token[s.sample_token]["prediction_valid"]),
                    key=lambda s: s.sample_token)[:config.reload_subset_size]
    sanity = {"decoder_memory_finite": True, "trajectory_output_finite": True}

    def check_memory(module: nn.Module, args: tuple, result: torch.Tensor) -> None:
        sanity["decoder_memory_finite"] &= bool(torch.isfinite(result).all())

    handle = selected.memory_norm.register_forward_hook(check_memory)
    _, reloaded_rows = evaluate(selected, runner, subset, records, "predicted_action")
    handle.remove()
    consistency = compare_reload([by_token[s.sample_token] for s in subset], reloaded_rows)
    consistency["subset_policy"] = "sorted_sample_token_first_valid_formal_predictions"
    consistency["sample_tokens"] = [s.sample_token for s in subset]
    write_json(output / "reload_consistency.json", consistency)
    if not consistency["reload_consistency"]:
        raise ValueError("checkpoint fresh reload is inconsistent")
    diagnostic_samples = [s for s in validation if s.target.longitudinal_valid and s.target.lateral_valid]
    diagnostic_metrics, diagnostic_rows = evaluate(selected, runner, diagnostic_samples, records,
                                                   "gt_action_diagnostic")
    write_predictions(output / "validation_gt_action_diagnostic_predictions.jsonl", diagnostic_rows)
    write_json(output / "validation_gt_action_diagnostic_metrics.json", diagnostic_metrics)
    sanity["trajectory_output_finite"] = all(r["prediction_valid"] for r in reloaded_rows)
    sanity["planner_parameters_finite"] = all(bool(torch.isfinite(p).all()) for p in selected.parameters())
    write_json(output / "final_sanity.json", sanity)
    if not all(sanity.values()):
        raise ValueError("final planner finite sanity check failed")
    write_json(output / "best_checkpoint.json", {**best, "provenance": provenance})
    result = {"status": "full_training_completed", "optimizer_steps": step, "samples_consumed": consumed,
              "provenance": provenance, "best_checkpoint": best,
              "predicted_action_conditioned": best["metrics"],
              "gt_action_diagnostic": diagnostic_metrics,
              "conditioning_gap": paired_gap(saved_rows, diagnostic_rows),
              "reload_consistency": consistency, "final_sanity": sanity,
              "elapsed_seconds": time.monotonic() - started, "full_model_saved": False}
    write_json(output / "training_summary.json", result)
    return result


def run_full(*, repository: Path, dataset_root: Path, derived_root: Path,
             config: FullConfig, git_provenance: GitProvenance, train_split: str = "train",
             validation_split: str = "validation") -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("full planner requires train optimization and validation evaluation only")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("planner artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"full planner output already exists: {output}")
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
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    model = runtime.adapter_loader(base, adapter).to(device)
    freeze_backbone(model)
    if model.config.text_config.hidden_size != 2560:
        raise ValueError("Qwen planner interface must have hidden size 2560")
    planner = WaypointDecoder(2560, config).to(device)
    runner = PlannerRunner(model, processor, runtime, dataset_root,
                           {**semantic.generation_kwargs, "use_cache": True}, device)
    provenance = {
        "execution_git_commit": git.commit, "config": asdict(config), "seed": config.seed,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION, "processor_revision": FIXED_REVISION,
        "selected_adapter": config.selected_adapter_relative_path, "selected_adapter_loaded": True,
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION,
        "serialization_version": SERIALIZATION_VERSION, "data": data,
        "train_sample_count": len(train), "validation_sample_count": len(validation),
        "conditioning_protocol": {"train": "gt_action_teacher_forced_train", "validation": "predicted_action",
                                  "diagnostic": "gt_action_diagnostic"},
        "planner_architecture": "2560->256->MemoryLayerNorm->6queries->2layer4headPostLN->2",
        "memory_norm_enabled": True, "waypoint_times_sec": [0.5, 1., 1.5, 2., 2.5, 3.],
        "coordinates": "current_ego_frame_x_forward_y_left_meters", "hidden_state_cache": False,
        "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
    }
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", asdict(config))
    write_json(output / "data_summary.json", data)
    return fit(model=model, planner=planner, runner=runner, train=train, validation=validation,
               records=records, config=config, output=output, provenance=provenance)

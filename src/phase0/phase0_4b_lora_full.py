from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
import gc
import json
from itertools import zip_longest
import math
from pathlib import Path
import random

import yaml

from src.phase0 import development_projection as development
from src.phase0.phase0_4b_evaluation import (
    checkpoint_score, factorized_metrics, failure_taxonomy,
    generalization_summary, joint_distribution, majority_baselines, select_checkpoint,
)
from src.phase0.phase0_4b_lora_smoke import load_temporal_samples, lora_b_report, predict_sample
from src.phase0.phase0_4b_protocol import (
    PARSER_VERSION, PROMPT_VERSION, SERIALIZATION_VERSION, SFTSample, StructuredCollator,
)
from src.phase0.qwen3vl_dataset_adapter import resolve_derived_path, validate_git_provenance
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import (
    LORA_TARGET_MODULES, _move_batch, default_runtime_dependencies, inject_lora,
    lora_config_kwargs, trainable_parameter_report,
)


@dataclass(frozen=True)
class FullConfig:
    protocol_version: str
    seed: int
    shuffle_version: str
    num_train_epochs: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    optimizer: str
    learning_rate: float
    warmup_ratio: float
    lr_scheduler: str
    precision: str
    gradient_checkpointing: bool
    attention_implementation: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    lora_task_type: str
    lora_target_modules: tuple[str, ...]
    local_files_only: bool
    checkpoint_fractions: tuple[float, ...]
    failure_examples_per_category: int
    lateral_collapse_straight_fraction: float
    generation_kwargs: dict
    output_relative_dir: str


def load_config(path: Path) -> FullConfig:
    values = yaml.safe_load(path.read_text())
    for key in ("lora_target_modules", "checkpoint_fractions"):
        values[key] = tuple(values[key])
    config = FullConfig(**values)
    frozen = {
        "protocol_version": "phase0.4b-full-action-sft-v0.1",
        "shuffle_version": "sorted-token-python-random-v0.1",
        "num_train_epochs": 1, "micro_batch_size": 1, "gradient_accumulation_steps": 4,
        "optimizer": "AdamW", "learning_rate": 1e-4, "warmup_ratio": 0.03,
        "lr_scheduler": "cosine", "precision": "bfloat16", "gradient_checkpointing": True,
        "attention_implementation": "sdpa", "lora_r": 8, "lora_alpha": 16,
        "lora_dropout": 0.05, "lora_bias": "none", "lora_task_type": "CAUSAL_LM",
        "lora_target_modules": LORA_TARGET_MODULES, "local_files_only": True,
        "checkpoint_fractions": (0.25, 0.5, 0.75, 1.0),
        "generation_kwargs": {"do_sample": False, "num_beams": 1, "max_new_tokens": 24},
        "output_relative_dir": "phase_0_4/structured_action_lora_full_v0_1",
    }
    for key, value in frozen.items():
        if getattr(config, key) != value:
            raise ValueError(f"{key} differs from frozen full-training protocol")
    if type(config.seed) is not int or config.failure_examples_per_category < 1:
        raise ValueError("seed must be integer and failure example count positive")
    if not 0 < config.lateral_collapse_straight_fraction <= 1:
        raise ValueError("collapse reporting fraction must be in (0,1]")
    return config


def epoch_groups(samples: list[SFTSample], accumulation: int, seed: int) -> Iterator[list[SFTSample]]:
    if not samples or any(s.split != "train" for s in samples):
        raise ValueError("optimization requires nonempty train samples only")
    ordered = sorted(samples, key=lambda s: s.sample_token)
    random.Random(seed).shuffle(ordered)
    for start in range(0, len(ordered), accumulation):
        yield ordered[start:start + accumulation]


def checkpoint_steps(total_steps: int, fractions: tuple[float, ...]) -> list[int]:
    return sorted({max(1, math.floor(total_steps * fraction + 0.5)) for fraction in fractions})


def learning_rate_at_step(step: int, total: int, config: FullConfig) -> float:
    warmup = math.ceil(total * config.warmup_ratio)
    if step <= warmup:
        return config.learning_rate * step / warmup
    progress = (step - warmup - 1) / (total - warmup)
    return config.learning_rate * (1 + math.cos(math.pi * progress)) / 2


def train_one_epoch(*, model: object, samples: list[SFTSample], collator: StructuredCollator,
                    optimizer: object, config: FullConfig, device: str,
                    on_step: Callable[[dict], None]) -> list[dict]:
    total = math.ceil(len(samples) / config.gradient_accumulation_steps)
    history = []
    consumed = 0
    model.train()
    optimizer.zero_grad()
    for step, group in enumerate(epoch_groups(samples, config.gradient_accumulation_steps, config.seed), 1):
        lr = learning_rate_at_step(step, total, config)
        for parameters in optimizer.param_groups:
            parameters["lr"] = lr
        supervised = [s for s in group if s.target.longitudinal_valid or s.target.lateral_valid]
        loss_sum = 0.0
        for sample in supervised:
            batch = collator([sample], expected_split="train")
            loss = model(**_move_batch(batch, device)).loss
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise ValueError("training loss must be finite")
            (loss / len(supervised)).backward()
            loss_sum += value
        # An all-invalid group has no loss and must not trigger AdamW weight decay.
        if supervised:
            optimizer.step()
        optimizer.zero_grad()
        consumed += len(group)
        entry = {"step": step, "sample_count": len(group), "consumed_sample_count": consumed,
                 "supervised_sample_count": len(supervised), "optimizer_step_performed": bool(supervised),
                 "loss": loss_sum / len(supervised) if supervised else None, "learning_rate": lr}
        history.append(entry)
        on_step(entry)
        model.train()
    return history


def evaluate_validation(model: object, samples: list[SFTSample], collator: StructuredCollator,
                        config: FullConfig, device: str, runtime: object) -> list[dict]:
    if not samples or any(s.split != "validation" for s in samples):
        raise ValueError("evaluation requires nonempty validation samples only")
    model.eval()
    with runtime.inference_context():
        return [predict_sample(model, s, collator, config.generation_kwargs, device,
                               expected_split="validation") for s in samples]


def prepare_data(repository: Path, derived_root: Path, config: FullConfig
                 ) -> tuple[list[SFTSample], list[SFTSample], dict]:
    train = load_temporal_samples(repository, derived_root, split="train")
    validation = load_temporal_samples(repository, derived_root, split="validation")
    if ({s.scene_token for s in train} & {s.scene_token for s in validation}
            or {s.sample_token for s in train} & {s.sample_token for s in validation}):
        raise ValueError("train/validation contamination")
    summary = {
        "train_sample_count": len(train), "train_scene_count": len({s.scene_token for s in train}),
        "validation_sample_count": len(validation),
        "validation_scene_count": len({s.scene_token for s in validation}),
        "train_joint_distribution": joint_distribution(train),
        "validation_joint_distribution": joint_distribution(validation),
        "train_sample_tokens": [s.sample_token for group in epoch_groups(
            train, config.gradient_accumulation_steps, config.seed) for s in group],
        "train_validity_counts": {
            axis: sum(getattr(s.target, f"{axis}_valid") for s in train)
            for axis in ("longitudinal", "lateral")
        },
        "checkpoint_selection_split": "full_validation",
        "test_records_read": 0, **asdict(development.IsolationCounters()),
        "test_evaluation_performed": False,
    }
    return train, validation, summary


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_full(*, repository: Path, dataset_root: Path, derived_root: Path,
             config: FullConfig, git_provenance: object) -> dict:
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("training artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"full-training output already exists: {output}")
    train, validation, data_summary = prepare_data(repository, derived_root, config)
    baselines = majority_baselines(train, validation)
    import torch

    torch.manual_seed(config.seed)
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector(config.precision)
    if dtype != torch.bfloat16:
        raise ValueError("formal Phase 0.4b-B requires BF16 support")
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, config.local_files_only)
    collator = StructuredCollator(processor, runtime.image_loader, dataset_root)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                config.attention_implementation, config.local_files_only)
    base.to(device)
    model = inject_lora(base, config=config, dependencies=runtime)
    parameter_report = trainable_parameter_report(model, config.lora_target_modules)
    if parameter_report["trainable_parameter_count"] == 0:
        raise ValueError("no trainable LoRA parameters")
    before_weights = lora_b_report(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = runtime.optimizer_factory([p for p in model.parameters() if p.requires_grad],
                                          config.learning_rate)
    total = math.ceil(len(train) / config.gradient_accumulation_steps)
    milestones = checkpoint_steps(total, config.checkpoint_fractions)
    output.mkdir(parents=True)
    resolved = {
        "config": asdict(config), "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION,
        "processor_revision": FIXED_REVISION, "prompt_version": PROMPT_VERSION,
        "serialization_version": SERIALIZATION_VERSION, "parser_version": PARSER_VERSION,
        "execution_git_commit": git.commit, "lora_config": lora_config_kwargs(config),
        "trainable_parameters": parameter_report, "full_model_saved": False,
        "total_optimizer_steps": total, "checkpoint_steps": milestones,
        "checkpoint_selection_protocol": "mean_direction_macro_f1_then_joint_accuracy_then_parser_success_then_earliest",
        "generation_use_cache": False,
        "warmup_steps": math.ceil(total * config.warmup_ratio),
        "scheduler_version": "linear-warmup-then-cosine-one-based-v0.1",
        "scheduler_definition": "warmup: lr*step/warmup; cosine: lr*(1+cos(pi*(step-warmup-1)/(total-warmup)))/2",
        "optimizer_defaults": {key: value for key, value in optimizer.defaults.items() if key != "params"},
        "loss_reduction": "mean_of_supervised_sample_token_means_in_actual_accumulation_group",
        "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
    }
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "data_summary.json", data_summary)
    write_json(output / "majority_baselines.json", baselines)
    checkpoints = []
    with (output / "training_history.jsonl").open("w") as stream:
        def on_step(entry: dict) -> None:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
            if entry["step"] in milestones:
                checkpoint = output / f"adapter_step_{entry['step']:04d}"
                model.save_pretrained(checkpoint)
                predictions = evaluate_validation(model, validation, collator, config, device, runtime)
                metric = factorized_metrics(predictions)
                prediction_path = output / f"validation_step_{entry['step']:04d}.json"
                write_json(prediction_path, predictions)
                metadata = {"step": entry["step"], "adapter_path": str(checkpoint),
                            "metrics": metric, "selection_score": checkpoint_score(metric),
                            "validation_predictions_path": str(prediction_path),
                            "validation_sample_count": len(predictions),
                            "lora_B": lora_b_report(model), "full_model_saved": False}
                checkpoints.append(metadata)
                write_json(output / "milestone_checkpoints.json", checkpoints)
                print(json.dumps({"milestone_step": entry["step"],
                                  "selection_score": metadata["selection_score"]}), flush=True)

        history = train_one_epoch(model=model, samples=train, collator=collator, optimizer=optimizer,
                                  config=config, device=device, on_step=on_step)
    after_weights = lora_b_report(model)
    updated = before_weights["nonzero_element_count"] == 0 and after_weights["nonzero_element_count"] > 0
    if not updated or not math.isfinite(after_weights["norm"]):
        raise ValueError("LoRA parameters did not update to finite weights")
    selected = select_checkpoint(checkpoints)
    write_json(output / "selected_checkpoint.json", selected)
    del on_step, optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                      config.attention_implementation, config.local_files_only)
    reloaded = runtime.adapter_loader(fresh_base, Path(selected["adapter_path"]))
    reloaded.to(device)
    reloaded.config.use_cache = False
    restored = lora_b_report(reloaded)
    expected = selected["lora_B"]
    if (any(restored[key] != expected[key] for key in ("tensor_count", "nonzero_tensor_count", "nonzero_element_count"))
            or not math.isclose(restored["norm"], expected["norm"], rel_tol=1e-5, abs_tol=1e-8)):
        raise ValueError("selected adapter reload weight report mismatch")
    predictions = evaluate_validation(reloaded, validation, collator, config, device, runtime)
    metric = factorized_metrics(predictions)
    selection_predictions = json.loads(Path(selected["validation_predictions_path"]).read_text())
    differences = [
        {"index": index, "selection": previous, "fresh_reload": current}
        for index, (previous, current) in enumerate(zip_longest(selection_predictions, predictions))
        if previous != current
    ]
    comparison = {"selected_step": selected["step"], "sample_differences": differences,
                  "selection_sample_count": len(selection_predictions),
                  "fresh_reload_sample_count": len(predictions),
                  "metrics_match": metric == selected["metrics"],
                  "selection_metrics": selected["metrics"], "fresh_reload_metrics": metric}
    comparison["matched"] = not differences and comparison["metrics_match"]
    write_json(output / "reload_consistency.json", comparison)
    if not comparison["matched"]:
        raise ValueError("fresh reload validation differs from selected checkpoint; see reload_consistency.json")
    failures = failure_taxonomy(predictions, config.failure_examples_per_category)
    with (output / "validation_predictions.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(output / "validation_metrics.json", metric)
    write_json(output / "failure_taxonomy.json", failures)
    result = {
        **resolved, **data_summary, "status": "formal_run_completed",
        "optimizer_steps": sum(h["optimizer_step_performed"] for h in history),
        "train_records_consumed": history[-1]["consumed_sample_count"], "epochs_completed": 1,
        "unsupervised_train_records": sum(h["sample_count"] - h["supervised_sample_count"] for h in history),
        "lora_B_before": before_weights, "lora_B_after": after_weights, "lora_parameters_updated": updated,
        "selected_checkpoint": selected, "reload_status": "success", "reload_lora_B": restored,
        "reload_validation_consistent": comparison["matched"],
        "formal_validation_model": "fresh_pinned_base_plus_selected_saved_adapter",
        "validation_records_consumed": len(predictions), "validation_metrics": metric,
        "generalization": generalization_summary(metric, baselines, config.lateral_collapse_straight_fraction),
        "validation_evaluation_performed": True,
    }
    write_json(output / "training_summary.json", result)
    return result

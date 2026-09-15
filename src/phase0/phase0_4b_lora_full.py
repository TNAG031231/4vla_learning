from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
import gc
import json
from itertools import zip_longest
import math
from pathlib import Path
import random
from time import perf_counter

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
    log_every_steps: int
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
    if type(config.log_every_steps) is not int or not 20 <= config.log_every_steps <= 50:
        raise ValueError("log_every_steps must be an integer between 20 and 50")
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


def synchronize_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def train_one_epoch(*, model: object, samples: list[SFTSample], collator: StructuredCollator,
                    optimizer: object, config: FullConfig, device: str,
                    on_step: Callable[[dict], None]) -> list[dict]:
    total = math.ceil(len(samples) / config.gradient_accumulation_steps)
    history = []
    consumed = 0
    micro_batches = 0
    optimizer_steps = 0
    model.train()
    optimizer.zero_grad()
    synchronize_device(device)
    training_started = perf_counter()
    callback_seconds = 0.0
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
            optimizer_steps += 1
        optimizer.zero_grad()
        synchronize_device(device)
        consumed += len(group)
        micro_batches += len(supervised)
        elapsed = perf_counter() - training_started - callback_seconds
        entry = {"step": step, "sample_count": len(group), "consumed_sample_count": consumed,
                 "supervised_sample_count": len(supervised), "optimizer_step_performed": bool(supervised),
                 "loss": loss_sum / len(supervised) if supervised else None, "learning_rate": lr,
                 "training_elapsed_seconds": elapsed, "micro_batches_completed": micro_batches,
                 "optimizer_steps_completed": optimizer_steps,
                 "mean_seconds_per_train_sample": elapsed / consumed,
                 "train_samples_per_second": consumed / elapsed if elapsed else None,
                 "estimated_remaining_training_seconds": elapsed / consumed * (len(samples) - consumed)}
        history.append(entry)
        if step % config.log_every_steps == 0 or step == total:
            print(json.dumps({"event": "training_progress", **entry}), flush=True)
        callback_started = perf_counter()
        on_step(entry)
        callback_seconds += perf_counter() - callback_started
        model.train()
    return history


def evaluate_validation(model: object, samples: list[SFTSample], collator: StructuredCollator,
                        config: FullConfig, device: str, runtime: object, *,
                        evaluation_name: str = "validation") -> tuple[list[dict], dict]:
    if not samples or any(s.split != "validation" for s in samples):
        raise ValueError("evaluation requires nonempty validation samples only")
    model.eval()
    synchronize_device(device)
    started = perf_counter()
    predictions = []
    with runtime.inference_context():
        for count, sample in enumerate(samples, 1):
            predictions.append(predict_sample(model, sample, collator, config.generation_kwargs,
                                              device, expected_split="validation"))
            if count % config.log_every_steps == 0 or count == len(samples):
                synchronize_device(device)
                elapsed = perf_counter() - started
                timing = {"evaluation_name": evaluation_name, "validation_samples_completed": count,
                          "validation_total_seconds": elapsed,
                          "mean_seconds_per_validation_sample": elapsed / count,
                          "validation_samples_per_second": count / elapsed if elapsed else None,
                          "estimated_remaining_validation_seconds": elapsed / count * (len(samples) - count)}
                print(json.dumps({"event": "validation_progress", **timing}), flush=True)
    return predictions, timing


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
    run_started = perf_counter()
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
    load_started = perf_counter()
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                config.attention_implementation, config.local_files_only)
    base.to(device)
    model = inject_lora(base, config=config, dependencies=runtime)
    synchronize_device(device)
    model_load_seconds = perf_counter() - load_started
    print(json.dumps({"event": "model_loaded", "stage": "training",
                      "model_load_seconds": model_load_seconds}), flush=True)
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
        "model_load_seconds": model_load_seconds,
        "timing_scope": "training excludes checkpoint/evaluation callbacks; validation includes preprocessing and generation; total ends before summary write",
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
                predictions, timing = evaluate_validation(
                    model, validation, collator, config, device, runtime,
                    evaluation_name=f"checkpoint_{entry['step']}",
                )
                metric = factorized_metrics(predictions)
                prediction_path = output / f"validation_step_{entry['step']:04d}.json"
                write_json(prediction_path, predictions)
                metadata = {"step": entry["step"], "adapter_path": str(checkpoint),
                            "metrics": metric, "selection_score": checkpoint_score(metric),
                            "validation_predictions_path": str(prediction_path),
                            "validation_sample_count": len(predictions),
                            **timing,
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
    load_started = perf_counter()
    fresh_base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype,
                                      config.attention_implementation, config.local_files_only)
    reloaded = runtime.adapter_loader(fresh_base, Path(selected["adapter_path"]))
    reloaded.to(device)
    reloaded.config.use_cache = False
    synchronize_device(device)
    fresh_model_load_seconds = perf_counter() - load_started
    print(json.dumps({"event": "model_loaded", "stage": "fresh_reload",
                      "model_load_seconds": fresh_model_load_seconds}), flush=True)
    restored = lora_b_report(reloaded)
    expected = selected["lora_B"]
    if (any(restored[key] != expected[key] for key in ("tensor_count", "nonzero_tensor_count", "nonzero_element_count"))
            or not math.isclose(restored["norm"], expected["norm"], rel_tol=1e-5, abs_tol=1e-8)):
        raise ValueError("selected adapter reload weight report mismatch")
    predictions, final_timing = evaluate_validation(
        reloaded, validation, collator, config, device, runtime, evaluation_name="fresh_reload",
    )
    write_json(output / "final_validation_timing.json", final_timing)
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
        "training_total_seconds": history[-1]["training_elapsed_seconds"],
        "mean_seconds_per_train_sample": history[-1]["mean_seconds_per_train_sample"],
        "micro_batches_completed": history[-1]["micro_batches_completed"],
        "fresh_model_load_seconds": fresh_model_load_seconds,
        "checkpoint_validation_timings": [{key: c[key] for key in (
            "step", "validation_total_seconds", "mean_seconds_per_validation_sample",
            "validation_samples_completed", "validation_samples_per_second",
        )} for c in checkpoints],
        "final_validation_timing": final_timing,
        "generalization": generalization_summary(metric, baselines, config.lateral_collapse_straight_fraction),
        "validation_evaluation_performed": True,
        "total_run_seconds": perf_counter() - run_started,
    }
    write_json(output / "training_summary.json", result)
    return result

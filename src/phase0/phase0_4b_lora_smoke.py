from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random

import yaml

from src.phase0 import development_projection as development
from src.phase0.phase0_4_source_projection import load_config as load_source_config
from src.phase0.phase0_4b_protocol import (
    PARSER_VERSION, PROMPT_VERSION, SERIALIZATION_VERSION, StructuredCollator,
    SFTSample, adapt_record, parse_action,
)
from src.phase0.qwen3vl_dataset_adapter import (
    load_config as load_ego_config, resolve_derived_path, validate_git_provenance,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import (
    LORA_TARGET_MODULES, _move_batch, default_runtime_dependencies, inject_lora,
    lora_config_kwargs, run_training_steps, trainable_parameter_report,
)
from src.phase0.scene_mapping import read_scene_mapping


@dataclass(frozen=True)
class SmokeConfig:
    train_subset_size: int
    seed: int
    max_steps: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    lora_task_type: str
    lora_target_modules: tuple[str, ...]
    local_files_only: bool
    generation_kwargs: dict
    output_relative_dir: str


def load_config(path: Path) -> SmokeConfig:
    values = yaml.safe_load(path.read_text())
    values["lora_target_modules"] = tuple(values["lora_target_modules"])
    config = SmokeConfig(**values)
    if not 1 <= config.train_subset_size <= 24 or not 1 <= config.max_steps <= 200:
        raise ValueError("Phase 0.4b-A requires a tiny subset (<=24) and <=200 steps")
    if config.gradient_accumulation_steps < 1 or not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("optimization settings must be positive and finite")
    if (config.lora_r, config.lora_alpha, config.lora_dropout, config.lora_bias,
        config.lora_task_type, config.lora_target_modules) != (
        8, 16, 0.05, "none", "CAUSAL_LM", LORA_TARGET_MODULES,
    ):
        raise ValueError("LoRA settings must match the approved Phase 0.3 configuration")
    if config.generation_kwargs != {"do_sample": False, "num_beams": 1, "max_new_tokens": 24}:
        raise ValueError("structured generation must match deterministic v0.1")
    return config


def load_samples(repository: Path, derived_root: Path, *, split: str = "train") -> list[SFTSample]:
    if split != "train":
        raise ValueError("Phase 0.4b-A only permits train")
    source = load_source_config(repository / "configs/phase0_4_source_projection.yaml", repository)
    mapping = read_scene_mapping(derived_root / source.source_contract.scene_mapping_relative_path)
    selection = development.select_development_scenes(mapping, source.source_contract)
    temporal = yaml.safe_load((repository / "configs/phase0_4_temporal_dataset.yaml").read_text())
    path = resolve_derived_path(derived_root, temporal["output_relative_dir"]) / "train.jsonl"
    ego_config = load_ego_config(repository / "configs/phase0_3_dataset_adapter.yaml")
    samples, seen = [], set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train" or row["scene_token"] not in selection.scene_tokens_by_split["train"]:
                raise ValueError("temporal record outside frozen train scenes")
            if row["split_mapping_sha256"] != mapping["scene_split_mapping_sha256"]:
                raise ValueError("temporal record split mapping mismatch")
            if row["history_length"] != temporal["history_length"]:
                raise ValueError("temporal history length mismatch")
            sample = adapt_record(row, ego_config)
            if sample.sample_token in seen:
                raise ValueError("duplicate temporal train sample")
            seen.add(sample.sample_token)
            samples.append(sample)
    if len(samples) != source.source_contract.expected_sample_counts["train"]:
        raise ValueError("temporal train count differs from frozen producer")
    return samples


def select_tiny(samples: list[SFTSample], size: int, seed: int) -> list[SFTSample]:
    if any(sample.split != "train" for sample in samples):
        raise ValueError("tiny subset only permits train")
    eligible = sorted((s for s in samples if s.target.longitudinal_valid or s.target.lateral_valid),
                      key=lambda s: s.sample_token)
    if not 1 <= size <= min(24, len(eligible)):
        raise ValueError("insufficient supervised samples for tiny subset")
    random.Random(seed).shuffle(eligible)
    groups = defaultdict(list)
    for sample in eligible:
        groups[(sample.target.longitudinal, sample.target.lateral)].append(sample)
    selected = [group[0] for group in groups.values()][:size]
    tokens = {sample.sample_token for sample in selected}
    selected.extend([s for s in eligible if s.sample_token not in tokens][:size - len(selected)])
    return selected


def metrics(predictions: list[dict]) -> dict:
    total = len(predictions)
    valid = sum(row["parsed_action"] is not None for row in predictions)
    result = {"sample_count": total, "parser_success_rate": valid / total,
              "invalid_output_rate": 1 - valid / total}
    for direction in ("longitudinal", "lateral", "joint"):
        axes = ("longitudinal", "lateral") if direction == "joint" else (direction,)
        eligible = [r for r in predictions if all(r["target"][f"{a}_valid"] for a in axes)]
        correct = sum(r["parsed_action"] is not None and all(
            r["parsed_action"][a] == r["target"][a] for a in axes) for r in eligible)
        result[f"{direction}_count"] = len(eligible)
        result[f"{direction}_accuracy"] = correct / len(eligible) if eligible else None
    return result


def evaluate(model: object, samples: list[SFTSample], collator: StructuredCollator,
             config: SmokeConfig, device: str, runtime: object) -> tuple[float, list[dict]]:
    model.eval()
    total_loss, token_count, predictions = 0.0, 0, []
    with runtime.inference_context():
        for sample in samples:
            batch = collator([sample], expected_split="train")
            count = int((batch["labels"] != -100).sum().item())
            loss = float(model(**_move_batch(batch, device)).loss.item())
            if not math.isfinite(loss):
                raise ValueError("evaluation loss must be finite")
            total_loss += loss * count
            token_count += count
            inputs = collator.processor.apply_chat_template(
                collator.messages(sample), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            output = model.generate(**_move_batch(inputs, device), **config.generation_kwargs)
            raw = collator.processor.batch_decode(
                output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            predictions.append({"sample_token": sample.sample_token, "split": sample.split,
                                "target": asdict(sample.target), "raw_output": raw,
                                "parsed_action": parse_action(raw)})
    return total_loss / token_count, predictions


def lora_b_report(model: object) -> dict:
    tensors = [parameter.detach() for name, parameter in model.named_parameters() if "lora_B" in name]
    return {"tensor_count": len(tensors),
            "nonzero_tensor_count": sum(bool(t.count_nonzero().item()) for t in tensors),
            "nonzero_element_count": sum(int(t.count_nonzero().item()) for t in tensors),
            "norm": math.sqrt(sum(float(t.float().square().sum().item()) for t in tensors))}


def run_smoke(*, repository: Path, dataset_root: Path, derived_root: Path,
              config: SmokeConfig, git_provenance: object, split: str = "train") -> dict:
    if split != "train":
        raise ValueError("Phase 0.4b-A only permits train")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("smoke artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    samples = select_tiny(load_samples(repository, derived_root), config.train_subset_size, config.seed)
    import torch

    torch.manual_seed(config.seed)
    runtime = default_runtime_dependencies()
    device, dtype = runtime.device_selector("cuda:0"), runtime.dtype_selector("bfloat16")
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, config.local_files_only)
    collator = StructuredCollator(processor, runtime.image_loader, dataset_root)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, "sdpa", config.local_files_only)
    base.to(device)
    initial_loss, before = evaluate(base, samples, collator, config, device, runtime)
    model = inject_lora(base, config=config, dependencies=runtime)
    parameter_report = trainable_parameter_report(model, config.lora_target_modules)
    before_weights = lora_b_report(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = runtime.optimizer_factory(
        [p for p in model.parameters() if p.requires_grad], config.learning_rate,
    )
    history = run_training_steps(model=model, samples=samples, collator=collator,
                                 optimizer=optimizer, config=config, device=device)
    after_weights = lora_b_report(model)
    output.mkdir(parents=True)
    checkpoint = output / "adapter_checkpoint"
    model.save_pretrained(checkpoint)
    del optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, "sdpa", config.local_files_only)
    reloaded = runtime.adapter_loader(fresh_base, checkpoint)
    reloaded.to(device)
    final_loss, after = evaluate(reloaded, samples, collator, config, device, runtime)
    before_metrics, after_metrics = metrics(before), metrics(after)
    updated = (before_weights["nonzero_element_count"] == 0
               and after_weights["nonzero_element_count"] > 0 and after_weights["norm"] > 0)
    joint_before, joint_after = before_metrics["joint_accuracy"], after_metrics["joint_accuracy"]
    improved = joint_before is not None and joint_after is not None and joint_after > joint_before
    passed = final_loss < initial_loss and updated and improved and after_metrics["parser_success_rate"] == 1
    result = {
        "status": "smoke_passed" if passed else "smoke_completed_without_learning_gate",
        "run_kind": "exact_tiny_train_overfit", "generalization_evaluated": False,
        "model_id": FIXED_MODEL_ID, "model_revision": FIXED_REVISION, "processor_revision": FIXED_REVISION,
        "execution_git_commit": git.commit, "prompt_version": PROMPT_VERSION,
        "serialization_version": SERIALIZATION_VERSION, "parser_version": PARSER_VERSION,
        "train_sample_tokens": [s.sample_token for s in samples], "tiny_subset_size": len(samples),
        "lora_config": lora_config_kwargs(config), "trainable_parameters": parameter_report,
        "optimizer_steps": len(history), "training_loss_history": history,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_measurement": "eval_mode_supervised_token_mean_on_exact_tiny_train_subset",
        "before_metrics": before_metrics, "after_metrics": after_metrics,
        "predictions_before": before, "predictions_after_reload": after,
        "lora_B_before": before_weights, "lora_B_after": after_weights, "lora_parameters_updated": updated,
        "checkpoint_path": str(checkpoint), "full_model_saved": False, "reload_status": "success",
        "test_records_read": 0, **asdict(development.IsolationCounters()),
        "test_evaluation_performed": False, "validation_evaluation_performed": False,
        "config": asdict(config), "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
    }
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

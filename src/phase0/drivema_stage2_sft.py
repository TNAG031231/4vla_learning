from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
import yaml

from src.phase0.drivema_stage2_protocol import (
    PROTOCOL_VERSION, AssistantSpan, assistant_log_probs, encode_teacher_forcing, generate_two_turn,
)
from src.phase0.phase0_4b_evaluation import factorized_metrics
from src.phase0.phase0_4b_lora_full import load_config as load_action_config
from src.phase0.phase0_4b_lora_smoke import select_tiny
from src.phase0.phase0_4b_protocol import SFTSample, StructuredCollator
from src.phase0.phase0_4c_evaluation import aggregate_metrics, compare_reload, prediction_metrics
from src.phase0.phase0_4c_full_train import prepare_data, write_predictions
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance, resolve_derived_path
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import _move_batch, default_runtime_dependencies


@dataclass(frozen=True)
class Stage2Config:
    seed: int
    train_subset_size: int
    validation_subset_size: int
    max_steps: int
    learning_rate: float
    gradient_accumulation_steps: int
    trajectory_max_new_tokens: int
    tiny_loss_ratio: float
    tiny_ade_threshold_m: float
    selected_adapter_relative_path: str
    output_relative_dir: str


def load_config(path: Path) -> Stage2Config:
    config = Stage2Config(**yaml.safe_load(path.read_text()))
    if not (1 <= config.train_subset_size <= 8 and 1 <= config.validation_subset_size <= 8
            and 1 <= config.max_steps <= 200 and config.gradient_accumulation_steps >= 1
            and 1 <= config.trajectory_max_new_tokens <= 512):
        raise ValueError("Stage 2 currently permits tiny train/validation smoke only")
    if (not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not 0 < config.tiny_loss_ratio < 1
            or not math.isfinite(config.tiny_ade_threshold_m) or config.tiny_ade_threshold_m <= 0):
        raise ValueError("invalid optimization or tiny-overfit gate")
    if config.selected_adapter_relative_path != (
        "phase_0_4/structured_action_lora_full_v0_1/adapter_step_3564"
    ):
        raise ValueError("Stage 2 must initialize from the selected step-3564 adapter")
    return config


class Stage2Runner:
    def __init__(self, processor: object, collator: StructuredCollator, device: str,
                 action_generation: dict, trajectory_max_new_tokens: int) -> None:
        self.processor = processor
        self.collator = collator
        self.device = device
        self.action_generation = action_generation
        self.trajectory_max_new_tokens = trajectory_max_new_tokens

    def images(self, sample: SFTSample) -> list[object]:
        messages = self.collator.messages(sample, expected_split=sample.split)
        return [item["image"] for item in messages[0]["content"] if item["type"] == "image"]

    def training_batch(self, sample: SFTSample, record: dict) -> tuple[dict, list[AssistantSpan]]:
        if sample.split != "train" or record["split"] != "train":
            raise ValueError("SFT updates only permit train samples")
        inputs, spans = encode_teacher_forcing(
            self.processor, sample.observation, self.images(sample),
            sample.target.training_text(), record["future_waypoints"],
        )
        return _move_batch(inputs, self.device), spans

    def predict(self, model: nn.Module, sample: SFTSample) -> dict:
        return generate_two_turn(
            model=model, processor=self.processor, observation=sample.observation,
            images=self.images(sample), device=self.device, action_generation=self.action_generation,
            trajectory_max_new_tokens=self.trajectory_max_new_tokens,
        )


def evaluate(model: nn.Module, runner: Stage2Runner, samples: list[SFTSample],
             records: dict[str, dict]) -> tuple[list[dict], dict]:
    rows = []
    for sample in samples:
        prediction = runner.predict(model, sample)
        record = records[sample.sample_token]
        points = prediction.pop("waypoints")
        metrics = prediction_metrics(
            torch.tensor(points) if points is not None else None,
            torch.tensor(record["future_waypoints"]), torch.tensor(record["trajectory_valid_mask"]),
        )
        if prediction["invalid_reason"] is not None:
            metrics["invalid_reason"] = prediction["invalid_reason"]
        rows.append({
            "sample_token": sample.sample_token, "scene_token": sample.scene_token,
            "split": sample.split, "target": asdict(sample.target),
            "target_waypoints": record["future_waypoints"],
            "trajectory_valid_mask": record["trajectory_valid_mask"], **prediction, **metrics,
        })
    return rows, {"trajectory": aggregate_metrics(rows, "predicted_action"),
                  "action": factorized_metrics(rows)}


def teacher_forced_losses(model: nn.Module, runner: Stage2Runner, samples: list[SFTSample],
                          records: dict[str, dict]) -> dict:
    model.eval()
    totals = [0.0, 0.0]
    counts = [0, 0]
    with torch.no_grad():
        for sample in samples:
            batch, spans = runner.training_batch(sample, records[sample.sample_token])
            for index, values in enumerate(assistant_log_probs(model, batch, spans)):
                totals[index] -= float(values.sum())
                counts[index] += values.numel()
    return {"action_ce": totals[0] / counts[0], "trajectory_ce": totals[1] / counts[1],
            "token_mean_ce": sum(totals) / sum(counts)}


def fit(*, model: nn.Module, runner: Stage2Runner, train: list[SFTSample],
        validation: list[SFTSample], records: dict[str, dict], config: Stage2Config,
        output: Path, metadata: dict) -> dict:
    from peft import PeftModel

    if any(s.split != "train" for s in train) or any(s.split != "validation" for s in validation):
        raise ValueError("training and validation samples must match their split")
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("only the loaded LoRA parameters may be updated")
    output.mkdir(parents=True, exist_ok=False)
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    write_json(output / "metadata.json", {
        **metadata, "protocol_version": PROTOCOL_VERSION, "config": asdict(config),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "train_sample_tokens": [sample.sample_token for sample in train],
        "validation_sample_tokens": [sample.sample_token for sample in validation],
        "supervision": "both_assistant_contents_and_EOS; token_mean_cross_entropy",
    })
    initial = teacher_forced_losses(model, runner, train, records)
    optimizer = torch.optim.AdamW(trainable.values(), lr=config.learning_rate, weight_decay=0.0)
    nonzero_gradient_steps = 0
    with (output / "training_history.jsonl").open("w") as stream:
        for step in range(config.max_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset in range(config.gradient_accumulation_steps):
                index = (step * config.gradient_accumulation_steps + offset) % len(train)
                sample = train[index]
                batch, spans = runner.training_batch(sample, records[sample.sample_token])
                values = assistant_log_probs(model, batch, spans)
                loss = -torch.cat(values).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite Stage 2 token loss")
                (loss / config.gradient_accumulation_steps).backward()
                losses.append(float(loss.detach()))
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
            nonzero_gradient_steps += int(gradient_norm > 0)
            optimizer.step()
            row = {"step": step + 1, "loss": sum(losses) / len(losses),
                   "gradient_norm": float(gradient_norm)}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
    del optimizer
    final = teacher_forced_losses(model, runner, train, records)
    delta = sum(float((parameter.detach().cpu() - before[name]).square().sum())
                for name, parameter in trainable.items()) ** 0.5
    if delta == 0 or nonzero_gradient_steps == 0:
        raise ValueError("LoRA did not receive a real parameter update")
    train_rows, train_metrics = evaluate(model, runner, train, records)
    validation_rows, validation_metrics = evaluate(model, runner, validation, records)
    write_predictions(output / "train_predictions.jsonl", train_rows)
    write_predictions(output / "validation_predictions.jsonl", validation_rows)
    write_json(output / "validation_metrics.json", validation_metrics)
    model.save_pretrained(output / "adapter")
    runner.processor.save_pretrained(output / "adapter")
    # The unchanged base is reused; adapter tensors are loaded from the saved directory.
    restored = PeftModel.from_pretrained(model.unload(), output / "adapter").to(runner.device)
    reloaded_rows, _ = evaluate(restored, runner, validation, records)
    reload_report = compare_reload(validation_rows, reloaded_rows)
    reload_report["token_ids_match"] = all(
        before_row["input_ids"] == after_row["input_ids"]
        for before_row, after_row in zip(validation_rows, reloaded_rows, strict=True)
    )
    reload_report["log_probs_match"] = all(
        torch.allclose(torch.tensor(a["token_log_probs"]), torch.tensor(b["token_log_probs"]),
                       atol=1e-4, rtol=1e-4)
        for first, second in zip(validation_rows, reloaded_rows, strict=True)
        for a, b in zip(first["assistant_spans"], second["assistant_spans"], strict=True)
    ) if reload_report["token_ids_match"] else False
    write_json(output / "reload_consistency.json", reload_report)
    trajectory = train_metrics["trajectory"]
    tiny_passed = (
        final["trajectory_ce"] <= initial["trajectory_ce"] * config.tiny_loss_ratio
        and trajectory["invalid_prediction_count"] == 0
        and trajectory["ade_3s_m"] <= config.tiny_ade_threshold_m
    )
    passed = (tiny_passed and validation_metrics["trajectory"]["invalid_prediction_count"] == 0
              and reload_report["reload_consistency"] and reload_report["token_ids_match"]
              and reload_report["log_probs_match"])
    summary = {
        "status": "smoke_passed" if passed else "smoke_failed",
        "initial_teacher_forced_losses": initial, "final_teacher_forced_losses": final,
        "lora_delta_l2": delta, "nonzero_gradient_steps": nonzero_gradient_steps,
        "tiny_overfit_passed": tiny_passed, "train_predicted_action_metrics": train_metrics,
        "validation_predicted_action_metrics": validation_metrics,
        "reload_consistency": reload_report["reload_consistency"],
        "formal_training_config": "pending_real_GPU_smoke_review",
    }
    write_json(output / "summary.json", summary)
    return summary


def run(*, repository: Path, dataset_root: Path, derived_root: Path, config: Stage2Config) -> dict:
    from peft import PeftModel

    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("training outputs must be outside the repository")
    if output.exists():
        raise FileExistsError(f"Stage 2 output already exists: {output}")
    runtime = default_runtime_dependencies()
    device = runtime.device_selector("cuda:0")
    dtype = runtime.dtype_selector("bfloat16")
    if dtype != torch.bfloat16:
        raise ValueError("the selected adapter requires a BF16-capable GPU")
    train, validation, records, data_summary = prepare_data(repository, derived_root)
    train = select_tiny(train, config.train_subset_size, config.seed)
    validation = sorted(validation, key=lambda sample: sample.sample_token)[:config.validation_subset_size]
    semantic = load_action_config(repository / "configs/phase0_4b_lora_full.yaml")
    torch.manual_seed(config.seed)
    processor = runtime.processor_loader(FIXED_MODEL_ID, FIXED_REVISION, True)
    base = runtime.model_loader(FIXED_MODEL_ID, FIXED_REVISION, dtype, "sdpa", True)
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = StructuredCollator(processor, runtime.image_loader, dataset_root)
    runner = Stage2Runner(processor, collator, device, semantic.generation_kwargs,
                          config.trajectory_max_new_tokens)
    return fit(model=model, runner=runner, train=train, validation=validation, records=records,
               config=config, output=output, metadata={
                   "run_kind": "real_GPU_stage2_tiny_SFT", "model_id": FIXED_MODEL_ID,
                   "model_revision": FIXED_REVISION, "processor_revision": FIXED_REVISION,
                   "execution_git_commit": collect_git_provenance(repository).commit,
                   "initial_adapter": config.selected_adapter_relative_path,
                   "data_intake": data_summary,
                   "transformers_version": runtime.package_version("transformers"),
                   "peft_version": runtime.package_version("peft"),
               })

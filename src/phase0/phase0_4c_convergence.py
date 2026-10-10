from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
import yaml

from src.phase0 import phase0_4c_full_train as full
from src.phase0.phase0_4b_lora_full import epoch_groups, load_config as load_semantic_config
from src.phase0.phase0_4b_protocol import PARSER_VERSION, PROMPT_VERSION, SERIALIZATION_VERSION, SFTSample
from src.phase0.phase0_4c_evaluation import aggregate_metrics, compare_reload
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder, freeze_backbone, masked_waypoint_loss
from src.phase0.qwen3vl_dataset_adapter import (
    GitProvenance, resolve_derived_path, validate_git_provenance,
)
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from src.phase0.qwen3vl_lora_smoke import default_runtime_dependencies

if TYPE_CHECKING:
    from src.phase0.phase0_4c_direct_waypoint import DirectSample
    from src.phase0.phase0_4c_no_action import NeutralRunner


EPOCH1_OPTIMIZER_STEP = 3564
DIAGNOSTIC_METRICS = ("ade_m", "fde_m", "ade_1s_m", "ade_2s_m", "ade_3s_m",
                      "fde_1s_m", "fde_2s_m", "fde_3s_m")


@dataclass(frozen=True)
class ConvergenceConfig(full.FullConfig):
    validation_schedule: str
    reproduction_rtol: float
    extension_min_relative_improvement: float
    expected_validation_count: int
    expected_motion_unavailable_count: int
    epoch1_reference: dict[str, float]
    mlp_reference: dict[str, float]
    motion_planner_reference: dict[str, float]
    motion_mlp_reference: dict[str, float]


def load_config(path: Path) -> ConvergenceConfig:
    config = ConvergenceConfig(train_subset_size=0, **yaml.safe_load(path.read_text()))
    formal = full.load_config(Path(__file__).resolve().parents[2] / "configs/phase0_4c_full_train.yaml")
    changes = {
        "protocol_version": "phase0.4c-action-conditioned-convergence-v0.1",
        "output_relative_dir": "phase_0_4/action_conditioned_convergence_v0_1",
        "num_train_epochs": 3, "checkpoint_interval": 1, "validation_interval": 1,
    }
    for key, value in asdict(formal).items():
        if getattr(config, key) != changes.get(key, value):
            raise ValueError(f"{key} differs from frozen convergence protocol")
    if config.validation_schedule != "epoch_end":
        raise ValueError("validation_schedule must be epoch_end")
    for key in ("reproduction_rtol", "extension_min_relative_improvement"):
        value = getattr(config, key)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"invalid {key}")
    for key in ("expected_validation_count", "expected_motion_unavailable_count"):
        if type(getattr(config, key)) is not int or getattr(config, key) < 1:
            raise ValueError(f"invalid {key}")
    horizons = {f"{metric}_{seconds}s_m" for metric in ("ade", "fde") for seconds in (1, 2, 3)}
    for key in ("epoch1_reference", "mlp_reference", "motion_planner_reference", "motion_mlp_reference"):
        reference = getattr(config, key)
        expected = horizons if key == "epoch1_reference" else {"ade_3s_m", "fde_3s_m"}
        if set(reference) != expected or any(not math.isfinite(v) or v <= 0 for v in reference.values()):
            raise ValueError(f"invalid {key}")
    return config


def deltas(metrics: dict, reference: dict) -> dict:
    return {key: metrics[key] - value if metrics[key] is not None and value is not None else None
            for key, value in reference.items()}


def reproduction_gate(metrics: dict, config: ConvergenceConfig) -> dict:
    difference = deltas(metrics, config.epoch1_reference)
    checks = {key: delta is not None and abs(delta) <= config.reproduction_rtol * config.epoch1_reference[key]
              for key, delta in difference.items()}
    checks["full_valid_coverage"] = (metrics["sample_count"] == config.expected_validation_count
                                   and metrics["invalid_prediction_count"] == 0)
    return {"passed": all(checks.values()), "checks": checks, "new_minus_old": difference,
            "relative_tolerance": config.reproduction_rtol, "reference": config.epoch1_reference}


def extension_gate(epochs: list[dict], config: ConvergenceConfig) -> dict:
    second, third = epochs[1]["metrics"], epochs[2]["metrics"]
    improvements = {key: (second[key] - third[key]) / second[key]
                    if second[key] is not None and second[key] > 0 and third[key] is not None else None
                    for key in ("ade_3s_m", "fde_3s_m")}
    allowed = (third["invalid_prediction_count"] <= second["invalid_prediction_count"]
               and all(v is not None and v >= config.extension_min_relative_improvement
                       for v in improvements.values()))
    return {"allowed": allowed, "epoch2_to_epoch3_relative_improvement": improvements,
            "minimum_relative_improvement": config.extension_min_relative_improvement}


def motion_subset(rows: list[dict], records: dict, config: ConvergenceConfig) -> dict:
    subset = [row for row in rows
              if records[row["sample_token"]]["ego_motion_history"][-1]["availability"] == "unavailable"]
    metrics = aggregate_metrics(subset, "predicted_action")
    if len(subset) != config.expected_motion_unavailable_count:
        raise ValueError("motion-unavailable subset count differs from frozen reference")
    return {"metrics": metrics, "sample_tokens": [r["sample_token"] for r in subset],
            "minus_old_planner": deltas(metrics, config.motion_planner_reference),
            "minus_mlp": deltas(metrics, config.motion_mlp_reference)}


def save_checkpoint(path: Path, planner: WaypointDecoder, optimizer: torch.optim.Optimizer,
                    config: ConvergenceConfig, provenance: dict, epoch: int, step: int,
                    epochs: list[dict]) -> None:
    full.save_checkpoint(path, planner, config, provenance, step)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved.update(optimizer_state_dict=optimizer.state_dict(), epoch=epoch, epochs=epochs,
                 torch_rng_state=torch.get_rng_state(),
                 cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(saved, path)


def load_resume(output: Path, config: ConvergenceConfig) -> dict:
    checkpoint = output / "epoch_3.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "optimizer_state_dict" not in saved or "epoch" not in saved:
        raise ValueError("resume requires a convergence optimizer-state checkpoint")
    if saved["training_config"] != asdict(config) or saved["epoch"] != 3:
        raise ValueError("resume requires this protocol's epoch-3 optimizer checkpoint")
    if not extension_gate(saved["epochs"], config)["allowed"]:
        raise ValueError("epoch-4/5 extension gate failed; STOP at epoch 3")
    if not saved["optimizer_state_dict"]["state"]:
        raise ValueError("resume requires saved AdamW state")
    if (output / "extension_metadata.json").exists() or any(output.glob("epoch_[45]*")):
        raise FileExistsError("epoch-4/5 artifacts already exist")
    summary = json.loads((output / "training_summary.json").read_text())
    if summary["status"] != "convergence_training_completed" or len(summary["epochs"]) != 3:
        raise ValueError("resume requires a successfully completed three-epoch run")
    if not reproduction_gate(saved["epochs"][0]["metrics"], config)["passed"]:
        for evidence in (saved["provenance"], summary):
            if (evidence.get("historical_epoch1_reproduction_passed") is not False
                    or evidence.get("continuation_after_failed_historical_reproduction") is not True):
                raise ValueError("epoch-1 reproduction failed without explicit continuation evidence")
    return saved


def load_failed_epoch1(output: Path, config: ConvergenceConfig) -> dict:
    required = ("epoch_1.pt", "epoch_1_metrics.json", "epoch_1_predictions.jsonl",
                "epoch1_reproduction.json", "epoch_comparison.json", "resolved_config.json",
                "run_metadata.json", "training_history.jsonl")
    for name in required:
        if not (output / name).is_file():
            raise ValueError(f"failed-epoch1 continuation requires {name}")
    if (any(output.glob("epoch_[2345]*")) or (output / "training_summary.json").exists()
            or (output / "continuation_metadata.json").exists()
            or (output / "extension_metadata.json").exists()):
        raise ValueError("continuation rejects existing epoch-2/3 or later/partial run artifacts")
    gate = json.loads((output / "epoch1_reproduction.json").read_text())
    if gate["passed"] is not False:
        raise ValueError("special continuation requires failed epoch-1 reproduction")
    saved = torch.load(output / "epoch_1.pt", map_location="cpu", weights_only=True)
    for key in ("epoch", "optimizer_step", "planner_state_dict", "optimizer_state_dict",
                "torch_rng_state", "cuda_rng_state", "training_config", "provenance", "epochs"):
        if key not in saved:
            raise ValueError(f"epoch-1 checkpoint missing {key}")
    if saved["epoch"] != 1 or saved["optimizer_step"] != EPOCH1_OPTIMIZER_STEP:
        raise ValueError("epoch-1 checkpoint epoch/optimizer_step mismatch")
    if not saved["planner_state_dict"] or not saved["optimizer_state_dict"].get("state"):
        raise ValueError("continuation requires planner and nonempty AdamW state")
    if (not isinstance(saved["torch_rng_state"], torch.Tensor)
            or saved["torch_rng_state"].numel() == 0 or not isinstance(saved["cuda_rng_state"], list)):
        raise ValueError("continuation requires valid torch/CUDA RNG state")
    if torch.cuda.is_available() and len(saved["cuda_rng_state"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG state does not match current devices")
    resolved = json.loads((output / "resolved_config.json").read_text())
    if saved["training_config"] != asdict(config) or resolved != asdict(config):
        raise ValueError("continuation training config mismatch")
    metrics = json.loads((output / "epoch_1_metrics.json").read_text())
    epochs = saved["epochs"]
    if (len(epochs) != 1 or epochs[0]["epoch"] != 1
            or epochs[0]["optimizer_step"] != EPOCH1_OPTIMIZER_STEP
            or epochs[0]["metrics"] != metrics):
        raise ValueError("checkpoint epoch-1 history/metrics mismatch")
    if (metrics["sample_count"] != config.expected_validation_count
            or metrics["valid_prediction_count"] != config.expected_validation_count
            or metrics["invalid_prediction_count"] != 0):
        raise ValueError("epoch-1 validation coverage mismatch")
    if reproduction_gate(metrics, config)["passed"]:
        raise ValueError("special continuation requires failed historical reproduction metrics")
    return saved


def train_group(planner: WaypointDecoder, runner: full.PlannerRunner | NeutralRunner,
                group: list[SFTSample] | list[DirectSample],
                records: dict, config: full.FullConfig, optimizer: torch.optim.Optimizer,
                conditioning_type: str = "gt_action_teacher_forced_train") -> float:
    planner.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for sample in group:
        prediction, _ = runner.predict(planner, sample, conditioning_type)
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
    return loss_sum


def fit(*, model: nn.Module, planner: WaypointDecoder, runner: full.PlannerRunner,
        train: list[SFTSample], validation: list[SFTSample], records: dict,
        config: ConvergenceConfig, output: Path, provenance: dict, resume: dict | None = None,
        continue_after_epoch1_reproduction_fail: bool = False) -> dict:
    if not train or any(s.split != "train" for s in train):
        raise ValueError("optimization requires train samples only")
    if not validation or any(s.split != "validation" for s in validation):
        raise ValueError("evaluation requires validation samples only")
    if len(validation) != config.expected_validation_count:
        raise ValueError("validation sample count differs from frozen reference")
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    provenance = {**provenance, "freeze_policy": full.verify_freeze(model, planner, optimizer),
                  "Direct multi-epoch control": "NOT RUN"}
    if continue_after_epoch1_reproduction_fail and (resume is None or resume["epoch"] != 1):
        raise ValueError("explicit failed-reproduction continuation requires epoch 1")
    if resume is not None and resume["epoch"] == 1 and not continue_after_epoch1_reproduction_fail:
        raise ValueError("epoch-1 resume requires explicit failed-reproduction continuation")
    start, step, epochs = 0, 0, []
    if resume is not None:
        if resume["provenance"]["data"] != provenance["data"]:
            raise ValueError("resume data provenance differs from original run")
        planner.load_state_dict(resume["planner_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start, step, epochs = resume["epoch"], resume["optimizer_step"], resume["epochs"]
        torch.set_rng_state(resume["torch_rng_state"])
        if resume["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        provenance["initial_run_provenance"] = resume["provenance"]
        provenance["historical_epoch1_reproduction_passed"] = reproduction_gate(epochs[0]["metrics"], config)["passed"]
        provenance["continuation_after_failed_historical_reproduction"] = (
            continue_after_epoch1_reproduction_fail
            or resume["provenance"].get("continuation_after_failed_historical_reproduction", False))
    end = 5 if start == 3 else config.num_train_epochs
    provenance["convergence_evidence_scope"] = f"within_run_epoch1_to_epoch{end}"
    metadata = ("continuation_metadata.json" if continue_after_epoch1_reproduction_fail
                else "extension_metadata.json" if resume else "run_metadata.json")
    write_json(output / metadata, provenance)
    with (output / "training_history.jsonl").open("a" if resume else "w") as history:
        for epoch in range(start, end):
            epoch_loss, consumed = 0., 0
            for group in epoch_groups(train, config.gradient_accumulation_steps, config.seed + epoch):
                loss_sum = train_group(planner, runner, group, records, config, optimizer)
                step += 1
                consumed += len(group)
                epoch_loss += loss_sum
                entry = {"epoch": epoch + 1, "optimizer_step": step, "train_loss": loss_sum / len(group),
                         "sample_tokens": [s.sample_token for s in group], "learning_rate": config.learning_rate}
                history.write(json.dumps(entry, allow_nan=False) + "\n")
                history.flush()
                if step % config.log_every_steps == 0:
                    print(json.dumps(entry), flush=True)
            metrics, rows = full.evaluate(planner, runner, validation, records, "predicted_action")
            name = f"epoch_{epoch + 1}"
            full.write_predictions(output / f"{name}_predictions.jsonl", rows)
            write_json(output / f"{name}_metrics.json", metrics)
            subset = motion_subset(rows, records, config)
            entry = {"epoch": epoch + 1, "optimizer_step": step, "checkpoint": f"{name}.pt",
                     "train_loss": epoch_loss / consumed, "metrics": metrics,
                     "minus_epoch1": deltas(metrics, {k: epochs[0]["metrics"][k] if epochs else metrics[k]
                                                      for k in config.epoch1_reference}),
                     "minus_mlp": deltas(metrics, config.mlp_reference), "motion_unavailable": subset}
            subset["minus_epoch1"] = deltas(subset["metrics"], {
                k: epochs[0]["motion_unavailable"]["metrics"][k] if epochs else subset["metrics"][k]
                for k in config.motion_planner_reference})
            entry["overall_improvement_with_motion_regression"] = {
                key: (entry["minus_epoch1"][key] < 0 and subset["minus_epoch1"][key] > 0)
                if entry["minus_epoch1"][key] is not None and subset["minus_epoch1"][key] is not None else None
                for key in config.motion_planner_reference}
            if epochs:
                keys = (*config.epoch1_reference, "ade_m", "fde_m", "invalid_prediction_count")
                entry["minus_epoch1"] = deltas(metrics, {k: epochs[0]["metrics"][k] for k in keys})
                entry["minus_previous_epoch"] = deltas(metrics, {k: epochs[-1]["metrics"][k] for k in keys})
                subset["minus_previous_epoch"] = deltas(subset["metrics"], {
                    k: epochs[-1]["motion_unavailable"]["metrics"][k] for k in config.motion_planner_reference})
            else:
                gate = reproduction_gate(metrics, config)
                provenance["historical_epoch1_reproduction_passed"] = gate["passed"]
                provenance["continuation_after_failed_historical_reproduction"] = False
            epochs.append(entry)
            save_checkpoint(output / f"{name}.pt", planner, optimizer, config, provenance, epoch + 1, step, epochs)
            write_json(output / "epoch_comparison.json", epochs)
            print(json.dumps(entry, allow_nan=False), flush=True)
            if epoch == 0:
                write_json(output / "epoch1_reproduction.json", gate)
                if not gate["passed"]:
                    raise ValueError("epoch-1 reproduction failed; STOP before epoch 2")
            if not metrics["valid_prediction_count"]:
                raise ValueError("formal validation produced no valid trajectories")
    best = min(epochs, key=lambda e: (e["metrics"]["invalid_prediction_count"], e["metrics"]["ade_m"],
                                     e["metrics"]["fde_m"], e["optimizer_step"]))
    selected = full.reload_planner(output / best["checkpoint"], runner.device)
    rows = [json.loads(line) for line in (output / f"epoch_{best['epoch']}_predictions.jsonl").read_text().splitlines()]
    by_token = {r["sample_token"]: r for r in rows}
    subset = sorted((s for s in validation if by_token[s.sample_token]["prediction_valid"]),
                    key=lambda s: s.sample_token)[:config.reload_subset_size]
    _, reloaded = full.evaluate(selected, runner, subset, records, "predicted_action")
    consistency = compare_reload([by_token[s.sample_token] for s in subset], reloaded)
    write_json(output / "reload_consistency.json", consistency)
    if not consistency["reload_consistency"]:
        raise ValueError("checkpoint fresh reload is inconsistent")
    write_json(output / "best_checkpoint.json", best)
    result = {"status": "convergence_training_completed", "epochs": epochs, "best_epoch": best["epoch"],
              "extension_gate": extension_gate(epochs, config), "optimizer_steps": step,
              "Direct multi-epoch control": "NOT RUN", "test_isolation": provenance["data"],
              "reload_consistency": consistency, "provenance": provenance,
              **{key: provenance[key] for key in (
                  "historical_epoch1_reproduction_passed", "continuation_after_failed_historical_reproduction",
                  "convergence_evidence_scope")}}
    write_json(output / "training_summary.json", result)
    return result


def run(*, repository: Path, dataset_root: Path, derived_root: Path, config: ConvergenceConfig,
        git_provenance: GitProvenance, extend_to_five: bool = False,
        continue_after_epoch1_reproduction_fail: bool = False,
        train_split: str = "train", validation_split: str = "validation") -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("convergence requires train/validation only")
    if extend_to_five and continue_after_epoch1_reproduction_fail:
        raise ValueError("failed-epoch1 continuation and extend-to-five are mutually exclusive")
    git = validate_git_provenance(git_provenance)
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("planner artifacts must be outside repository")
    if output.exists() and not (extend_to_five or continue_after_epoch1_reproduction_fail):
        raise FileExistsError(f"convergence output already exists: {output}")
    resume = load_resume(output, config) if extend_to_five else None
    if continue_after_epoch1_reproduction_fail:
        resume = load_failed_epoch1(output, config)
    inputs = prepare_run(repository, dataset_root, derived_root, config, git)
    if resume is None:
        output.mkdir(parents=True)
        write_json(output / "resolved_config.json", asdict(config))
        write_json(output / "data_summary.json", inputs["provenance"]["data"])
    return fit(**inputs, config=config, output=output, resume=resume,
               continue_after_epoch1_reproduction_fail=continue_after_epoch1_reproduction_fail)


def prepare_run(repository: Path, dataset_root: Path, derived_root: Path,
                config: full.FullConfig, git: GitProvenance, *, prepared_data: tuple | None = None) -> dict:
    train, validation, records, data = (
        full.prepare_data(repository, derived_root) if prepared_data is None else prepared_data)
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
    runner = full.PlannerRunner(model, processor, runtime, dataset_root,
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
    return dict(model=model, planner=planner, runner=runner, train=train, validation=validation,
                records=records, provenance=provenance)


def diagnostic_comparison(second: dict, third: dict) -> dict:
    changes = {}
    for track, prefix in (("predicted_action", "predicted_action"), ("gt_action_diagnostic", "gt_action")):
        changes[f"{prefix}_delta"] = deltas(third[track], {k: second[track][k] for k in DIAGNOSTIC_METRICS})
        changes[f"{prefix}_relative_change"] = {
            k: changes[f"{prefix}_delta"][k] / second[track][k]
            if second[track][k] is not None and second[track][k] > 0
            and changes[f"{prefix}_delta"][k] is not None else None for k in DIAGNOSTIC_METRICS}
    predicted = [changes["predicted_action_relative_change"][k] for k in ("ade_3s_m", "fde_3s_m")]
    gt = [changes["gt_action_relative_change"][k] for k in ("ade_3s_m", "fde_3s_m")]
    predicted_worse = all(v is not None and v > 0 for v in predicted)
    mismatch = predicted_worse and all(v is not None and v < .01 for v in gt)
    regression = predicted_worse and all(v is not None and v >= .01 for v in gt)
    return {"epoch_2": second, "epoch_3": third, "epoch2_to_epoch3": changes,
            "conditioning_gap": {
                f"epoch_{epoch}_predicted_minus_gt": deltas(
                    entry["predicted_action"], {k: entry["gt_action_diagnostic"][k] for k in DIAGNOSTIC_METRICS})
                for epoch, entry in ((2, second), (3, third))},
            "diagnostic_interpretation": {
                "diagnostic": "case_a" if mismatch else "case_b" if regression else "inconclusive",
                "teacher_forcing_predicted_action_mismatch_supported": mismatch,
                "ordinary_overfitting_alone_supported": regression,
                "ordinary_overfitting_or_generalization_regression_supported": regression,
                "descriptive_relative_threshold": .01, "threshold_is_experiment_gate": False,
                "explanation": (
                    "GT-action performance improves or remains approximately equal while predicted action regresses; "
                    "this supports sensitivity to predicted-action errors, not a causal proof."
                    if mismatch else "Both action tracks regress; ordinary overfitting or generalization regression "
                    "is supported, not uniquely identified."
                    if regression else "Mixed or unavailable changes do not distinguish the hypotheses.")}}


def diagnose_gt_action(*, repository: Path, dataset_root: Path, derived_root: Path,
                       config: ConvergenceConfig, git_provenance: GitProvenance,
                       validation_split: str = "validation") -> dict:
    if validation_split != "validation":
        raise ValueError("diagnostic requires validation only")
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("planner artifacts must be outside repository")
    names = [f"epoch_{epoch}_gt_action_{suffix}" for epoch in (2, 3)
             for suffix in ("metrics.json", "predictions.jsonl")]
    names.append("gt_action_diagnostic_comparison.json")
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"diagnostic output already exists: {name}")
    summary = json.loads((output / "training_summary.json").read_text())
    if summary["status"] != "convergence_training_completed" or summary["best_epoch"] != 2:
        raise ValueError("diagnostic requires completed convergence with best_epoch == 2")
    checkpoints, historical = {}, {}
    for epoch in (2, 3):
        saved = torch.load(output / f"epoch_{epoch}.pt", map_location="cpu", weights_only=True)
        if saved["epoch"] != epoch or saved["optimizer_step"] != epoch * EPOCH1_OPTIMIZER_STEP:
            raise ValueError("diagnostic checkpoint epoch/optimizer_step mismatch")
        if saved["training_config"] != asdict(config):
            raise ValueError("diagnostic training config mismatch")
        metrics = json.loads((output / f"epoch_{epoch}_metrics.json").read_text())
        rows = [json.loads(line) for line in (output / f"epoch_{epoch}_predictions.jsonl").read_text().splitlines()]
        if (metrics != aggregate_metrics(rows, "predicted_action")
                or metrics["sample_count"] != config.expected_validation_count
                or metrics["valid_prediction_count"] != config.expected_validation_count
                or metrics["invalid_prediction_count"] != 0):
            raise ValueError("diagnostic historical validation coverage/metrics mismatch")
        checkpoints[epoch], historical[epoch] = saved, rows
    prepared = prepare_run(repository, dataset_root, derived_root, config,
                           validate_git_provenance(git_provenance))
    validation, records = prepared["validation"], prepared["records"]
    if (len(validation) != config.expected_validation_count
            or any(s.split != "validation" for s in validation)
            or any(not (s.target.longitudinal_valid and s.target.lateral_valid) for s in validation)):
        raise ValueError("diagnostic requires full valid GT-action validation coverage")
    tokens = {s.sample_token for s in validation}
    for epoch in (2, 3):
        if checkpoints[epoch]["provenance"]["data"] != prepared["provenance"]["data"]:
            raise ValueError("diagnostic data provenance mismatch")
        rows = historical[epoch]
        if len(tokens) != len(validation) or {r["sample_token"] for r in rows} != tokens:
            raise ValueError("diagnostic historical validation sample mismatch")
        for row in rows:
            record = records[row["sample_token"]]
            target = torch.tensor(record["future_waypoints"], dtype=torch.float32).tolist()
            if (row["split"] != "validation" or row["scene_token"] != record["scene_token"]
                    or row["target_waypoints"] != target
                    or row["trajectory_valid_mask"] != record["trajectory_valid_mask"]):
                raise ValueError("diagnostic historical validation target/mask mismatch")
    missing = {s.sample_token for s in validation
               if records[s.sample_token]["ego_motion_history"][-1]["availability"] == "unavailable"}
    if len(missing) != config.expected_motion_unavailable_count:
        raise ValueError("diagnostic motion-unavailable subset count mismatch")
    model, planner, runner = prepared["model"], prepared["planner"], prepared["runner"]
    freeze_backbone(model)
    model.eval()
    evaluated, overall, subsets = {}, {}, {}
    with torch.no_grad():
        for epoch in (2, 3):
            planner.load_state_dict(checkpoints[epoch]["planner_state_dict"])
            planner.eval()
            metrics, rows = full.evaluate(planner, runner, validation, records, "gt_action_diagnostic")
            if (metrics["sample_count"] != config.expected_validation_count
                    or metrics["valid_prediction_count"] != config.expected_validation_count
                    or metrics["invalid_prediction_count"] != 0):
                raise ValueError("diagnostic GT-action validation coverage mismatch")
            evaluated[epoch] = rows
            tracks = {"predicted_action": historical[epoch], "gt_action_diagnostic": rows}
            overall[epoch] = {track: aggregate_metrics(values, track) for track, values in tracks.items()}
            subsets[epoch] = {track: aggregate_metrics(
                [r for r in values if r["sample_token"] in missing], track) for track, values in tracks.items()}
    result = {"status": "gt_action_epoch2_epoch3_diagnostic_completed",
              **diagnostic_comparison(overall[2], overall[3]),
              "motion_unavailable": {"sample_tokens": sorted(missing),
                                     **diagnostic_comparison(subsets[2], subsets[3])},
              "provenance": prepared["provenance"], "test_isolation": prepared["provenance"]["data"]}
    improvements = {}
    for track, source in (("predicted_action", historical), ("gt_action_diagnostic", evaluated)):
        previous = {r["sample_token"]: r for r in source[2]}
        improvements[track] = {}
        for key in ("ade_3s_m", "fde_3s_m"):
            pairs = [(previous[r["sample_token"]][key], r[key]) for r in source[3]
                     if previous[r["sample_token"]][key] is not None and r[key] is not None]
            count = sum(b < a for a, b in pairs)
            improvements[track][key] = {"improved_count": count, "paired_sample_count": len(pairs),
                                       "improved_percentage": 100 * count / len(pairs) if pairs else None}
    result["sample_level_improvement"] = improvements
    for epoch in (2, 3):
        full.write_predictions(output / f"epoch_{epoch}_gt_action_predictions.jsonl", evaluated[epoch])
        write_json(output / f"epoch_{epoch}_gt_action_metrics.json", overall[epoch]["gt_action_diagnostic"])
    write_json(output / "gt_action_diagnostic_comparison.json", result)
    return result

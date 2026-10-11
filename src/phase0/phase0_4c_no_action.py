from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import yaml

from src.baselines.ego_history_mlp import index_predictions, read_predictions
from src.phase0 import phase0_4c_convergence as convergence
from src.phase0 import phase0_4c_direct_waypoint as direct
from src.phase0 import phase0_4c_full_train as full
from src.phase0.phase0_4b_lora_full import epoch_groups
from src.phase0.phase0_4b_protocol import Observation, inference_messages
from src.phase0.phase0_4c_evaluation import aggregate_metrics
from src.phase0.phase0_4c_trajectory_stratification import recompute
from src.phase0.phase0_4c_two_turn_planner import PLANNING_PROMPT, WaypointDecoder
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, resolve_derived_path, validate_git_provenance

TRACK = "no_action_neutral_context"
NEUTRAL_CONTEXT = "structured_action=unavailable"
REFERENCE_FILES = {
    "predicted_action": "action_conditioned_convergence_v0_1/epoch_2_predictions.jsonl",
    "gt_action_diagnostic": "action_conditioned_convergence_v0_1/epoch_2_gt_action_predictions.jsonl",
    "ego_history_mlp": "ego_history_mlp_baseline_v0_1/validation_predictions.jsonl",
}


@dataclass(frozen=True)
class NoActionConfig(full.FullConfig):
    conditioning_type: str
    neutral_context_text: str
    expected_validation_count: int
    expected_motion_unavailable_count: int


def load_config(path: Path, repository: Path) -> NoActionConfig:
    config = NoActionConfig(train_subset_size=0, **yaml.safe_load(path.read_text()))
    expected = asdict(full.load_config(repository / "configs/phase0_4c_full_train.yaml"))
    expected.update(protocol_version="phase0.4c-no-action-ablation-v0.1",
                    output_relative_dir="phase_0_4/no_action_ablation_v0_1",
                    num_train_epochs=2, checkpoint_interval=1, validation_interval=1,
                    selection_policy="fixed_epoch_2", conditioning_type=TRACK,
                    neutral_context_text=NEUTRAL_CONTEXT, expected_validation_count=3594,
                    expected_motion_unavailable_count=94)
    if asdict(config) != expected:
        raise ValueError("config differs from frozen two-epoch no-action protocol")
    return config


def neutral_messages(observation: Observation, images: Sequence[object]) -> list[dict]:
    return inference_messages(observation, images) + [
        {"role": "assistant", "content": [{"type": "text", "text": NEUTRAL_CONTEXT}]},
        {"role": "user", "content": [{"type": "text", "text": PLANNING_PROMPT}]},
    ]


class NeutralRunner(direct.DirectRunner):
    context_evidence: dict | None = None

    def messages(self, observation: Observation, images: Sequence[object]) -> list[dict]:
        messages = neutral_messages(observation, images)
        if self.context_evidence is None:
            self.context_evidence = {
                "conditioning_type": TRACK, "neutral_context_text": NEUTRAL_CONTEXT,
                "rendered_context": self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True),
            }
        return messages

    def predict(self, planner: WaypointDecoder, sample: direct.DirectSample,
                conditioning_type: str = TRACK) -> tuple[torch.Tensor, dict]:
        if conditioning_type != TRACK:
            raise ValueError("neutral runner requires no-action conditioning")
        return super().predict(planner, sample)


def aligned_metrics(rows: list[dict], track: str, records: dict) -> dict[str, dict]:
    indexed = index_predictions(rows, track)
    if indexed.keys() != records.keys():
        raise ValueError(f"{track} exact sample_token alignment failed")
    result = {}
    for token, row in indexed.items():
        record = records[token]
        target = torch.tensor(record["future_waypoints"], dtype=torch.float32).tolist()
        if (record["split"] != "validation" or row["scene_token"] != record["scene_token"]
                or row["target_waypoints"] != target
                or any(type(v) is not bool for v in row["trajectory_valid_mask"])
                or row["trajectory_valid_mask"] != record["trajectory_valid_mask"]):
            raise ValueError(f"{track} scene/target/mask mismatch: {token}")
        result[token] = recompute(row)
    return result


def compare(models: dict[str, dict[str, dict]], tokens: list[str]) -> dict:
    overall = {track: aggregate_metrics([rows[t] for t in tokens], track)
               for track, rows in models.items()}
    pairs = {}
    for track in REFERENCE_FILES:
        metrics = {}
        for key in ("ade_3s_m", "fde_3s_m"):
            values = [(models[TRACK][t][key], models[track][t][key]) for t in tokens
                      if models[TRACK][t]["prediction_valid"] and models[track][t]["prediction_valid"]
                      and models[TRACK][t][key] is not None and models[track][t][key] is not None]
            differences = [a - b for a, b in values]
            count = len(values)
            delta = sum(differences) / count if count else None
            reference = sum(b for _, b in values) / count if count else None
            wins = sum(d < 0 for d in differences)
            metrics[key] = {
                "paired_sample_count": count, "excluded_sample_count": len(tokens) - count,
                "mean_delta_m": delta,
                "relative_delta": delta / reference if reference else None,
                "win_count": wins, "tie_count": sum(d == 0 for d in differences),
                "loss_count": sum(d > 0 for d in differences), "win_rate": wins / count if count else None,
            }
        pairs[f"no_action_minus_{track}"] = metrics
    return {"sample_count": len(tokens), "sample_tokens": tokens, "models": overall, "pairwise": pairs,
            "delta_definition": "no_action_minus_reference; negative_is_better",
            "relative_delta_definition": "mean_delta / paired_reference_mean; null_if_zero_or_unavailable",
            "tie_definition": "exact_zero_difference", "interpretation": "descriptive_only"}


def intake(repository: Path, derived_root: Path, config: NoActionConfig) -> tuple[tuple, dict, list[str]]:
    prepared = direct.prepare_data(repository, derived_root)
    _, validation, records, data = prepared
    temporal = {s.sample_token: records[s.sample_token] for s in validation}
    if len(validation) != config.expected_validation_count or len(temporal) != len(validation):
        raise ValueError("validation count differs from frozen population")
    missing = sorted(t for t, r in temporal.items()
                     if r["ego_motion_history"][-1]["availability"] == "unavailable")
    if len(missing) != config.expected_motion_unavailable_count:
        raise ValueError("motion-unavailable subset count differs from frozen population")
    directory = derived_root / "phase_0_4/action_conditioned_convergence_v0_1"
    metadata = json.loads((directory / "run_metadata.json").read_text())
    historical_config = convergence.load_config(repository / "configs/phase0_4c_convergence.yaml")
    if (metadata["config"] != asdict(historical_config)
            or metadata["data"]["source_provenance"] != data["source_provenance"]
            or any(metadata["data"][key] != data[key] for key in ("train", "validation"))):
        raise ValueError("reference config/data provenance mismatch")
    references = {track: aligned_metrics(read_predictions(derived_root / "phase_0_4" / path, track),
                                         track, temporal) for track, path in REFERENCE_FILES.items()}
    return prepared, references, missing


def fit(*, model: torch.nn.Module, planner: WaypointDecoder, runner: NeutralRunner,
        train: list[direct.DirectSample], validation: list[direct.DirectSample], records: dict,
        config: NoActionConfig, output: Path, provenance: dict) -> list[dict]:
    if not train or any(s.split != "train" for s in train):
        raise ValueError("no-action optimization requires train only")
    if not validation or any(s.split != "validation" for s in validation):
        raise ValueError("no-action evaluation requires validation only")
    optimizer = torch.optim.AdamW(planner.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    provenance = {**provenance, "freeze_policy": full.verify_freeze(model, planner, optimizer)}
    write_json(output / "run_metadata.json", provenance)
    step, epochs = 0, []
    with (output / "training_history.jsonl").open("w") as history:
        for epoch in range(config.num_train_epochs):
            loss_sum = 0.
            for group in epoch_groups(train, config.gradient_accumulation_steps, config.seed + epoch):
                loss = convergence.train_group(planner, runner, group, records, config, optimizer, TRACK)
                loss_sum += loss
                step += 1
                entry = {"epoch": epoch + 1, "optimizer_step": step, "train_loss": loss / len(group),
                         "sample_tokens": [s.sample_token for s in group], "conditioning_type": TRACK}
                history.write(json.dumps(entry, allow_nan=False) + "\n")
                history.flush()
                if step % config.log_every_steps == 0:
                    print(json.dumps(entry), flush=True)
            metrics, rows = direct.evaluate(planner, runner, validation, records, TRACK)
            full.write_predictions(output / f"epoch_{epoch + 1}_predictions.jsonl", rows)
            write_json(output / f"epoch_{epoch + 1}_metrics.json", metrics)
            full.save_checkpoint(output / f"epoch_{epoch + 1}.pt", planner, config, provenance, step)
            epochs.append({"epoch": epoch + 1, "optimizer_step": step,
                           "train_loss": loss_sum / len(train), "metrics": metrics})
            print(json.dumps(epochs[-1], allow_nan=False), flush=True)
    full.verify_freeze(model, planner, optimizer)
    write_json(output / "input_evidence.json", runner.context_evidence)
    selected = full.reload_planner(output / "epoch_2.pt", runner.device)
    before = read_predictions(output / "epoch_2_predictions.jsonl", TRACK)
    subset = sorted(validation, key=lambda s: s.sample_token)[:config.reload_subset_size]
    _, reloaded = direct.evaluate(selected, runner, subset, records, TRACK)
    by_token = {r["sample_token"]: r for r in before}
    consistency = direct.compare_reload([by_token[s.sample_token] for s in subset], reloaded, TRACK)
    consistency["scope"] = "sorted_validation_subset"
    write_json(output / "reload_consistency.json", consistency)
    if not consistency["reload_consistency"]:
        raise ValueError("no-action checkpoint reload mismatch")
    write_json(output / "training_summary.json", {
        "status": "two_epoch_training_completed", "epochs": epochs, "selected_epoch": 2,
        "selection_policy": "fixed_epoch_2", "optimizer_steps": step,
        "test_isolation": provenance["data"], "reload_consistency": consistency,
    })
    return before


def run(*, repository: Path, dataset_root: Path, derived_root: Path, config: NoActionConfig,
        git_provenance: GitProvenance | None, train_split: str = "train",
        validation_split: str = "validation", dry_run: bool = False) -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("no-action requires train/validation only")
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("no-action artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"no-action output already exists: {output}")
    if not dataset_root.is_dir():
        raise FileNotFoundError("dataset root does not exist")
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("selected adapter files do not exist")
    prepared, references, missing = intake(repository, derived_root, config)
    if dry_run:
        return {"status": "dry_run_intake_passed_no_model_or_image_access", "config": asdict(config),
                "output_directory": str(output), "selected_adapter": str(adapter),
                "train_sample_count": len(prepared[0]), "validation_sample_count": len(prepared[1]),
                "motion_unavailable_count": len(missing), "test_isolation": prepared[3]}
    inputs = convergence.prepare_run(repository, dataset_root, derived_root, config,
                                     validate_git_provenance(git_provenance), prepared_data=prepared)
    original = inputs["runner"]
    inputs["runner"] = NeutralRunner(original.model, original.processor, original.runtime,
                                     original.dataset_root, original.device)
    provenance = inputs["provenance"]
    provenance.update(conditioning_type=TRACK, neutral_context_text=NEUTRAL_CONTEXT,
                      conditioning_protocol={"train": TRACK, "validation": TRACK},
                      action_generation_calls=0, action_values_used_for_prediction=0,
                      action_values_used_for_loss=0, future_information_inputs=0,
                      validation_schedule="epoch_end", selection_policy="fixed_epoch_2",
                      reference_files=REFERENCE_FILES)
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", asdict(config))
    write_json(output / "data_summary.json", prepared[3])
    rows = fit(**inputs, config=config, output=output)
    temporal = {s.sample_token: prepared[2][s.sample_token] for s in prepared[1]}
    models = {TRACK: aligned_metrics(rows, TRACK, temporal), **references}
    result = compare(models, sorted(temporal))
    write_json(output / "comparison.json", result)
    write_json(output / "motion_unavailable_comparison.json", compare(models, missing))
    return result

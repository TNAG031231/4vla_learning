from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
import yaml

from src.baselines import ego_history_mlp as ego
from src.phase0 import phase0_4c_convergence as convergence
from src.phase0 import phase0_4c_direct_waypoint as direct
from src.phase0 import phase0_4c_no_action as neutral
from src.phase0.phase0_4c_two_turn_planner import WaypointDecoder
from src.phase0.phase0_4c_tiny_overfit import write_json
from src.phase0.qwen3vl_dataset_adapter import GitProvenance, resolve_derived_path, validate_git_provenance

TRACK = "ego_fusion_neutral_context"
REFERENCE_FILES = {
    neutral.TRACK: "no_action_ablation_v0_1/epoch_2_predictions.jsonl",
    **neutral.REFERENCE_FILES,
}


@dataclass(frozen=True)
class FusionConfig(neutral.NoActionConfig):
    history_length: int
    ego_hidden_dimension: int
    normalization_epsilon: float


def load_config(path: Path, repository: Path) -> FusionConfig:
    config = FusionConfig(train_subset_size=0, **yaml.safe_load(path.read_text()))
    expected = asdict(neutral.load_config(repository / "configs/phase0_4c_no_action.yaml", repository))
    expected.update(protocol_version="phase0.4c-ego-fusion-v0.1",
                    output_relative_dir="phase_0_4/ego_fusion_v0_1", conditioning_type=TRACK,
                    history_length=3, ego_hidden_dimension=128, normalization_epsilon=1e-6)
    if asdict(config) != expected:
        raise ValueError("config differs from two-epoch ego-fusion protocol")
    return config


@dataclass(frozen=True)
class FusionSample(direct.DirectSample):
    ego_motion_history: list[dict | None]


class EgoWaypointDecoder(WaypointDecoder):
    normalization: dict

    def __init__(self, hidden_size: int, config: FusionConfig) -> None:
        super().__init__(hidden_size, config)
        self.ego_encoder = nn.Sequential(
            nn.Linear(7 * config.history_length, config.ego_hidden_dimension), nn.GELU(),
            nn.Linear(config.ego_hidden_dimension, config.planner_dimension),
        )


class FusionRunner(neutral.NeutralRunner):
    def predict(self, planner: EgoWaypointDecoder, sample: FusionSample,
                conditioning_type: str = TRACK) -> tuple[torch.Tensor, dict]:
        if conditioning_type != TRACK:
            raise ValueError("fusion runner requires ego-fusion conditioning")
        hidden, attention_mask, evidence = self.context(sample)
        features = ego.encode(sample.ego_motion_history, list(sample.observation.history_valid_mask),
                              planner.normalization).unsqueeze(0).to(self.device)
        condition = planner.ego_encoder(features.to(planner.ego_encoder[0].weight.dtype))
        prediction = planner(hidden, attention_mask, query_condition=condition)
        self.context_evidence.update(conditioning_type=TRACK, ego_feature_shape=list(features.shape),
                                     ego_feature_source="past_current_ego_motion_and_availability")
        return prediction, {**evidence, "ego_feature_shape": list(features.shape),
                            "waypoint_output_shape": list(prediction.shape)}


def reload_planner(path: Path, device: str) -> EgoWaypointDecoder:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = FusionConfig(**saved["training_config"])
    planner = EgoWaypointDecoder(saved["hidden_size"], config).to(device)
    planner.load_state_dict(saved["planner_state_dict"])
    planner.normalization = saved["provenance"]["ego_normalization"]
    return planner


def run(*, repository: Path, dataset_root: Path, derived_root: Path, config: FusionConfig,
        git_provenance: GitProvenance | None, train_split: str = "train",
        validation_split: str = "validation", dry_run: bool = False) -> dict:
    if train_split != "train" or validation_split != "validation":
        raise ValueError("fusion requires train/validation only")
    output = resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository.resolve()):
        raise ValueError("fusion artifacts must be outside repository")
    if output.exists():
        raise FileExistsError(f"fusion output already exists: {output}")
    if not dataset_root.is_dir():
        raise FileNotFoundError("dataset root does not exist")
    adapter = resolve_derived_path(derived_root, config.selected_adapter_relative_path)
    if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("selected adapter files do not exist")
    prepared, references, missing = neutral.intake(repository, derived_root, config)
    train, validation, records, data = prepared
    neutral_dir = derived_root / "phase_0_4/no_action_ablation_v0_1"
    metadata = json.loads((neutral_dir / "run_metadata.json").read_text())
    reference_config = neutral.load_config(repository / "configs/phase0_4c_no_action.yaml", repository)
    if (metadata["config"] != asdict(reference_config)
            or metadata["data"]["source_provenance"] != data["source_provenance"]
            or any(metadata["data"][key] != data[key] for key in ("train", "validation"))):
        raise ValueError("no-action reference config/data provenance mismatch")
    temporal = {s.sample_token: records[s.sample_token] for s in validation}
    references[neutral.TRACK] = neutral.aligned_metrics(
        ego.read_predictions(neutral_dir / "epoch_2_predictions.jsonl", neutral.TRACK), neutral.TRACK, temporal)
    stats = ego.normalization_stats([records[s.sample_token] for s in train], config.normalization_epsilon)
    groups = [[FusionSample(s.sample_token, s.scene_token, s.split, s.observation,
                            records[s.sample_token]["ego_motion_history"]) for s in group]
              for group in (train, validation)]
    if dry_run:
        return {"status": "dry_run_intake_passed_no_model_or_image_access", "config": asdict(config),
                "output_directory": str(output), "selected_adapter": str(adapter),
                "train_sample_count": len(train), "validation_sample_count": len(validation),
                "motion_unavailable_count": len(missing), "ego_normalization_fit_split": stats["fit_split"],
                "test_isolation": data}
    inputs = convergence.prepare_run(
        repository, dataset_root, derived_root, config, validate_git_provenance(git_provenance),
        prepared_data=(groups[0], groups[1], records, data), planner_type=EgoWaypointDecoder)
    original = inputs["runner"]
    inputs["runner"] = FusionRunner(original.model, original.processor, original.runtime,
                                    original.dataset_root, original.device)
    inputs["planner"].normalization = stats
    provenance = inputs["provenance"]
    provenance.update(
        conditioning_type=TRACK, neutral_context_text=neutral.NEUTRAL_CONTEXT,
        conditioning_protocol={"train": TRACK, "validation": TRACK},
        planner_architecture="2560->256->MemoryLayerNorm; queries+ego_MLP(21->128->256)->2layer4headPostLN->2",
        ego_normalization=stats, ego_history_order="oldest_to_current",
        ego_feature_layout="per_frame: normalized_speed, speed_available, normalized_acceleration, "
                           "acceleration_available, normalized_yaw_rate, yaw_rate_available, history_valid",
        action_generation_calls=0, action_values_used_for_prediction=0,
        action_values_used_for_loss=0, future_information_inputs=0,
        validation_schedule="epoch_end", selection_policy="fixed_epoch_2", reference_files=REFERENCE_FILES)
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", asdict(config))
    write_json(output / "data_summary.json", data)
    write_json(output / "normalization_stats.json", stats)
    rows = neutral.fit(**inputs, config=config, output=output, reload=reload_planner)
    models = {TRACK: neutral.aligned_metrics(rows, TRACK, temporal), **references}
    result = neutral.compare(models, sorted(temporal), primary_track=TRACK, primary_name="fusion")
    write_json(output / "comparison.json", result)
    write_json(output / "motion_unavailable_comparison.json",
               neutral.compare(models, missing, primary_track=TRACK, primary_name="fusion"))
    return result

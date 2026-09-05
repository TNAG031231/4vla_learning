from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile

import yaml

from src.phase0 import development_projection as development
from src.phase0.manifest import (
    COORDINATE_METADATA,
    json_record,
    write_canonical_json,
    write_jsonl_records,
)
from src.phase0.qwen3vl_smoke import read_selected_rule
from src.phase0.qwen_preflight import check_manifest_integrity, sha256_file
from src.phase0.scene_mapping import read_scene_mapping


SOURCE_VERSION = "phase0.4a1-source-projection-v0.1"
SOURCE_SCHEMA = "phase0_4_source_projection_v0.1"


@dataclass(frozen=True)
class SourceProjectionConfig:
    source_contract: development.ProjectionConfig
    trainval_config: Path
    output_relative_dir: str
    config_sha256: str
    raw_point_count: int
    horizon_sec: float
    sample_interval_sec: float
    time_tolerance_sec: float
    anchor_absolute_tolerance: float


def load_config(path: Path, repository_root: Path) -> SourceProjectionConfig:
    loaded = yaml.safe_load(path.read_bytes())
    if not isinstance(loaded, dict):
        raise ValueError("source projection config must be a mapping")
    if (
        loaded.get("source_projection_version") != SOURCE_VERSION
        or loaded.get("source_projection_schema_version") != SOURCE_SCHEMA
    ):
        raise ValueError("source projection version mismatch")
    trajectory = loaded.get("trajectory")
    if trajectory != {
        "raw_point_count": 7,
        "horizon_sec": 3.0,
        "sample_interval_sec": 0.5,
        "time_tolerance_sec": 0.075,
        "anchor_absolute_tolerance": 1e-12,
    }:
        raise ValueError("trajectory configuration differs from frozen contract")
    fixed_paths = {
        "source_contract_config": "configs/phase0_3_development_projection.yaml",
        "trainval_config": "configs/trainval_manifest.yaml",
        "output_relative_dir": "phase_0_4/source_projection_v0_1",
    }
    for key, expected in fixed_paths.items():
        if loaded.get(key) != expected:
            raise ValueError(f"{key} differs from source projection contract")
    return SourceProjectionConfig(
        source_contract=development.load_config(
            repository_root / loaded["source_contract_config"]
        ),
        trainval_config=repository_root / loaded["trainval_config"],
        output_relative_dir=loaded["output_relative_dir"],
        config_sha256=sha256_file(path),
        **trajectory,
    )


def project_record(
    source_record: object,
    *,
    split: str,
    config: SourceProjectionConfig,
    split_mapping_sha256: str,
    selection: development.DevelopmentSceneSelection,
) -> dict[str, object]:
    source = json_record(source_record)
    if split not in development.PROJECT_SPLITS:
        raise ValueError("source projection only permits train and validation")
    if source.get("scene_token") not in selection.scene_tokens_by_split[split]:
        raise ValueError("producer scene is outside the selected split")
    development.project_record(
        source,
        split=split,
        config=config.source_contract,
        split_mapping_sha256=split_mapping_sha256,
    )
    coordinates = source["coordinate_metadata"]
    if not isinstance(coordinates, Mapping):
        raise ValueError("coordinate metadata must be a mapping")
    trajectory_coordinates = coordinates.get("future_ego_trajectory")
    if not isinstance(trajectory_coordinates, Mapping):
        raise ValueError("future trajectory coordinate metadata is missing")
    expected_coordinates = COORDINATE_METADATA["future_ego_trajectory"]
    for key in (
        "source_frame", "target_frame", "x_axis", "y_axis", "unit", "transform",
    ):
        if trajectory_coordinates.get(key) != expected_coordinates[key]:
            raise ValueError(f"future trajectory coordinate {key} mismatch")
    pose = source["current_ego_pose"]
    if (
        not isinstance(pose, Mapping)
        or pose.get("timestamp_source") != "CAM_FRONT_sample_data"
    ):
        raise ValueError("current ego pose must retain CAM_FRONT timestamp provenance")
    if not isinstance(pose.get("timestamp_us"), int):
        raise ValueError("current ego pose timestamp_us must be an integer")
    points = source["future_ego_trajectory"]
    if not isinstance(points, list) or len(points) != config.raw_point_count:
        raise ValueError("raw future trajectory must contain 7 points")
    tokens = set()
    previous_time = -math.inf
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError("trajectory point must be a mapping")
        token = point.get("future_sample_token")
        if not isinstance(token, str) or not token or token in tokens:
            raise ValueError("trajectory tokens must be nonempty and unique")
        tokens.add(token)
        for field in ("t_sec", "x_m", "y_m", "heading_delta_rad"):
            value = point.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"trajectory {field} must be finite")
        time = point["t_sec"]
        if time <= previous_time:
            raise ValueError("trajectory time must be strictly increasing")
        previous_time = time
        if index == 0:
            if token != source["sample_token"] or time != 0.0:
                raise ValueError("trajectory current anchor token/time mismatch")
            if any(
                abs(point[field]) > config.anchor_absolute_tolerance
                for field in ("x_m", "y_m", "heading_delta_rad")
            ):
                raise ValueError("trajectory current anchor must be at ego origin")
        elif abs(time - index * config.sample_interval_sec) > config.time_tolerance_sec:
            raise ValueError("future trajectory time outside frozen tolerance")
    preserved = (
        "sample_token", "scene_token", "split", "timestamp", "cam_front_path",
        "current_ego_pose", "current_ego_motion", "future_ego_trajectory",
        "label_rule_version", "source_audit_record", "split_seed",
        "split_strategy_version", "split_mapping_sha256",
    )
    return {
        **{key: source[key] for key in preserved},
        "source_projection_version": SOURCE_VERSION,
        "source_projection_schema_version": SOURCE_SCHEMA,
        "coordinate_metadata": {
            key: coordinates[key]
            for key in (
                "current_ego_pose", "current_ego_motion", "future_ego_trajectory",
            )
        },
        "source_legacy_meta_action": source["meta_action"],
        "source_manifest_schema_version": source["manifest_schema_version"],
        "source_combined_manifest_sha256": (
            config.source_contract.expected_combined_manifest_sha256
        ),
    }


def build_source_projection(
    *,
    config: SourceProjectionConfig,
    repository_root: Path,
    derived_root: Path,
    nuscenes_root: Path,
    reader_factory: Callable[[], development.NuScenesReader],
    producer: development.RecordProducer,
    producer_inputs: development.ProducerInputs,
    git_provenance: development.GitProvenance,
) -> dict[str, object]:
    git = development.validate_git_provenance(git_provenance)
    if (
        producer_inputs.horizon_sec != config.horizon_sec
        or producer_inputs.sample_interval_sec != config.sample_interval_sec
        or producer_inputs.time_tolerance_sec != config.time_tolerance_sec
    ):
        raise ValueError("producer timing differs from frozen trajectory contract")
    source = config.source_contract
    manifest, failures = check_manifest_integrity(
        development.resolve_derived_path(
            derived_root, source.combined_manifest_relative_path,
        ),
        source.expected_combined_manifest_sha256,
    )
    if failures:
        raise ValueError(str(failures[0]["code"]))
    rule = read_selected_rule(
        development.resolve_derived_path(
            derived_root, source.selected_rule_relative_path,
        ),
        source.expected_combined_manifest_sha256,
    )
    mapping = read_scene_mapping(
        development.resolve_derived_path(
            derived_root, source.scene_mapping_relative_path,
        )
    )
    mapping_sha = mapping["scene_split_mapping_sha256"]
    if rule["split_mapping_sha256"] != mapping_sha:
        raise ValueError("selected rule and scene mapping SHA-256 mismatch")
    selection = development.select_development_scenes(mapping, source)
    output = development.resolve_derived_path(derived_root, config.output_relative_dir)
    if output.is_relative_to(repository_root.resolve()):
        raise ValueError("source projection must be outside the repository")
    if output.exists():
        raise FileExistsError("source projection already exists; refusing to overwrite")
    counters = development.IsolationCounters()
    reader = development.GuardedNuScenesReader(
        reader_factory(), selection.allowed_scene_tokens,
        selection.forbidden_test_scene_tokens, counters,
    )
    records_by_split = {}
    seen_tokens = set()
    for split in development.PROJECT_SPLITS:
        result = producer(
            nuscenes=reader,
            scene_tokens=selection.scene_tokens_by_split[split],
            scene_splits=selection.scene_splits,
            official_splits=selection.official_splits,
            split_seed=source.expected_split_seed,
            split_strategy_version=source.expected_split_strategy_version,
            split_mapping_sha256=mapping_sha,
            audit_index={},
            dataroot=nuscenes_root,
            rules=producer_inputs.rules,
            horizon_sec=producer_inputs.horizon_sec,
            sample_interval_sec=producer_inputs.sample_interval_sec,
            time_tolerance_sec=producer_inputs.time_tolerance_sec,
            agent_radius_m=producer_inputs.agent_radius_m,
        )
        records = [
            project_record(
                record, split=split, config=config,
                split_mapping_sha256=mapping_sha, selection=selection,
            )
            for record in result.records
        ]
        if len(records) != source.expected_sample_counts[split]:
            raise ValueError(f"{split} record count differs from frozen contract")
        for record in records:
            token = record["sample_token"]
            if token in seen_tokens:
                raise ValueError("source sample tokens must be globally unique")
            seen_tokens.add(token)
        records_by_split[split] = sorted(
            records,
            key=lambda record: (
                record["scene_token"], record["timestamp"], record["sample_token"],
            ),
        )
    if counters != development.IsolationCounters():
        raise ValueError("test isolation counters must remain zero")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        outputs = {}
        for split, records in records_by_split.items():
            path = staging / f"{split}.jsonl"
            write_jsonl_records(records, path)
            outputs[split] = {
                "relative_path": path.name,
                "record_count": len(records),
                "sha256": sha256_file(path),
            }
        receipt = {
            "source_projection_version": SOURCE_VERSION,
            "source_projection_schema_version": SOURCE_SCHEMA,
            "combined_manifest": {
                "relative_path": source.combined_manifest_relative_path,
                "sha256": manifest["sha256"],
                "records_parsed": 0,
            },
            "source_manifest_schema_version": source.expected_manifest_schema_version,
            "source_label_rule_version": source.expected_label_rule_version,
            "split_seed": source.expected_split_seed,
            "split_strategy_version": source.expected_split_strategy_version,
            "split_mapping_sha256": mapping_sha,
            "git": asdict(git),
            "config_sha256": config.config_sha256,
            "source_contract_config_sha256": source.config_sha256,
            "outputs": outputs,
            **asdict(counters),
        }
        write_canonical_json(receipt, staging / "source_projection_receipt.json")
        development._fsync_directory(staging)
        os.rename(staging, output)
        development._fsync_directory(output.parent)
        return receipt
    finally:
        if staging.exists():
            for name in (
                "train.jsonl", "validation.jsonl", "source_projection_receipt.json",
            ):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()

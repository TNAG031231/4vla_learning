from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Protocol

import yaml

from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION, normalize_action
from src.phase0.manifest import (
    json_record,
    write_canonical_json,
    write_jsonl_records,
)
from src.phase0.protocol import (
    MOTION_SOURCE,
    PHASE0_SPLIT_SEED,
    POSE_TIMESTAMP_SOURCE,
    TRAINVAL_MANIFEST_SCHEMA_VERSION,
    is_valid_cam_front_path,
    validate_sha256,
)
from src.phase0.qwen3vl_smoke import read_selected_rule
from src.phase0.qwen_preflight import check_manifest_integrity, sha256_file
from src.phase0.scene_mapping import (
    MAPPING_SCHEMA_VERSION,
    canonical_json_bytes,
    read_scene_mapping,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


PROJECT_SPLITS = ("train", "validation")
MOTION_AVAILABILITY = ("full", "partial", "unavailable")
PROJECTION_VERSION = "phase0.3b-development-projection-v0.1"
PROJECTION_SCHEMA_VERSION = "phase0_3_development_projection_v0.1"
COMBINED_MANIFEST_RELATIVE_PATH = (
    "phase_0_1b/trainval_manifest_v1/manifest.jsonl"
)
COMBINED_MANIFEST_SHA256 = (
    "60517f985fec8fe3977a31660a5204942e9fd36baf09ea4d950328b1f225d1b3"
)
SCENE_MAPPING_RELATIVE_PATH = (
    "phase_0_1b/trainval_manifest_v1/scene_split_mapping_v1.json"
)
SELECTED_RULE_RELATIVE_PATH = (
    "phase_0_2/ego_motion_rule_v0_1/selected_rule.json"
)
OUTPUT_RELATIVE_DIR = "phase_0_3/development_projection_v0_1"
FROZEN_SCENE_COUNTS = {"train": 560, "validation": 140}
FROZEN_SAMPLE_COUNTS = {"train": 14253, "validation": 3594}
FROZEN_MOTION_AVAILABILITY = {
    "train": {"full": 13476, "partial": 392, "unavailable": 385},
    "validation": {"full": 3401, "partial": 99, "unavailable": 94},
}
OUTPUT_FILENAMES = (
    "train.jsonl",
    "validation.jsonl",
    "projection_receipt.json",
)


class NuScenesReader(Protocol):
    def get(self, table_name: str, token: str) -> dict[str, object]:
        ...


class ProducerResult(Protocol):
    records: Sequence[object]


class RecordProducer(Protocol):
    def __call__(
        self,
        *,
        nuscenes: NuScenesReader,
        scene_tokens: Sequence[str],
        scene_splits: Mapping[str, str],
        official_splits: Mapping[str, str],
        split_seed: int,
        split_strategy_version: str,
        split_mapping_sha256: str,
        audit_index: Mapping[str, object],
        dataroot: Path,
        rules: object,
        horizon_sec: float,
        sample_interval_sec: float,
        time_tolerance_sec: float,
        agent_radius_m: float,
    ) -> ProducerResult:
        ...


@dataclass(frozen=True)
class ProjectionConfig:
    projection_version: str
    projection_schema_version: str
    nuscenes_version: str
    combined_manifest_relative_path: str
    expected_combined_manifest_sha256: str
    scene_mapping_relative_path: str
    selected_rule_relative_path: str
    expected_manifest_schema_version: str
    expected_label_rule_version: str
    expected_split_seed: int
    expected_split_strategy_version: str
    allowed_project_splits: tuple[str, ...]
    expected_scene_counts: dict[str, int]
    expected_sample_counts: dict[str, int]
    expected_motion_availability: dict[str, dict[str, int]]
    output_relative_dir: str
    config_sha256: str


@dataclass(frozen=True)
class ProducerInputs:
    rules: object
    horizon_sec: float
    sample_interval_sec: float
    time_tolerance_sec: float
    agent_radius_m: float


@dataclass(frozen=True)
class DevelopmentSceneSelection:
    scene_tokens_by_split: dict[str, tuple[str, ...]]
    scene_splits: dict[str, str]
    official_splits: dict[str, str]
    allowed_scene_tokens: frozenset[str]
    forbidden_test_scene_tokens: frozenset[str]


@dataclass
class IsolationCounters:
    combined_manifest_records_parsed: int = 0
    test_scene_traversal_attempts: int = 0
    test_sample_records_read: int = 0
    test_images_opened: int = 0
    test_labels_read: int = 0


class GuardedNuScenesReader:
    def __init__(
        self,
        reader: NuScenesReader,
        allowed_scene_tokens: frozenset[str],
        forbidden_test_scene_tokens: frozenset[str],
        counters: IsolationCounters,
    ) -> None:
        self._reader = reader
        self._allowed_scene_tokens = allowed_scene_tokens
        self._forbidden_test_scene_tokens = forbidden_test_scene_tokens
        self._counters = counters

    def get(self, table_name: str, token: str) -> dict[str, object]:
        if table_name == "scene":
            if token in self._forbidden_test_scene_tokens:
                self._counters.test_scene_traversal_attempts += 1
                raise ValueError("project test scene traversal is forbidden")
            if token not in self._allowed_scene_tokens:
                raise ValueError("scene traversal is outside development selection")
        record = self._reader.get(table_name, token)
        if table_name == "sample":
            scene_token = record.get("scene_token")
            if scene_token in self._forbidden_test_scene_tokens:
                self._counters.test_sample_records_read += 1
                raise ValueError("project test sample access is forbidden")
            if scene_token not in self._allowed_scene_tokens:
                raise ValueError("sample access is outside development selection")
        return record


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _required_integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"configuration missing integer {key}")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration missing mapping {key}")
    return value


def _relative_posix_path(mapping: Mapping[str, object], key: str) -> str:
    value = _required_string(mapping, key)
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or posix_path.as_posix() != value
    ):
        raise ValueError(f"{key} must be a traversal-free relative POSIX path")
    return value


def _count_mapping(
    raw: Mapping[str, object],
    field_name: str,
    expected_keys: tuple[str, ...],
) -> dict[str, int]:
    if set(raw) != set(expected_keys):
        raise ValueError(f"{field_name} must contain {expected_keys}")
    counts = {}
    for key in expected_keys:
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name}.{key} must be a non-negative integer")
        counts[key] = value
    return counts


def load_config(config_path: Path) -> ProjectionConfig:
    config_bytes = config_path.read_bytes()
    loaded = yaml.safe_load(config_bytes)
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    allowed_splits = loaded.get("allowed_project_splits")
    if allowed_splits != list(PROJECT_SPLITS):
        raise ValueError("allowed_project_splits must be train and validation")
    expected_scene_counts = _count_mapping(
        _required_mapping(loaded, "expected_scene_counts"),
        "expected_scene_counts",
        PROJECT_SPLITS,
    )
    expected_sample_counts = _count_mapping(
        _required_mapping(loaded, "expected_sample_counts"),
        "expected_sample_counts",
        PROJECT_SPLITS,
    )
    motion_root = _required_mapping(loaded, "expected_motion_availability")
    if set(motion_root) != set(PROJECT_SPLITS):
        raise ValueError(
            "expected_motion_availability must contain train and validation"
        )
    expected_motion = {
        split: _count_mapping(
            _required_mapping(motion_root, split),
            f"expected_motion_availability.{split}",
            MOTION_AVAILABILITY,
        )
        for split in PROJECT_SPLITS
    }
    for split in PROJECT_SPLITS:
        if sum(expected_motion[split].values()) != expected_sample_counts[split]:
            raise ValueError(
                f"{split} motion availability counts must match sample count"
            )
    config = ProjectionConfig(
        projection_version=_required_string(loaded, "projection_version"),
        projection_schema_version=_required_string(
            loaded,
            "projection_schema_version",
        ),
        nuscenes_version=_required_string(loaded, "nuscenes_version"),
        combined_manifest_relative_path=_relative_posix_path(
            loaded,
            "combined_manifest_relative_path",
        ),
        expected_combined_manifest_sha256=validate_sha256(
            loaded.get("expected_combined_manifest_sha256"),
            "expected_combined_manifest_sha256",
        ),
        scene_mapping_relative_path=_relative_posix_path(
            loaded,
            "scene_mapping_relative_path",
        ),
        selected_rule_relative_path=_relative_posix_path(
            loaded,
            "selected_rule_relative_path",
        ),
        expected_manifest_schema_version=_required_string(
            loaded,
            "expected_manifest_schema_version",
        ),
        expected_label_rule_version=_required_string(
            loaded,
            "expected_label_rule_version",
        ),
        expected_split_seed=_required_integer(loaded, "expected_split_seed"),
        expected_split_strategy_version=_required_string(
            loaded,
            "expected_split_strategy_version",
        ),
        allowed_project_splits=PROJECT_SPLITS,
        expected_scene_counts=expected_scene_counts,
        expected_sample_counts=expected_sample_counts,
        expected_motion_availability=expected_motion,
        output_relative_dir=_relative_posix_path(loaded, "output_relative_dir"),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    if config.nuscenes_version != "v1.0-trainval":
        raise ValueError("nuscenes_version must be v1.0-trainval")
    frozen_contract = {
        "projection_version": (config.projection_version, PROJECTION_VERSION),
        "projection_schema_version": (
            config.projection_schema_version,
            PROJECTION_SCHEMA_VERSION,
        ),
        "combined_manifest_relative_path": (
            config.combined_manifest_relative_path,
            COMBINED_MANIFEST_RELATIVE_PATH,
        ),
        "expected_combined_manifest_sha256": (
            config.expected_combined_manifest_sha256,
            COMBINED_MANIFEST_SHA256,
        ),
        "scene_mapping_relative_path": (
            config.scene_mapping_relative_path,
            SCENE_MAPPING_RELATIVE_PATH,
        ),
        "selected_rule_relative_path": (
            config.selected_rule_relative_path,
            SELECTED_RULE_RELATIVE_PATH,
        ),
        "expected_manifest_schema_version": (
            config.expected_manifest_schema_version,
            TRAINVAL_MANIFEST_SCHEMA_VERSION,
        ),
        "expected_label_rule_version": (
            config.expected_label_rule_version,
            LABEL_RULE_VERSION,
        ),
        "expected_split_seed": (
            config.expected_split_seed,
            PHASE0_SPLIT_SEED,
        ),
        "expected_split_strategy_version": (
            config.expected_split_strategy_version,
            SPLIT_STRATEGY_VERSION,
        ),
        "expected_scene_counts": (
            config.expected_scene_counts,
            FROZEN_SCENE_COUNTS,
        ),
        "expected_sample_counts": (
            config.expected_sample_counts,
            FROZEN_SAMPLE_COUNTS,
        ),
        "expected_motion_availability": (
            config.expected_motion_availability,
            FROZEN_MOTION_AVAILABILITY,
        ),
        "output_relative_dir": (
            config.output_relative_dir,
            OUTPUT_RELATIVE_DIR,
        ),
    }
    for field_name, (actual, expected) in frozen_contract.items():
        if actual != expected:
            raise ValueError(f"{field_name} does not match frozen projection contract")
    return config


def resolve_derived_path(derived_root: Path, relative_path: str) -> Path:
    root = derived_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("derived path escapes VLA_DERIVED_ROOT") from error
    return resolved


def select_development_scenes(
    mapping: Mapping[str, object],
    config: ProjectionConfig,
) -> DevelopmentSceneSelection:
    if mapping.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION:
        raise ValueError("scene mapping schema does not match projection contract")
    if mapping.get("split_seed") != config.expected_split_seed:
        raise ValueError("scene mapping split seed does not match projection config")
    if (
        mapping.get("split_strategy_version")
        != config.expected_split_strategy_version
    ):
        raise ValueError(
            "scene mapping split strategy does not match projection config"
        )
    if mapping.get("label_rule_version") != config.expected_label_rule_version:
        raise ValueError("scene mapping label rule does not match projection config")
    scenes = mapping.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("scene mapping scenes must be a list")
    selected: dict[str, list[str]] = {split: [] for split in PROJECT_SPLITS}
    scene_splits: dict[str, str] = {}
    official_splits: dict[str, str] = {}
    forbidden_test_scene_tokens = set()
    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise ValueError("scene mapping entry must be an object")
        scene_token = scene.get("scene_token")
        official_split = scene.get("official_split")
        project_split = scene.get("project_split")
        if not isinstance(scene_token, str):
            raise ValueError("scene mapping entry is missing scene_token")
        if project_split in PROJECT_SPLITS:
            if official_split != "train":
                raise ValueError(
                    "development scene must be official train before producer init"
                )
            selected[project_split].append(scene_token)
            scene_splits[scene_token] = project_split
            official_splits[scene_token] = "train"
        elif project_split == "test":
            if official_split != "val":
                raise ValueError("project test scene must be official val")
            forbidden_test_scene_tokens.add(scene_token)
        else:
            raise ValueError("scene mapping has unsupported project split")
    scene_tokens_by_split = {
        split: tuple(sorted(selected[split])) for split in PROJECT_SPLITS
    }
    actual_scene_counts = {
        split: len(scene_tokens_by_split[split]) for split in PROJECT_SPLITS
    }
    if actual_scene_counts != config.expected_scene_counts:
        raise ValueError("development scene counts do not match projection config")
    if len(forbidden_test_scene_tokens) != 150:
        raise ValueError("scene mapping must contain 150 forbidden project test scenes")
    allowed_scene_tokens = frozenset(scene_splits)
    forbidden_tokens = frozenset(forbidden_test_scene_tokens)
    if allowed_scene_tokens & forbidden_tokens:
        raise ValueError("development and project test scenes overlap")
    return DevelopmentSceneSelection(
        scene_tokens_by_split=scene_tokens_by_split,
        scene_splits=scene_splits,
        official_splits=official_splits,
        allowed_scene_tokens=allowed_scene_tokens,
        forbidden_test_scene_tokens=forbidden_tokens,
    )


def _validated_motion(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("current_ego_motion must be a mapping")
    expected_fields = (
        "speed_mps",
        "longitudinal_acceleration_mps2",
        "yaw_rate_radps",
        "source",
        "timestamp_source",
        "availability",
        "history_interval_sec",
        "acceleration_interval_sec",
        "unavailable_reason",
    )
    if set(value) != set(expected_fields):
        raise ValueError("current_ego_motion fields do not match projection contract")
    availability = value.get("availability")
    if availability not in MOTION_AVAILABILITY:
        raise ValueError("unsupported current_ego_motion availability")
    if value.get("source") != MOTION_SOURCE:
        raise ValueError("unsupported current_ego_motion source")
    if value.get("timestamp_source") != POSE_TIMESTAMP_SOURCE:
        raise ValueError("unsupported current_ego_motion timestamp source")
    numeric_fields = (
        "speed_mps",
        "longitudinal_acceleration_mps2",
        "yaw_rate_radps",
        "history_interval_sec",
        "acceleration_interval_sec",
    )
    for field in numeric_fields:
        number = value.get(field)
        if number is not None and (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
        ):
            raise ValueError(f"current_ego_motion.{field} must be finite or null")
    if availability == "full" and any(
        value.get(field) is None for field in numeric_fields
    ):
        raise ValueError("full motion must provide all numeric fields")
    if availability == "partial" and (
        value.get("speed_mps") is None
        or value.get("yaw_rate_radps") is None
        or value.get("history_interval_sec") is None
        or value.get("longitudinal_acceleration_mps2") is not None
        or value.get("acceleration_interval_sec") is not None
    ):
        raise ValueError("partial motion fields are inconsistent")
    if availability == "unavailable" and any(
        value.get(field) is not None for field in numeric_fields
    ):
        raise ValueError("unavailable motion numeric fields must be null")
    unavailable_reason = value.get("unavailable_reason")
    if availability == "full" and unavailable_reason is not None:
        raise ValueError("full motion cannot have unavailable_reason")
    if availability != "full" and not isinstance(unavailable_reason, str):
        raise ValueError("partial or unavailable motion requires a reason")
    return {field: value[field] for field in expected_fields}


def project_record(
    source_record: object,
    *,
    split: str,
    config: ProjectionConfig,
    split_mapping_sha256: str,
) -> dict[str, object]:
    source = json_record(source_record)
    if source.get("split") != split or split not in PROJECT_SPLITS:
        raise ValueError("producer record split does not match requested split")
    if source.get("official_split") != "train":
        raise ValueError("development producer record must be official train")
    if source.get("manifest_schema_version") != config.expected_manifest_schema_version:
        raise ValueError("producer manifest schema version mismatch")
    if source.get("label_rule_version") != config.expected_label_rule_version:
        raise ValueError("producer label rule version mismatch")
    if source.get("split_seed") != config.expected_split_seed:
        raise ValueError("producer split seed mismatch")
    if (
        source.get("split_strategy_version")
        != config.expected_split_strategy_version
    ):
        raise ValueError("producer split strategy version mismatch")
    if source.get("split_mapping_sha256") != split_mapping_sha256:
        raise ValueError("producer split mapping SHA-256 mismatch")
    sample_token = source.get("sample_token")
    scene_token = source.get("scene_token")
    timestamp = source.get("timestamp")
    cam_front_path = source.get("cam_front_path")
    if not isinstance(sample_token, str) or not sample_token:
        raise ValueError("producer record is missing sample_token")
    if not isinstance(scene_token, str) or not scene_token:
        raise ValueError("producer record is missing scene_token")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise ValueError("producer record timestamp must be an integer")
    if not isinstance(cam_front_path, str) or not is_valid_cam_front_path(
        cam_front_path
    ):
        raise ValueError("producer record has invalid CAM_FRONT path")
    target_action = source.get("meta_action")
    if not isinstance(target_action, str):
        raise ValueError("producer record is missing meta_action")
    target_action = normalize_action(target_action)
    motion = _validated_motion(source.get("current_ego_motion"))
    material: dict[str, object] = {
        "projection_schema_version": config.projection_schema_version,
        "sample_token": sample_token,
        "scene_token": scene_token,
        "timestamp": timestamp,
        "split": split,
        "cam_front_path": cam_front_path,
        "current_ego_motion": motion,
        "target_action": target_action,
        "source_manifest_schema_version": config.expected_manifest_schema_version,
        "label_rule_version": config.expected_label_rule_version,
        "split_mapping_sha256": split_mapping_sha256,
        "source_combined_manifest_sha256": (
            config.expected_combined_manifest_sha256
        ),
    }
    material["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    return material


def _projection_records(
    source_records: Sequence[object],
    *,
    split: str,
    config: ProjectionConfig,
    split_mapping_sha256: str,
    selected_scene_tokens: frozenset[str],
) -> tuple[dict[str, object], ...]:
    records = tuple(
        project_record(
            record,
            split=split,
            config=config,
            split_mapping_sha256=split_mapping_sha256,
        )
        for record in source_records
    )
    if len(records) != config.expected_sample_counts[split]:
        raise ValueError(f"{split} sample count does not match projection config")
    if any(record["scene_token"] not in selected_scene_tokens for record in records):
        raise ValueError("producer returned a record outside selected scenes")
    motion_counts = Counter(
        str(record["current_ego_motion"]["availability"])
        for record in records
    )
    if {
        key: motion_counts[key] for key in MOTION_AVAILABILITY
    } != config.expected_motion_availability[split]:
        raise ValueError(
            f"{split} motion availability does not match projection config"
        )
    return records


def _run_producer(
    *,
    producer: RecordProducer,
    reader: NuScenesReader,
    selection: DevelopmentSceneSelection,
    split: str,
    config: ProjectionConfig,
    split_mapping_sha256: str,
    nuscenes_root: Path,
    producer_inputs: ProducerInputs,
) -> Sequence[object]:
    scene_tokens = selection.scene_tokens_by_split[split]
    if set(scene_tokens) & selection.forbidden_test_scene_tokens:
        raise ValueError("producer scene tokens intersect project test scenes")
    if tuple(sorted(scene_tokens)) != scene_tokens:
        raise ValueError("producer scene tokens must be deterministically sorted")
    result = producer(
        nuscenes=reader,
        scene_tokens=scene_tokens,
        scene_splits=selection.scene_splits,
        official_splits=selection.official_splits,
        split_seed=config.expected_split_seed,
        split_strategy_version=config.expected_split_strategy_version,
        split_mapping_sha256=split_mapping_sha256,
        audit_index={},
        dataroot=nuscenes_root,
        rules=producer_inputs.rules,
        horizon_sec=producer_inputs.horizon_sec,
        sample_interval_sec=producer_inputs.sample_interval_sec,
        time_tolerance_sec=producer_inputs.time_tolerance_sec,
        agent_radius_m=producer_inputs.agent_radius_m,
    )
    return result.records


def _output_statistics(
    records_by_split: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    action_distribution = {}
    motion_distribution = {}
    for split in PROJECT_SPLITS:
        records = records_by_split[split]
        action_counts = Counter(str(record["target_action"]) for record in records)
        action_distribution[split] = {
            action: action_counts[action] for action in ACTION_SCHEMA
        }
        motion_counts = Counter(
            str(record["current_ego_motion"]["availability"])
            for record in records
        )
        motion_distribution[split] = {
            state: motion_counts[state] for state in MOTION_AVAILABILITY
        }
    return action_distribution, motion_distribution


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_staging_directory(path: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        (path / filename).unlink(missing_ok=True)
    path.rmdir()


def _validate_existing_artifact(
    output_dir: Path,
    *,
    config: ProjectionConfig,
    split_mapping_sha256: str,
) -> dict[str, object]:
    if not output_dir.is_dir():
        raise ValueError("existing projection output path is not a directory")
    receipt_path = output_dir / "projection_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("existing projection is missing its receipt")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("existing projection receipt must be an object")
    config_metadata = payload.get("config")
    combined_metadata = payload.get("combined_manifest")
    mapping_metadata = payload.get("scene_mapping")
    outputs = payload.get("outputs")
    if not all(
        isinstance(value, Mapping)
        for value in (config_metadata, combined_metadata, mapping_metadata, outputs)
    ):
        raise ValueError("existing projection receipt metadata is incomplete")
    if (
        payload.get("projection_version") != config.projection_version
        or payload.get("projection_schema_version")
        != config.projection_schema_version
        or config_metadata.get("sha256") != config.config_sha256
        or combined_metadata.get("sha256")
        != config.expected_combined_manifest_sha256
        or mapping_metadata.get("split_mapping_sha256")
        != split_mapping_sha256
    ):
        raise ValueError("existing projection provenance does not match config")
    for split in PROJECT_SPLITS:
        output_metadata = outputs.get(split)
        if not isinstance(output_metadata, Mapping):
            raise ValueError(f"existing projection is missing {split} metadata")
        output_path = output_dir / f"{split}.jsonl"
        if not output_path.is_file():
            raise ValueError(f"existing projection is missing {split}.jsonl")
        if sha256_file(output_path) != output_metadata.get("sha256"):
            raise ValueError(f"existing {split} projection SHA-256 mismatch")
        if output_metadata.get("record_count") != config.expected_sample_counts[split]:
            raise ValueError(f"existing {split} projection count mismatch")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_development_projection(
    *,
    config: ProjectionConfig,
    config_relative_path: str,
    repository_root: Path,
    nuscenes_root: Path,
    derived_root: Path,
    nuscenes: NuScenesReader,
    producer: RecordProducer,
    producer_inputs: ProducerInputs,
    git_commit: str,
    now_utc: Callable[[], str] = _utc_now,
) -> dict[str, object]:
    combined_path = resolve_derived_path(
        derived_root,
        config.combined_manifest_relative_path,
    )
    manifest_metadata, failures = check_manifest_integrity(
        combined_path,
        config.expected_combined_manifest_sha256,
    )
    if failures:
        raise ValueError(str(failures[0]["code"]))
    selected_rule = read_selected_rule(
        resolve_derived_path(derived_root, config.selected_rule_relative_path),
        config.expected_combined_manifest_sha256,
    )
    mapping = read_scene_mapping(
        resolve_derived_path(derived_root, config.scene_mapping_relative_path)
    )
    split_mapping_sha256 = validate_sha256(
        mapping.get("scene_split_mapping_sha256"),
        "scene_split_mapping_sha256",
    )
    if selected_rule["split_mapping_sha256"] != split_mapping_sha256:
        raise ValueError("selected rule and scene mapping SHA-256 mismatch")
    selection = select_development_scenes(mapping, config)
    output_dir = resolve_derived_path(derived_root, config.output_relative_dir)
    try:
        output_dir.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("development projection must not be written in repository")
    if output_dir.exists():
        receipt = _validate_existing_artifact(
            output_dir,
            config=config,
            split_mapping_sha256=split_mapping_sha256,
        )
        return {"status": "already_exists", "receipt": receipt}

    counters = IsolationCounters()
    guarded_reader = GuardedNuScenesReader(
        nuscenes,
        selection.allowed_scene_tokens,
        selection.forbidden_test_scene_tokens,
        counters,
    )
    records_by_split: dict[str, tuple[dict[str, object], ...]] = {}
    seen_sample_tokens = set()
    for split in PROJECT_SPLITS:
        source_records = _run_producer(
            producer=producer,
            reader=guarded_reader,
            selection=selection,
            split=split,
            config=config,
            split_mapping_sha256=split_mapping_sha256,
            nuscenes_root=nuscenes_root,
            producer_inputs=producer_inputs,
        )
        records = _projection_records(
            source_records,
            split=split,
            config=config,
            split_mapping_sha256=split_mapping_sha256,
            selected_scene_tokens=frozenset(
                selection.scene_tokens_by_split[split]
            ),
        )
        sample_tokens = {str(record["sample_token"]) for record in records}
        if len(sample_tokens) != len(records) or seen_sample_tokens & sample_tokens:
            raise ValueError("projection sample tokens must be globally unique")
        seen_sample_tokens.update(sample_tokens)
        records_by_split[split] = records
    if counters != IsolationCounters():
        raise ValueError("test isolation counters must remain zero")

    action_distribution, motion_distribution = _output_statistics(
        records_by_split
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=output_dir.parent,
        )
    )
    published = False
    try:
        output_metadata: dict[str, dict[str, object]] = {}
        for split in PROJECT_SPLITS:
            staged_path = staging_dir / f"{split}.jsonl"
            write_jsonl_records(records_by_split[split], staged_path)
            output_metadata[split] = {
                "relative_path": (
                    PurePosixPath(config.output_relative_dir) / f"{split}.jsonl"
                ).as_posix(),
                "sha256": sha256_file(staged_path),
                "record_count": len(records_by_split[split]),
            }
        receipt: dict[str, object] = {
            "projection_version": config.projection_version,
            "projection_schema_version": config.projection_schema_version,
            "generated_at_utc": now_utc(),
            "git_commit": git_commit,
            "config": {
                "relative_path": config_relative_path,
                "sha256": config.config_sha256,
            },
            "nuscenes_version": config.nuscenes_version,
            "combined_manifest": {
                "relative_path": config.combined_manifest_relative_path,
                "file_size_bytes": manifest_metadata["file_size_bytes"],
                "sha256": manifest_metadata["sha256"],
                "records_parsed": 0,
            },
            "scene_mapping": {
                "relative_path": config.scene_mapping_relative_path,
                "mapping_schema_version": mapping["mapping_schema_version"],
                "split_mapping_sha256": split_mapping_sha256,
                "train_scene_count": config.expected_scene_counts["train"],
                "validation_scene_count": config.expected_scene_counts[
                    "validation"
                ],
                "test_scene_count": len(selection.forbidden_test_scene_tokens),
            },
            "selected_rule": {
                "relative_path": config.selected_rule_relative_path,
                "manifest_sha256": selected_rule["manifest_sha256"],
                "split_mapping_sha256": selected_rule["split_mapping_sha256"],
                "test_evaluation_performed": False,
            },
            "outputs": output_metadata,
            "action_distribution_by_split": action_distribution,
            "motion_availability_distribution_by_split": motion_distribution,
            "absolute_path_leak_count": 0,
            "invalid_cam_front_path_count": 0,
            "future_trajectory_used_for_target_derivation": True,
            "future_trajectory_written_to_projection": False,
            "nearby_agents_written_to_projection": False,
            "current_ego_pose_written_to_projection": False,
            "audit_records_read": 0,
            "combined_manifest_records_parsed": 0,
            "test_scene_traversal_attempts": 0,
            "test_sample_records_read": 0,
            "test_images_opened": 0,
            "test_labels_read": 0,
            "test_evaluation_performed": False,
            "model_load_performed": False,
            "processor_load_performed": False,
        }
        write_canonical_json(receipt, staging_dir / "projection_receipt.json")
        _fsync_directory(staging_dir)
        os.rename(staging_dir, output_dir)
        published = True
        _fsync_directory(output_dir.parent)
        return {"status": "created", "receipt": receipt}
    finally:
        if not published and staging_dir.exists():
            _cleanup_staging_directory(staging_dir)

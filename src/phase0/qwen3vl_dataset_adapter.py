from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import tempfile
from typing import BinaryIO

import yaml

from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION, normalize_action
from src.phase0.protocol import (
    TRAINVAL_MANIFEST_SCHEMA_VERSION,
    is_valid_cam_front_path,
    validate_sha256,
)


PROJECT_SPLITS = ("train", "validation")
VARIANTS = ("image_only", "image_ego_state")
MOTION_AVAILABILITY = ("full", "partial", "unavailable")
ADAPTER_VERSION = "phase0.3b-qwen3vl-dataset-adapter-v0.1"
ADAPTER_SCHEMA_VERSION = "phase0_3_qwen3vl_dataset_adapter_v0.1"
SOURCE_PROJECTION_RELATIVE_DIR = "phase_0_3/development_projection_v0_1"
SOURCE_RECEIPT_RELATIVE_PATH = "projection_receipt.json"
SOURCE_PROJECTION_VERSION = "phase0.3b-development-projection-v0.1"
SOURCE_PROJECTION_SCHEMA_VERSION = "phase0_3_development_projection_v0.1"
SOURCE_PROJECTION_GIT_COMMIT = "ee50005caaa2ba3fa28074a70bbe9178b089b6a8"
SOURCE_RECEIPT_SHA256 = (
    "16fd73b069b61061fdf3f81e0b31ee7367003e13fcfe3196df2482d50150673b"
)
SOURCE_FILE_CONTRACT = {
    "train": {
        "relative_path": "train.jsonl",
        "sha256": "32dd520c87921804e6640273bb1a4ad663acc72c82a43e0f6f092759389dfb5a",
        "record_count": 14253,
    },
    "validation": {
        "relative_path": "validation.jsonl",
        "sha256": "0cbd4cf94bde422d6810caa5a302d079e4770a49dd6c3f05548d9b8c12fac94a",
        "record_count": 3594,
    },
}
FROZEN_MOTION_AVAILABILITY = {
    "train": {"full": 13476, "partial": 392, "unavailable": 385},
    "validation": {"full": 3401, "partial": 99, "unavailable": 94},
}
EGO_PREFIX = "Current ego state:"
EGO_FIELD_ORDER = (
    "speed_mps",
    "longitudinal_acceleration_mps2",
    "yaw_rate_radps",
    "history_interval_sec",
    "acceleration_interval_sec",
    "availability",
)
FLOAT_PRECISION = 6
UNAVAILABLE_TOKEN = "unavailable"
OUTPUT_RELATIVE_DIR = "phase_0_3/qwen3vl_dataset_adapter_v0_1"
TASK_PROMPT = (
    "Observe the front-facing driving image and select exactly one current "
    "driving meta-action. Allowed labels: keep, accelerate, decelerate, "
    "stop, left_lateral, right_lateral. Return only the label and no "
    "explanation."
)
SOURCE_RECORD_FIELDS = frozenset(
    (
        "projection_schema_version",
        "sample_token",
        "scene_token",
        "timestamp",
        "split",
        "cam_front_path",
        "current_ego_motion",
        "target_action",
        "source_manifest_schema_version",
        "label_rule_version",
        "split_mapping_sha256",
        "source_combined_manifest_sha256",
        "projection_record_sha256",
    )
)
ADAPTER_RECORD_FIELDS = frozenset(
    (
        "adapter_schema_version",
        "adapter_version",
        "sample_token",
        "scene_token",
        "timestamp",
        "split",
        "variant",
        "cam_front_path",
        "messages",
        "target_action",
        "source_projection_record_sha256",
        "source_projection_file_sha256",
        "source_projection_receipt_sha256",
        "adapter_record_sha256",
    )
)
OUTPUT_FILENAMES = (
    "train_image_only.jsonl",
    "validation_image_only.jsonl",
    "train_image_ego_state.jsonl",
    "validation_image_ego_state.jsonl",
    "adapter_receipt.json",
)
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SourceFileConfig:
    relative_path: str
    sha256: str
    record_count: int


@dataclass(frozen=True)
class AdapterConfig:
    adapter_version: str
    adapter_schema_version: str
    source_projection_relative_dir: str
    source_projection_receipt: str
    expected_source_projection_version: str
    expected_source_projection_schema_version: str
    expected_source_projection_git_commit: str
    expected_source_receipt_sha256: str
    source_files: dict[str, SourceFileConfig]
    variants: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    task_prompt: str
    ego_prefix: str
    ego_field_order: tuple[str, ...]
    float_precision: int
    unavailable_token: str
    output_relative_dir: str
    config_sha256: str


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    branch: str | None
    detached_head: bool
    worktree_clean: bool


@dataclass
class SourceValidationState:
    seen_sample_tokens: set[str] = field(default_factory=set)
    split_mapping_sha256: str | None = None
    source_combined_manifest_sha256: str | None = None
    forbidden_split_records_seen: int = 0


GitRunner = Callable[[Path, tuple[str, ...]], str]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing string {key}")
    return value


def _required_mapping(
    mapping: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping {key}")
    return value


def _required_integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"missing integer {key}")
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


def _string_tuple(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def load_config(path: Path) -> AdapterConfig:
    config_bytes = path.read_bytes()
    loaded = yaml.safe_load(config_bytes)
    if not isinstance(loaded, Mapping):
        raise ValueError("adapter config must be a mapping")
    source_files_value = _required_mapping(loaded, "source_files")
    if set(source_files_value) != set(PROJECT_SPLITS):
        raise ValueError("source_files must contain train and validation only")
    source_files: dict[str, SourceFileConfig] = {}
    for split in PROJECT_SPLITS:
        source_value = _required_mapping(source_files_value, split)
        source_files[split] = SourceFileConfig(
            relative_path=_relative_posix_path(source_value, "relative_path"),
            sha256=validate_sha256(
                source_value.get("sha256"), f"source_files.{split}.sha256"
            ),
            record_count=_required_integer(source_value, "record_count"),
        )
    ego = _required_mapping(loaded, "ego_state")
    config = AdapterConfig(
        adapter_version=_required_string(loaded, "adapter_version"),
        adapter_schema_version=_required_string(loaded, "adapter_schema_version"),
        source_projection_relative_dir=_relative_posix_path(
            loaded, "source_projection_relative_dir"
        ),
        source_projection_receipt=_relative_posix_path(
            loaded, "source_projection_receipt"
        ),
        expected_source_projection_version=_required_string(
            loaded, "expected_source_projection_version"
        ),
        expected_source_projection_schema_version=_required_string(
            loaded, "expected_source_projection_schema_version"
        ),
        expected_source_projection_git_commit=_required_string(
            loaded, "expected_source_projection_git_commit"
        ),
        expected_source_receipt_sha256=validate_sha256(
            loaded.get("expected_source_receipt_sha256"),
            "expected_source_receipt_sha256",
        ),
        source_files=source_files,
        variants=_string_tuple(loaded, "variants"),
        allowed_actions=_string_tuple(loaded, "allowed_actions"),
        task_prompt=_required_string(loaded, "task_prompt"),
        ego_prefix=_required_string(ego, "prefix"),
        ego_field_order=_string_tuple(ego, "field_order"),
        float_precision=_required_integer(ego, "float_precision"),
        unavailable_token=_required_string(ego, "unavailable_token"),
        output_relative_dir=_relative_posix_path(loaded, "output_relative_dir"),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    frozen_contract = {
        "adapter_version": (config.adapter_version, ADAPTER_VERSION),
        "adapter_schema_version": (
            config.adapter_schema_version,
            ADAPTER_SCHEMA_VERSION,
        ),
        "source_projection_relative_dir": (
            config.source_projection_relative_dir,
            SOURCE_PROJECTION_RELATIVE_DIR,
        ),
        "source_projection_receipt": (
            config.source_projection_receipt,
            SOURCE_RECEIPT_RELATIVE_PATH,
        ),
        "expected_source_projection_version": (
            config.expected_source_projection_version,
            SOURCE_PROJECTION_VERSION,
        ),
        "expected_source_projection_schema_version": (
            config.expected_source_projection_schema_version,
            SOURCE_PROJECTION_SCHEMA_VERSION,
        ),
        "expected_source_projection_git_commit": (
            config.expected_source_projection_git_commit,
            SOURCE_PROJECTION_GIT_COMMIT,
        ),
        "expected_source_receipt_sha256": (
            config.expected_source_receipt_sha256,
            SOURCE_RECEIPT_SHA256,
        ),
        "source_files": (
            {
                split: {
                    "relative_path": source.relative_path,
                    "sha256": source.sha256,
                    "record_count": source.record_count,
                }
                for split, source in config.source_files.items()
            },
            SOURCE_FILE_CONTRACT,
        ),
        "variants": (config.variants, VARIANTS),
        "allowed_actions": (config.allowed_actions, ACTION_SCHEMA),
        "task_prompt": (config.task_prompt, TASK_PROMPT),
        "ego_prefix": (config.ego_prefix, EGO_PREFIX),
        "ego_field_order": (config.ego_field_order, EGO_FIELD_ORDER),
        "float_precision": (config.float_precision, FLOAT_PRECISION),
        "unavailable_token": (config.unavailable_token, UNAVAILABLE_TOKEN),
        "output_relative_dir": (config.output_relative_dir, OUTPUT_RELATIVE_DIR),
    }
    for field_name, (actual, expected) in frozen_contract.items():
        if actual != expected:
            raise ValueError(f"{field_name} does not match frozen adapter contract")
    return config


def resolve_derived_path(derived_root: Path, relative_path: str) -> Path:
    root = derived_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("derived path escapes VLA_DERIVED_ROOT") from error
    return resolved


def _run_git(repository_root: Path, arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_git_provenance(provenance: object) -> GitProvenance:
    if not isinstance(provenance, GitProvenance):
        raise ValueError("git_provenance must be a GitProvenance instance")
    if GIT_COMMIT_PATTERN.fullmatch(provenance.commit) is None:
        raise ValueError("git commit must be a lowercase 40-character SHA")
    if provenance.worktree_clean is not True:
        raise ValueError("adapter execution requires a clean worktree")
    if provenance.detached_head:
        if provenance.branch is not None:
            raise ValueError("detached Git provenance must not record a branch")
    elif not isinstance(provenance.branch, str) or not provenance.branch:
        raise ValueError("attached Git provenance must record a branch")
    return provenance


def collect_git_provenance(
    repository_root: Path,
    git_runner: Callable[[Path, tuple[str, ...]], str] = _run_git,
) -> GitProvenance:
    resolved_root = repository_root.resolve()
    top_level = Path(
        git_runner(resolved_root, ("rev-parse", "--show-toplevel"))
    ).resolve()
    if top_level != resolved_root:
        raise ValueError("repository root does not match Git top-level")
    branch = git_runner(resolved_root, ("branch", "--show-current"))
    provenance = GitProvenance(
        commit=git_runner(resolved_root, ("rev-parse", "HEAD")),
        branch=branch or None,
        detached_head=not branch,
        worktree_clean=not git_runner(
            resolved_root,
            ("status", "--porcelain", "--untracked-files=all"),
        ),
    )
    return validate_git_provenance(provenance)


def _require_exact(
    mapping: Mapping[str, object], key: str, expected: object, context: str
) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{context} {key} mismatch")


def validate_source_receipt(
    payload: Mapping[str, object], config: AdapterConfig
) -> dict[str, dict[str, int]]:
    _require_exact(
        payload,
        "projection_version",
        config.expected_source_projection_version,
        "source receipt",
    )
    _require_exact(
        payload,
        "projection_schema_version",
        config.expected_source_projection_schema_version,
        "source receipt",
    )
    _require_exact(payload, "nuscenes_version", "v1.0-trainval", "source receipt")
    git = _required_mapping(payload, "git")
    _require_exact(
        git,
        "commit",
        config.expected_source_projection_git_commit,
        "source receipt git",
    )
    _require_exact(git, "worktree_clean", True, "source receipt git")
    outputs = _required_mapping(payload, "outputs")
    if set(outputs) != set(PROJECT_SPLITS):
        raise ValueError(
            "source receipt outputs must contain train and validation only"
        )
    for split in PROJECT_SPLITS:
        output = _required_mapping(outputs, split)
        source = config.source_files[split]
        _require_exact(output, "relative_path", source.relative_path, "source output")
        _require_exact(output, "sha256", source.sha256, "source output")
        _require_exact(output, "record_count", source.record_count, "source output")
    motion = _required_mapping(
        payload, "motion_availability_distribution_by_split"
    )
    validated_motion: dict[str, dict[str, int]] = {}
    for split in PROJECT_SPLITS:
        counts = _required_mapping(motion, split)
        expected = FROZEN_MOTION_AVAILABILITY[split]
        _require_exact(motion, split, expected, "source motion distribution")
        validated_motion[split] = {key: int(counts[key]) for key in expected}
    frozen_flags = {
        "combined_manifest_records_parsed": 0,
        "test_scene_traversal_attempts": 0,
        "test_sample_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "model_load_performed": False,
        "processor_load_performed": False,
    }
    for key, expected in frozen_flags.items():
        _require_exact(payload, key, expected, "source receipt")
    action = _required_mapping(payload, "action_distribution_by_split")
    action_distribution: dict[str, dict[str, int]] = {}
    for split in PROJECT_SPLITS:
        counts = _required_mapping(action, split)
        if set(counts) != set(ACTION_SCHEMA):
            raise ValueError("source action distribution schema mismatch")
        values: dict[str, int] = {}
        for name in ACTION_SCHEMA:
            count = counts.get(name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("source action distribution count is invalid")
            values[name] = count
        if sum(values.values()) != config.source_files[split].record_count:
            raise ValueError("source action distribution total mismatch")
        action_distribution[split] = values
    return action_distribution


def _finite_number(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"current_ego_motion.{field_name} must be finite or null")
    return float(value)


def validate_motion(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("current_ego_motion must be a mapping")
    expected_fields = frozenset(
        (
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
    )
    if set(value) != expected_fields:
        raise ValueError("current_ego_motion fields mismatch")
    if value.get("source") != "ego_pose_past_difference":
        raise ValueError("unsupported current_ego_motion source")
    if value.get("timestamp_source") != "CAM_FRONT_sample_data":
        raise ValueError("unsupported current_ego_motion timestamp source")
    availability = value.get("availability")
    if availability not in MOTION_AVAILABILITY:
        raise ValueError("unsupported motion availability")
    numeric_fields = EGO_FIELD_ORDER[:-1]
    numbers = {name: _finite_number(value.get(name), name) for name in numeric_fields}
    if availability == "full" and any(number is None for number in numbers.values()):
        raise ValueError("full motion must provide all numeric fields")
    if availability == "partial" and (
        numbers["speed_mps"] is None
        or numbers["yaw_rate_radps"] is None
        or numbers["history_interval_sec"] is None
        or numbers["longitudinal_acceleration_mps2"] is not None
        or numbers["acceleration_interval_sec"] is not None
    ):
        raise ValueError("partial motion fields are inconsistent")
    if availability == "unavailable" and any(
        number is not None for number in numbers.values()
    ):
        raise ValueError("unavailable motion numeric fields must be null")
    unavailable_reason = value.get("unavailable_reason")
    if availability == "full" and unavailable_reason is not None:
        raise ValueError("full motion cannot have unavailable_reason")
    if availability != "full" and (
        not isinstance(unavailable_reason, str) or not unavailable_reason
    ):
        raise ValueError("incomplete motion requires unavailable_reason")
    return {**numbers, "availability": availability}


def serialize_ego_state(motion: object, config: AdapterConfig) -> str:
    validated = validate_motion(motion)
    fields = []
    for name in config.ego_field_order:
        value = validated[name]
        if name == "availability":
            rendered = str(value)
        elif value is None:
            rendered = config.unavailable_token
        else:
            rounded = round(float(value), config.float_precision)
            if rounded == 0.0:
                rounded = 0.0
            rendered = f"{rounded:.{config.float_precision}f}"
        fields.append(f"{name}={rendered}")
    return f"{config.ego_prefix}\n" + "; ".join(fields)


def validate_source_record(
    record: Mapping[str, object],
    *,
    expected_split: str,
    state: SourceValidationState,
) -> dict[str, object]:
    split = record.get("split")
    if split != expected_split or split not in PROJECT_SPLITS:
        state.forbidden_split_records_seen += 1
        raise ValueError("forbidden or mismatched source split")
    if set(record) != SOURCE_RECORD_FIELDS:
        raise ValueError("source record fields mismatch")
    if record.get("projection_schema_version") != SOURCE_PROJECTION_SCHEMA_VERSION:
        raise ValueError("source projection schema version mismatch")
    material = {
        key: value
        for key, value in record.items()
        if key != "projection_record_sha256"
    }
    expected_hash = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    if record.get("projection_record_sha256") != expected_hash:
        raise ValueError("invalid source projection record SHA-256")
    sample_token = _required_string(record, "sample_token")
    if sample_token in state.seen_sample_tokens:
        raise ValueError("duplicate source sample_token")
    state.seen_sample_tokens.add(sample_token)
    scene_token = _required_string(record, "scene_token")
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise ValueError("source timestamp must be an integer")
    cam_front_path = record.get("cam_front_path")
    if not isinstance(cam_front_path, str) or not is_valid_cam_front_path(
        cam_front_path
    ):
        raise ValueError("source record has invalid CAM_FRONT path")
    action_value = record.get("target_action")
    if not isinstance(action_value, str):
        raise ValueError("source target_action must be a string")
    target_action = normalize_action(action_value)
    if record.get("source_manifest_schema_version") != TRAINVAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source manifest schema version mismatch")
    if record.get("label_rule_version") != LABEL_RULE_VERSION:
        raise ValueError("source label rule version mismatch")
    split_mapping_sha256 = validate_sha256(
        record.get("split_mapping_sha256"), "split_mapping_sha256"
    )
    source_manifest_sha256 = validate_sha256(
        record.get("source_combined_manifest_sha256"),
        "source_combined_manifest_sha256",
    )
    if state.split_mapping_sha256 is None:
        state.split_mapping_sha256 = split_mapping_sha256
    elif state.split_mapping_sha256 != split_mapping_sha256:
        raise ValueError("source split mapping SHA-256 is inconsistent")
    if state.source_combined_manifest_sha256 is None:
        state.source_combined_manifest_sha256 = source_manifest_sha256
    elif state.source_combined_manifest_sha256 != source_manifest_sha256:
        raise ValueError("source combined manifest SHA-256 is inconsistent")
    validate_motion(record.get("current_ego_motion"))
    return {
        "sample_token": sample_token,
        "scene_token": scene_token,
        "timestamp": timestamp,
        "split": split,
        "cam_front_path": cam_front_path,
        "current_ego_motion": record["current_ego_motion"],
        "target_action": target_action,
        "projection_record_sha256": expected_hash,
    }


def build_messages(
    source: Mapping[str, object], variant: str, config: AdapterConfig
) -> list[dict[str, object]]:
    if variant not in config.variants:
        raise ValueError("unsupported adapter variant")
    prompt = config.task_prompt
    if variant == "image_ego_state":
        ego_state = serialize_ego_state(source["current_ego_motion"], config)
        prompt = f"{ego_state}\n\n{prompt}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source["cam_front_path"]},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": source["target_action"]}],
        },
    ]


def adapter_record(
    source: Mapping[str, object],
    *,
    variant: str,
    config: AdapterConfig,
    source_file_sha256: str,
) -> dict[str, object]:
    material: dict[str, object] = {
        "adapter_schema_version": config.adapter_schema_version,
        "adapter_version": config.adapter_version,
        "sample_token": source["sample_token"],
        "scene_token": source["scene_token"],
        "timestamp": source["timestamp"],
        "split": source["split"],
        "variant": variant,
        "cam_front_path": source["cam_front_path"],
        "messages": build_messages(source, variant, config),
        "target_action": source["target_action"],
        "source_projection_record_sha256": source["projection_record_sha256"],
        "source_projection_file_sha256": source_file_sha256,
        "source_projection_receipt_sha256": config.expected_source_receipt_sha256,
    }
    material["adapter_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    return material


def _load_json_object(path: Path, context: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    return payload


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield payload


def _write_json_line(
    handle: BinaryIO, record: Mapping[str, object]
) -> None:
    handle.write(canonical_json_bytes(record))
    handle.write(b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_staging(path: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        (path / filename).unlink(missing_ok=True)
    path.rmdir()


def _validate_adapter_output_record(
    record: Mapping[str, object],
    *,
    split: str,
    variant: str,
    config: AdapterConfig,
    source_file_sha256: str,
) -> str:
    if set(record) != ADAPTER_RECORD_FIELDS:
        raise ValueError("existing adapter record fields mismatch")
    _require_exact(
        record,
        "adapter_schema_version",
        config.adapter_schema_version,
        "adapter record",
    )
    _require_exact(
        record,
        "adapter_version",
        config.adapter_version,
        "adapter record",
    )
    _require_exact(record, "split", split, "adapter record")
    _require_exact(record, "variant", variant, "adapter record")
    _require_exact(
        record,
        "source_projection_file_sha256",
        source_file_sha256,
        "adapter record",
    )
    _require_exact(
        record,
        "source_projection_receipt_sha256",
        config.expected_source_receipt_sha256,
        "adapter record",
    )
    cam_front_path = record.get("cam_front_path")
    if not isinstance(cam_front_path, str) or not is_valid_cam_front_path(
        cam_front_path
    ):
        raise ValueError("existing adapter record CAM_FRONT path is invalid")
    action = record.get("target_action")
    if not isinstance(action, str):
        raise ValueError("existing adapter target_action must be a string")
    normalize_action(action)
    _required_string(record, "sample_token")
    _required_string(record, "scene_token")
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise ValueError("existing adapter timestamp must be an integer")
    validate_sha256(
        record.get("source_projection_record_sha256"),
        "source_projection_record_sha256",
    )
    _validate_messages(
        record.get("messages"),
        variant=variant,
        cam_front_path=cam_front_path,
        target_action=action,
        config=config,
    )
    material = {
        key: value
        for key, value in record.items()
        if key != "adapter_record_sha256"
    }
    expected_hash = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    if record.get("adapter_record_sha256") != expected_hash:
        raise ValueError("existing adapter record SHA-256 mismatch")
    return action


def _validate_messages(
    value: object,
    *,
    variant: str,
    cam_front_path: str,
    target_action: str,
    config: AdapterConfig,
) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("existing adapter messages mismatch")
    user, assistant = value
    if not isinstance(user, Mapping) or not isinstance(assistant, Mapping):
        raise ValueError("existing adapter message must be an object")
    if set(user) != {"role", "content"} or user.get("role") != "user":
        raise ValueError("existing adapter user message mismatch")
    content = user.get("content")
    if not isinstance(content, list) or len(content) != 2:
        raise ValueError("existing adapter user content mismatch")
    expected_image = {"type": "image", "image": cam_front_path}
    if content[0] != expected_image:
        raise ValueError("existing adapter image content mismatch")
    text_item = content[1]
    if not isinstance(text_item, Mapping) or set(text_item) != {"type", "text"}:
        raise ValueError("existing adapter text content mismatch")
    if text_item.get("type") != "text":
        raise ValueError("existing adapter text content type mismatch")
    prompt = text_item.get("text")
    if not isinstance(prompt, str):
        raise ValueError("existing adapter prompt must be a string")
    if variant == "image_only":
        if prompt != config.task_prompt:
            raise ValueError("existing image-only prompt mismatch")
    else:
        separator = f"\n\n{config.task_prompt}"
        if not prompt.startswith(f"{config.ego_prefix}\n") or not prompt.endswith(
            separator
        ):
            raise ValueError("existing ego-state prompt mismatch")
        ego_line = prompt[len(config.ego_prefix) + 1 : -len(separator)]
        parts = ego_line.split("; ")
        if len(parts) != len(config.ego_field_order):
            raise ValueError("existing ego-state field count mismatch")
        names = tuple(part.partition("=")[0] for part in parts)
        if names != config.ego_field_order or any("=" not in part for part in parts):
            raise ValueError("existing ego-state field order mismatch")
        serialized = {
            name: part.partition("=")[2]
            for name, part in zip(config.ego_field_order, parts, strict=True)
        }
        availability = serialized["availability"]
        if availability not in MOTION_AVAILABILITY:
            raise ValueError("existing ego-state availability mismatch")
        number_pattern = re.compile(r"-?\d+\.\d{6}")
        for name in config.ego_field_order[:-1]:
            rendered = serialized[name]
            if rendered != config.unavailable_token and (
                number_pattern.fullmatch(rendered) is None
                or rendered == "-0.000000"
            ):
                raise ValueError("existing ego-state numeric value mismatch")
        unavailable_fields = {
            name
            for name in config.ego_field_order[:-1]
            if serialized[name] == config.unavailable_token
        }
        expected_unavailable = {
            "full": set(),
            "partial": {
                "longitudinal_acceleration_mps2",
                "acceleration_interval_sec",
            },
            "unavailable": set(config.ego_field_order[:-1]),
        }
        if unavailable_fields != expected_unavailable[availability]:
            raise ValueError("existing ego-state availability fields mismatch")
    expected_assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": target_action}],
    }
    if assistant != expected_assistant:
        raise ValueError("existing adapter assistant message mismatch")


def _validate_existing_artifact(
    output_dir: Path,
    *,
    config: AdapterConfig,
    config_relative_path: str,
) -> Mapping[str, object]:
    receipt_path = output_dir / "adapter_receipt.json"
    if not output_dir.is_dir() or not receipt_path.is_file():
        raise ValueError("existing adapter artifact is incomplete")
    receipt = _load_json_object(receipt_path, "existing adapter receipt")
    _require_exact(
        receipt,
        "adapter_version",
        config.adapter_version,
        "adapter receipt",
    )
    _require_exact(
        receipt,
        "adapter_schema_version",
        config.adapter_schema_version,
        "adapter receipt",
    )
    receipt_config = _required_mapping(receipt, "config")
    _require_exact(
        receipt_config,
        "relative_path",
        config_relative_path,
        "adapter config",
    )
    _require_exact(receipt_config, "sha256", config.config_sha256, "adapter config")
    source_receipt = _required_mapping(receipt, "source_projection")
    expected_source_metadata: dict[str, object] = {
        "relative_dir": config.source_projection_relative_dir,
        "receipt_relative_path": config.source_projection_receipt,
        "receipt_sha256": config.expected_source_receipt_sha256,
        "projection_version": config.expected_source_projection_version,
        "projection_schema_version": (
            config.expected_source_projection_schema_version
        ),
        "git_commit": config.expected_source_projection_git_commit,
        "train_relative_path": config.source_files["train"].relative_path,
        "train_sha256": config.source_files["train"].sha256,
        "train_record_count": config.source_files["train"].record_count,
        "validation_relative_path": (
            config.source_files["validation"].relative_path
        ),
        "validation_sha256": config.source_files["validation"].sha256,
        "validation_record_count": (
            config.source_files["validation"].record_count
        ),
    }
    if dict(source_receipt) != expected_source_metadata:
        raise ValueError("existing adapter source metadata mismatch")
    prompt = _required_mapping(receipt, "prompt")
    expected_prompt: dict[str, object] = {
        "task_prompt": config.task_prompt,
        "allowed_actions": list(config.allowed_actions),
        "ego_state_field_order": list(config.ego_field_order),
        "float_precision": config.float_precision,
        "unavailable_token": config.unavailable_token,
    }
    if dict(prompt) != expected_prompt:
        raise ValueError("existing adapter prompt metadata mismatch")
    receipt_git = _required_mapping(receipt, "git")
    validate_git_provenance(
        GitProvenance(
            commit=_required_string(receipt_git, "commit"),
            branch=(
                receipt_git.get("branch")
                if isinstance(receipt_git.get("branch"), str)
                else None
            ),
            detached_head=receipt_git.get("detached_head") is True,
            worktree_clean=receipt_git.get("worktree_clean") is True,
        )
    )
    outputs = _required_mapping(receipt, "outputs")
    expected_output_keys = {
        f"{split}_{variant}" for split in PROJECT_SPLITS for variant in VARIANTS
    }
    if set(outputs) != expected_output_keys:
        raise ValueError("existing adapter output metadata mismatch")
    for split in PROJECT_SPLITS:
        for variant in VARIANTS:
            key = f"{split}_{variant}"
            metadata = _required_mapping(outputs, key)
            filename = f"{key}.jsonl"
            _require_exact(metadata, "relative_path", filename, "adapter output")
            output_sha = validate_sha256(
                metadata.get("sha256"), f"outputs.{key}.sha256"
            )
            path = output_dir / filename
            if not path.is_file() or sha256_file(path) != output_sha:
                raise ValueError(f"existing {key} output SHA-256 mismatch")
            count = 0
            actions = Counter()
            sample_tokens = set()
            for record in _iter_jsonl(path):
                action = _validate_adapter_output_record(
                    record,
                    split=split,
                    variant=variant,
                    config=config,
                    source_file_sha256=config.source_files[split].sha256,
                )
                sample_token = _required_string(record, "sample_token")
                if sample_token in sample_tokens:
                    raise ValueError(
                        "existing adapter output has duplicate sample_token"
                    )
                sample_tokens.add(sample_token)
                actions[action] += 1
                count += 1
            _require_exact(metadata, "record_count", count, "adapter output")
            if count != config.source_files[split].record_count:
                raise ValueError("existing adapter output count mismatch")
            expected_actions = _required_mapping(
                _required_mapping(receipt, "action_distribution_by_split"), split
            )
            actual_actions = {
                action: actions[action] for action in ACTION_SCHEMA
            }
            if actual_actions != dict(expected_actions):
                raise ValueError("existing adapter action distribution mismatch")
    source_counts = _required_mapping(receipt, "source_records_parsed")
    expected_source_counts = {
        split: config.source_files[split].record_count for split in PROJECT_SPLITS
    }
    if dict(source_counts) != expected_source_counts:
        raise ValueError("existing adapter source record counts mismatch")
    written_counts = _required_mapping(receipt, "adapter_records_written")
    expected_written_counts = {
        f"{split}_{variant}": config.source_files[split].record_count
        for split in PROJECT_SPLITS
        for variant in VARIANTS
    }
    if dict(written_counts) != expected_written_counts:
        raise ValueError("existing adapter written record counts mismatch")
    motion_distribution = _required_mapping(
        receipt, "motion_availability_distribution_by_split"
    )
    if dict(motion_distribution) != FROZEN_MOTION_AVAILABILITY:
        raise ValueError("existing adapter motion distribution mismatch")
    frozen_flags = {
        "absolute_path_leak_count": 0,
        "invalid_cam_front_path_count": 0,
        "invalid_source_record_hash_count": 0,
        "duplicate_sample_token_count": 0,
        "forbidden_split_records_seen": 0,
        "combined_manifest_accessed": False,
        "test_files_opened": 0,
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "image_files_opened": 0,
        "model_load_performed": False,
        "processor_load_performed": False,
        "tokenization_performed": False,
        "training_performed": False,
    }
    for key, expected in frozen_flags.items():
        _require_exact(receipt, key, expected, "adapter receipt")
    return receipt


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_qwen3vl_dataset_adapter(
    *,
    config: AdapterConfig,
    config_relative_path: str,
    repository_root: Path,
    derived_root: Path,
    git_provenance: GitProvenance,
    now_utc: Callable[[], str] = _utc_now,
) -> dict[str, object]:
    validated_git = validate_git_provenance(git_provenance)
    source_dir = resolve_derived_path(
        derived_root, config.source_projection_relative_dir
    )
    if not source_dir.is_dir():
        raise FileNotFoundError("source development projection directory is missing")
    if (source_dir / "test.jsonl").exists():
        raise ValueError("source development projection must not contain test.jsonl")
    receipt_path = source_dir / config.source_projection_receipt
    if not receipt_path.is_file():
        raise FileNotFoundError("source projection receipt is missing")
    if sha256_file(receipt_path) != config.expected_source_receipt_sha256:
        raise ValueError("source projection receipt SHA-256 mismatch")
    source_receipt = _load_json_object(receipt_path, "source projection receipt")
    expected_action_distribution = validate_source_receipt(source_receipt, config)
    source_combined_metadata = _required_mapping(
        source_receipt, "combined_manifest"
    )
    expected_source_combined_sha256 = validate_sha256(
        source_combined_metadata.get("sha256"),
        "source_receipt.combined_manifest.sha256",
    )
    source_mapping_metadata = _required_mapping(source_receipt, "scene_mapping")
    expected_split_mapping_sha256 = validate_sha256(
        source_mapping_metadata.get("split_mapping_sha256"),
        "source_receipt.scene_mapping.split_mapping_sha256",
    )
    source_paths: dict[str, Path] = {}
    for split in PROJECT_SPLITS:
        source = config.source_files[split]
        path = source_dir / source.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"source {split} projection is missing")
        if sha256_file(path) != source.sha256:
            raise ValueError(f"source {split} projection SHA-256 mismatch")
        source_paths[split] = path

    output_dir = resolve_derived_path(derived_root, config.output_relative_dir)
    try:
        output_dir.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("adapter output must not be written inside repository")
    if output_dir.exists():
        receipt = _validate_existing_artifact(
            output_dir,
            config=config,
            config_relative_path=config_relative_path,
        )
        return {"status": "already_exists", "receipt": receipt}

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=output_dir.parent,
        )
    )
    state = SourceValidationState()
    action_distribution: dict[str, dict[str, int]] = {}
    motion_distribution: dict[str, dict[str, int]] = {}
    source_records_parsed: dict[str, int] = {}
    adapter_records_written: dict[str, int] = {}
    handles: dict[str, BinaryIO] = {}
    published = False
    try:
        for split in PROJECT_SPLITS:
            for variant in VARIANTS:
                key = f"{split}_{variant}"
                handles[key] = (staging_dir / f"{key}.jsonl").open("wb")
            action_counts = Counter()
            motion_counts = Counter()
            record_count = 0
            for record in _iter_jsonl(source_paths[split]):
                source = validate_source_record(
                    record, expected_split=split, state=state
                )
                action_counts[str(source["target_action"])] += 1
                motion = source["current_ego_motion"]
                if not isinstance(motion, Mapping):
                    raise ValueError("validated motion must be a mapping")
                motion_counts[str(motion["availability"])] += 1
                for variant in VARIANTS:
                    key = f"{split}_{variant}"
                    _write_json_line(
                        handles[key],
                        adapter_record(
                            source,
                            variant=variant,
                            config=config,
                            source_file_sha256=config.source_files[split].sha256,
                        ),
                    )
                    adapter_records_written[key] = (
                        adapter_records_written.get(key, 0) + 1
                    )
                record_count += 1
            source_records_parsed[split] = record_count
            if record_count != config.source_files[split].record_count:
                raise ValueError(f"source {split} record count mismatch")
            action_distribution[split] = {
                action: action_counts[action] for action in ACTION_SCHEMA
            }
            if action_distribution[split] != expected_action_distribution[split]:
                raise ValueError(f"source {split} action distribution mismatch")
            motion_distribution[split] = {
                name: motion_counts[name] for name in MOTION_AVAILABILITY
            }
            if motion_distribution[split] != FROZEN_MOTION_AVAILABILITY[split]:
                raise ValueError(f"source {split} motion distribution mismatch")
        if state.source_combined_manifest_sha256 != expected_source_combined_sha256:
            raise ValueError("source combined manifest SHA-256 differs from receipt")
        if state.split_mapping_sha256 != expected_split_mapping_sha256:
            raise ValueError("source split mapping SHA-256 differs from receipt")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        outputs = {}
        for split in PROJECT_SPLITS:
            for variant in VARIANTS:
                key = f"{split}_{variant}"
                path = staging_dir / f"{key}.jsonl"
                outputs[key] = {
                    "relative_path": f"{key}.jsonl",
                    "sha256": sha256_file(path),
                    "record_count": adapter_records_written[key],
                }
        receipt: dict[str, object] = {
            "adapter_version": config.adapter_version,
            "adapter_schema_version": config.adapter_schema_version,
            "generated_at_utc": now_utc(),
            "git": {
                "commit": validated_git.commit,
                "branch": validated_git.branch,
                "detached_head": validated_git.detached_head,
                "worktree_clean": validated_git.worktree_clean,
            },
            "config": {
                "relative_path": config_relative_path,
                "sha256": config.config_sha256,
            },
            "source_projection": {
                "relative_dir": config.source_projection_relative_dir,
                "receipt_relative_path": config.source_projection_receipt,
                "receipt_sha256": config.expected_source_receipt_sha256,
                "projection_version": config.expected_source_projection_version,
                "projection_schema_version": (
                    config.expected_source_projection_schema_version
                ),
                "git_commit": config.expected_source_projection_git_commit,
                "train_relative_path": (
                    config.source_files["train"].relative_path
                ),
                "train_sha256": config.source_files["train"].sha256,
                "train_record_count": (
                    config.source_files["train"].record_count
                ),
                "validation_relative_path": (
                    config.source_files["validation"].relative_path
                ),
                "validation_sha256": (
                    config.source_files["validation"].sha256
                ),
                "validation_record_count": (
                    config.source_files["validation"].record_count
                ),
            },
            "prompt": {
                "task_prompt": config.task_prompt,
                "allowed_actions": list(config.allowed_actions),
                "ego_state_field_order": list(config.ego_field_order),
                "float_precision": config.float_precision,
                "unavailable_token": config.unavailable_token,
            },
            "outputs": outputs,
            "action_distribution_by_split": action_distribution,
            "motion_availability_distribution_by_split": motion_distribution,
            "source_records_parsed": source_records_parsed,
            "adapter_records_written": adapter_records_written,
            "absolute_path_leak_count": 0,
            "invalid_cam_front_path_count": 0,
            "invalid_source_record_hash_count": 0,
            "duplicate_sample_token_count": 0,
            "forbidden_split_records_seen": (
                state.forbidden_split_records_seen
            ),
            "combined_manifest_accessed": False,
            "test_files_opened": 0,
            "test_records_read": 0,
            "test_images_opened": 0,
            "test_labels_read": 0,
            "test_evaluation_performed": False,
            "image_files_opened": 0,
            "model_load_performed": False,
            "processor_load_performed": False,
            "tokenization_performed": False,
            "training_performed": False,
        }
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        receipt_output_path = staging_dir / "adapter_receipt.json"
        with receipt_output_path.open("wb") as receipt_file:
            receipt_file.write(receipt_bytes)
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        _fsync_directory(staging_dir)
        _validate_existing_artifact(
            staging_dir,
            config=config,
            config_relative_path=config_relative_path,
        )
        os.rename(staging_dir, output_dir)
        published = True
        _fsync_directory(output_dir.parent)
        return {"status": "created", "receipt": receipt}
    finally:
        for handle in handles.values():
            handle.close()
        if not published and staging_dir.exists():
            _cleanup_staging(staging_dir)

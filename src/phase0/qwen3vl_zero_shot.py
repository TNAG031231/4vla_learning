from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Protocol

from PIL import Image
import yaml

from src.actions.schema import ACTION_SCHEMA, normalize_action
from src.phase0.protocol import evaluate_classification, validate_sha256
from src.phase0.qwen3vl_dataset_adapter import (
    ADAPTER_RECORD_FIELDS,
    AdapterConfig,
    GitProvenance,
    _validate_adapter_output_record,
    canonical_json_bytes,
    load_config as load_adapter_config,
    resolve_derived_path,
    sha256_file,
    validate_git_provenance,
)
from src.phase0.qwen3vl_interface import (
    FIXED_MODEL_ID,
    FIXED_REVISION,
    FIXED_TASK_PROMPT,
    PROMPT_VERSION,
    VARIANTS,
    build_multimodal_messages,
    validate_processor_inputs,
)
from src.phase0.qwen3vl_smoke import parse_action_output, resolve_image_path


ARTIFACT_VERSION = "zero_shot_smoke_v0_1"
ARTIFACT_SCHEMA_VERSION = "phase0_3c1_zero_shot_artifact_v0.1"
PARSER_VERSION = "phase0.3a2-strict-legacy-action-v0.1"
GENERATION_CONFIG_VERSION = "phase0.3a2-deterministic-generation-v0.1"
ALLOWED_SPLIT = "validation"
FIXED_GENERATION_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 16,
}
OUTPUT_FILENAMES = ("predictions.jsonl", "metrics.json", "run_receipt.json")
LEGACY_PREDICTION_FIELDS = frozenset(
    (
        "sample_token",
        "scene_token",
        "split",
        "input_variant",
        "target_action",
        "raw_output",
        "parsed_action",
        "parser_success",
        "invalid_reason",
        "is_correct",
        "model_id",
        "model_revision",
        "processor_revision",
        "prompt_version",
        "parser_version",
        "generation_config_version",
        "adapter_schema_version",
        "adapter_record_sha256",
    )
)


class CudaRuntime(Protocol):
    def is_available(self) -> bool:
        ...

    def is_bf16_supported(self) -> bool:
        ...


class TorchRuntime(Protocol):
    bfloat16: object
    float16: object
    cuda: CudaRuntime

    def inference_mode(self) -> AbstractContextManager[object]:
        ...


@dataclass(frozen=True)
class ZeroShotConfig:
    artifact_version: str
    artifact_schema_version: str
    prompt_version: str
    parser_version: str
    generation_config_version: str
    model_id: str
    model_revision: str
    processor_revision: str
    dtype_preference: str
    attention_implementation: str
    device: str
    local_files_only: bool
    generation_kwargs: Mapping[str, object]
    retry_count: int
    allowed_split: str
    variants: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    task_prompt: str
    adapter_relative_dir: str
    adapter_receipt_relative_path: str
    adapter_config_relative_path: str
    adapter_config_sha256: str
    adapter_version: str
    adapter_schema_version: str
    adapter_validation_record_count: int
    adapter_validation_files: Mapping[str, str]
    output_relative_dir: str
    config_sha256: str


@dataclass(frozen=True)
class AdapterArtifact:
    receipt_sha256: str
    validation_paths: Mapping[str, Path]
    validation_sha256: Mapping[str, str]
    adapter_config: AdapterConfig


@dataclass(frozen=True)
class AdapterSample:
    sample_token: str
    scene_token: str
    split: str
    variant: str
    cam_front_path: str
    target_action: str
    ego_state_text: str | None
    adapter_record_sha256: str


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: TorchRuntime
    processor_loader: Callable[..., object]
    model_loader: Callable[..., object]
    image_loader: Callable[[Path], object]
    package_version: Callable[[str], str]


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_mapping(
    mapping: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _relative_posix_path(value: str, field_name: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or posix_path.as_posix() != value
    ):
        raise ValueError(
            f"{field_name} must be a traversal-free relative POSIX path"
        )
    return value


def _string_tuple(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def load_config(path: Path) -> ZeroShotConfig:
    config_bytes = path.read_bytes()
    loaded = yaml.safe_load(config_bytes)
    if not isinstance(loaded, Mapping):
        raise ValueError("zero-shot config must be a mapping")
    adapter = _required_mapping(loaded, "adapter")
    validation_files = _required_mapping(adapter, "validation_files")
    if set(validation_files) != set(VARIANTS):
        raise ValueError("adapter validation_files must contain both variants")
    parsed_validation_files = {
        variant: _relative_posix_path(
            _required_string(validation_files, variant),
            f"adapter.validation_files.{variant}",
        )
        for variant in VARIANTS
    }
    generation_kwargs = _required_mapping(loaded, "generation_kwargs")
    local_files_only = loaded.get("local_files_only")
    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only must be a boolean")
    config = ZeroShotConfig(
        artifact_version=_required_string(loaded, "artifact_version"),
        artifact_schema_version=_required_string(
            loaded, "artifact_schema_version"
        ),
        prompt_version=_required_string(loaded, "prompt_version"),
        parser_version=_required_string(loaded, "parser_version"),
        generation_config_version=_required_string(
            loaded, "generation_config_version"
        ),
        model_id=_required_string(loaded, "model_id"),
        model_revision=_required_string(loaded, "model_revision"),
        processor_revision=_required_string(loaded, "processor_revision"),
        dtype_preference=_required_string(loaded, "dtype_preference"),
        attention_implementation=_required_string(
            loaded, "attention_implementation"
        ),
        device=_required_string(loaded, "device"),
        local_files_only=local_files_only,
        generation_kwargs=dict(generation_kwargs),
        retry_count=_required_integer(loaded, "retry_count"),
        allowed_split=_required_string(loaded, "allowed_split"),
        variants=_string_tuple(loaded, "variants"),
        allowed_actions=_string_tuple(loaded, "allowed_actions"),
        task_prompt=_required_string(loaded, "task_prompt"),
        adapter_relative_dir=_relative_posix_path(
            _required_string(adapter, "relative_dir"), "adapter.relative_dir"
        ),
        adapter_receipt_relative_path=_relative_posix_path(
            _required_string(adapter, "receipt_relative_path"),
            "adapter.receipt_relative_path",
        ),
        adapter_config_relative_path=_relative_posix_path(
            _required_string(adapter, "config_relative_path"),
            "adapter.config_relative_path",
        ),
        adapter_config_sha256=validate_sha256(
            adapter.get("config_sha256"), "adapter.config_sha256"
        ),
        adapter_version=_required_string(adapter, "version"),
        adapter_schema_version=_required_string(adapter, "schema_version"),
        adapter_validation_record_count=_required_integer(
            adapter, "validation_record_count"
        ),
        adapter_validation_files=parsed_validation_files,
        output_relative_dir=_relative_posix_path(
            _required_string(loaded, "output_relative_dir"),
            "output_relative_dir",
        ),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    frozen_contract = {
        "artifact_version": (config.artifact_version, ARTIFACT_VERSION),
        "artifact_schema_version": (
            config.artifact_schema_version,
            ARTIFACT_SCHEMA_VERSION,
        ),
        "prompt_version": (config.prompt_version, PROMPT_VERSION),
        "parser_version": (config.parser_version, PARSER_VERSION),
        "generation_config_version": (
            config.generation_config_version,
            GENERATION_CONFIG_VERSION,
        ),
        "model_id": (config.model_id, FIXED_MODEL_ID),
        "model_revision": (config.model_revision, FIXED_REVISION),
        "processor_revision": (config.processor_revision, FIXED_REVISION),
        "dtype_preference": (config.dtype_preference, "bfloat16"),
        "attention_implementation": (
            config.attention_implementation,
            "sdpa",
        ),
        "device": (config.device, "cuda:0"),
        "generation_kwargs": (
            dict(config.generation_kwargs),
            FIXED_GENERATION_KWARGS,
        ),
        "retry_count": (config.retry_count, 0),
        "allowed_split": (config.allowed_split, ALLOWED_SPLIT),
        "variants": (config.variants, VARIANTS),
        "allowed_actions": (config.allowed_actions, ACTION_SCHEMA),
        "task_prompt": (config.task_prompt, FIXED_TASK_PROMPT),
    }
    for field_name, (actual, expected) in frozen_contract.items():
        if actual != expected:
            raise ValueError(f"{field_name} does not match frozen contract")
    if config.adapter_validation_record_count <= 0:
        raise ValueError("adapter validation_record_count must be positive")
    if PurePosixPath(config.output_relative_dir).parts[0] != "phase_0_3":
        raise ValueError("output_relative_dir must be under phase_0_3")
    return config


def _load_json_object(path: Path, context: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    return payload


def _require_exact(
    mapping: Mapping[str, object], key: str, expected: object, context: str
) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{context} {key} mismatch")


def _validate_receipt_git(receipt: Mapping[str, object]) -> None:
    git = _required_mapping(receipt, "git")
    branch_value = git.get("branch")
    if branch_value is not None and not isinstance(branch_value, str):
        raise ValueError("adapter receipt git branch mismatch")
    detached_head = git.get("detached_head")
    worktree_clean = git.get("worktree_clean")
    if not isinstance(detached_head, bool) or not isinstance(worktree_clean, bool):
        raise ValueError("adapter receipt git flags mismatch")
    validate_git_provenance(
        GitProvenance(
            commit=_required_string(git, "commit"),
            branch=branch_value,
            detached_head=detached_head,
            worktree_clean=worktree_clean,
        )
    )


def _validate_adapter_receipt(
    receipt: Mapping[str, object],
    config: ZeroShotConfig,
    adapter_config: AdapterConfig,
) -> dict[str, str]:
    _require_exact(
        receipt, "adapter_version", config.adapter_version, "adapter receipt"
    )
    _require_exact(
        receipt,
        "adapter_schema_version",
        config.adapter_schema_version,
        "adapter receipt",
    )
    _validate_receipt_git(receipt)
    receipt_config = _required_mapping(receipt, "config")
    _require_exact(
        receipt_config,
        "relative_path",
        config.adapter_config_relative_path,
        "adapter receipt config",
    )
    _require_exact(
        receipt_config,
        "sha256",
        config.adapter_config_sha256,
        "adapter receipt config",
    )
    source = _required_mapping(receipt, "source_projection")
    expected_source = {
        "relative_dir": adapter_config.source_projection_relative_dir,
        "receipt_relative_path": adapter_config.source_projection_receipt,
        "receipt_sha256": adapter_config.expected_source_receipt_sha256,
        "projection_version": adapter_config.expected_source_projection_version,
        "projection_schema_version": (
            adapter_config.expected_source_projection_schema_version
        ),
        "git_commit": adapter_config.expected_source_projection_git_commit,
        "train_relative_path": adapter_config.source_files["train"].relative_path,
        "train_sha256": adapter_config.source_files["train"].sha256,
        "train_record_count": adapter_config.source_files["train"].record_count,
        "validation_relative_path": (
            adapter_config.source_files["validation"].relative_path
        ),
        "validation_sha256": adapter_config.source_files["validation"].sha256,
        "validation_record_count": (
            adapter_config.source_files["validation"].record_count
        ),
    }
    if dict(source) != expected_source:
        raise ValueError("adapter receipt source provenance mismatch")
    prompt = _required_mapping(receipt, "prompt")
    expected_prompt = {
        "task_prompt": adapter_config.task_prompt,
        "allowed_actions": list(adapter_config.allowed_actions),
        "ego_state_field_order": list(adapter_config.ego_field_order),
        "float_precision": adapter_config.float_precision,
        "unavailable_token": adapter_config.unavailable_token,
    }
    if dict(prompt) != expected_prompt:
        raise ValueError("adapter receipt prompt contract mismatch")
    outputs = _required_mapping(receipt, "outputs")
    expected_output_keys = {
        f"{split}_{variant}"
        for split in ("train", "validation")
        for variant in VARIANTS
    }
    if set(outputs) != expected_output_keys:
        raise ValueError("adapter receipt outputs mismatch")
    validation_hashes = {}
    for variant in VARIANTS:
        key = f"validation_{variant}"
        output = _required_mapping(outputs, key)
        _require_exact(
            output,
            "relative_path",
            config.adapter_validation_files[variant],
            f"adapter receipt {key}",
        )
        _require_exact(
            output,
            "record_count",
            config.adapter_validation_record_count,
            f"adapter receipt {key}",
        )
        validation_hashes[variant] = validate_sha256(
            output.get("sha256"), f"adapter receipt {key}.sha256"
        )
    source_counts = _required_mapping(receipt, "source_records_parsed")
    _require_exact(
        source_counts,
        "validation",
        config.adapter_validation_record_count,
        "adapter receipt source_records_parsed",
    )
    written_counts = _required_mapping(receipt, "adapter_records_written")
    for variant in VARIANTS:
        _require_exact(
            written_counts,
            f"validation_{variant}",
            config.adapter_validation_record_count,
            "adapter receipt adapter_records_written",
        )
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
    return validation_hashes


def load_adapter_artifact(
    *,
    config: ZeroShotConfig,
    repository_root: Path,
    derived_root: Path,
) -> AdapterArtifact:
    adapter_config_path = (
        repository_root / config.adapter_config_relative_path
    ).resolve()
    try:
        adapter_config_path.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError("adapter config escapes repository") from error
    if sha256_file(adapter_config_path) != config.adapter_config_sha256:
        raise ValueError("adapter config SHA-256 mismatch")
    adapter_config = load_adapter_config(adapter_config_path)
    if adapter_config.adapter_version != config.adapter_version:
        raise ValueError("adapter version mismatch")
    if adapter_config.adapter_schema_version != config.adapter_schema_version:
        raise ValueError("adapter schema version mismatch")
    if (
        adapter_config.source_files[ALLOWED_SPLIT].record_count
        != config.adapter_validation_record_count
    ):
        raise ValueError("adapter validation record count mismatch")
    adapter_dir = resolve_derived_path(derived_root, config.adapter_relative_dir)
    receipt_path = adapter_dir / config.adapter_receipt_relative_path
    if not receipt_path.is_file():
        raise FileNotFoundError("frozen adapter receipt is missing")
    receipt = _load_json_object(receipt_path, "adapter receipt")
    validation_hashes = _validate_adapter_receipt(
        receipt, config, adapter_config
    )
    validation_paths = {}
    for variant in VARIANTS:
        path = adapter_dir / config.adapter_validation_files[variant]
        if not path.is_file():
            raise FileNotFoundError(f"adapter {variant} validation file is missing")
        if sha256_file(path) != validation_hashes[variant]:
            raise ValueError(f"adapter {variant} validation SHA-256 mismatch")
        with path.open("rb") as validation_file:
            line_count = sum(1 for _ in validation_file)
        if line_count != config.adapter_validation_record_count:
            raise ValueError(f"adapter {variant} validation record count mismatch")
        validation_paths[variant] = path
    return AdapterArtifact(
        receipt_sha256=sha256_file(receipt_path),
        validation_paths=validation_paths,
        validation_sha256=validation_hashes,
        adapter_config=adapter_config,
    )


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path.name}:{line_number}: record must be an object")
            yield payload


def _ego_state_text(
    record: Mapping[str, object], adapter_config: AdapterConfig
) -> str | None:
    if record["variant"] == "image_only":
        return None
    messages = record["messages"]
    if not isinstance(messages, list):
        raise ValueError("validated adapter messages must be a list")
    user = messages[0]
    if not isinstance(user, Mapping):
        raise ValueError("validated adapter user message must be an object")
    content = user["content"]
    if not isinstance(content, list):
        raise ValueError("validated adapter content must be a list")
    text_item = content[1]
    if not isinstance(text_item, Mapping):
        raise ValueError("validated adapter text item must be an object")
    prompt = text_item["text"]
    if not isinstance(prompt, str):
        raise ValueError("validated adapter prompt must be a string")
    suffix = f"\n\n{adapter_config.task_prompt}"
    return prompt[: -len(suffix)]


def _validate_adapter_record(
    record: Mapping[str, object],
    *,
    variant: str,
    artifact: AdapterArtifact,
) -> None:
    if set(record) != ADAPTER_RECORD_FIELDS:
        raise ValueError("adapter record fields mismatch")
    _validate_adapter_output_record(
        record,
        split=ALLOWED_SPLIT,
        variant=variant,
        config=artifact.adapter_config,
        source_file_sha256=(
            artifact.adapter_config.source_files[ALLOWED_SPLIT].sha256
        ),
    )


def select_adapter_samples(
    *,
    artifact: AdapterArtifact,
    input_variant: str,
    max_samples: int | None,
) -> tuple[AdapterSample, ...]:
    if input_variant not in VARIANTS:
        raise ValueError("unsupported input_variant")
    record_count = artifact.adapter_config.source_files[ALLOWED_SPLIT].record_count
    limit = record_count if max_samples is None else max_samples
    if limit <= 0 or limit > record_count:
        raise ValueError("max_samples must be within the validation record count")
    counterpart = (
        "image_ego_state" if input_variant == "image_only" else "image_only"
    )
    selected_iter = _iter_jsonl(artifact.validation_paths[input_variant])
    counterpart_iter = _iter_jsonl(artifact.validation_paths[counterpart])
    samples = []
    for _ in range(limit):
        try:
            selected_record = next(selected_iter)
            counterpart_record = next(counterpart_iter)
        except StopIteration as error:
            raise ValueError("adapter validation file ended early") from error
        _validate_adapter_record(
            selected_record, variant=input_variant, artifact=artifact
        )
        _validate_adapter_record(
            counterpart_record, variant=counterpart, artifact=artifact
        )
        consistency_fields = (
            "sample_token",
            "scene_token",
            "timestamp",
            "split",
            "cam_front_path",
            "target_action",
            "source_projection_record_sha256",
        )
        if any(
            selected_record[field] != counterpart_record[field]
            for field in consistency_fields
        ):
            raise ValueError("adapter variants are not sample-aligned")
        samples.append(
            AdapterSample(
                sample_token=str(selected_record["sample_token"]),
                scene_token=str(selected_record["scene_token"]),
                split=str(selected_record["split"]),
                variant=input_variant,
                cam_front_path=str(selected_record["cam_front_path"]),
                target_action=str(selected_record["target_action"]),
                ego_state_text=_ego_state_text(
                    selected_record, artifact.adapter_config
                ),
                adapter_record_sha256=str(
                    selected_record["adapter_record_sha256"]
                ),
            )
        )
    if max_samples is None:
        try:
            next(selected_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("adapter validation file exceeds receipt count")
        try:
            next(counterpart_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("counterpart validation file exceeds receipt count")
    return tuple(samples)


def build_inference_messages(
    sample: AdapterSample,
    image: object,
    config: ZeroShotConfig,
) -> list[dict[str, object]]:
    if config.task_prompt != FIXED_TASK_PROMPT:
        raise ValueError("task prompt does not match producer interface")
    return build_multimodal_messages(
        variant=sample.variant,
        image=image,
        ego_state_text=sample.ego_state_text,
    )


def build_metrics(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ground_truth = [str(record["target_action"]) for record in predictions]
    parsed_predictions = [
        (
            str(record["parsed_action"])
            if record["parsed_action"] is not None
            else "INVALID"
        )
        for record in predictions
    ]
    classification = evaluate_classification(ground_truth, parsed_predictions)
    result = asdict(classification)
    result["confusion_matrix"] = [
        list(row) for row in classification.confusion_matrix
    ]
    target_distribution = result.pop("class_distribution")
    parser_success_count = classification.valid_prediction_count
    result["parser_success_count"] = parser_success_count
    result["parser_success_rate"] = classification.action_parsing_success_rate
    result["invalid_output_count"] = classification.invalid_prediction_count
    result["prediction_class_distribution"] = {
        action: sum(
            record["parsed_action"] == action for record in predictions
        )
        for action in ACTION_SCHEMA
    }
    result["target_class_distribution"] = target_distribution
    return result


def _default_processor_loader(
    model_id: str, revision: str, local_files_only: bool
) -> object:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )


def _default_model_loader(
    model_id: str,
    revision: str,
    dtype: object,
    attention_implementation: str,
    local_files_only: bool,
) -> object:
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )


def _default_image_loader(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def default_runtime_dependencies() -> RuntimeDependencies:
    import torch

    return RuntimeDependencies(
        torch=torch,
        processor_loader=_default_processor_loader,
        model_loader=_default_model_loader,
        image_loader=_default_image_loader,
        package_version=metadata.version,
    )


def _resolved_revision(model: object) -> str | None:
    model_config = getattr(model, "config", None)
    revision = getattr(model_config, "_commit_hash", None)
    return revision if isinstance(revision, str) else None


def _model_dtype(model: object) -> str | None:
    dtype = getattr(model, "dtype", None)
    return str(dtype) if dtype is not None else None


def _attention_implementation(model: object) -> str | None:
    model_config = getattr(model, "config", None)
    value = getattr(model_config, "_attn_implementation", None)
    return str(value) if value is not None else None


def _input_token_count(inputs: Mapping[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("processor did not return batched input_ids")
    return int(shape[-1])


def _run_output_dir(
    config: ZeroShotConfig,
    derived_root: Path,
    input_variant: str,
    max_samples: int | None,
) -> Path:
    suffix = "all" if max_samples is None else str(max_samples)
    relative = (
        PurePosixPath(config.output_relative_dir)
        / ALLOWED_SPLIT
        / input_variant
        / f"max_samples_{suffix}"
    )
    return resolve_derived_path(derived_root, relative.as_posix())


def _jsonl_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, value: bytes) -> None:
    with path.open("wb") as output_file:
        output_file.write(value)
        output_file.flush()
        os.fsync(output_file.fileno())


def _cleanup_staging(path: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        (path / filename).unlink(missing_ok=True)
    path.rmdir()


def _receipt_matches_run(
    receipt: Mapping[str, object],
    *,
    config: ZeroShotConfig,
    config_relative_path: str,
    artifact: AdapterArtifact,
    git: GitProvenance,
    input_variant: str,
    max_samples: int | None,
    sample_count: int,
) -> None:
    expected = {
        "artifact_version": config.artifact_version,
        "artifact_schema_version": config.artifact_schema_version,
        "run_kind": "validation_smoke" if max_samples is not None else "baseline",
        "input_variant": input_variant,
        "split": ALLOWED_SPLIT,
        "max_samples": max_samples,
        "sample_count": sample_count,
        "prompt_version": config.prompt_version,
        "parser_version": config.parser_version,
        "generation_config_version": config.generation_config_version,
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "validation_label_used_as_model_input": False,
    }
    for key, value in expected.items():
        _require_exact(receipt, key, value, "zero-shot receipt")
    receipt_config = _required_mapping(receipt, "config")
    _require_exact(
        receipt_config,
        "relative_path",
        config_relative_path,
        "zero-shot config",
    )
    _require_exact(
        receipt_config, "sha256", config.config_sha256, "zero-shot config"
    )
    receipt_git = _required_mapping(receipt, "git")
    expected_git = {
        "commit": git.commit,
        "branch": git.branch,
        "detached_head": git.detached_head,
        "worktree_clean": git.worktree_clean,
    }
    if dict(receipt_git) != expected_git:
        raise ValueError("existing zero-shot Git provenance mismatch")
    adapter = _required_mapping(receipt, "adapter")
    _require_exact(
        adapter,
        "receipt_sha256",
        artifact.receipt_sha256,
        "zero-shot adapter",
    )
    validation_sha = _required_mapping(adapter, "validation_jsonl_sha256")
    if dict(validation_sha) != dict(artifact.validation_sha256):
        raise ValueError("existing zero-shot adapter validation SHA mismatch")
    model = _required_mapping(receipt, "model")
    for key, value in {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "attention_implementation": config.attention_implementation,
        "device": config.device,
    }.items():
        _require_exact(model, key, value, "zero-shot model")
    generation = _required_mapping(receipt, "generation")
    _require_exact(
        generation,
        "kwargs",
        dict(config.generation_kwargs),
        "zero-shot generation",
    )
    _require_exact(
        generation, "retry_count", 0, "zero-shot generation"
    )


def _validate_prediction_record(
    record: Mapping[str, object], sample: AdapterSample, config: ZeroShotConfig
) -> None:
    if set(record) != LEGACY_PREDICTION_FIELDS:
        raise ValueError("prediction fields mismatch")
    expected = {
        "sample_token": sample.sample_token,
        "scene_token": sample.scene_token,
        "split": ALLOWED_SPLIT,
        "input_variant": sample.variant,
        "target_action": sample.target_action,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "prompt_version": config.prompt_version,
        "parser_version": config.parser_version,
        "generation_config_version": config.generation_config_version,
        "adapter_schema_version": config.adapter_schema_version,
        "adapter_record_sha256": sample.adapter_record_sha256,
    }
    for key, value in expected.items():
        _require_exact(record, key, value, "prediction")
    parser = parse_action_output(record.get("raw_output"))
    _require_exact(
        record, "parsed_action", parser["predicted_action"], "prediction"
    )
    _require_exact(
        record, "parser_success", parser["parser_success"], "prediction"
    )
    _require_exact(
        record, "invalid_reason", parser["invalid_reason"], "prediction"
    )
    expected_correct = parser["predicted_action"] == sample.target_action
    _require_exact(record, "is_correct", expected_correct, "prediction")


def _validate_existing_artifact(
    output_dir: Path,
    *,
    config: ZeroShotConfig,
    config_relative_path: str,
    artifact: AdapterArtifact,
    git: GitProvenance,
    input_variant: str,
    max_samples: int | None,
    samples: Sequence[AdapterSample],
) -> Mapping[str, object]:
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} != set(
        OUTPUT_FILENAMES
    ):
        raise ValueError("existing zero-shot artifact is incomplete")
    receipt = _load_json_object(output_dir / "run_receipt.json", "run receipt")
    _receipt_matches_run(
        receipt,
        config=config,
        config_relative_path=config_relative_path,
        artifact=artifact,
        git=git,
        input_variant=input_variant,
        max_samples=max_samples,
        sample_count=len(samples),
    )
    artifacts = _required_mapping(receipt, "artifacts")
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    predictions_sha = validate_sha256(
        artifacts.get("predictions_sha256"), "predictions_sha256"
    )
    metrics_sha = validate_sha256(
        artifacts.get("metrics_sha256"), "metrics_sha256"
    )
    if sha256_file(predictions_path) != predictions_sha:
        raise ValueError("existing predictions SHA-256 mismatch")
    if sha256_file(metrics_path) != metrics_sha:
        raise ValueError("existing metrics SHA-256 mismatch")
    prediction_records = tuple(_iter_jsonl(predictions_path))
    if len(prediction_records) != len(samples):
        raise ValueError("existing prediction count mismatch")
    for record, sample in zip(prediction_records, samples, strict=True):
        _validate_prediction_record(record, sample, config)
    metrics = _load_json_object(metrics_path, "metrics")
    if dict(metrics) != build_metrics(prediction_records):
        raise ValueError("existing metrics content mismatch")
    return receipt


def _publish_artifact(
    *,
    output_dir: Path,
    predictions: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    receipt: dict[str, object],
    validate_staging: Callable[[Path], None],
) -> None:
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
        predictions_bytes = _jsonl_bytes(predictions)
        metrics_bytes = canonical_json_bytes(metrics) + b"\n"
        receipt["artifacts"] = {
            "predictions_relative_path": "predictions.jsonl",
            "predictions_sha256": _sha256_bytes(predictions_bytes),
            "metrics_relative_path": "metrics.json",
            "metrics_sha256": _sha256_bytes(metrics_bytes),
        }
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        _write_fsynced(staging_dir / "predictions.jsonl", predictions_bytes)
        _write_fsynced(staging_dir / "metrics.json", metrics_bytes)
        _write_fsynced(staging_dir / "run_receipt.json", receipt_bytes)
        _fsync_directory(staging_dir)
        validate_staging(staging_dir)
        os.rename(staging_dir, output_dir)
        published = True
        _fsync_directory(output_dir.parent)
    finally:
        if not published and staging_dir.exists():
            _cleanup_staging(staging_dir)


def _git_mapping(git: GitProvenance) -> dict[str, object]:
    return {
        "commit": git.commit,
        "branch": git.branch,
        "detached_head": git.detached_head,
        "worktree_clean": git.worktree_clean,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_zero_shot(
    *,
    config: ZeroShotConfig,
    config_relative_path: str,
    repository_root: Path,
    nuscenes_root: Path,
    derived_root: Path,
    input_variant: str,
    split: str,
    max_samples: int | None,
    git_provenance: GitProvenance,
    dependencies: RuntimeDependencies | None = None,
    now_utc: Callable[[], str] = _utc_now,
) -> dict[str, object]:
    if split != ALLOWED_SPLIT:
        raise ValueError("Phase 0.3c zero-shot runner only permits validation")
    if input_variant not in VARIANTS:
        raise ValueError("unsupported input_variant")
    if isinstance(max_samples, bool) or (
        max_samples is not None and max_samples <= 0
    ):
        raise ValueError("max_samples must be positive")
    git = validate_git_provenance(git_provenance)
    artifact = load_adapter_artifact(
        config=config,
        repository_root=repository_root,
        derived_root=derived_root,
    )
    samples = select_adapter_samples(
        artifact=artifact,
        input_variant=input_variant,
        max_samples=max_samples,
    )
    output_dir = _run_output_dir(
        config, derived_root, input_variant, max_samples
    )
    try:
        output_dir.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("zero-shot output must not be inside repository")
    if output_dir.exists():
        receipt = _validate_existing_artifact(
            output_dir,
            config=config,
            config_relative_path=config_relative_path,
            artifact=artifact,
            git=git,
            input_variant=input_variant,
            max_samples=max_samples,
            samples=samples,
        )
        return {
            "status": "already_exists",
            "output_dir": output_dir,
            "receipt": receipt,
        }

    runtime = dependencies or default_runtime_dependencies()
    torch = runtime.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL zero-shot inference")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = runtime.processor_loader(
        config.model_id,
        config.processor_revision,
        config.local_files_only,
    )
    model = runtime.model_loader(
        config.model_id,
        config.model_revision,
        dtype,
        config.attention_implementation,
        config.local_files_only,
    )
    model.to(config.device)
    model.eval()
    if _resolved_revision(model) != config.model_revision:
        raise ValueError("resolved model revision does not match config")
    if _model_dtype(model) != str(dtype):
        raise ValueError("loaded model dtype does not match runtime policy")
    if _attention_implementation(model) != config.attention_implementation:
        raise ValueError("loaded attention implementation does not match config")

    predictions = []
    validation_images_opened = 0
    for sample in samples:
        image_path = resolve_image_path(nuscenes_root, sample.cam_front_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"CAM_FRONT image is missing for sample {sample.sample_token}"
            )
        image = runtime.image_loader(image_path)
        validation_images_opened += 1
        messages = build_inference_messages(sample, image, config)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        validate_processor_inputs(
            inputs,
            expected_batch_size=1,
            expected_image_count=1,
        )
        input_token_count = _input_token_count(inputs)
        inputs = inputs.to(config.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **config.generation_kwargs)
        generated_ids = output_ids[:, input_token_count:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_output = decoded[0]
        parser = parse_action_output(raw_output)
        target_action = normalize_action(sample.target_action)
        parsed_action = parser["predicted_action"]
        predictions.append(
            {
                "sample_token": sample.sample_token,
                "scene_token": sample.scene_token,
                "split": sample.split,
                "input_variant": sample.variant,
                "target_action": target_action,
                "raw_output": raw_output,
                "parsed_action": parsed_action,
                "parser_success": parser["parser_success"],
                "invalid_reason": parser["invalid_reason"],
                "is_correct": parsed_action == target_action,
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "processor_revision": config.processor_revision,
                "prompt_version": config.prompt_version,
                "parser_version": config.parser_version,
                "generation_config_version": config.generation_config_version,
                "adapter_schema_version": config.adapter_schema_version,
                "adapter_record_sha256": sample.adapter_record_sha256,
            }
        )
    metrics = build_metrics(predictions)
    receipt: dict[str, object] = {
        "artifact_version": config.artifact_version,
        "artifact_schema_version": config.artifact_schema_version,
        "run_kind": "validation_smoke" if max_samples is not None else "baseline",
        "generated_at_utc": now_utc(),
        "git": _git_mapping(git),
        "config": {
            "relative_path": config_relative_path,
            "sha256": config.config_sha256,
        },
        "adapter": {
            "relative_dir": config.adapter_relative_dir,
            "receipt_sha256": artifact.receipt_sha256,
            "adapter_version": config.adapter_version,
            "adapter_schema_version": config.adapter_schema_version,
            "validation_jsonl_sha256": dict(artifact.validation_sha256),
        },
        "model": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "processor_revision": config.processor_revision,
            "dtype_policy": "bfloat16_with_float16_fallback",
            "actual_dtype": str(dtype),
            "attention_implementation": config.attention_implementation,
            "device": config.device,
            "transformers_version": runtime.package_version("transformers"),
        },
        "prompt_version": config.prompt_version,
        "parser_version": config.parser_version,
        "generation_config_version": config.generation_config_version,
        "generation": {
            "kwargs": dict(config.generation_kwargs),
            "retry_count": config.retry_count,
        },
        "input_variant": input_variant,
        "split": split,
        "max_samples": max_samples,
        "sample_count": len(predictions),
        "validation_images_opened": validation_images_opened,
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "validation_label_used_as_model_input": False,
    }
    _publish_artifact(
        output_dir=output_dir,
        predictions=predictions,
        metrics=metrics,
        receipt=receipt,
        validate_staging=lambda path: _validate_existing_artifact(
            path,
            config=config,
            config_relative_path=config_relative_path,
            artifact=artifact,
            git=git,
            input_variant=input_variant,
            max_samples=max_samples,
            samples=samples,
        ),
    )
    validated_receipt = _validate_existing_artifact(
        output_dir,
        config=config,
        config_relative_path=config_relative_path,
        artifact=artifact,
        git=git,
        input_variant=input_variant,
        max_samples=max_samples,
        samples=samples,
    )
    return {
        "status": "created",
        "output_dir": output_dir,
        "receipt": validated_receipt,
    }

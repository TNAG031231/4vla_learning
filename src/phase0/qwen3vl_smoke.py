from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import time
from typing import Protocol

from PIL import Image
import yaml

from src.actions.schema import ACTION_SCHEMA
from src.phase0.protocol import (
    is_valid_cam_front_path,
    iter_manifest_rows,
    validate_sha256,
)
from src.phase0.qwen_preflight import (
    check_manifest_integrity,
    collect_git_provenance,
    resolve_output_path,
    write_preflight_artifact,
)


ARTIFACT_SCHEMA_VERSION = "phase0.3a2_qwen3vl_smoke_artifact_v0.1"
FIXED_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
FIXED_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
ALLOWED_SPLITS = frozenset({"train", "validation"})
SELECTION_STRATEGY = "first_manifest_record_in_allowed_split"
FIXED_GENERATION_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 16,
}
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "smoke_version",
        "model_id",
        "model_revision",
        "processor_revision",
        "allowed_split",
        "selection_strategy",
        "task_prompt",
        "allowed_actions",
        "dtype_preference",
        "attention_implementation",
        "generation_kwargs",
        "manifest_relative_path",
        "expected_manifest_sha256",
        "output_relative_path",
        "local_files_only",
    }
)


class CudaRuntime(Protocol):
    OutOfMemoryError: type[BaseException]

    def is_available(self) -> bool:
        ...

    def is_bf16_supported(self) -> bool:
        ...

    def reset_peak_memory_stats(self) -> None:
        ...

    def memory_allocated(self) -> int:
        ...

    def max_memory_allocated(self) -> int:
        ...

    def max_memory_reserved(self) -> int:
        ...

    def get_device_properties(self, index: int) -> object:
        ...


class TorchRuntime(Protocol):
    __version__: str
    bfloat16: object
    float16: object
    cuda: CudaRuntime
    version: object

    def inference_mode(self) -> AbstractContextManager[object]:
        ...


@dataclass(frozen=True)
class SmokeConfig:
    smoke_version: str
    model_id: str
    model_revision: str
    processor_revision: str
    allowed_split: str
    selection_strategy: str
    task_prompt: str
    allowed_actions: tuple[str, ...]
    dtype_preference: str
    attention_implementation: str
    generation_kwargs: Mapping[str, object]
    manifest_relative_path: str
    expected_manifest_sha256: str
    output_relative_path: str
    local_files_only: bool
    config_sha256: str


@dataclass(frozen=True)
class SmokeSample:
    manifest_line_number: int
    sample_token: str
    scene_token: str
    split: str
    cam_front_path: str


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: TorchRuntime
    processor_loader: Callable[..., object]
    model_loader: Callable[..., object]
    image_loader: Callable[[Path], object]
    package_version: Callable[[str], str]
    timer: Callable[[], float] = time.perf_counter


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _relative_posix_path(value: str, field_name: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or "." in posix_path.parts
    ):
        raise ValueError(
            f"{field_name} must be a traversal-free relative POSIX path"
        )
    return posix_path.as_posix()


def load_smoke_config(path: Path) -> SmokeConfig:
    config_bytes = path.read_bytes()
    raw: object = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("smoke config root must be a mapping")
    missing_fields = REQUIRED_CONFIG_FIELDS - set(raw)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"smoke config missing required fields: {missing}")

    model_id = _required_string(raw, "model_id")
    if model_id != FIXED_MODEL_ID:
        raise ValueError(f"model_id must be {FIXED_MODEL_ID}")
    model_revision = _required_string(raw, "model_revision")
    processor_revision = _required_string(raw, "processor_revision")
    if model_revision == "main" or processor_revision == "main":
        raise ValueError("model and processor revisions must not be main")
    if GIT_REVISION_PATTERN.fullmatch(model_revision) is None:
        raise ValueError("model_revision must be a lowercase 40-character commit")
    if GIT_REVISION_PATTERN.fullmatch(processor_revision) is None:
        raise ValueError(
            "processor_revision must be a lowercase 40-character commit"
        )
    if model_revision != FIXED_REVISION or processor_revision != FIXED_REVISION:
        raise ValueError(f"model and processor revisions must be {FIXED_REVISION}")

    allowed_split = _required_string(raw, "allowed_split")
    if allowed_split not in ALLOWED_SPLITS:
        raise ValueError("allowed_split must be train or validation")
    selection_strategy = _required_string(raw, "selection_strategy")
    if selection_strategy != SELECTION_STRATEGY:
        raise ValueError(f"selection_strategy must be {SELECTION_STRATEGY}")

    allowed_actions_value = raw.get("allowed_actions")
    if not isinstance(allowed_actions_value, list):
        raise ValueError("allowed_actions must be a list")
    allowed_actions = tuple(allowed_actions_value)
    if allowed_actions != ACTION_SCHEMA:
        raise ValueError("allowed_actions must exactly match ACTION_SCHEMA")

    dtype_preference = _required_string(raw, "dtype_preference")
    if dtype_preference != "bfloat16":
        raise ValueError("dtype_preference must be bfloat16")
    attention_implementation = _required_string(
        raw,
        "attention_implementation",
    )
    if attention_implementation != "sdpa":
        raise ValueError("attention_implementation must be sdpa")

    generation_kwargs = raw.get("generation_kwargs")
    if not isinstance(generation_kwargs, Mapping):
        raise ValueError("generation_kwargs must be a mapping")
    if dict(generation_kwargs) != FIXED_GENERATION_KWARGS:
        raise ValueError(
            "generation_kwargs must exactly match the deterministic contract"
        )

    manifest_relative_path = _relative_posix_path(
        _required_string(raw, "manifest_relative_path"),
        "manifest_relative_path",
    )
    output_relative_path = _relative_posix_path(
        _required_string(raw, "output_relative_path"),
        "output_relative_path",
    )
    if PurePosixPath(output_relative_path).parts[0] != "phase_0_3":
        raise ValueError("output_relative_path must be under phase_0_3")
    if manifest_relative_path == output_relative_path:
        raise ValueError("output_relative_path must not overwrite the manifest")

    local_files_only = raw.get("local_files_only")
    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only must be a boolean")

    return SmokeConfig(
        smoke_version=_required_string(raw, "smoke_version"),
        model_id=model_id,
        model_revision=model_revision,
        processor_revision=processor_revision,
        allowed_split=allowed_split,
        selection_strategy=selection_strategy,
        task_prompt=_required_string(raw, "task_prompt"),
        allowed_actions=allowed_actions,
        dtype_preference=dtype_preference,
        attention_implementation=attention_implementation,
        generation_kwargs=dict(generation_kwargs),
        manifest_relative_path=manifest_relative_path,
        expected_manifest_sha256=validate_sha256(
            raw.get("expected_manifest_sha256"),
            "expected_manifest_sha256",
        ),
        output_relative_path=output_relative_path,
        local_files_only=local_files_only,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def parse_action_output(raw_output: object) -> dict[str, object]:
    if not isinstance(raw_output, str):
        return {
            "normalized_output": None,
            "parser_success": False,
            "predicted_action": None,
            "invalid_reason": "non_string_output",
        }
    normalized_output = raw_output.strip().lower()
    if not normalized_output:
        invalid_reason = "empty_output"
    elif normalized_output not in ACTION_SCHEMA:
        invalid_reason = "output_not_exact_allowed_action"
    else:
        invalid_reason = None
    return {
        "normalized_output": normalized_output,
        "parser_success": invalid_reason is None,
        "predicted_action": (
            normalized_output if invalid_reason is None else None
        ),
        "invalid_reason": invalid_reason,
    }


def select_first_sample(
    manifest_path: Path,
    allowed_split: str,
) -> tuple[SmokeSample, int]:
    if allowed_split not in ALLOWED_SPLITS:
        raise ValueError("runtime allowed_split must be train or validation")
    records_parsed = 0
    for line_number, row in enumerate(iter_manifest_rows(manifest_path), 1):
        records_parsed += 1
        split = _required_string(row, "split")
        if split == "test":
            raise ValueError("test manifest records must not be consumed")
        if split != allowed_split:
            continue
        cam_front_path = _required_string(row, "cam_front_path")
        if not is_valid_cam_front_path(cam_front_path):
            raise ValueError(
                "cam_front_path must be under samples/CAM_FRONT/ and relative "
                "to NUSCENES_ROOT"
            )
        return (
            SmokeSample(
                manifest_line_number=line_number,
                sample_token=_required_string(row, "sample_token"),
                scene_token=_required_string(row, "scene_token"),
                split=split,
                cam_front_path=cam_front_path,
            ),
            records_parsed,
        )
    raise ValueError(f"manifest contains no {allowed_split} sample")


def resolve_image_path(nuscenes_root: Path, relative_path: str) -> Path:
    if not is_valid_cam_front_path(relative_path):
        raise ValueError("invalid CAM_FRONT relative path")
    root = nuscenes_root.resolve()
    image_path = (root / relative_path).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise ValueError("CAM_FRONT path escapes NUSCENES_ROOT") from error
    return image_path


def _failure(code: str, message: str, kind: str) -> dict[str, str]:
    return {"code": code, "message": message, "kind": kind}


def _shape(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dimension) for dimension in shape]


def _tensor_metadata(inputs: Mapping[str, object]) -> dict[str, object]:
    pixel_values = inputs.get("pixel_values")
    input_ids_shape = _shape(inputs.get("input_ids"))
    input_token_count = None
    if input_ids_shape is not None and len(input_ids_shape) >= 2:
        input_token_count = input_ids_shape[-1]
    return {
        "input_keys": sorted(str(key) for key in inputs),
        "input_ids_shape": input_ids_shape,
        "attention_mask_shape": _shape(inputs.get("attention_mask")),
        "pixel_values_shape": _shape(pixel_values),
        "image_grid_thw_shape": _shape(inputs.get("image_grid_thw")),
        "input_token_count": input_token_count,
        "pixel_values_dtype": (
            str(getattr(pixel_values, "dtype"))
            if pixel_values is not None
            else None
        ),
        "pixel_values_device": (
            str(getattr(pixel_values, "device"))
            if pixel_values is not None
            else None
        ),
    }


def _resolved_revision(model: object) -> str | None:
    model_config = getattr(model, "config", None)
    revision = getattr(model_config, "_commit_hash", None)
    return revision if isinstance(revision, str) else None


def _model_dtype(model: object) -> str | None:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return str(dtype)
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return None
    return str(parameter.dtype)


def _attention_implementation(model: object) -> str | None:
    model_config = getattr(model, "config", None)
    implementation = getattr(model_config, "_attn_implementation", None)
    return str(implementation) if implementation is not None else None


def _default_processor_loader(
    model_id: str,
    revision: str,
    local_files_only: bool,
) -> object:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
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


def _status_from_failures(failures: list[dict[str, str]]) -> str:
    return "failed" if any(item["kind"] == "failed" for item in failures) else "blocked"


def _base_artifact(
    config: SmokeConfig,
    git: Mapping[str, object],
    local_files_only: bool,
) -> dict[str, object]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "smoke_version": config.smoke_version,
        "status": "blocked",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"sha256": config.config_sha256},
        "git": dict(git),
        "environment": {
            "torch_version": None,
            "transformers_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "total_vram_bytes": None,
            "bf16_supported": None,
            "hf_home": os.environ.get("HF_HOME"),
            "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        },
        "model": {
            "model_id": config.model_id,
            "configured_revision": config.model_revision,
            "resolved_revision": None,
            "model_class": "Qwen3VLForConditionalGeneration",
            "dtype_preference": config.dtype_preference,
            "actual_dtype": None,
            "device": None,
            "attention_implementation": None,
            "local_files_only": local_files_only,
            "load_performed": False,
        },
        "processor": {
            "processor_class": "AutoProcessor",
            "configured_revision": config.processor_revision,
            "input_keys": None,
            "input_ids_shape": None,
            "attention_mask_shape": None,
            "pixel_values_shape": None,
            "image_grid_thw_shape": None,
            "input_token_count": None,
            "image_width_pixels": None,
            "image_height_pixels": None,
            "pixel_values_dtype": None,
            "pixel_values_device": None,
        },
        "sample": None,
        "input_contract": {
            "modality": "single_CAM_FRONT_image_only",
            "forbidden_inputs": [
                "ego_state",
                "past_motion",
                "speed",
                "acceleration",
                "yaw_rate",
                "route_command",
                "natural_language_instruction",
                "future_ego_trajectory",
                "nearby_agents",
                "gt_boxes",
                "occupancy",
                "gt_meta_action",
            ],
        },
        "prompt": {
            "prompt_version": config.smoke_version,
            "task_prompt": config.task_prompt,
        },
        "generation": {
            "kwargs": dict(config.generation_kwargs),
            "generated_token_shape": None,
            "retry_count": 0,
            "completed": False,
        },
        "raw_output": None,
        "parser": {
            "normalized_output": None,
            "parser_success": False,
            "predicted_action": None,
            "invalid_reason": "generation_not_completed",
        },
        "timing": {
            "processor_duration_seconds": None,
            "model_load_duration_seconds": None,
            "generation_duration_seconds": None,
            "total_duration_seconds": None,
        },
        "cuda_memory": {
            "memory_allocated_before_load_bytes": None,
            "memory_allocated_after_load_bytes": None,
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
        },
        "manifest": None,
        "manifest_records_parsed": 0,
        "warnings": [],
        "failures": [],
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
    }


def _check_cache_location(
    repository_root: Path,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    cache_value = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get(
        "HF_HOME"
    )
    if not cache_value:
        return None, _failure(
            "huggingface_cache_unset",
            "HF_HOME or HUGGINGFACE_HUB_CACHE must be set for the formal smoke",
            "blocked",
        )
    cache_path = Path(cache_value).expanduser().resolve()
    try:
        cache_path.relative_to(repository_root.resolve())
    except ValueError:
        return {"path": str(cache_path)}, None
    return None, _failure(
        "huggingface_cache_inside_repository",
        "Hugging Face model cache must not be inside the Git repository",
        "blocked",
    )


def run_smoke(
    *,
    config: SmokeConfig,
    repository_root: Path,
    nuscenes_root: Path,
    derived_root: Path,
    local_files_only: bool | None = None,
    dependencies: RuntimeDependencies | None = None,
    require_clean_git: bool = True,
    git_runner: Callable[[Path, tuple[str, ...]], str] | None = None,
) -> dict[str, object]:
    total_started = time.perf_counter()
    if config.allowed_split not in ALLOWED_SPLITS:
        raise ValueError("runtime allowed_split must be train or validation")
    effective_local_only = (
        config.local_files_only
        if local_files_only is None
        else local_files_only
    )
    if git_runner is None:
        git, git_failures = collect_git_provenance(repository_root)
    else:
        git, git_failures = collect_git_provenance(repository_root, git_runner)
    artifact = _base_artifact(config, git, effective_local_only)
    failures = artifact["failures"]
    assert isinstance(failures, list)

    def finish(status: str) -> dict[str, object]:
        artifact["status"] = status
        timing = artifact["timing"]
        assert isinstance(timing, dict)
        timing["total_duration_seconds"] = time.perf_counter() - total_started
        return artifact

    if not require_clean_git:
        git_failures = [
            failure
            for failure in git_failures
            if failure["code"] != "git_worktree_dirty"
        ]
    failures.extend(git_failures)

    cache, cache_failure = _check_cache_location(repository_root)
    environment = artifact["environment"]
    assert isinstance(environment, dict)
    environment["model_cache"] = cache
    if cache_failure is not None:
        failures.append(cache_failure)

    manifest_path = (derived_root / config.manifest_relative_path).resolve()
    manifest, manifest_failures = check_manifest_integrity(
        manifest_path,
        config.expected_manifest_sha256,
    )
    artifact["manifest"] = manifest
    failures.extend(manifest_failures)
    if failures:
        return finish(_status_from_failures(failures))

    try:
        sample, records_parsed = select_first_sample(
            manifest_path,
            config.allowed_split,
        )
        image_path = resolve_image_path(nuscenes_root, sample.cam_front_path)
    except (OSError, ValueError) as error:
        failures.append(_failure("sample_selection_failed", str(error), "failed"))
        return finish("failed")
    artifact["manifest_records_parsed"] = records_parsed
    artifact["sample"] = {
        "manifest_line_number": sample.manifest_line_number,
        "sample_token": sample.sample_token,
        "scene_token": sample.scene_token,
        "split": sample.split,
        "cam_front_path": sample.cam_front_path,
    }
    if sample.split == "test":
        failures.append(
            _failure("test_split_runtime_rejected", "test split is forbidden", "failed")
        )
        return finish("failed")
    if not image_path.is_file():
        failures.append(
            _failure("image_missing", "selected CAM_FRONT image is missing", "blocked")
        )
        return finish("blocked")

    runtime = dependencies or default_runtime_dependencies()
    torch = runtime.torch
    environment.update(
        {
            "torch_version": str(torch.__version__),
            "transformers_version": runtime.package_version("transformers"),
            "cuda_version": str(getattr(torch.version, "cuda", None)),
        }
    )
    if not torch.cuda.is_available():
        failures.append(
            _failure("cuda_unavailable", "CUDA is unavailable", "blocked")
        )
        return finish("blocked")

    properties = torch.cuda.get_device_properties(0)
    environment["gpu_name"] = str(getattr(properties, "name"))
    environment["total_vram_bytes"] = int(getattr(properties, "total_memory"))
    torch.cuda.reset_peak_memory_stats()
    cuda_memory = artifact["cuda_memory"]
    assert isinstance(cuda_memory, dict)
    cuda_memory["memory_allocated_before_load_bytes"] = int(
        torch.cuda.memory_allocated()
    )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    environment["bf16_supported"] = bf16_supported
    chosen_dtype = torch.bfloat16 if bf16_supported else torch.float16

    try:
        image = runtime.image_loader(image_path)
    except FileNotFoundError:
        failures.append(
            _failure("image_missing", "selected CAM_FRONT image is missing", "blocked")
        )
        return finish("blocked")
    except (OSError, ValueError) as error:
        failures.append(
            _failure(
                "image_decode_failed",
                f"CAM_FRONT image decode failed: {error}",
                "failed",
            )
        )
        return finish("failed")

    processor_started = runtime.timer()
    try:
        processor = runtime.processor_loader(
            config.model_id,
            config.processor_revision,
            effective_local_only,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config.task_prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as error:
        failures.append(
            _failure("processor_failed", f"processor failed: {error}", "failed")
        )
        return finish("failed")
    artifact["timing"]["processor_duration_seconds"] = (
        runtime.timer() - processor_started
    )
    image_size = getattr(image, "size", None)
    processor_record = artifact["processor"]
    assert isinstance(processor_record, dict)
    processor_record.update(_tensor_metadata(inputs))
    if image_size is not None:
        processor_record["image_width_pixels"] = int(image_size[0])
        processor_record["image_height_pixels"] = int(image_size[1])

    load_started = runtime.timer()
    try:
        model = runtime.model_loader(
            config.model_id,
            config.model_revision,
            chosen_dtype,
            config.attention_implementation,
            effective_local_only,
        )
        model.to("cuda:0")
        model.eval()
    except Exception as error:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        code = "model_load_oom" if isinstance(error, oom_type) else "model_load_failed"
        failures.append(_failure(code, f"model load failed: {error}", "failed"))
        return finish("failed")
    artifact["timing"]["model_load_duration_seconds"] = (
        runtime.timer() - load_started
    )
    cuda_memory["memory_allocated_after_load_bytes"] = int(
        torch.cuda.memory_allocated()
    )
    model_record = artifact["model"]
    assert isinstance(model_record, dict)
    model_record.update(
        {
            "resolved_revision": _resolved_revision(model),
            "actual_dtype": _model_dtype(model),
            "device": str(getattr(model, "device", "cuda:0")),
            "attention_implementation": _attention_implementation(model),
            "load_performed": True,
        }
    )
    if model_record["resolved_revision"] != config.model_revision:
        failures.append(
            _failure(
                "resolved_revision_mismatch",
                "resolved model revision does not match configured revision",
                "failed",
            )
        )
        return finish("failed")
    if model_record["actual_dtype"] != str(chosen_dtype):
        failures.append(
            _failure(
                "model_dtype_mismatch",
                "loaded model dtype does not match the controlled runtime choice",
                "failed",
            )
        )
        return finish("failed")
    if model_record["attention_implementation"] != config.attention_implementation:
        failures.append(
            _failure(
                "attention_implementation_mismatch",
                "loaded model attention implementation does not match config",
                "failed",
            )
        )
        return finish("failed")

    inputs = inputs.to("cuda:0")
    processor_record.update(_tensor_metadata(inputs))
    input_ids_shape = processor_record["input_ids_shape"]
    if not isinstance(input_ids_shape, list) or len(input_ids_shape) < 2:
        failures.append(
            _failure(
                "input_ids_missing",
                "processor did not return input_ids",
                "failed",
            )
        )
        return finish("failed")
    input_token_count = int(input_ids_shape[-1])

    generation_started = runtime.timer()
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **config.generation_kwargs)
        generated_ids = output_ids[:, input_token_count:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_output = decoded[0]
    except Exception as error:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        code = "generation_oom" if isinstance(error, oom_type) else "generation_failed"
        failures.append(_failure(code, f"generation failed: {error}", "failed"))
        return finish("failed")
    artifact["timing"]["generation_duration_seconds"] = (
        runtime.timer() - generation_started
    )
    generation_record = artifact["generation"]
    assert isinstance(generation_record, dict)
    generation_record.update(
        {
            "generated_token_shape": _shape(generated_ids),
            "completed": True,
        }
    )
    artifact["raw_output"] = raw_output
    parser = parse_action_output(raw_output)
    artifact["parser"] = parser
    final_status = (
        "passed" if parser["parser_success"] else "completed_with_invalid_output"
    )
    cuda_memory["peak_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated()
    )
    cuda_memory["peak_memory_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved()
    )
    return finish(final_status)


def smoke_exit_code(status: str) -> int:
    if status in {"passed", "completed_with_invalid_output"}:
        return 0
    if status == "blocked":
        return 2
    return 1


def smoke_output_path(
    config: SmokeConfig,
    derived_root: Path,
    repository_root: Path,
) -> Path:
    return resolve_output_path(
        derived_root,
        config.output_relative_path,
        repository_root,
    )


def write_smoke_artifact(
    artifact: Mapping[str, object],
    output_path: Path,
) -> None:
    write_preflight_artifact(artifact, output_path)

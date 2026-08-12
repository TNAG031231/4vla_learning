from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

import yaml

from src.actions.schema import ACTION_SCHEMA, normalize_action
from src.phase0.qwen3vl_dataset_adapter import (
    AdapterConfig,
    GitProvenance,
    _validate_adapter_output_record,
    canonical_json_bytes,
    load_config as load_adapter_config,
    resolve_derived_path,
    sha256_file,
    validate_git_provenance,
)
from src.phase0.qwen3vl_zero_shot import (
    AdapterSample,
    FIXED_GENERATION_KWARGS,
    FIXED_MODEL_ID,
    FIXED_REVISION,
    FIXED_TASK_PROMPT,
    GENERATION_CONFIG_VERSION,
    PARSER_VERSION,
    PROMPT_VERSION,
    _ego_state_text,
    _input_token_count,
    _validate_adapter_receipt,
    build_inference_messages,
    build_metrics,
    parse_action_output,
)
from src.phase0.qwen3vl_smoke import resolve_image_path


ARTIFACT_VERSION = "phase0.3d1-qwen3vl-lora-smoke-v0.1"
ARTIFACT_SCHEMA_VERSION = "phase0_3d1_qwen3vl_lora_smoke_v0.1"
INPUT_VARIANT = "image_ego_state"
OPTIMIZATION_SPLIT = "train"
EVALUATION_SPLIT = "validation"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
IGNORE_INDEX = -100


class Optimizer(Protocol):
    def zero_grad(self) -> None:
        ...

    def step(self) -> None:
        ...


@dataclass(frozen=True)
class LoraSmokeConfig:
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
    input_variant: str
    optimization_split: str
    evaluation_split: str
    allowed_actions: tuple[str, ...]
    task_prompt: str
    adapter_relative_dir: str
    adapter_receipt_relative_path: str
    adapter_config_relative_path: str
    adapter_config_sha256: str
    adapter_version: str
    adapter_schema_version: str
    adapter_train_record_count: int
    adapter_validation_record_count: int
    adapter_train_files: Mapping[str, str]
    adapter_validation_files: Mapping[str, str]
    train_subset_size: int
    validation_subset_size: int
    seed: int
    tiny_overfit: bool
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    lora_task_type: str
    lora_target_modules: tuple[str, ...]
    generation_kwargs: Mapping[str, object]
    retry_count: int
    output_relative_dir: str
    checkpoint_dirname: str
    config_sha256: str


@dataclass(frozen=True)
class TrainingArtifact:
    receipt_sha256: str
    adapter_config: AdapterConfig
    train_path: Path
    validation_path: Path
    train_sha256: str
    validation_sha256: str


@dataclass(frozen=True)
class RuntimeDependencies:
    processor_loader: Callable[[str, str, bool], object]
    model_loader: Callable[[str, str, object, str, bool], object]
    lora_config_factory: Callable[..., object]
    lora_injector: Callable[[object, object], object]
    adapter_loader: Callable[[object, Path], object]
    optimizer_factory: Callable[[Sequence[object], float], Optimizer]
    image_loader: Callable[[Path], object]
    dtype_selector: Callable[[str], object]
    device_selector: Callable[[str], str]
    inference_context: Callable[[], AbstractContextManager[object]]
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


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


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


def _variant_paths(
    adapter: Mapping[str, object], key: str
) -> dict[str, str]:
    paths = _required_mapping(adapter, key)
    expected = {"image_only", INPUT_VARIANT}
    if set(paths) != expected:
        raise ValueError(f"adapter.{key} must contain both adapter variants")
    return {
        variant: _relative_posix_path(
            _required_string(paths, variant), f"adapter.{key}.{variant}"
        )
        for variant in expected
    }


def load_config(path: Path) -> LoraSmokeConfig:
    config_bytes = path.read_bytes()
    loaded = yaml.safe_load(config_bytes)
    if not isinstance(loaded, Mapping):
        raise ValueError("LoRA smoke config must be a mapping")
    adapter = _required_mapping(loaded, "adapter")
    selection = _required_mapping(loaded, "selection")
    training = _required_mapping(loaded, "training")
    lora = _required_mapping(loaded, "lora")
    local_files_only = loaded.get("local_files_only")
    tiny_overfit = selection.get("tiny_overfit")
    gradient_checkpointing = training.get("gradient_checkpointing")
    if not isinstance(local_files_only, bool):
        raise ValueError("local_files_only must be a boolean")
    if tiny_overfit is not True:
        raise ValueError("selection.tiny_overfit must be true")
    if gradient_checkpointing is not True:
        raise ValueError("training.gradient_checkpointing must be true")
    config = LoraSmokeConfig(
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
        input_variant=_required_string(loaded, "input_variant"),
        optimization_split=_required_string(loaded, "optimization_split"),
        evaluation_split=_required_string(loaded, "evaluation_split"),
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
        adapter_config_sha256=_required_string(adapter, "config_sha256"),
        adapter_version=_required_string(adapter, "version"),
        adapter_schema_version=_required_string(adapter, "schema_version"),
        adapter_train_record_count=_required_integer(
            adapter, "train_record_count"
        ),
        adapter_validation_record_count=_required_integer(
            adapter, "validation_record_count"
        ),
        adapter_train_files=_variant_paths(adapter, "train_files"),
        adapter_validation_files=_variant_paths(adapter, "validation_files"),
        train_subset_size=_required_integer(selection, "train_subset_size"),
        validation_subset_size=_required_integer(
            selection, "validation_subset_size"
        ),
        seed=_required_integer(selection, "seed"),
        tiny_overfit=tiny_overfit,
        batch_size=_required_integer(training, "batch_size"),
        gradient_accumulation_steps=_required_integer(
            training, "gradient_accumulation_steps"
        ),
        max_steps=_required_integer(training, "max_steps"),
        learning_rate=_required_float(training, "learning_rate"),
        gradient_checkpointing=gradient_checkpointing,
        lora_r=_required_integer(lora, "r"),
        lora_alpha=_required_integer(lora, "alpha"),
        lora_dropout=_required_float(lora, "dropout"),
        lora_bias=_required_string(lora, "bias"),
        lora_task_type=_required_string(lora, "task_type"),
        lora_target_modules=_string_tuple(lora, "target_modules"),
        generation_kwargs=dict(
            _required_mapping(loaded, "generation_kwargs")
        ),
        retry_count=_required_integer(loaded, "retry_count"),
        output_relative_dir=_relative_posix_path(
            _required_string(loaded, "output_relative_dir"),
            "output_relative_dir",
        ),
        checkpoint_dirname=_relative_posix_path(
            _required_string(loaded, "checkpoint_dirname"),
            "checkpoint_dirname",
        ),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )
    frozen = {
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
        "input_variant": (config.input_variant, INPUT_VARIANT),
        "optimization_split": (
            config.optimization_split,
            OPTIMIZATION_SPLIT,
        ),
        "evaluation_split": (config.evaluation_split, EVALUATION_SPLIT),
        "allowed_actions": (config.allowed_actions, ACTION_SCHEMA),
        "task_prompt": (config.task_prompt, FIXED_TASK_PROMPT),
        "lora_r": (config.lora_r, 8),
        "lora_alpha": (config.lora_alpha, 16),
        "lora_dropout": (config.lora_dropout, 0.05),
        "lora_bias": (config.lora_bias, "none"),
        "lora_task_type": (config.lora_task_type, "CAUSAL_LM"),
        "lora_target_modules": (
            config.lora_target_modules,
            LORA_TARGET_MODULES,
        ),
        "generation_kwargs": (
            dict(config.generation_kwargs),
            FIXED_GENERATION_KWARGS,
        ),
        "retry_count": (config.retry_count, 0),
        "batch_size": (config.batch_size, 1),
    }
    for field_name, (actual, expected) in frozen.items():
        if actual != expected:
            raise ValueError(f"{field_name} does not match frozen contract")
    positive_integers = (
        config.adapter_train_record_count,
        config.adapter_validation_record_count,
        config.train_subset_size,
        config.validation_subset_size,
        config.gradient_accumulation_steps,
        config.max_steps,
    )
    if any(value <= 0 for value in positive_integers):
        raise ValueError("record counts and training sizes must be positive")
    if config.train_subset_size < len(ACTION_SCHEMA):
        raise ValueError("train subset must cover all action classes")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if PurePosixPath(config.output_relative_dir).parts[0] != "phase_0_3":
        raise ValueError("output_relative_dir must be under phase_0_3")
    return config


def _load_json_object(path: Path, context: str) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be an object")
    return payload


def _line_count(path: Path) -> int:
    with path.open("rb") as source_file:
        return sum(1 for _ in source_file)


def load_training_artifact(
    *,
    config: LoraSmokeConfig,
    repository_root: Path,
    derived_root: Path,
) -> TrainingArtifact:
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
        adapter_config.source_files[OPTIMIZATION_SPLIT].record_count
        != config.adapter_train_record_count
        or adapter_config.source_files[EVALUATION_SPLIT].record_count
        != config.adapter_validation_record_count
    ):
        raise ValueError("adapter record count mismatch")
    adapter_dir = resolve_derived_path(derived_root, config.adapter_relative_dir)
    receipt_path = adapter_dir / config.adapter_receipt_relative_path
    if not receipt_path.is_file():
        raise FileNotFoundError("frozen adapter receipt is missing")
    receipt = _load_json_object(receipt_path, "adapter receipt")
    validation_hashes = _validate_adapter_receipt(
        receipt, config, adapter_config
    )
    outputs = _required_mapping(receipt, "outputs")
    train_output = _required_mapping(outputs, f"train_{INPUT_VARIANT}")
    expected_train_path = config.adapter_train_files[INPUT_VARIANT]
    if train_output.get("relative_path") != expected_train_path:
        raise ValueError("adapter train output path mismatch")
    if train_output.get("record_count") != config.adapter_train_record_count:
        raise ValueError("adapter train output count mismatch")
    train_sha256 = _required_string(train_output, "sha256")
    written = _required_mapping(receipt, "adapter_records_written")
    if written.get(f"train_{INPUT_VARIANT}") != config.adapter_train_record_count:
        raise ValueError("adapter train written count mismatch")
    train_path = adapter_dir / expected_train_path
    validation_path = (
        adapter_dir / config.adapter_validation_files[INPUT_VARIANT]
    )
    expected = (
        (train_path, train_sha256, config.adapter_train_record_count),
        (
            validation_path,
            validation_hashes[INPUT_VARIANT],
            config.adapter_validation_record_count,
        ),
    )
    for path, expected_sha, expected_count in expected:
        if not path.is_file():
            raise FileNotFoundError(f"adapter dataset is missing: {path.name}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"adapter dataset SHA-256 mismatch: {path.name}")
        if _line_count(path) != expected_count:
            raise ValueError(f"adapter dataset record count mismatch: {path.name}")
    return TrainingArtifact(
        receipt_sha256=sha256_file(receipt_path),
        adapter_config=adapter_config,
        train_path=train_path,
        validation_path=validation_path,
        train_sha256=train_sha256,
        validation_sha256=validation_hashes[INPUT_VARIANT],
    )


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path.name}:{line_number}: record must be an object")
            yield payload


def load_adapter_samples(
    *,
    path: Path,
    split: str,
    artifact: TrainingArtifact,
) -> tuple[AdapterSample, ...]:
    if split not in (OPTIMIZATION_SPLIT, EVALUATION_SPLIT):
        raise ValueError("LoRA smoke only permits train and validation")
    source_sha = artifact.adapter_config.source_files[split].sha256
    samples = []
    for record in _iter_jsonl(path):
        if record.get("split") != split:
            raise ValueError("forbidden adapter split")
        _validate_adapter_output_record(
            record,
            split=split,
            variant=INPUT_VARIANT,
            config=artifact.adapter_config,
            source_file_sha256=source_sha,
        )
        samples.append(
            AdapterSample(
                sample_token=str(record["sample_token"]),
                scene_token=str(record["scene_token"]),
                split=split,
                variant=INPUT_VARIANT,
                cam_front_path=str(record["cam_front_path"]),
                target_action=str(record["target_action"]),
                ego_state_text=_ego_state_text(
                    record, artifact.adapter_config
                ),
                adapter_record_sha256=str(record["adapter_record_sha256"]),
            )
        )
    return tuple(samples)


def _selection_key(sample: AdapterSample, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{sample.sample_token}".encode("utf-8")
    ).hexdigest()


def select_train_subset(
    samples: Sequence[AdapterSample], *, size: int, seed: int
) -> tuple[AdapterSample, ...]:
    if any(sample.split != OPTIMIZATION_SPLIT for sample in samples):
        raise ValueError("optimization samples must come from train only")
    if size < len(ACTION_SCHEMA) or size > len(samples):
        raise ValueError("train subset size cannot provide class coverage")
    grouped = {
        action: sorted(
            (sample for sample in samples if sample.target_action == action),
            key=lambda sample: _selection_key(sample, seed),
        )
        for action in ACTION_SCHEMA
    }
    missing = [action for action, group in grouped.items() if not group]
    if missing:
        raise ValueError(f"train subset cannot cover actions: {missing}")
    selected = [grouped[action][0] for action in ACTION_SCHEMA]
    selected_tokens = {sample.sample_token for sample in selected}
    remaining = sorted(
        (
            sample
            for sample in samples
            if sample.sample_token not in selected_tokens
        ),
        key=lambda sample: _selection_key(sample, seed),
    )
    selected.extend(remaining[: size - len(selected)])
    return tuple(selected)


def select_validation_subset(
    samples: Sequence[AdapterSample], *, size: int, seed: int
) -> tuple[AdapterSample, ...]:
    if any(sample.split != EVALUATION_SPLIT for sample in samples):
        raise ValueError("evaluation samples must come from validation only")
    if size <= 0 or size > len(samples):
        raise ValueError("invalid validation subset size")
    return tuple(
        sorted(samples, key=lambda sample: _selection_key(sample, seed))[:size]
    )


def build_training_messages(
    sample: AdapterSample, image: object, config: LoraSmokeConfig
) -> list[dict[str, object]]:
    messages = build_inference_messages(sample, image, config)
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample.target_action}],
        }
    )
    return messages


def _active_ids(input_ids: object, attention_mask: object) -> list[int]:
    ids = input_ids.tolist()
    mask = attention_mask.tolist()
    return [token for token, active in zip(ids, mask, strict=True) if active]


class Qwen3VLSupervisedCollator:
    def __init__(
        self,
        *,
        processor: object,
        image_loader: Callable[[Path], object],
        nuscenes_root: Path,
        config: LoraSmokeConfig,
    ) -> None:
        self.processor = processor
        self.image_loader = image_loader
        self.nuscenes_root = nuscenes_root
        self.config = config

    def __call__(
        self,
        samples: Sequence[AdapterSample],
        *,
        expected_split: str,
    ) -> dict[str, object]:
        if not samples:
            raise ValueError("collator batch must not be empty")
        if expected_split not in (OPTIMIZATION_SPLIT, EVALUATION_SPLIT):
            raise ValueError("collator only permits train and validation")
        if any(sample.split != expected_split for sample in samples):
            raise ValueError("collator sample split mismatch")
        if any(sample.variant != INPUT_VARIANT for sample in samples):
            raise ValueError("collator only permits image_ego_state")
        images = [
            self.image_loader(
                resolve_image_path(self.nuscenes_root, sample.cam_front_path)
            )
            for sample in samples
        ]
        conversations = [
            build_training_messages(sample, image, self.config)
            for sample, image in zip(samples, images, strict=True)
        ]
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        if not isinstance(batch, Mapping) or not required.issubset(batch):
            raise ValueError("processor did not return required multimodal fields")
        input_rows = batch["input_ids"]
        mask_rows = batch["attention_mask"]
        if input_rows.shape != mask_rows.shape or input_rows.shape[0] != len(samples):
            raise ValueError("processor text batch is misaligned")
        if batch["image_grid_thw"].shape[0] != len(samples):
            raise ValueError("processor image grid is misaligned")
        labels = [[IGNORE_INDEX] * len(row) for row in input_rows.tolist()]
        for index, (sample, conversation) in enumerate(
            zip(samples, conversations, strict=True)
        ):
            prefix = self.processor.apply_chat_template(
                conversation[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prefix_ids = _active_ids(
                prefix["input_ids"][0], prefix["attention_mask"][0]
            )
            active_positions = [
                position
                for position, active in enumerate(mask_rows[index].tolist())
                if active
            ]
            full_ids = [
                input_rows[index][position].item()
                for position in active_positions
            ]
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    "assistant prefix is not a prefix of full conversation"
                )
            target_positions = active_positions[len(prefix_ids) :]
            if not target_positions:
                raise ValueError("assistant target token span is empty")
            supervised_ids = [
                input_rows[index][position].item() for position in target_positions
            ]
            decoded = self.processor.batch_decode(
                [supervised_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            if decoded != sample.target_action:
                raise ValueError(
                    "supervised token span does not decode to target action"
                )
            for position in target_positions:
                labels[index][position] = input_rows[index][position].item()
        result = dict(batch)
        result["labels"] = input_rows.new_tensor(labels)
        return result


def lora_config_kwargs(config: LoraSmokeConfig) -> dict[str, object]:
    return {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": config.lora_bias,
        "task_type": config.lora_task_type,
        "target_modules": list(config.lora_target_modules),
    }


def inject_lora(
    model: object,
    *,
    config: LoraSmokeConfig,
    dependencies: RuntimeDependencies,
) -> object:
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora_config = dependencies.lora_config_factory(**lora_config_kwargs(config))
    peft_model = dependencies.lora_injector(model, lora_config)
    for name, parameter in peft_model.named_parameters():
        if "lora_" in name.lower():
            if not parameter.requires_grad:
                raise ValueError("LoRA parameter is frozen")
        elif parameter.requires_grad:
            raise ValueError("non-LoRA base parameter is trainable")
    return peft_model


def trainable_parameter_report(
    model: object, target_modules: Sequence[str]
) -> dict[str, object]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
        "lora_target_module_names": list(target_modules),
    }


def _default_image_loader(path: Path) -> object:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def default_runtime_dependencies() -> RuntimeDependencies:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    def processor_loader(model_id: str, revision: str, local_only: bool) -> object:
        return AutoProcessor.from_pretrained(
            model_id, revision=revision, local_files_only=local_only
        )

    def model_loader(
        model_id: str,
        revision: str,
        dtype: object,
        attention: str,
        local_only: bool,
    ) -> object:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            attn_implementation=attention,
            local_files_only=local_only,
        )

    def dtype_selector(preference: str) -> object:
        if preference != "bfloat16":
            raise ValueError("unsupported dtype preference")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real LoRA smoke training")
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def device_selector(device: str) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real LoRA smoke training")
        return device

    return RuntimeDependencies(
        processor_loader=processor_loader,
        model_loader=model_loader,
        lora_config_factory=LoraConfig,
        lora_injector=get_peft_model,
        adapter_loader=lambda model, path: PeftModel.from_pretrained(model, path),
        optimizer_factory=lambda parameters, learning_rate: torch.optim.AdamW(
            parameters, lr=learning_rate
        ),
        image_loader=_default_image_loader,
        dtype_selector=dtype_selector,
        device_selector=device_selector,
        inference_context=torch.inference_mode,
        package_version=metadata.version,
    )


def _move_batch(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {name: value.to(device) for name, value in batch.items()}


def run_training_steps(
    *,
    model: object,
    samples: Sequence[AdapterSample],
    collator: Qwen3VLSupervisedCollator,
    optimizer: Optimizer,
    config: LoraSmokeConfig,
    device: str,
) -> list[float]:
    model.train()
    optimizer.zero_grad()
    history = []
    sample_index = 0
    for _ in range(config.max_steps):
        step_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            sample = samples[sample_index % len(samples)]
            sample_index += 1
            batch = collator([sample], expected_split=OPTIMIZATION_SPLIT)
            output = model(**_move_batch(batch, device))
            loss = output.loss
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise ValueError("training loss must be finite")
            (loss / config.gradient_accumulation_steps).backward()
            step_loss += value / config.gradient_accumulation_steps
        optimizer.step()
        optimizer.zero_grad()
        history.append(step_loss)
    return history


def _generate_prediction(
    *,
    model: object,
    processor: object,
    sample: AdapterSample,
    image: object,
    config: LoraSmokeConfig,
    device: str,
    inference_context: Callable[[], AbstractContextManager[object]],
) -> dict[str, object]:
    messages = build_inference_messages(sample, image, config)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_token_count = _input_token_count(inputs)
    with inference_context():
        output_ids = model.generate(
            **_move_batch(inputs, device), **config.generation_kwargs
        )
    generated_ids = output_ids[:, input_token_count:]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_action_output(raw_output)
    return {
        "sample_token": sample.sample_token,
        "scene_token": sample.scene_token,
        "split": sample.split,
        "input_variant": sample.variant,
        "target_action": normalize_action(sample.target_action),
        "raw_output": raw_output,
        "parsed_action": parsed["predicted_action"],
        "parser_success": parsed["parser_success"],
        "invalid_reason": parsed["invalid_reason"],
    }


def _predict_samples(
    *,
    model: object,
    processor: object,
    samples: Sequence[AdapterSample],
    nuscenes_root: Path,
    config: LoraSmokeConfig,
    device: str,
    dependencies: RuntimeDependencies,
) -> list[dict[str, object]]:
    model.eval()
    return [
        _generate_prediction(
            model=model,
            processor=processor,
            sample=sample,
            image=dependencies.image_loader(
                resolve_image_path(nuscenes_root, sample.cam_front_path)
            ),
            config=config,
            device=device,
            inference_context=dependencies.inference_context,
        )
        for sample in samples
    ]


def _action_distribution(samples: Sequence[AdapterSample]) -> dict[str, int]:
    return {
        action: sum(sample.target_action == action for sample in samples)
        for action in ACTION_SCHEMA
    }


def _checkpoint_sha256(checkpoint_dir: Path) -> tuple[Path, str]:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint_dir / filename
        if path.is_file():
            return path, sha256_file(path)
    raise FileNotFoundError("PEFT adapter weights were not saved")


def run_lora_smoke(
    *,
    config: LoraSmokeConfig,
    config_relative_path: str,
    repository_root: Path,
    nuscenes_root: Path,
    derived_root: Path,
    optimization_split: str,
    evaluation_split: str,
    git_provenance: GitProvenance,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, object]:
    if optimization_split != OPTIMIZATION_SPLIT:
        raise ValueError("LoRA optimization only permits train")
    if evaluation_split != EVALUATION_SPLIT:
        raise ValueError("LoRA smoke evaluation only permits validation")
    git = validate_git_provenance(git_provenance)
    artifact = load_training_artifact(
        config=config,
        repository_root=repository_root,
        derived_root=derived_root,
    )
    train_records = load_adapter_samples(
        path=artifact.train_path,
        split=OPTIMIZATION_SPLIT,
        artifact=artifact,
    )
    validation_records = load_adapter_samples(
        path=artifact.validation_path,
        split=EVALUATION_SPLIT,
        artifact=artifact,
    )
    train_samples = select_train_subset(
        train_records, size=config.train_subset_size, seed=config.seed
    )
    validation_samples = select_validation_subset(
        validation_records,
        size=config.validation_subset_size,
        seed=config.seed,
    )
    runtime = dependencies or default_runtime_dependencies()
    device = runtime.device_selector(config.device)
    dtype = runtime.dtype_selector(config.dtype_preference)
    processor = runtime.processor_loader(
        config.model_id,
        config.processor_revision,
        config.local_files_only,
    )
    base_model = runtime.model_loader(
        config.model_id,
        config.model_revision,
        dtype,
        config.attention_implementation,
        config.local_files_only,
    )
    model = inject_lora(base_model, config=config, dependencies=runtime)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    parameter_report = trainable_parameter_report(
        model, config.lora_target_modules
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = runtime.optimizer_factory(
        trainable_parameters, config.learning_rate
    )
    collator = Qwen3VLSupervisedCollator(
        processor=processor,
        image_loader=runtime.image_loader,
        nuscenes_root=nuscenes_root,
        config=config,
    )
    loss_history = run_training_steps(
        model=model,
        samples=train_samples,
        collator=collator,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    output_dir = resolve_derived_path(derived_root, config.output_relative_dir)
    try:
        output_dir.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("LoRA smoke output must not be inside repository")
    if output_dir.exists():
        raise FileExistsError("LoRA smoke output already exists")
    checkpoint_dir = output_dir / config.checkpoint_dirname
    checkpoint_dir.parent.mkdir(parents=True)
    model.save_pretrained(checkpoint_dir)
    checkpoint_file, checkpoint_sha256 = _checkpoint_sha256(checkpoint_dir)
    reloaded_base = runtime.model_loader(
        config.model_id,
        config.model_revision,
        dtype,
        config.attention_implementation,
        config.local_files_only,
    )
    reloaded_model = runtime.adapter_loader(reloaded_base, checkpoint_dir)
    reloaded_model.to(device)
    train_predictions = _predict_samples(
        model=reloaded_model,
        processor=processor,
        samples=train_samples,
        nuscenes_root=nuscenes_root,
        config=config,
        device=device,
        dependencies=runtime,
    )
    validation_predictions = _predict_samples(
        model=reloaded_model,
        processor=processor,
        samples=validation_samples,
        nuscenes_root=nuscenes_root,
        config=config,
        device=device,
        dependencies=runtime,
    )
    result: dict[str, object] = {
        "artifact_version": config.artifact_version,
        "artifact_schema_version": config.artifact_schema_version,
        "run_kind": "tiny_overfit_smoke",
        "config": {
            "relative_path": config_relative_path,
            "sha256": config.config_sha256,
        },
        "git": asdict(git),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "processor_revision": config.processor_revision,
        "transformers_version": runtime.package_version("transformers"),
        "peft_version": runtime.package_version("peft"),
        "input_variant": config.input_variant,
        "optimization_split": optimization_split,
        "evaluation_split": evaluation_split,
        "training_sample_count": len(train_samples),
        "validation_smoke_sample_count": len(validation_samples),
        "selected_train_action_distribution": _action_distribution(train_samples),
        "training_loss_history": loss_history,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "trainable_parameter_summary": parameter_report,
        "lora_config": lora_config_kwargs(config),
        "tiny_overfit_predictions": train_predictions,
        "tiny_overfit_metrics": build_metrics(train_predictions),
        "validation_smoke_predictions": validation_predictions,
        "validation_smoke_metrics": build_metrics(validation_predictions),
        "checkpoint": {
            "adapter_path": str(checkpoint_dir),
            "weights_path": str(checkpoint_file),
            "sha256": checkpoint_sha256,
            "full_model_saved": False,
        },
        "checkpoint_reload_result": {
            "completed": True,
            "parser_version": config.parser_version,
            "prediction_count": len(train_predictions)
            + len(validation_predictions),
        },
        "adapter_receipt_sha256": artifact.receipt_sha256,
        "test_files_opened": 0,
        "test_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
    }
    (output_dir / "smoke_result.json").write_bytes(
        canonical_json_bytes(result) + b"\n"
    )
    return result

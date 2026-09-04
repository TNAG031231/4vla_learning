from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.phase0.qwen3vl_dataset_adapter import VARIANTS


PRODUCER_INTERFACE_VERSION = "phase0.3e2-qwen3vl-producer-interface-v0.1"
FIXED_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
FIXED_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROMPT_VERSION = "phase0.3c-zero-shot-v0.1"
FIXED_TASK_PROMPT = (
    "Based only on the provided current observation, predict the ego vehicle's\n"
    "near-future coarse driving action.\n\n"
    "Choose exactly one action from:\n\n"
    "keep\n"
    "accelerate\n"
    "decelerate\n"
    "stop\n"
    "left_lateral\n"
    "right_lateral\n\n"
    "Output exactly the action name and nothing else."
)
PROCESSOR_REQUIRED_FIELDS = frozenset(
    ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
)
PROCESSOR_DTYPES = {
    "input_ids": "torch.int64",
    "attention_mask": "torch.int64",
    "pixel_values": "torch.float32",
    "image_grid_thw": "torch.int64",
}
PIXEL_VALUES_PATCH_WIDTH = 1536


@dataclass(frozen=True)
class ProcessorInputMetadata:
    input_keys: tuple[str, ...]
    input_ids_shape: tuple[int, int]
    attention_mask_shape: tuple[int, int]
    pixel_values_shape: tuple[int, int]
    image_grid_thw_shape: tuple[int, int]
    dtypes: Mapping[str, str]


def build_multimodal_messages(
    *,
    variant: str,
    image: object,
    ego_state_text: str | None,
) -> list[dict[str, object]]:
    if variant not in VARIANTS:
        raise ValueError("unsupported input variant")
    prompt = FIXED_TASK_PROMPT
    if variant == "image_ego_state":
        if ego_state_text is None:
            raise ValueError("image_ego_state input is missing adapter serialization")
        prompt = f"{ego_state_text}\n\n{prompt}"
    elif ego_state_text is not None:
        raise ValueError("image_only input must not contain ego-state text")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _tensor_shape(value: object, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"processor {field_name} is not tensor-like")
    dimensions = tuple(int(dimension) for dimension in shape)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"processor {field_name} has an empty dimension")
    return dimensions


def _tensor_dtype(value: object, field_name: str) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise ValueError(f"processor {field_name} has no dtype")
    return str(dtype)


def validate_processor_inputs(
    inputs: object,
    *,
    expected_batch_size: int,
    expected_image_count: int,
) -> ProcessorInputMetadata:
    if expected_batch_size <= 0 or expected_image_count <= 0:
        raise ValueError("expected processor counts must be positive")
    if not isinstance(inputs, Mapping):
        raise ValueError("processor output must be a mapping")
    missing = PROCESSOR_REQUIRED_FIELDS - set(inputs)
    if missing:
        raise ValueError(
            f"processor output is missing required fields: {sorted(missing)}"
        )

    shapes = {
        field_name: _tensor_shape(inputs[field_name], field_name)
        for field_name in PROCESSOR_REQUIRED_FIELDS
    }
    input_ids_shape = shapes["input_ids"]
    attention_mask_shape = shapes["attention_mask"]
    pixel_values_shape = shapes["pixel_values"]
    image_grid_thw_shape = shapes["image_grid_thw"]
    if len(input_ids_shape) != 2 or input_ids_shape[0] != expected_batch_size:
        raise ValueError("processor input_ids must have shape [batch, sequence]")
    if attention_mask_shape != input_ids_shape:
        raise ValueError("processor attention_mask must match input_ids shape")
    if (
        len(pixel_values_shape) != 2
        or pixel_values_shape[1] != PIXEL_VALUES_PATCH_WIDTH
    ):
        raise ValueError(
            "processor pixel_values must have shape [visual_patches, 1536]"
        )
    if image_grid_thw_shape != (expected_image_count, 3):
        raise ValueError("processor image_grid_thw must have shape [images, 3]")

    dtypes = {
        field_name: _tensor_dtype(inputs[field_name], field_name)
        for field_name in PROCESSOR_REQUIRED_FIELDS
    }
    for field_name, expected_dtype in PROCESSOR_DTYPES.items():
        if dtypes[field_name] != expected_dtype:
            raise ValueError(
                f"processor {field_name} dtype must be {expected_dtype}"
            )
    return ProcessorInputMetadata(
        input_keys=tuple(sorted(str(key) for key in inputs)),
        input_ids_shape=(input_ids_shape[0], input_ids_shape[1]),
        attention_mask_shape=(
            attention_mask_shape[0],
            attention_mask_shape[1],
        ),
        pixel_values_shape=(pixel_values_shape[0], pixel_values_shape[1]),
        image_grid_thw_shape=(
            image_grid_thw_shape[0],
            image_grid_thw_shape[1],
        ),
        dtypes=dtypes,
    )

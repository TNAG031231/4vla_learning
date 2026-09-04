from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PRODUCER_INTERFACE_VERSION = "phase0.3e2-qwen3vl-producer-interface-v0.1"
FIXED_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
FIXED_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
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
PLANNING_FEATURE_INTERFACE_VERSION = (
    "phase0.3e2-qwen3vl-planning-feature-v0.1"
)
PLANNING_FEATURE_EXTRACTION_POLICY = (
    "Qwen3VLForConditionalGeneration.get_image_features"
)
PLANNING_FEATURE_DIM = 2560
PLANNING_FEATURE_DTYPE = "torch.bfloat16"
VISION_SPATIAL_MERGE_SIZE = 2
PLANNING_DYNAMIC_TOKEN_DIMENSION = "N_i"
PLANNING_FRAME_ALIGNMENT = (
    "image_embeds[i] corresponds to input image i and image_grid_thw[i]"
)


@dataclass(frozen=True)
class ProcessorInputMetadata:
    input_keys: tuple[str, ...]
    input_ids_shape: tuple[int, int]
    attention_mask_shape: tuple[int, int]
    pixel_values_shape: tuple[int, int]
    image_grid_thw_shape: tuple[int, int]
    dtypes: Mapping[str, str]


@dataclass(frozen=True)
class PlanningFeatureMetadata:
    feature_interface_version: str
    model_id: str
    model_revision: str
    processor_revision: str
    extraction_policy: str
    per_image_shapes: tuple[tuple[int, int], ...]
    per_image_token_counts: tuple[int, ...]
    feature_dtype: str
    dynamic_token_dimension: str
    stable_feature_dimension: int
    frame_alignment: str


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


def _image_grid_rows(image_grid_thw: object) -> tuple[tuple[int, int, int], ...]:
    shape = _tensor_shape(image_grid_thw, "image_grid_thw")
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError("processor image_grid_thw must have shape [images, 3]")
    tolist = getattr(image_grid_thw, "tolist", None)
    if not callable(tolist):
        raise ValueError("processor image_grid_thw must provide tensor values")
    values = tolist()
    if not isinstance(values, list):
        raise ValueError("processor image_grid_thw values must be row-major")
    rows = tuple(tuple(int(value) for value in row) for row in values)
    if any(len(row) != 3 or any(value <= 0 for value in row) for row in rows):
        raise ValueError("processor image_grid_thw values must be positive THW rows")
    return rows


def _visual_patch_counts(image_grid_thw: object) -> tuple[int, ...]:
    return tuple(
        temporal * height * width
        for temporal, height, width in _image_grid_rows(image_grid_thw)
    )


def planning_visual_token_counts(image_grid_thw: object) -> tuple[int, ...]:
    merge_area = VISION_SPATIAL_MERGE_SIZE**2
    patch_counts = _visual_patch_counts(image_grid_thw)
    if any(patch_count % merge_area for patch_count in patch_counts):
        raise ValueError("image_grid_thw patch count must align to spatial merge")
    return tuple(patch_count // merge_area for patch_count in patch_counts)


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
    if pixel_values_shape[0] != sum(
        _visual_patch_counts(inputs["image_grid_thw"])
    ):
        raise ValueError(
            "processor pixel_values rows must match image_grid_thw patch count"
        )

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


def validate_planning_image_embeds(
    image_embeds: object,
    *,
    image_grid_thw: object,
) -> PlanningFeatureMetadata:
    if not isinstance(image_embeds, Sequence):
        raise ValueError("primary image_embeds must be a per-image sequence")
    token_counts = planning_visual_token_counts(image_grid_thw)
    if len(image_embeds) != len(token_counts):
        raise ValueError("primary image_embeds must align one-to-one with images")

    shapes = tuple(
        _tensor_shape(image_embed, f"image_embeds[{index}]")
        for index, image_embed in enumerate(image_embeds)
    )
    for shape, token_count in zip(shapes, token_counts, strict=True):
        if shape != (token_count, PLANNING_FEATURE_DIM):
            raise ValueError(
                "primary image_embeds visual token count or feature dimension mismatch"
            )
    dtypes = tuple(
        _tensor_dtype(image_embed, f"image_embeds[{index}]")
        for index, image_embed in enumerate(image_embeds)
    )
    if any(dtype != PLANNING_FEATURE_DTYPE for dtype in dtypes):
        raise ValueError(
            f"primary image_embeds dtype must be {PLANNING_FEATURE_DTYPE}"
        )
    return PlanningFeatureMetadata(
        feature_interface_version=PLANNING_FEATURE_INTERFACE_VERSION,
        model_id=FIXED_MODEL_ID,
        model_revision=FIXED_REVISION,
        processor_revision=FIXED_REVISION,
        extraction_policy=PLANNING_FEATURE_EXTRACTION_POLICY,
        per_image_shapes=tuple(
            (shape[0], shape[1])
            for shape in shapes
        ),
        per_image_token_counts=token_counts,
        feature_dtype=PLANNING_FEATURE_DTYPE,
        dynamic_token_dimension=PLANNING_DYNAMIC_TOKEN_DIMENSION,
        stable_feature_dimension=PLANNING_FEATURE_DIM,
        frame_alignment=PLANNING_FRAME_ALIGNMENT,
    )

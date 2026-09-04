from __future__ import annotations

from dataclasses import fields
import hashlib
from inspect import signature
from pathlib import Path
import sys

import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.qwen3vl_dataset_adapter import (
    ADAPTER_RECORD_FIELDS,
    adapter_record,
    load_config as load_adapter_config,
    serialize_ego_state,
)
import src.phase0.qwen3vl_interface as interface_module
from src.phase0.qwen3vl_interface import (
    FIXED_MODEL_ID,
    FIXED_REVISION,
    PIXEL_VALUES_PATCH_WIDTH,
    PLANNING_FEATURE_DIM,
    PLANNING_FEATURE_DTYPE,
    PLANNING_FEATURE_EXTRACTION_POLICY,
    PLANNING_FEATURE_INTERFACE_VERSION,
    PRODUCER_INTERFACE_VERSION,
    VISION_SPATIAL_MERGE_SIZE,
    planning_visual_token_counts,
    validate_planning_image_embeds,
    validate_processor_inputs,
)
from src.phase0.qwen3vl_lora_smoke import load_config as load_lora_config
from src.phase0.qwen3vl_zero_shot import (
    FIXED_TASK_PROMPT,
    LEGACY_PREDICTION_FIELDS,
    PROMPT_VERSION,
    build_multimodal_messages,
    load_config as load_zero_shot_config,
    parse_action_output,
)

ADAPTER_CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_dataset_adapter.yaml"
ZERO_SHOT_CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_zero_shot.yaml"
LORA_CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_lora_smoke.yaml"


def _motion() -> dict[str, object]:
    return {
        "speed_mps": 4.25,
        "longitudinal_acceleration_mps2": -0.5,
        "yaw_rate_radps": 0.125,
        "source": "ego_pose_past_difference",
        "timestamp_source": "CAM_FRONT_sample_data",
        "availability": "full",
        "history_interval_sec": 0.5,
        "acceleration_interval_sec": 0.5,
        "unavailable_reason": None,
    }


def _processor_output(
    *, sequence_length: int, image_grid_thw: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch_size = image_grid_thw.shape[0]
    visual_patches = int(image_grid_thw.prod(dim=1).sum().item())
    return {
        "input_ids": torch.ones((batch_size, sequence_length), dtype=torch.long),
        "attention_mask": torch.ones(
            (batch_size, sequence_length), dtype=torch.long
        ),
        "pixel_values": torch.ones(
            (visual_patches, PIXEL_VALUES_PATCH_WIDTH), dtype=torch.float32
        ),
        "image_grid_thw": image_grid_thw,
    }


def test_phase03_consumers_share_frozen_model_processor_and_prompt() -> None:
    assert PRODUCER_INTERFACE_VERSION == (
        "phase0.3e2-qwen3vl-producer-interface-v0.1"
    )
    zero_shot = load_zero_shot_config(ZERO_SHOT_CONFIG_PATH)
    lora = load_lora_config(LORA_CONFIG_PATH)
    for config in (zero_shot, lora):
        assert config.model_id == FIXED_MODEL_ID
        assert config.model_revision == FIXED_REVISION
        assert config.processor_revision == FIXED_REVISION
        assert config.prompt_version == PROMPT_VERSION
        assert config.task_prompt == FIXED_TASK_PROMPT


def test_adapter_producer_freezes_record_and_ego_serialization() -> None:
    config = load_adapter_config(ADAPTER_CONFIG_PATH)
    source = {
        "sample_token": "sample-token",
        "scene_token": "scene-token",
        "timestamp": 123,
        "split": "validation",
        "cam_front_path": "samples/CAM_FRONT/example.jpg",
        "current_ego_motion": _motion(),
        "target_action": "keep",
        "projection_record_sha256": "a" * 64,
    }
    record = adapter_record(
        source,
        variant="image_ego_state",
        config=config,
        source_file_sha256="b" * 64,
    )
    assert set(record) == ADAPTER_RECORD_FIELDS
    assert serialize_ego_state(_motion(), config) == (
        "Current ego state:\n"
        "speed_mps=4.250000; longitudinal_acceleration_mps2=-0.500000; "
        "yaw_rate_radps=0.125000; history_interval_sec=0.500000; "
        "acceleration_interval_sec=0.500000; availability=full"
    )
    for forbidden in (
        "future_ego_trajectory",
        "nearby_agents",
        "gt_boxes",
        "gt_occupancy",
        "future_agents",
        "test_labels",
    ):
        assert forbidden not in record


@pytest.mark.parametrize("variant", ("image_only", "image_ego_state"))
def test_legacy_messages_contain_only_inference_inputs(variant: str) -> None:
    ego_state = (
        "Current ego state:\nspeed_mps=4.250000"
        if variant == "image_ego_state"
        else None
    )
    messages = build_multimodal_messages(
        variant=variant,
        image=object(),
        ego_state_text=ego_state,
    )
    assert len(messages) == 1
    assert [item["type"] for item in messages[0]["content"]] == ["image", "text"]
    serialized = repr(messages)
    for forbidden in (
        "target_action",
        "future_ego_trajectory",
        "nearby_agents",
        "gt_boxes",
        "occupancy",
        "test_labels",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("sequence_length", "image_grid_thw"),
    (
        (1413, torch.tensor([[1, 56, 100]], dtype=torch.long)),
        (
            317,
            torch.tensor(
                [[1, 8, 8], [1, 4, 16]],
                dtype=torch.long,
            ),
        ),
    ),
)
def test_processor_contract_keeps_sequence_and_visual_axes_dynamic(
    sequence_length: int,
    image_grid_thw: torch.Tensor,
) -> None:
    batch_size = image_grid_thw.shape[0]
    visual_patches = int(image_grid_thw.prod(dim=1).sum().item())
    inputs = _processor_output(
        sequence_length=sequence_length,
        image_grid_thw=image_grid_thw,
    )
    metadata = validate_processor_inputs(
        inputs,
        expected_batch_size=batch_size,
        expected_image_count=batch_size,
    )
    assert metadata.input_ids_shape == (batch_size, sequence_length)
    assert metadata.attention_mask_shape == (batch_size, sequence_length)
    assert metadata.pixel_values_shape == (
        visual_patches,
        PIXEL_VALUES_PATCH_WIDTH,
    )
    assert metadata.image_grid_thw_shape == (batch_size, 3)
    assert metadata.input_keys == (
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    )


def test_processor_contract_rejects_missing_or_incompatible_fields() -> None:
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    missing = _processor_output(sequence_length=8, image_grid_thw=grid)
    del missing["image_grid_thw"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_processor_inputs(
            missing,
            expected_batch_size=1,
            expected_image_count=1,
        )

    wrong_patch_width = _processor_output(
        sequence_length=8,
        image_grid_thw=grid,
    )
    wrong_patch_width["pixel_values"] = torch.ones((4, 768))
    with pytest.raises(ValueError, match="visual_patches, 1536"):
        validate_processor_inputs(
            wrong_patch_width,
            expected_batch_size=1,
            expected_image_count=1,
        )

    wrong_patch_count = _processor_output(
        sequence_length=8,
        image_grid_thw=grid,
    )
    wrong_patch_count["pixel_values"] = torch.ones(
        (8, PIXEL_VALUES_PATCH_WIDTH),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="image_grid_thw patch count"):
        validate_processor_inputs(
            wrong_patch_count,
            expected_batch_size=1,
            expected_image_count=1,
        )


def test_planning_feature_contract_uses_public_primary_image_embeds() -> None:
    assert PLANNING_FEATURE_INTERFACE_VERSION == (
        "phase0.3e2-qwen3vl-planning-feature-v0.1"
    )
    assert PLANNING_FEATURE_EXTRACTION_POLICY == (
        "Qwen3VLForConditionalGeneration.get_image_features"
    )
    assert PLANNING_FEATURE_DIM == 2560
    assert PLANNING_FEATURE_DTYPE == "torch.bfloat16"
    assert VISION_SPATIAL_MERGE_SIZE == 2

    grid = torch.tensor(
        [[1, 56, 100], [1, 28, 50]],
        dtype=torch.long,
    )
    assert planning_visual_token_counts(grid) == (1400, 350)
    metadata = validate_planning_image_embeds(
        (
            torch.ones((1400, 2560), dtype=torch.bfloat16),
            torch.ones((350, 2560), dtype=torch.bfloat16),
        ),
        image_grid_thw=grid,
    )
    assert metadata.per_image_shapes == ((1400, 2560), (350, 2560))
    assert metadata.per_image_token_counts == (1400, 350)
    assert metadata.feature_dtype == "torch.bfloat16"


def test_planning_feature_contract_rejects_fixed_or_misaligned_tokens() -> None:
    grid = torch.tensor([[1, 28, 50]], dtype=torch.long)
    with pytest.raises(ValueError, match="visual token count"):
        validate_planning_image_embeds(
            (torch.ones((1400, 2560), dtype=torch.bfloat16),),
            image_grid_thw=grid,
        )


def test_legacy_contract_is_frozen_but_not_required_by_planning_features() -> None:
    assert hashlib.sha256(FIXED_TASK_PROMPT.encode()).hexdigest() == (
        "0db9a9c679aa088eab9268eadcd2a2a96c0bb622f376601b34e6d46be5667c17"
    )
    for action in ACTION_SCHEMA:
        assert parse_action_output(action)["predicted_action"] == action
    assert "parsed_action" in LEGACY_PREDICTION_FIELDS
    assert "raw_output" in LEGACY_PREDICTION_FIELDS
    assert "parser_version" in LEGACY_PREDICTION_FIELDS
    assert "target_action" not in signature(build_multimodal_messages).parameters
    planning_fields = {
        field.name
        for field in fields(interface_module.PlanningFeatureMetadata)
    }
    assert "prompt" not in planning_fields
    assert "parser" not in planning_fields
    assert "deepstack" not in planning_fields
    assert not hasattr(interface_module, "FIXED_TASK_PROMPT")
    assert not hasattr(interface_module, "build_multimodal_messages")
    assert not hasattr(interface_module, "parse_action_output")
    assert not hasattr(interface_module, "LEGACY_PREDICTION_FIELDS")

from __future__ import annotations

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
    FIXED_TASK_PROMPT,
    PIXEL_VALUES_PATCH_WIDTH,
    PRODUCER_INTERFACE_VERSION,
    PROMPT_VERSION,
    build_multimodal_messages,
    validate_processor_inputs,
)
from src.phase0.qwen3vl_lora_smoke import load_config as load_lora_config
from src.phase0.qwen3vl_zero_shot import (
    LEGACY_PREDICTION_FIELDS,
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
    *, batch_size: int, sequence_length: int, visual_patches: int
) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.ones((batch_size, sequence_length), dtype=torch.long),
        "attention_mask": torch.ones(
            (batch_size, sequence_length), dtype=torch.long
        ),
        "pixel_values": torch.ones(
            (visual_patches, PIXEL_VALUES_PATCH_WIDTH), dtype=torch.float32
        ),
        "image_grid_thw": torch.ones((batch_size, 3), dtype=torch.long),
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
def test_phase04_messages_contain_only_inference_inputs(variant: str) -> None:
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
    ("batch_size", "sequence_length", "visual_patches"),
    ((1, 1438, 5600), (2, 317, 128)),
)
def test_processor_contract_keeps_sequence_and_visual_axes_dynamic(
    batch_size: int,
    sequence_length: int,
    visual_patches: int,
) -> None:
    inputs = _processor_output(
        batch_size=batch_size,
        sequence_length=sequence_length,
        visual_patches=visual_patches,
    )
    inputs["mm_token_type_ids"] = inputs["attention_mask"].clone()
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
        "mm_token_type_ids",
        "pixel_values",
    )


def test_processor_contract_rejects_missing_or_incompatible_fields() -> None:
    missing = _processor_output(batch_size=1, sequence_length=8, visual_patches=4)
    del missing["image_grid_thw"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_processor_inputs(
            missing,
            expected_batch_size=1,
            expected_image_count=1,
        )

    wrong_patch_width = _processor_output(
        batch_size=1, sequence_length=8, visual_patches=4
    )
    wrong_patch_width["pixel_values"] = torch.ones((4, 768))
    with pytest.raises(ValueError, match="visual_patches, 1536"):
        validate_processor_inputs(
            wrong_patch_width,
            expected_batch_size=1,
            expected_image_count=1,
        )


def test_legacy_contract_is_frozen_but_not_required_by_phase04_messages() -> None:
    for action in ACTION_SCHEMA:
        assert parse_action_output(action)["predicted_action"] == action
    assert "parsed_action" in LEGACY_PREDICTION_FIELDS
    assert "raw_output" in LEGACY_PREDICTION_FIELDS
    assert "parser_version" in LEGACY_PREDICTION_FIELDS
    assert "target_action" not in signature(build_multimodal_messages).parameters
    assert not hasattr(interface_module, "parse_action_output")
    assert not hasattr(interface_module, "LEGACY_PREDICTION_FIELDS")

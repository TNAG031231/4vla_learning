from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4_factorized_targets import (
    LATERAL_ACTIONS, LONGITUDINAL_ACTIONS, derive_feature_targets,
)
from src.phase0.phase0_4_temporal_dataset import build_record, collect_history
from src.phase0.phase0_4b_protocol import (
    ActionTarget, StructuredCollator, adapt_record, inference_messages,
    parse_action, serialize_action, target_token_mask,
)
from src.phase0.qwen3vl_dataset_adapter import load_config
from src.phase0.qwen3vl_interface import FIXED_MODEL_ID, FIXED_REVISION
from test_phase0_4_temporal_dataset import case


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(FIXED_MODEL_ID, revision=FIXED_REVISION, local_files_only=True)


@pytest.fixture
def samples(case):
    reader, records = case
    config = load_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    return [adapt_record(build_record(row, collect_history(reader, row, 3), 3), config)
            for row in records]


@pytest.mark.parametrize("longitudinal,lateral", tuple(product(LONGITUDINAL_ACTIONS, LATERAL_ACTIONS)))
def test_round_trip(longitudinal, lateral):
    assert parse_action(serialize_action(longitudinal, lateral)) == {
        "longitudinal": longitudinal, "lateral": lateral,
    }


@pytest.mark.parametrize("text", [
    "", "decelerate", "left_lateral", "The car should slow down and move left.",
    "longitudinal=keep", "lateral=left", "longitudinal=yield; lateral=left",
    "longitudinal=keep; lateral=turn_left", "longitudinal=keep,lateral=left",
    "longitudinal=keep;lateral=left", "longitudinal=keep; lateral=left; lateral=right",
    "longitudinal=keep; lateral=left; speed=1", "longitudinal=keep; lateral=left\n",
    " longitudinal=keep; lateral=left", "lateral=left; longitudinal=keep",
    "longitudinal=keep; lateral=", "longitudinal=; lateral=left",
])
def test_strict_parser_rejects_noncanonical(text):
    assert parse_action(text) is None


@pytest.mark.parametrize("long_valid,lat_valid", tuple(product((False, True), repeat=2)))
@pytest.mark.parametrize("longitudinal,lateral", [("decelerate", "straight"), ("stop", "left")])
def test_real_processor_independent_masks(processor, samples, tmp_path, long_valid, lat_valid,
                                           longitudinal, lateral):
    target = ActionTarget(longitudinal if long_valid else None, lateral if lat_valid else None,
                          long_valid, lat_valid)
    selected = [replace(samples[0], target=target), replace(samples[3], target=target)]
    collator = StructuredCollator(processor, lambda path: Image.new("RGB", (64, 64)), tmp_path)
    batch = collator(selected, expected_split="train")
    assert batch["image_grid_thw"].shape[0] == 4
    assert batch["attention_mask"][0].sum() != batch["attention_mask"][1].sum()
    for index, sample in enumerate(selected):
        prefix = processor.apply_chat_template(collator.messages(sample), tokenize=True,
                                               add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")["input_ids"][0].tolist()
        active = batch["attention_mask"][index].nonzero().flatten().tolist()
        target_ids, mask = target_token_mask(processor.tokenizer, target)
        supervised = batch["labels"][index] != -100
        positions = supervised.nonzero().flatten().tolist()
        assert positions == [active[len(prefix) + i] for i, value in enumerate(mask) if value]
        assert batch["labels"][index, active[:len(prefix)]].eq(-100).all()
        assert batch["labels"][index, active[len(prefix) + len(target_ids):]].eq(-100).all()
        assert batch["labels"][index, batch["attention_mask"][index] == 0].eq(-100).all()
        decoded = processor.tokenizer.decode(batch["labels"][index, supervised].tolist())
        expected = (f"longitudinal={longitudinal}" if long_valid else "")
        expected += ";" if long_valid and lat_valid else ""
        expected += f" lateral={lateral}" if lat_valid else ""
        assert decoded == expected


def test_real_merged_equals_straight_and_multitoken_values(processor):
    target = ActionTarget("decelerate", "straight", True, True)
    ids, mask = target_token_mask(processor.tokenizer, target)
    assert ids == [4825, 12842, 977, 28, 450, 3672, 58668, 26, 44469, 15932, 7386]
    assert processor.tokenizer.convert_ids_to_tokens(ids)[-2:] == ["=str", "aight"]
    assert all(mask)
    ids, _ = target_token_mask(processor.tokenizer, ActionTarget("accelerate", "left", True, True))
    assert processor.tokenizer.convert_ids_to_tokens(ids)[4:6] == ["accel", "erate"]


def test_adapter_uses_producer_and_removes_target_geometry(case):
    reader, records = case
    row = build_record(records[3], collect_history(reader, records[3], 3), 3)
    config = load_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    original = adapt_record(row, config)
    row.update(derive_feature_targets({"path_length_m": 3, "end_speed_proxy_mps": 2,
                                      "delta_speed_proxy_mps": 2, "final_lateral_displacement_m": -2}))
    row["future_waypoints"] = [[999, 888]] * 6
    for field in ("future_ego_trajectory", "gt_boxes", "gt_occupancy", "future_agents",
                  "legacy_meta_action", "test_labels", "reviewer_only"):
        row[field] = "LEAK_SENTINEL"
    adapted = adapt_record(row, config)
    assert adapted.observation == original.observation
    assert adapted.target != original.target
    messages = inference_messages(adapted.observation, [object()] * 3)
    assert "LEAK_SENTINEL" not in repr(messages)
    assert "999" not in repr(messages)
    assert "target" not in repr(adapted.observation)


@pytest.mark.parametrize("split", ["test", "validation"])
def test_split_rejected_before_other_field_or_image_access(split, processor, samples, tmp_path):
    with pytest.raises(ValueError, match="only permits train"):
        adapt_record({"split": split}, None)
    collator = StructuredCollator(processor, lambda path: pytest.fail("image accessed"), tmp_path)
    with pytest.raises(ValueError, match="only permits train"):
        collator([replace(samples[0], split=split)], expected_split="train")

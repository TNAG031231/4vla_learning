from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase0_4c_planner_smoke import main
from src.phase0.phase0_4b_protocol import ActionTarget
from src.phase0.phase0_4c_two_turn_planner import (
    PLANNING_PROMPT, WaypointDecoder, contextual_hidden_states, freeze_backbone,
    load_config, masked_waypoint_loss, planning_inputs, planning_messages, predicted_action_inference,
)
from test_phase0_4b_protocol import case, processor, samples


@pytest.fixture
def config():
    return load_config(ROOT / "configs/phase0_4c_planner_smoke.yaml")


@pytest.fixture
def tiny_qwen():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, rope_scaling=dict(rope_type="default", mrope_section=[1, 1, 2])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, deepstack_visual_indexes=[0, 1],
                           num_position_embeddings=16),
    )
    model = Qwen3VLForConditionalGeneration(config)
    freeze_backbone(model)
    return model


def test_real_processor_two_turn_context_has_exact_action_and_no_targets(processor, samples):
    sample = samples[3]
    images = [Image.new("RGB", (64, 64))] * 3
    predicted = "longitudinal=decelerate; lateral=left"
    messages = planning_messages(sample.observation, images, predicted)
    inputs, evidence = planning_inputs(processor, messages, "cpu")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0]["text"] == predicted
    assert messages[2]["content"][0]["text"] == PLANNING_PROMPT
    assert evidence["action_decoded"] == predicted
    assert evidence["rendered_context"].endswith("<|im_start|>assistant\n")
    assert inputs["image_grid_thw"].shape[0] == 3
    changed = replace(sample, target=ActionTarget("stop", "right", True, True))
    assert planning_messages(changed.observation, images, predicted) == messages
    for field in ("future_waypoints", "future_ego_trajectory", "gt_action", "gt_occupancy", "future_agents"):
        assert field not in repr(messages)


def test_decoder_shape_masked_memory_and_backward(config):
    torch.manual_seed(config.seed)
    planner = WaypointDecoder(32, config).eval()
    memory = torch.randn(2, 9, 32)
    mask = torch.tensor([[1] * 9, [1] * 4 + [0] * 5])
    prediction = planner(memory, mask)
    changed = memory.clone()
    changed[1, 4:] = 1234
    torch.testing.assert_close(prediction, planner(changed, mask))
    assert prediction.shape == (2, 6, 2)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    for module in planner.children():
        assert all(p.requires_grad for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


def test_loss_masks_invalid_targets_and_gradients():
    prediction = torch.zeros(1, 6, 2, requires_grad=True)
    target = torch.tensor([[[1., 3.], [float("nan"), float("nan")]] * 3])
    mask = torch.tensor([[True, False] * 3])
    loss = masked_waypoint_loss(prediction, target, mask, beta=1.)
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert prediction.grad[:, 1::2].eq(0).all()
    assert prediction.grad[:, ::2].ne(0).all()
    with pytest.raises(ValueError, match="at least one valid"):
        masked_waypoint_loss(prediction, target, torch.zeros_like(mask), beta=1.)


def test_real_tiny_qwen_output_contract_and_frozen_backward(tiny_qwen, processor, samples, config):
    messages = planning_messages(samples[0].observation, [Image.new("RGB", (64, 64))],
                                 "longitudinal=keep; lateral=straight")
    inputs, _ = planning_inputs(processor, messages, "cpu")
    hidden = contextual_hidden_states(tiny_qwen, inputs)
    assert hidden.shape == (*inputs["input_ids"].shape, 32)
    assert not hidden.requires_grad and not hidden.is_inference()
    planner = WaypointDecoder(32, config)
    prediction = planner(hidden, inputs["attention_mask"])
    loss = masked_waypoint_loss(prediction, torch.ones_like(prediction),
                                torch.ones(1, 6, dtype=torch.bool), beta=1.)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(not p.requires_grad and p.grad is None for p in tiny_qwen.parameters())
    assert all(p.requires_grad and p.grad is not None for p in planner.parameters())
    assert sum(p.grad.abs().sum() for p in planner.parameters()) > 0


def test_peft_backbone_and_adapter_freeze(tiny_qwen):
    peft = pytest.importorskip("peft", reason="PEFT adapter freeze also checked by AutoDL smoke")
    model = peft.get_peft_model(tiny_qwen, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    assert any(p.requires_grad for name, p in model.named_parameters() if "lora_" in name)
    freeze_backbone(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_predicted_path_uses_generated_action_not_gt(tiny_qwen, processor, samples, config, monkeypatch):
    generated = "longitudinal=decelerate; lateral=left"
    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 24
        suffix = torch.tensor([processor.tokenizer(generated, add_special_tokens=False)["input_ids"]])
        return torch.cat((kwargs["input_ids"], suffix), dim=1)
    monkeypatch.setattr(tiny_qwen, "generate", generate)
    planner = WaypointDecoder(32, config)
    sample = samples[0]
    output, evidence = predicted_action_inference(
        model=tiny_qwen, planner=planner, observation=sample.observation,
        images=[Image.new("RGB", (64, 64))], processor=processor,
        generation_kwargs={"do_sample": False, "num_beams": 1, "max_new_tokens": 24}, device="cpu",
    )
    assert output.shape == (1, 6, 2) and evidence["waypoints_finite"]
    assert evidence["conditioning_type"] == "predicted_action"
    assert evidence["raw_output"] == evidence["assistant_action_text"] == evidence["action_decoded"] == generated
    assert "target" not in inspect.signature(predicted_action_inference).parameters
    with pytest.raises(TypeError, match="gt_action"):
        predicted_action_inference(gt_action=sample.target)
    with pytest.raises(ValueError, match="valid frozen structured action"):
        planning_messages(sample.observation, [], "invalid action")


def test_cli_dry_run_and_test_rejection(capsys):
    assert main(["--dry-run"]) == 0
    assert "dry_run_no_data_or_model_access" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--split", "test"])
    assert error.value.code == 2

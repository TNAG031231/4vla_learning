from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_drivema_stage2_sft import main
from src.phase0 import drivema_stage2_sft as training
from src.phase0.drivema_stage2_protocol import (
    assistant_log_probs, encode_teacher_forcing, parse_trajectory,
    serialize_trajectory, trajectory_to_meta_action,
)
from src.phase0.phase0_4b_protocol import ActionTarget, StructuredCollator, adapt_record
from src.phase0.qwen3vl_dataset_adapter import load_config as load_ego_config
from test_phase0_4b_protocol import processor
from test_phase0_4c_full_train import case, records
from test_phase0_4c_two_turn_planner import tiny_qwen


@pytest.fixture
def samples(records):
    ego = load_ego_config(ROOT / "configs/phase0_3_dataset_adapter.yaml")
    return {split: [adapt_record(row, ego, expected_split=split) for row in rows]
            for split, rows in records.items()}


@pytest.fixture
def runner(processor, tmp_path):
    collator = StructuredCollator(processor, lambda path: Image.new("RGB", (64, 64)), tmp_path)
    return training.Stage2Runner(processor, collator, "cpu",
                                {"do_sample": False, "num_beams": 1, "max_new_tokens": 24}, 192)


@pytest.fixture
def lora_model(tiny_qwen):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(tiny_qwen, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    ))


def test_coordinate_precision_roundtrip_and_frozen_projection():
    points = [[index + 0.1234, -0.0001] for index in range(1, 7)]
    text = serialize_trajectory(points)
    assert "-0.00" not in text
    parsed = parse_trajectory(text)
    assert max(abs(a - b) for p, q in zip(points, parsed) for a, b in zip(p, q)) <= 0.005
    projected = trajectory_to_meta_action([[float(i), 0.] for i in range(1, 7)])
    assert projected["longitudinal_action"] == "keep"
    assert projected["lateral_action"] == "straight"
    assert trajectory_to_meta_action([[0., 0.]] * 6)["longitudinal_action"] == "stop"
    assert trajectory_to_meta_action([[float(i), float(i)] for i in range(1, 7)])["lateral_action"] == "left"


@pytest.mark.parametrize("text", [
    "", "[1.00, 0.00]", "[nan, 0.00]", "[[1.00, 0.00]]", "[1, 0], " * 6,
    ", ".join(["[1.00, 0.00]"] * 6) + "\n",
    "Here are the points: " + ", ".join(["[1.00, 0.00]"] * 6),
    ", ".join(["[1e2, 0.00]"] * 6),
])
def test_parser_rejects_invalid_generation(text):
    assert parse_trajectory(text) is None


@pytest.mark.parametrize("index", [0, 3])
def test_real_processor_spans_mask_images_and_eos(processor, samples, records, runner, index):
    sample = samples["train"][index]
    batch, spans = runner.training_batch(sample, records["train"][index])
    assert batch["image_grid_thw"].shape[0] == (1 if index == 0 else 3)
    supervised = batch["labels"].ne(-100)
    expected = torch.zeros_like(supervised)
    for span in spans:
        expected[:, span.start:span.end] = True
        assert batch["input_ids"][0, span.end - 1] == processor.tokenizer.eos_token_id
    assert torch.equal(supervised, expected)
    assert processor.tokenizer.decode(batch["labels"][0, spans[0].start:spans[0].end - 1]) == sample.target.training_text()
    text = processor.tokenizer.decode(batch["labels"][0, spans[1].start:spans[1].end - 1])
    assert parse_trajectory(text) is not None
    assert batch["labels"][0, :spans[0].start].eq(-100).all()
    assert batch["labels"][0, spans[0].end:spans[1].start].eq(-100).all()


def test_selected_log_probs_equal_causal_ce_and_update_only_lora(lora_model, runner, samples, records):
    batch, spans = runner.training_batch(samples["train"][0], records["train"][0])
    lora_model.eval()
    selected = assistant_log_probs(lora_model, batch, spans)
    loss = -torch.cat(selected).mean()
    with torch.no_grad():
        reference = lora_model(**batch, use_cache=False).loss
    torch.testing.assert_close(loss, reference)
    before = {name: value.detach().clone() for name, value in lora_model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=0.01)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_model.parameters())
    optimizer.step()
    changed = [name for name, value in lora_model.named_parameters() if not torch.equal(before[name], value)]
    assert changed and all("lora_" in name for name in changed)
    assert all(p.grad is None for name, p in lora_model.named_parameters() if "lora_" not in name)


def scripted_generate(processor, responses, calls):
    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        calls.append(kwargs["input_ids"][0].tolist())
        turn = 0 if kwargs["max_new_tokens"] == 24 else 1
        response = responses[turn]
        suffix = processor.tokenizer(response, add_special_tokens=False)["input_ids"]
        suffix += [processor.tokenizer.eos_token_id]
        return torch.cat((kwargs["input_ids"], torch.tensor([suffix])), dim=1)
    return generate


def test_predicted_action_exact_context_and_no_gt_inputs(lora_model, processor, runner, samples, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    action = "longitudinal=decelerate; lateral=left"
    trajectory = serialize_trajectory([[i, 0.] for i in range(1, 7)])
    calls = []
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", scripted_generate(processor, [action, trajectory], calls))
    sample = replace(samples["validation"][0], target=ActionTarget("stop", "right", True, True))
    prediction = runner.predict(lora_model, sample)
    assert prediction["action_context"] == action
    assert prediction["waypoints"] == [[i, 0.] for i in range(1, 7)]
    assert len(calls) == 2
    assert "longitudinal=stop; lateral=right" not in processor.tokenizer.decode(calls[1])
    spans = prediction["assistant_spans"]
    assert len(spans) == 2
    for span in spans:
        assert len(span["token_ids"]) == len(span["token_log_probs"]) == span["end"] - span["start"]
        assert span["log_prob_sum"] == pytest.approx(sum(span["token_log_probs"]), rel=1e-5)
    assert calls[1][spans[0]["start"]:spans[0]["end"]] == spans[0]["token_ids"]


def test_real_random_qwen_generation_is_reported_invalid(lora_model, runner, samples):
    runner.action_generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2}
    result = runner.predict(lora_model, samples["validation"][0])
    assert result["invalid_reason"] is not None
    assert result["waypoints"] is None
    assert len(result["assistant_spans"]) == 1
    assert len(result["assistant_spans"][0]["token_log_probs"]) > 0


def test_sampled_action_ids_survive_noncanonical_tokenization(lora_model, runner, processor,
                                                            samples, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    action = "longitudinal=keep; lateral=straight"
    action_ids = [token for character in action
                  for token in processor.tokenizer(character, add_special_tokens=False)["input_ids"]]
    action_ids += [processor.tokenizer.eos_token_id]
    assert action_ids[:-1] != processor.tokenizer(action, add_special_tokens=False)["input_ids"]
    calls = []
    def generate(self, **kwargs):
        calls.append(kwargs["input_ids"][0].tolist())
        suffix = (action_ids if len(calls) == 1 else
                  processor.tokenizer(serialize_trajectory([[0., 0.]] * 6), add_special_tokens=False)["input_ids"]
                  + [processor.tokenizer.eos_token_id])
        return torch.cat((kwargs["input_ids"], torch.tensor([suffix])), dim=1)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", generate)
    result = runner.predict(lora_model, samples["validation"][0])
    span = result["assistant_spans"][0]
    assert span["token_ids"] == action_ids
    assert calls[1][span["start"]:span["end"]] == action_ids
    assert result["invalid_reason"] is None


def test_valid_text_without_eos_is_truncated(lora_model, runner, processor, samples, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    def generate(self, **kwargs):
        suffix = processor.tokenizer("longitudinal=keep; lateral=straight", add_special_tokens=False)["input_ids"]
        return torch.cat((kwargs["input_ids"], torch.tensor([suffix])), dim=1)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", generate)
    result = runner.predict(lora_model, samples["validation"][0])
    assert result["invalid_reason"] == "unterminated_action"
    assert len(result["assistant_spans"]) == 1


@pytest.mark.parametrize("action,trajectory,reason", [
    ("invalid action", "", "invalid_structured_action"),
    ("longitudinal=keep; lateral=straight", "[1, 2]", "invalid_trajectory_text"),
])
def test_invalid_outputs_remain_in_evaluation_denominator(lora_model, runner, processor, samples,
                                                         records, monkeypatch, action, trajectory, reason):
    from transformers import Qwen3VLForConditionalGeneration
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", scripted_generate(processor, [action, trajectory], []))
    targets = {r["sample_token"]: r for r in records["validation"]}
    rows, metrics = training.evaluate(lora_model, runner, samples["validation"][:1], targets)
    assert rows[0]["invalid_reason"] == reason
    assert metrics["trajectory"]["sample_count"] == metrics["trajectory"]["invalid_prediction_count"] == 1
    assert metrics["trajectory"]["ade_3s_m"] is None


def test_fit_real_cpu_lora_save_reload_with_scripted_rollout(lora_model, runner, processor, samples,
                                                           records, tmp_path, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    config = replace(training.load_config(ROOT / "configs/drivema_stage2_smoke.yaml"),
                     max_steps=2, gradient_accumulation_steps=1, learning_rate=0.01)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", scripted_generate(
        processor, [samples["train"][0].target.training_text(),
                    serialize_trajectory(records["train"][0]["future_waypoints"])], [],
    ))
    targets = {r["sample_token"]: r for rows in records.values() for r in rows}
    output = tmp_path / "cpu-fit"
    result = training.fit(model=lora_model, runner=runner, train=samples["train"][:1],
                          validation=samples["validation"][:1], records=targets, config=config,
                          output=output, metadata={"run_kind": "CPU_random_Qwen_scripted_rollout"})
    assert result["lora_delta_l2"] > 0 and result["nonzero_gradient_steps"] == 2
    assert result["reload_consistency"]
    reload = json.loads((output / "reload_consistency.json").read_text())
    assert reload["token_ids_match"] and reload["log_probs_match"]
    assert (output / "adapter/adapter_model.safetensors").is_file()
    assert len((output / "validation_predictions.jsonl").read_text().splitlines()) == 1
    assert result["status"] == "smoke_failed"  # Two CPU updates do not establish tiny overfit.


def test_split_and_invalid_teacher_forcing_rejected(runner, samples, records, processor):
    with pytest.raises(ValueError, match="only permit train"):
        runner.training_batch(samples["validation"][0], records["validation"][0])
    with pytest.raises(ValueError, match="train/validation"):
        runner.predict(None, replace(samples["train"][0], split="test"))
    with pytest.raises(ValueError, match="valid joint action"):
        encode_teacher_forcing(processor, samples["train"][0].observation, [], "invalid", [[0, 0]] * 6)


def test_cli_config_and_rerun_guard(tmp_path, capsys):
    assert main(["--dry-run"]) == 0
    assert "dry_run_no_data_or_model_access" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--split", "test"])
    config = training.load_config(ROOT / "configs/drivema_stage2_smoke.yaml")
    (tmp_path / config.output_relative_dir).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        training.run(repository=ROOT, dataset_root=tmp_path, derived_root=tmp_path, config=config)

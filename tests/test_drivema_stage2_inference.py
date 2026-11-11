from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_stage2_sft import evaluate
from test_drivema_stage2_sft import (
    case, records, samples, processor, runner, tiny_qwen, lora_model, scripted_generate,
)


def comparable(row):
    return {key: value for key, value in row.items() if key not in ("assistant_spans", "log_prob_definition")} | {
        "assistant_spans": [{k: v for k, v in span.items() if k not in ("token_log_probs", "log_prob_sum")}
                            for span in row["assistant_spans"]]}


@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_batch_exact_tokens_mixed_history_and_invalid(processor, runner, samples, records,
                                                     lora_model, monkeypatch, size, count):
    from transformers import Qwen3VLForConditionalGeneration
    selected = samples["validation"][:count]
    targets = {r["sample_token"]: r for r in records["validation"]}
    actions = ["longitudinal=keep; lateral=straight", "invalid", "longitudinal=decelerate; lateral=right",
               "longitudinal=decelerate; lateral=left"]
    action_ids = [processor.tokenizer(text, add_special_tokens=False)["input_ids"]
                  + [processor.tokenizer.eos_token_id] for text in actions]
    # Noncanonical tokenization must remain byte-for-byte in the second turn.
    action_ids[0] = [token for char in actions[0]
                     for token in processor.tokenizer(char, add_special_tokens=False)["input_ids"]]
    action_ids[0].append(processor.tokenizer.eos_token_id)
    action_ids[2] = [token for char in actions[2]
                     for token in processor.tokenizer(char, add_special_tokens=False)["input_ids"]]
    runner.action_generation = {**runner.action_generation, "max_new_tokens": len(action_ids[2])}
    trajectories = [serialize_trajectory([[float(i), float(i)] for i in range(1, 7)]), "", "", "[1, 2]"]
    lookup = {}
    for index, sample in enumerate(selected):
        from src.phase0.phase0_4b_protocol import inference_messages
        inputs = processor.apply_chat_template(inference_messages(sample.observation, runner.images(sample)),
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        lookup[tuple(inputs["input_ids"][0].tolist())] = index
    calls = []
    def generate(self, **kwargs):
        batch_ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        calls.append(batch_ids.shape[0])
        outputs = []
        image_count = 0
        for row, attention in zip(batch_ids, mask, strict=True):
            context = row[attention.bool()].tolist()
            assert attention[-1] == 1
            prefix = next(key for key in lookup if context[:len(key)] == list(key))
            index = lookup[prefix]
            image_count += len(runner.images(selected[index]))
            if kwargs["max_new_tokens"] == runner.action_generation["max_new_tokens"]:
                ids = action_ids[index]
            else:
                assert context[len(prefix):len(prefix) + len(action_ids[index])] == action_ids[index]
                assert index not in (1, 2)
                ids = processor.tokenizer(trajectories[index], add_special_tokens=False)["input_ids"]
                ids += [processor.tokenizer.eos_token_id]
            outputs.append(ids)
        assert kwargs["image_grid_thw"].shape[0] == image_count
        width = max(map(len, outputs))
        suffix = batch_ids.new_full((len(outputs), width), processor.tokenizer.pad_token_id)
        for i, ids in enumerate(outputs):
            suffix[i, :len(ids)] = suffix.new_tensor(ids)
        return torch.cat((batch_ids, suffix), dim=1)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", generate)
    baseline, metrics = evaluate(lora_model, runner, selected, targets)
    calls.clear()
    runner.optimized, runner.compute_log_probs, runner.inference_batch_size = True, False, size
    performance = []
    batched, actual = evaluate(lora_model, runner, selected, targets, performance=performance)
    assert [comparable(row) for row in batched] == [comparable(row) for row in baseline]
    assert actual == metrics
    assert batched[2]["invalid_reason"] == "unterminated_action"
    assert max(calls) == min(size, count)
    assert all(span["token_log_probs"] is None for row in batched for span in row["assistant_spans"])
    assert sum(len(item["sample_tokens"]) for item in performance) == len(selected)
    assert all(item["total_seconds"] >= item["turn1_seconds"] for item in performance)


def test_skip_probability_forward_retains_stage2a_default(runner, samples, lora_model, processor, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    import src.phase0.drivema_stage2_protocol as protocol
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", scripted_generate(
        processor, ["longitudinal=keep; lateral=straight", serialize_trajectory([[0., 0.]] * 6)], []))
    reference = runner.predict(lora_model, samples["validation"][0])
    assert reference["assistant_spans"][0]["token_log_probs"] is not None
    def forbidden(*args):
        pytest.fail("unnecessary probability forward")
    monkeypatch.setattr(protocol, "assistant_log_probs", forbidden)
    runner.compute_log_probs = False
    result = runner.predict(lora_model, samples["validation"][0])
    assert comparable(result) == comparable(reference)
    assert result["log_prob_definition"] == "not_computed"


def test_real_qwen_batch_cache_matches_single_and_prefills_once(runner, samples, records, tiny_qwen):
    torch.manual_seed(11)
    selected = [samples["validation"][0], samples["validation"][3]]
    targets = {r["sample_token"]: r for r in records["validation"]}
    runner.action_generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": 3}
    runner.compute_log_probs = False
    reference, metrics = evaluate(tiny_qwen, runner, selected, targets)
    visual_calls = []
    hook = tiny_qwen.model.visual.register_forward_hook(lambda *args: visual_calls.append(1))
    runner.optimized, runner.inference_batch_size = True, 2
    actual, actual_metrics = evaluate(tiny_qwen, runner, selected, targets)
    hook.remove()
    assert [comparable(row) for row in actual] == [comparable(row) for row in reference]
    assert actual_metrics == metrics
    assert len(visual_calls) == 1


def test_real_two_turn_cache_and_multimodal_positions(processor, runner, samples, records,
                                                       tiny_qwen, monkeypatch):
    from transformers import LogitsProcessorList
    tiny_qwen.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    original = tiny_qwen.generate
    action = "longitudinal=keep; lateral=straight"
    trajectory = serialize_trajectory([[0., 0.]] * 6)
    logits = []
    def generate(**kwargs):
        text = action if kwargs["max_new_tokens"] == 24 else trajectory
        ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"] + [processor.tokenizer.eos_token_id]
        width = kwargs["input_ids"].shape[1]
        def force(input_ids, scores):
            step = input_ids.shape[1] - width
            if step == 0:
                logits.append(scores.detach().clone())
            forced = torch.full_like(scores, -torch.inf)
            forced[:, ids[step]] = 0
            return forced
        return original(**kwargs, logits_processor=LogitsProcessorList([force]))
    monkeypatch.setattr(tiny_qwen, "generate", generate)
    selected = [samples["validation"][0], samples["validation"][3]]
    targets = {r["sample_token"]: r for r in records["validation"]}
    runner.compute_log_probs = True
    baseline, metrics = evaluate(tiny_qwen, runner, selected, targets)
    runner.optimized, runner.inference_batch_size = True, 2
    visual_calls = []
    hook = tiny_qwen.model.visual.register_forward_hook(lambda *args: visual_calls.append(1))
    actual, actual_metrics = evaluate(tiny_qwen, runner, selected, targets)
    hook.remove()
    assert actual_metrics == metrics
    assert [comparable(row) for row in actual] == [comparable(row) for row in baseline]
    assert len(visual_calls) == 4  # two prefills plus two requested probability forwards
    for left, right in zip(baseline, actual, strict=True):
        for a, b in zip(left["assistant_spans"], right["assistant_spans"], strict=True):
            torch.testing.assert_close(torch.tensor(a["token_log_probs"]),
                                       torch.tensor(b["token_log_probs"]))
    for turn in (0, 1):
        expected = torch.cat([logits[turn], logits[turn + 2]])
        torch.testing.assert_close(logits[4 + turn], expected, atol=1e-5, rtol=1e-4)


def test_benchmark_summary_and_mismatch_gate(processor, runner, samples, records,
                                             lora_model, monkeypatch):
    from copy import deepcopy
    from transformers import Qwen3VLForConditionalGeneration
    from scripts.benchmark_drivema_stage2c import compare_predictions, summarize, markdown_summary
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "generate", scripted_generate(
        processor, ["longitudinal=keep; lateral=straight", serialize_trajectory([[0., 0.]] * 6)], []))
    targets = {r["sample_token"]: r for r in records["validation"]}
    performance = []
    rows, metrics = evaluate(lora_model, runner, samples["validation"][:1], targets, performance=performance)
    report = summarize(performance, 2., metrics, None)
    assert report["samples_per_second"] == .5
    assert report["generated_tokens"]["1"] == len(rows[0]["assistant_spans"][0]["token_ids"])
    consistency = compare_predictions(rows, rows, metrics, metrics)
    assert consistency["passed"]
    summary = markdown_summary({"baseline": report, "optimized": report, "consistency": consistency,
                                "throughput_speedup": 1., "latency_reduction_fraction": 0.})
    assert "Valid trajectory rate" in summary and "P95 latency" in summary
    for field in ("sample_token", "action_context", "trajectory_text", "input_ids", "invalid_reason"):
        changed = deepcopy(rows)
        changed[0][field] = "changed"
        result = compare_predictions(rows, changed, metrics, metrics)
        assert not result["passed"] and field in result["mismatches"][0]["fields"]

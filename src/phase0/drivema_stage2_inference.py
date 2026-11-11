from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.phase0.drivema_stage2_protocol import (
    AssistantSpan, assistant_log_probs, measure, parse_trajectory, second_turn_messages,
    trajectory_to_meta_action,
)
from src.phase0.phase0_4b_protocol import Observation, inference_messages, parse_action
from src.phase0.qwen3vl_interface import validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch


def _batch_inputs(items: list[dict], pad: int) -> dict:
    lengths = [item["input_ids"].shape[1] for item in items]
    width = max(lengths)
    ids = items[0]["input_ids"].new_full((len(items), width), pad)
    mask = torch.zeros_like(ids)
    for row, (item, length) in enumerate(zip(items, lengths, strict=True)):
        ids[row, -length:] = item["input_ids"][0]
        mask[row, -length:] = 1
    # Visual patches and grids are flattened in sample/image order, not padded.
    return {"input_ids": ids, "attention_mask": mask,
            "pixel_values": torch.cat([item["pixel_values"] for item in items]),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in items])}


def _generated_ids(sequence: torch.Tensor, width: int, eos: set[int]) -> list[int]:
    ids = sequence[width:].tolist()
    for index, token in enumerate(ids):
        if token in eos:
            return ids[:index + 1]
    return ids


def generate_two_turn_batch(*, model: nn.Module, processor: object,
                            observations: Sequence[Observation], images: Sequence[Sequence[object]],
                            device: str, action_generation: dict, trajectory_max_new_tokens: int,
                            compute_log_probs: bool = False, timings: dict | None = None) -> list[dict]:
    model.eval()
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token_id
    generation_eos = model.generation_config.eos_token_id
    stops = set(generation_eos if isinstance(generation_eos, list) else [generation_eos])
    stops.add(eos)
    pad = tokenizer.pad_token_id
    first = []
    messages = []
    for observation, sample_images in zip(observations, images, strict=True):
        message = inference_messages(observation, sample_images)
        inputs = processor.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        validate_processor_inputs(inputs, expected_batch_size=1, expected_image_count=len(sample_images))
        messages.append(message)
        first.append(_move_batch(inputs, device))
    batched = _batch_inputs(first, pad)
    with torch.no_grad():
        with measure(timings, "turn1_seconds", device):
            generated = model.generate(**batched, **{**action_generation, "use_cache": True})
        actions = [_generated_ids(row, batched["input_ids"].shape[1], stops) for row in generated]
        texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                 for ids in actions]
        parsed = [parse_action(text) for text in texts]
        complete = [bool(ids) and ids[-1] == eos for ids in actions]
        contexts, spans, active = [], [], []
        for index, (item, ids) in enumerate(zip(first, actions, strict=True)):
            prefix = item["input_ids"][0].tolist()
            spans.append([AssistantSpan(1, len(prefix), len(prefix) + len(ids))])
            context = prefix + ids
            if parsed[index] is not None and complete[index]:
                # Render only text to obtain the template suffix; never re-encode action IDs.
                before = processor.apply_chat_template(messages[index], tokenize=False, add_generation_prompt=True)
                after = processor.apply_chat_template(
                    second_turn_messages(messages[index], texts[index]), tokenize=False, add_generation_prompt=True,
                )
                boundary = len(before + texts[index] + tokenizer.eos_token)
                suffix = tokenizer(after[boundary:], add_special_tokens=False)["input_ids"]
                context += suffix
                active.append(index)
            contexts.append(context)
        trajectory_ids = [[] for _ in first]
        if active:
            second = [{**first[i], "input_ids": first[i]["input_ids"].new_tensor([contexts[i]])} for i in active]
            batched = _batch_inputs(second, pad)
            with measure(timings, "turn2_seconds", device):
                generated = model.generate(**batched, do_sample=False, num_beams=1,
                                           max_new_tokens=trajectory_max_new_tokens, use_cache=True)
            for index, row in zip(active, generated, strict=True):
                ids = _generated_ids(row, batched["input_ids"].shape[1], stops)
                trajectory_ids[index] = ids
                start = len(contexts[index])
                contexts[index] += ids
                spans[index].append(AssistantSpan(2, start, start + len(ids)))
        results = []
        for index, context in enumerate(contexts):
            ids = trajectory_ids[index]
            text = (tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if index in active else None)
            trajectory_complete = bool(ids) and ids[-1] == eos
            points = parse_trajectory(text) if trajectory_complete else None
            reason = ("invalid_structured_action" if parsed[index] is None else
                      "unterminated_action" if not complete[index] else
                      "unterminated_trajectory" if not trajectory_complete else
                      "invalid_trajectory_text" if points is None else None)
            values = [None] * len(spans[index])
            if compute_log_probs:
                inputs = {**first[index], "input_ids": first[index]["input_ids"].new_tensor([context])}
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                with measure(timings, "log_prob_seconds", device):
                    values = assistant_log_probs(model, inputs, spans[index])
            results.append({
                "conditioning_type": "predicted_action", "action_context": texts[index],
                "raw_output": texts[index], "parsed_action": parsed[index], "trajectory_text": text,
                "waypoints": points, "invalid_reason": reason, "input_ids": context,
                "assistant_spans": [
                    {"turn": span.turn, "start": span.start, "end": span.end,
                     "token_ids": context[span.start:span.end],
                     "token_log_probs": value.cpu().tolist() if value is not None else None,
                     "log_prob_sum": float(value.sum()) if value is not None else None}
                    for span, value in zip(spans[index], values, strict=True)
                ],
                "log_prob_definition": ("raw_policy_teacher_forced; EOS_included; not_sampling_distribution"
                                        if compute_log_probs else "not_computed"),
                "trajectory_projection": trajectory_to_meta_action(points) if points is not None else None,
            })
    return results

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import re

import torch
from torch import nn

from src.phase0.phase0_4_factorized_targets import derive_feature_targets
from src.phase0.phase0_4b_protocol import Observation, inference_messages, parse_action
from src.phase0.qwen3vl_interface import validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch

PROTOCOL_VERSION = "drivema-stage2-two-turn-v0.1"
WAYPOINT_COUNT = 6
INTERVAL_SEC = 0.5
COORDINATE_DECIMALS = 2
TRAJECTORY_PROMPT = (
    "Given the driving decision above and the same observations, predict six future "
    "ego waypoints at 0.5, 1.0, 1.5, 2.0, 2.5 and 3.0 seconds. "
    "Use the current CAM_FRONT ego frame: x forward, y left, units meters. "
    "Output exactly [x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], "
    "[x_5, y_5], [x_6, y_6], with two decimal places per coordinate and no other text."
)
_NUMBER = r"-?(?:0|[1-9][0-9]*)\.[0-9]{2}"
_POINT = rf"\[({_NUMBER}), ({_NUMBER})\]"
_TRAJECTORY = re.compile(", ".join([_POINT] * WAYPOINT_COUNT))


def serialize_trajectory(waypoints: Sequence[Sequence[float]]) -> str:
    if len(waypoints) != WAYPOINT_COUNT or any(len(point) != 2 for point in waypoints):
        raise ValueError("trajectory requires six x/y pairs")
    if not all(math.isfinite(value) for point in waypoints for value in point):
        raise ValueError("trajectory coordinates must be finite")
    return ", ".join(
        "[" + ", ".join(f"{round(value, COORDINATE_DECIMALS) + 0.0:.2f}" for value in point) + "]"
        for point in waypoints
    )


def parse_trajectory(text: str) -> list[list[float]] | None:
    match = _TRAJECTORY.fullmatch(text)
    if match is None:
        return None
    values = [float(value) for value in match.groups()]
    if not all(math.isfinite(value) for value in values):
        return None
    return [values[index:index + 2] for index in range(0, len(values), 2)]


def trajectory_to_meta_action(waypoints: Sequence[Sequence[float]]) -> dict:
    """Project generated nominal-grid points; reuse frozen label thresholds."""
    serialize_trajectory(waypoints)
    points = [(0.0, 0.0), *waypoints]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    return derive_feature_targets({
        "path_length_m": sum(lengths),
        "end_speed_proxy_mps": lengths[-1] / INTERVAL_SEC,
        "delta_speed_proxy_mps": (lengths[-1] - lengths[0]) / INTERVAL_SEC,
        "final_lateral_displacement_m": waypoints[-1][1],
    })


def second_turn_messages(messages: list[dict], action_text: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        {"role": "user", "content": [{"type": "text", "text": TRAJECTORY_PROMPT}]},
    ]


@dataclass(frozen=True)
class AssistantSpan:
    turn: int
    start: int
    end: int


def encode_teacher_forcing(processor: object, observation: Observation,
                           images: Sequence[object], action_text: str,
                           waypoints: Sequence[Sequence[float]]) -> tuple[dict, list[AssistantSpan]]:
    if parse_action(action_text) is None:
        raise ValueError("teacher forcing requires a valid joint action")
    first = inference_messages(observation, images)
    second = second_turn_messages(first, action_text)
    text = serialize_trajectory(waypoints)
    full = second + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt",
    )
    validate_processor_inputs(inputs, expected_batch_size=1, expected_image_count=len(images))
    spans = []
    ids = inputs["input_ids"][0].tolist()
    for turn, messages, response in ((1, first, action_text), (2, second, text)):
        prefix = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )["input_ids"][0].tolist()
        target = processor.tokenizer(response, add_special_tokens=False)["input_ids"]
        target += [processor.tokenizer.eos_token_id]
        if ids[:len(prefix) + len(target)] != prefix + target:
            raise ValueError("real processor assistant content/EOS boundary mismatch")
        spans.append(AssistantSpan(turn, len(prefix), len(prefix) + len(target)))
    labels = inputs["input_ids"].new_full(inputs["input_ids"].shape, -100)
    for span in spans:
        labels[:, span.start:span.end] = inputs["input_ids"][:, span.start:span.end]
    return {**inputs, "labels": labels}, spans


def assistant_log_probs(model: nn.Module, inputs: dict,
                        spans: Sequence[AssistantSpan]) -> list[torch.Tensor]:
    """Unwarped policy log p(token | prefix), including EOS; retains autograd."""
    model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
    positions = torch.cat([torch.arange(span.start, span.end, device=inputs["input_ids"].device)
                           for span in spans])
    output = model(**model_inputs, use_cache=False, return_dict=True, logits_to_keep=positions - 1)
    logits = output.logits[0].float()
    targets = inputs["input_ids"][0, positions]
    log_probs = logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
    return list(log_probs.split([span.end - span.start for span in spans]))


def generate_two_turn(*, model: nn.Module, processor: object, observation: Observation,
                      images: Sequence[object], device: str, action_generation: dict,
                      trajectory_max_new_tokens: int) -> dict:
    model.eval()
    messages = inference_messages(observation, images)
    first = _move_batch(processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ), device)
    validate_processor_inputs(first, expected_batch_size=1, expected_image_count=len(images))
    eos = processor.tokenizer.eos_token_id
    with torch.no_grad():
        action_sequence = model.generate(**first, **action_generation)
        action_start = first["input_ids"].shape[1]
        action_ids = action_sequence[0, action_start:].tolist()
        action_text = processor.tokenizer.decode(
            action_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        parsed_action = parse_action(action_text)
        spans = [AssistantSpan(1, action_start, action_sequence.shape[1])]
        inputs = {**first, "input_ids": action_sequence, "attention_mask": torch.ones_like(action_sequence)}
        trajectory_ids, trajectory_text = [], None
        action_complete = bool(action_ids) and action_ids[-1] == eos
        if parsed_action is not None and action_complete:
            second = processor.apply_chat_template(
                second_turn_messages(messages, action_text), tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            )["input_ids"].to(device)
            canonical_action = processor.tokenizer(action_text, add_special_tokens=False)["input_ids"]
            # Preserve the sampled action IDs, even when decode/re-encode changes tokenization.
            suffix = second[:, action_start + len(canonical_action) + 1:]
            context = torch.cat((action_sequence, suffix), dim=1)
            inputs = {**first, "input_ids": context, "attention_mask": torch.ones_like(context)}
            sequence = model.generate(**inputs, do_sample=False, num_beams=1,
                                      max_new_tokens=trajectory_max_new_tokens)
            start = context.shape[1]
            trajectory_ids = sequence[0, start:].tolist()
            trajectory_text = processor.tokenizer.decode(
                trajectory_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            spans.append(AssistantSpan(2, start, sequence.shape[1]))
            inputs = {**first, "input_ids": sequence, "attention_mask": torch.ones_like(sequence)}
        log_probs = assistant_log_probs(model, inputs, spans)
    trajectory_complete = bool(trajectory_ids) and trajectory_ids[-1] == eos
    waypoints = parse_trajectory(trajectory_text) if trajectory_complete else None
    reason = ("invalid_structured_action" if parsed_action is None else
              "unterminated_action" if not action_complete else
              "unterminated_trajectory" if not trajectory_complete else
              "invalid_trajectory_text" if waypoints is None else None)
    return {
        "conditioning_type": "predicted_action", "action_context": action_text,
        "raw_output": action_text, "parsed_action": parsed_action,
        "trajectory_text": trajectory_text, "waypoints": waypoints, "invalid_reason": reason,
        "input_ids": inputs["input_ids"][0].tolist(),
        "assistant_spans": [
            {"turn": span.turn, "start": span.start, "end": span.end,
             "token_ids": inputs["input_ids"][0, span.start:span.end].tolist(),
             "token_log_probs": probs.cpu().tolist(), "log_prob_sum": float(probs.sum())}
            for span, probs in zip(spans, log_probs, strict=True)
        ],
        "log_prob_definition": "raw_policy_teacher_forced; EOS_included; not_sampling_distribution",
        "trajectory_projection": trajectory_to_meta_action(waypoints) if waypoints is not None else None,
    }

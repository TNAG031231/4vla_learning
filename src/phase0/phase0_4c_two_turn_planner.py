from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.phase0.phase0_4b_lora_smoke import generate_action
from src.phase0.phase0_4b_protocol import Observation, inference_messages, parse_action
from src.phase0.qwen3vl_lora_smoke import _move_batch

PLANNING_PROMPT = (
    "Given the structured driving decision above and the same observations, "
    "prepare a representation for planning the ego trajectory over the next "
    "3.0 seconds at 0.5-second intervals."
)


@dataclass(frozen=True)
class PlannerConfig:
    seed: int
    train_subset_size: int
    selected_adapter_relative_path: str
    output_relative_dir: str
    planner_dimension: int
    num_decoder_layers: int
    num_heads: int
    dropout: float
    num_waypoint_queries: int
    smooth_l1_beta: float


def load_config(path: Path) -> PlannerConfig:
    config = PlannerConfig(**yaml.safe_load(path.read_text()))
    if config.train_subset_size not in (1, 2) or config.num_waypoint_queries != 6:
        raise ValueError("smoke requires 1–2 train samples and six waypoint queries")
    if (config.planner_dimension <= 0 or config.num_heads <= 0
            or config.planner_dimension % config.num_heads or config.num_decoder_layers <= 0
            or not 0 <= config.dropout < 1 or not config.smooth_l1_beta > 0):
        raise ValueError("invalid planner dimensions, dropout or SmoothL1 beta")
    return config


def planning_messages(observation: Observation, images: Sequence[object],
                      action_text: str) -> list[dict]:
    if parse_action(action_text) is None:
        raise ValueError("planning context requires a valid frozen structured action")
    return inference_messages(observation, images) + [
        {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        {"role": "user", "content": [{"type": "text", "text": PLANNING_PROMPT}]},
    ]


def planning_inputs(processor: object, messages: list[dict], device: str) -> tuple[dict, dict]:
    batch = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    # Measure the assistant boundary with this processor's actual chat template.
    prefix = processor.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )["input_ids"][0].tolist()
    action_text = messages[1]["content"][0]["text"]
    action_ids = processor.tokenizer(action_text, add_special_tokens=False)["input_ids"]
    ids = batch["input_ids"][0].tolist()
    if ids[:len(prefix) + len(action_ids)] != prefix + action_ids:
        raise ValueError("Turn-1 action tokens differ from the actual Turn-2 context")
    evidence = {
        "assistant_action_text": action_text,
        "action_context_matches": True,
        "action_token_start": len(prefix),
        "action_token_ids": ids[len(prefix):len(prefix) + len(action_ids)],
        "action_tokens": processor.tokenizer.convert_ids_to_tokens(action_ids),
        "action_decoded": processor.tokenizer.decode(action_ids),
        "rendered_context": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ),
    }
    return _move_batch(batch, device), evidence


def freeze_backbone(model: nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


def contextual_hidden_states(model: nn.Module, inputs: dict) -> torch.Tensor:
    # no_grad keeps frozen memory usable by the trainable projection's backward.
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True, return_dict=True,
                       use_cache=False, logits_to_keep=1)
    return output.hidden_states[-1]


class WaypointDecoder(nn.Module):
    def __init__(self, hidden_size: int, config: PlannerConfig) -> None:
        super().__init__()
        dimension = config.planner_dimension
        self.context_projection = nn.Linear(hidden_size, dimension)
        self.waypoint_queries = nn.Embedding(config.num_waypoint_queries, dimension)
        layer = nn.TransformerDecoderLayer(
            dimension, config.num_heads, dim_feedforward=4 * dimension,
            dropout=config.dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, config.num_decoder_layers)
        self.waypoint_projection = nn.Linear(dimension, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        memory = self.context_projection(hidden_states.to(self.context_projection.weight.dtype))
        queries = self.waypoint_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=~attention_mask.bool())
        return self.waypoint_projection(decoded)


def masked_waypoint_loss(prediction: torch.Tensor, target: torch.Tensor,
                         valid_mask: torch.Tensor, *, beta: float) -> torch.Tensor:
    if not valid_mask.any():
        raise ValueError("waypoint loss requires at least one valid timestep")
    return F.smooth_l1_loss(prediction[valid_mask], target[valid_mask], beta=beta)


def predicted_action_inference(*, model: nn.Module, planner: WaypointDecoder,
                               observation: Observation, images: Sequence[object],
                               processor: object, generation_kwargs: dict, device: str
                               ) -> tuple[torch.Tensor, dict]:
    model.eval()
    planner.eval()
    with torch.no_grad():
        prediction = generate_action(
            model, inference_messages(observation, images), processor, generation_kwargs, device,
        )
        messages = planning_messages(observation, images, prediction["raw_output"])
        inputs, evidence = planning_inputs(processor, messages, device)
        hidden = contextual_hidden_states(model, inputs)
        waypoints = planner(hidden, inputs["attention_mask"])
    return waypoints, {
        "conditioning_type": "predicted_action", **prediction, **evidence,
        "hidden_state_field": "hidden_states[-1]", "hidden_state_shape": list(hidden.shape),
        "hidden_state_dtype": str(hidden.dtype), "waypoint_shape": list(waypoints.shape),
        "waypoints_finite": bool(torch.isfinite(waypoints).all()),
        "predicted_waypoints": waypoints.cpu().tolist(),
    }

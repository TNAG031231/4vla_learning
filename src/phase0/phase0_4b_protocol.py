from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.phase0.phase0_4_factorized_targets import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from src.phase0.phase0_4_temporal_dataset import validate_record
from src.phase0.qwen3vl_dataset_adapter import AdapterConfig, serialize_ego_state
from src.phase0.qwen3vl_interface import validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import IGNORE_INDEX, _active_ids
from src.phase0.qwen3vl_smoke import resolve_image_path

PROMPT_VERSION = "phase0.4b-factorized-action-prompt-v0.1"
SERIALIZATION_VERSION = "phase0.4b-factorized-action-serialization-v0.1"
PARSER_VERSION = "phase0.4b-factorized-action-parser-v0.1"
TASK_PROMPT = (
    "Based only on the provided past/current CAM_FRONT observations and ego states, "
    "predict exactly one near-future structured factorized driving action. "
    "Observations are ordered oldest to current; unavailable history slots have no image. "
    "Longitudinal must be one of: stop, decelerate, keep, accelerate. "
    "Lateral must be one of: left, straight, right. "
    "Left/right describe lateral motion, without specifying a turn or lane change. "
    "Output exactly longitudinal=<value>; lateral=<value> and nothing else."
)


def serialize_action(longitudinal: str, lateral: str) -> str:
    if longitudinal not in LONGITUDINAL_ACTIONS or lateral not in LATERAL_ACTIONS:
        raise ValueError("action outside frozen factorized value sets")
    return f"longitudinal={longitudinal}; lateral={lateral}"


def parse_action(text: str) -> dict[str, str] | None:
    for longitudinal in LONGITUDINAL_ACTIONS:
        for lateral in LATERAL_ACTIONS:
            if text == serialize_action(longitudinal, lateral):
                return {"longitudinal": longitudinal, "lateral": lateral}
    return None


@dataclass(frozen=True)
class ActionTarget:
    longitudinal: str | None
    lateral: str | None
    longitudinal_valid: bool
    lateral_valid: bool

    def __post_init__(self) -> None:
        for value, valid, allowed in (
            (self.longitudinal, self.longitudinal_valid, LONGITUDINAL_ACTIONS),
            (self.lateral, self.lateral_valid, LATERAL_ACTIONS),
        ):
            if type(valid) is not bool or (valid and value not in allowed):
                raise ValueError("valid direction requires a frozen action value")
            if not valid and value is not None:
                raise ValueError("invalid frozen direction must have a null target")

    def training_text(self) -> str:
        # Empty invalid values are masked training scaffolding, never parser outputs.
        return f"longitudinal={self.longitudinal or ''}; lateral={self.lateral or ''}"


@dataclass(frozen=True)
class Observation:
    image_paths: tuple[str, ...]
    frame_texts: tuple[str, ...]
    history_valid_mask: tuple[bool, ...]


@dataclass(frozen=True)
class SFTSample:
    sample_token: str
    scene_token: str
    split: str
    observation: Observation
    target: ActionTarget


def adapt_record(record: dict, ego_config: AdapterConfig, *, expected_split: str = "train") -> SFTSample:
    if expected_split not in ("train", "validation") or record["split"] != expected_split:
        raise ValueError("default intake only permits train; explicit validation intake requires matching split")
    validate_record(record)
    paths, texts = [], []
    for path, time, motion, valid in zip(
        record["historical_cam_front_paths"], record["historical_relative_times_sec"],
        record["ego_motion_history"], record["history_valid_mask"], strict=True,
    ):
        if valid:
            paths.append(path)
            texts.append(f"Observation at t={time:.6f} s relative to current:\n"
                         + serialize_ego_state(motion, ego_config))
    target = ActionTarget(
        record["longitudinal_action"], record["lateral_action"],
        record["longitudinal_action_valid"], record["lateral_action_valid"],
    )
    if record["factorized_action_joint_valid"] != (
        target.longitudinal_valid and target.lateral_valid
    ):
        raise ValueError("joint validity differs from independent masks")
    return SFTSample(
        record["sample_token"], record["scene_token"], record["split"],
        Observation(tuple(paths), tuple(texts), tuple(record["history_valid_mask"])), target,
    )


def inference_messages(observation: Observation, images: Sequence[object]) -> list[dict]:
    content = [{"type": "text", "text": TASK_PROMPT + "\nHistory availability: "
                + ", ".join("available" if valid else "unavailable"
                            for valid in observation.history_valid_mask)}]
    for image, text in zip(images, observation.frame_texts, strict=True):
        content.extend(({"type": "text", "text": text}, {"type": "image", "image": image}))
    return [{"role": "user", "content": content}]


def target_token_mask(tokenizer: object, target: ActionTarget) -> tuple[list[int], list[bool]]:
    text = target.training_text()
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    delimiter = text.index(";")
    spans = ((0, delimiter, target.longitudinal_valid),
             (delimiter, delimiter + 1, target.longitudinal_valid and target.lateral_valid),
             (delimiter + 1, len(text), target.lateral_valid))
    mask = []
    for start, end in encoded["offset_mapping"]:
        owners = [valid for left, right, valid in spans if left <= start < end <= right]
        if len(owners) != 1:
            raise ValueError("token crosses independent supervision boundary")
        mask.append(owners[0])
    return encoded["input_ids"], mask


def verify_processor_protocol(processor: object) -> list[dict]:
    """Inspect real processor tokens with synthetic images, without dataset/model access."""
    from PIL import Image

    evidence = []
    for longitudinal in (*LONGITUDINAL_ACTIONS, None):
        for lateral in (*LATERAL_ACTIONS, None):
            target = ActionTarget(longitudinal, lateral, longitudinal is not None, lateral is not None)
            target_ids, mask = target_token_mask(processor.tokenizer, target)
            for count in (1, 3):
                images = [Image.new("RGB", (64 + 32 * i, 64)) for i in range(count)]
                observation = Observation(tuple(f"image-{i}" for i in range(count)),
                                          tuple(f"Synthetic observation {i}" for i in range(count)),
                                          (False,) * (3 - count) + (True,) * count)
                messages = inference_messages(observation, images)
                prefix = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )["input_ids"][0].tolist()
                full_messages = messages + [{"role": "assistant", "content": [
                    {"type": "text", "text": target.training_text()},
                ]}]
                full = processor.apply_chat_template(
                    full_messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt",
                )
                validate_processor_inputs(full, expected_batch_size=1, expected_image_count=count)
                ids = full["input_ids"][0].tolist()
                if ids[:len(prefix) + len(target_ids)] != prefix + target_ids:
                    raise ValueError("real processor assistant boundary mismatch")
                if set(target_ids) & set(processor.tokenizer.all_special_ids):
                    raise ValueError("structured target contains special tokens")
                evidence.append({
                    "target": target.training_text(), "image_count": count,
                    "assistant_start": len(prefix), "target_positions": list(range(len(prefix), len(prefix) + len(target_ids))),
                    "token_ids": target_ids, "tokens": processor.tokenizer.convert_ids_to_tokens(target_ids),
                    "supervised_positions": [len(prefix) + i for i, valid in enumerate(mask) if valid],
                    "assistant_prefix_tail": processor.tokenizer.decode(prefix[-3:]),
                    "suffix": processor.tokenizer.decode(ids[len(prefix) + len(target_ids):]),
                    "chat_template": processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False),
                })
    return evidence


class StructuredCollator:
    def __init__(self, processor: object, image_loader: Callable[[Path], object],
                 dataset_root: Path) -> None:
        self.processor = processor
        self.image_loader = image_loader
        self.dataset_root = dataset_root

    def messages(self, sample: SFTSample, *, expected_split: str = "train") -> list[dict]:
        if expected_split not in ("train", "validation") or sample.split != expected_split:
            raise ValueError("inference sample must match the permitted train/validation split")
        images = [self.image_loader(resolve_image_path(self.dataset_root, path))
                  for path in sample.observation.image_paths]
        return inference_messages(sample.observation, images)

    def __call__(self, samples: Sequence[SFTSample], *, expected_split: str) -> dict:
        if expected_split != "train" or any(sample.split != "train" for sample in samples):
            raise ValueError("Phase 0.4b-A only permits train")
        conversations = [self.messages(sample) + [{"role": "assistant", "content": [
            {"type": "text", "text": sample.target.training_text()},
        ]}] for sample in samples]
        batch = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False,
            padding=True, return_dict=True, return_tensors="pt",
        )
        validate_processor_inputs(batch, expected_batch_size=len(samples),
                                  expected_image_count=sum(len(s.observation.image_paths) for s in samples))
        labels = batch["input_ids"].new_full(batch["input_ids"].shape, IGNORE_INDEX)
        for index, (sample, conversation) in enumerate(zip(samples, conversations, strict=True)):
            prefix = self.processor.apply_chat_template(
                conversation[:-1], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            prefix_ids = _active_ids(prefix["input_ids"][0], prefix["attention_mask"][0])
            positions = batch["attention_mask"][index].nonzero().flatten().tolist()
            full_ids = batch["input_ids"][index, positions].tolist()
            target_ids, mask = target_token_mask(self.processor.tokenizer, sample.target)
            if full_ids[:len(prefix_ids) + len(target_ids)] != prefix_ids + target_ids:
                raise ValueError("real chat template assistant target boundary mismatch")
            for offset, supervised in enumerate(mask):
                if supervised:
                    position = positions[len(prefix_ids) + offset]
                    labels[index, position] = batch["input_ids"][index, position]
        return {**batch, "labels": labels}

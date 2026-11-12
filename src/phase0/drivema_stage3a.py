from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from transformers import GenerationConfig
import yaml

from src.phase0.drivema_stage2_protocol import (
    AssistantSpan, parse_trajectory, second_turn_messages,
)
from src.phase0.drivema_stage3_rewards import RewardConfig, assign_advantages, candidate_rewards
from src.phase0.phase0_4b_lora_smoke import load_temporal_records
from src.phase0.phase0_4b_protocol import Observation, SFTSample, adapt_record, inference_messages, parse_action
from src.phase0.qwen3vl_dataset_adapter import load_config as load_ego_config
from src.phase0.qwen3vl_interface import validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch

SCHEMA_VERSION = 'drivema-stage3a-group-v0.1'


@dataclass(frozen=True)
class RolloutConfig:
    seed: int
    group_size: int
    rollout_micro_batch: int
    train_samples: int
    checkpoint_relative_path: str
    output_relative_dir: str
    action_max_new_tokens: int
    trajectory_max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    reward: RewardConfig


def load_config(path: Path) -> RolloutConfig:
    values = yaml.safe_load(path.read_text())
    values['reward'] = RewardConfig(**values['reward'])
    config = RolloutConfig(**values)
    if (config.group_size != 4 or config.rollout_micro_batch != 1 or config.do_sample is not True
            or config.temperature != 1.0 or config.top_p != 1.0 or config.top_k != 0):
        raise ValueError('Stage 3A requires G=4, micro-batch=1 and unwarped categorical sampling')
    if config.checkpoint_relative_path != 'drivema/stage2b_full_run01/checkpoint_3564':
        raise ValueError('Stage 3A must initialize from selected Full SFT checkpoint')
    if config.reward != RewardConfig(0.7, 0.3, 1.0, 0.5, 0.0001):
        raise ValueError('Stage 3A reward definition differs from the approved contract')
    if any(type(value) is not int or value < 1 for value in
           (config.train_samples, config.action_max_new_tokens, config.trajectory_max_new_tokens)):
        raise ValueError('sample count and generation lengths must be positive integers')
    return config


def prepare_train(repository: Path, derived_root: Path) -> tuple[list[SFTSample], dict, dict]:
    rows = load_temporal_records(repository, derived_root, split='train')
    ego = load_ego_config(repository / 'configs/phase0_3_dataset_adapter.yaml')
    samples, excluded_action, excluded_trajectory = [], 0, 0
    for row in rows:
        sample = adapt_record(row, ego, expected_split='train')
        if not (sample.target.longitudinal_valid or sample.target.lateral_valid):
            excluded_action += 1
        elif not any(row['trajectory_valid_mask']):
            excluded_trajectory += 1
        else:
            samples.append(sample)
    return sorted(samples, key=lambda s: s.sample_token), {r['sample_token']: r for r in rows}, {
        'split': 'train', 'total_records': len(rows), 'eligible_records': len(samples),
        'excluded_no_action_supervision': excluded_action, 'excluded_no_valid_waypoints': excluded_trajectory,
        'validation_records_read': 0, 'test_records_read': 0}


def generation_configs(processor: object, config: RolloutConfig) -> list[GenerationConfig]:
    return [GenerationConfig(
        do_sample=config.do_sample, temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
        num_beams=1, max_new_tokens=length, eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id, use_cache=True,
        return_dict_in_generate=True, output_scores=True,
    ) for length in (config.action_max_new_tokens, config.trajectory_max_new_tokens)]


def observation_inputs(processor: object, observation: Observation, images: list, device: str) -> dict:
    inputs = processor.apply_chat_template(
        inference_messages(observation, images), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt')
    validate_processor_inputs(inputs, expected_batch_size=1, expected_image_count=len(images))
    return _move_batch(inputs, device)


@torch.no_grad()
def sample_candidate(model: nn.Module, processor: object, observation: Observation, images: list,
                     first: dict, generation: list[GenerationConfig]) -> dict:
    model.eval()
    inputs = first
    spans, texts, complete = [], [], []
    tokenizer = processor.tokenizer
    action = None
    for turn in range(2):
        prefix = inputs['input_ids'][0].tolist()
        output = model.generate(**inputs, generation_config=generation[turn], use_model_defaults=False)
        sequence = output.sequences
        ids = sequence[0, len(prefix):].tolist()
        log_probs = [float(score[0].float().log_softmax(-1)[token])
                     for score, token in zip(output.scores, ids, strict=True)]
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ended = bool(ids) and ids[-1] == tokenizer.eos_token_id
        spans.append({'turn': turn + 1, 'start': len(prefix), 'end': sequence.shape[1],
                      'prefix_ids': prefix, 'token_ids': ids, 'old_token_log_probs': log_probs,
                      'eos': ended, 'termination': 'eos' if ended else 'length'})
        texts.append(text)
        complete.append(ended)
        inputs = {**first, 'input_ids': sequence, 'attention_mask': torch.ones_like(sequence)}
        if turn == 0:
            action = parse_action(text)
            if action is None or not ended:
                break
            messages = inference_messages(observation, images)
            before = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            after = processor.apply_chat_template(second_turn_messages(messages, text), tokenize=False,
                                                  add_generation_prompt=True)
            suffix = tokenizer(after[len(before + text + tokenizer.eos_token):],
                               add_special_tokens=False)['input_ids']
            context = torch.cat((sequence, sequence.new_tensor([suffix])), dim=1)
            inputs = {**first, 'input_ids': context, 'attention_mask': torch.ones_like(context)}
    present = len(spans) == 2
    complete += [False] * (2 - len(complete))
    points = parse_trajectory(texts[1]) if present and complete[1] else None
    valid = [action is not None and complete[0], points is not None]
    status = [('valid' if valid[0] else 'invalid_action' if complete[0] else 'length'),
              ('missing' if not present else 'length' if not complete[1] else
               'valid' if points is not None else 'invalid_trajectory')]
    return {'input_ids': inputs['input_ids'][0].tolist(), 'assistant_spans': spans,
            'action_text': texts[0], 'parsed_action': action,
            'trajectory_text': texts[1] if present else None, 'waypoints': points,
            'turn_mask': [True, present], 'turn_complete': complete, 'turn_valid': valid,
            'parser_status': status}


def rollout_group(*, model: nn.Module, processor: object, sample: SFTSample, images: list,
                  record: dict, config: RolloutConfig, device: str) -> dict:
    if sample.split != 'train' or record['split'] != 'train' or sample.sample_token != record['sample_token']:
        raise ValueError('group rollout requires aligned train sample/record')
    if not (sample.target.longitudinal_valid or sample.target.lateral_valid):
        raise ValueError('group rollout requires action supervision')
    first = observation_inputs(processor, sample.observation, images, device)
    generation = generation_configs(processor, config)
    candidates = []
    for index in range(config.group_size):
        candidate = sample_candidate(model, processor, sample.observation, images, first, generation)
        candidate.update(sample_token=sample.sample_token, candidate_index=index)
        candidate['rewards'] = candidate_rewards(candidate, sample.target, record['future_waypoints'],
                                                  record['trajectory_valid_mask'], config.reward)
        candidates.append(candidate)
    statistics = assign_advantages(candidates, config.reward.advantage_eps)
    return {'schema_version': SCHEMA_VERSION, 'sample_token': sample.sample_token,
            'scene_token': sample.scene_token, 'split': sample.split, 'observation': asdict(sample.observation),
            'generation_config': [g.to_dict() for g in generation],
            'use_model_defaults': False,
            'probability_definition': 'categorical_raw_policy; temperature=1; no_logits_processors_or_warpers',
            'std_correction': 1, 'candidates': candidates, 'statistics': statistics}


def replay_inputs(processor: object, observation: Observation, images: list, candidate: dict,
                  device: str) -> tuple[dict, list[AssistantSpan]]:
    inputs = observation_inputs(processor, observation, images, device)
    ids = candidate['input_ids']
    first = candidate['assistant_spans'][0]
    if inputs['input_ids'][0].tolist() != ids[:first['start']]:
        raise ValueError('rollout observation prefix differs from replay processor inputs')
    spans = []
    for saved in candidate['assistant_spans']:
        if (saved['prefix_ids'] != ids[:saved['start']]
                or saved['token_ids'] != ids[saved['start']:saved['end']]):
            raise ValueError('rollout prefix/token span mismatch')
        spans.append(AssistantSpan(saved['turn'], saved['start'], saved['end']))
    inputs['input_ids'] = inputs['input_ids'].new_tensor([ids])
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    return inputs, spans


def group_markdown(group: dict) -> str:
    lines = ['# Stage 3A Group Rollout', '', f"Sample: `{group['sample_token']}`",
             f"Observation: `{json.dumps(group['observation'], ensure_ascii=False)}`", '',
             '| Candidate | Action | Trajectory | Action / Trajectory / Consistency | A1 / A2 | Turn mask |',
             '| --- | --- | --- | --- | --- | --- |']
    for c in group['candidates']:
        r = c['rewards']
        action = c['action_text'].replace('|', '\\|').replace('\n', ' ')
        trajectory = str(c['trajectory_text']).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {c['candidate_index']} | {action} | {trajectory} | "
                     f"{r['action']:.6f} / {r['trajectory']:.6f} / {r['consistency']:.6f} | "
                     f"{c['turn_advantages'][0]:.6f} / {c['turn_advantages'][1]:.6f} | {c['turn_mask']} |")
    return '\n'.join(lines) + '\n'


def policy_targets(candidate: dict, device: str) -> dict[str, torch.Tensor]:
    positions, old, advantages, turns = [], [], [], []
    for span in candidate['assistant_spans']:
        count = span['end'] - span['start']
        positions.extend(range(span['start'], span['end']))
        old.extend(span['old_token_log_probs'])
        advantages.extend([candidate['turn_advantages'][span['turn'] - 1]] * count)
        turns.extend([span['turn']] * count)
    return {'target_positions': torch.tensor(positions, device=device),
            'logit_positions': torch.tensor(positions, device=device) - 1,
            'old_log_probs': torch.tensor(old, device=device),
            'advantages': torch.tensor(advantages, device=device),
            'turn_ids': torch.tensor(turns, device=device),
            'loss_mask': torch.ones(len(positions), dtype=torch.bool, device=device)}

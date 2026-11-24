from __future__ import annotations

from collections import Counter, defaultdict
import random
from statistics import mean

import torch
from transformers import GenerationConfig

from src.phase0.drivema_stage2_inference import _batch_inputs
from src.phase0.drivema_stage2_protocol import parse_trajectory, second_turn_messages
from src.phase0.drivema_stage3_rewards import RewardConfig, candidate_rewards
from src.phase0.phase0_4b_protocol import SFTSample, serialize_action
from src.phase0.qwen3vl_interface import validate_processor_inputs
from src.phase0.qwen3vl_lora_smoke import _move_batch

DIRECTIONS = ('accelerate', 'decelerate')
DRAWS = 4


def select_pairs(records: dict[str, dict], schedule: list[dict], per_stratum: int,
                 seed: int) -> dict:
    if per_stratum < 1:
        raise ValueError('per-stratum must be positive')
    pools = defaultdict(list)
    visited = {s['sample_token']: s['scene_token'] for s in schedule}
    for token, scene in sorted(visited.items()):
        row = records[token]
        if row['split'] != 'train' or row['scene_token'] != scene:
            raise ValueError(f'train schedule/manifest mismatch for {token}')
        if (row['factorized_action_joint_valid'] and row['lateral_action'] == 'straight'
                and row['trajectory_valid_mask'][-1]):
            pools[row['longitudinal_action']].append(token)
    rng = random.Random(seed)
    for label in ('keep', *DIRECTIONS):
        rng.shuffle(pools[label])
    eligible = {label: len(tokens) for label, tokens in pools.items()}
    if min(eligible['keep'] // 2, *(eligible[d] for d in DIRECTIONS)) < per_stratum:
        raise ValueError(f'too few eligible straight observations per stratum: {eligible}')
    pairs = []
    for index, direction in enumerate(DIRECTIONS):
        keep = pools['keep'][index * per_stratum:(index + 1) * per_stratum]
        for label, tokens in ((direction, pools[direction][:per_stratum]), ('keep', keep)):
            for token in tokens:
                pairs.append({'sample_token': token, 'scene_token': visited[token],
                              'target_longitudinal': label, 'alternative': direction,
                              'lateral': 'straight'})
    return {'split': 'train', 'visited_observations': len(visited),
            'eligible_by_longitudinal': eligible,
            'selection_seed': seed, 'per_stratum': per_stratum, 'pairs': pairs,
            'observations': len(pairs), 'scenes': len({p['scene_token'] for p in pairs}),
            'second_turn_generations': len(pairs) * 2 * (2 * DRAWS + 1)}


@torch.inference_mode()
def evaluate_pair(model: torch.nn.Module, processor: object, messages: list[dict], sample: SFTSample,
                  record: dict, alternative: str, seed: int, device: str,
                  sampling: GenerationConfig, greedy_max_tokens: int, reward: RewardConfig) -> dict:
    tokenizer = processor.tokenizer
    branches = {}
    for longitudinal in ('keep', alternative):
        conversation = second_turn_messages(messages, serialize_action(longitudinal, 'straight'))
        inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt')
        validate_processor_inputs(inputs, expected_batch_size=1,
                                  expected_image_count=len(sample.observation.image_paths))
        inputs = _batch_inputs([_move_batch(inputs, device)], tokenizer.pad_token_id)
        outputs = []
        for draw in range(2 * DRAWS + 1):
            if draw == 2 * DRAWS:
                draw_seed = None
                sequence = model.generate(**inputs, do_sample=False, num_beams=1,
                                          max_new_tokens=greedy_max_tokens, use_cache=True)
            else:
                draw_seed = seed + draw
                torch.manual_seed(draw_seed)
                sequence = model.generate(**inputs, generation_config=sampling, use_model_defaults=False)
            ids = sequence[0, inputs['input_ids'].shape[1]:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ended = bool(ids) and ids[-1] == tokenizer.eos_token_id
            points = parse_trajectory(text) if ended else None
            candidate = {'parsed_action': {'longitudinal': longitudinal, 'lateral': 'straight'},
                         'turn_complete': [True, ended], 'waypoints': points}
            scores = candidate_rewards(candidate, sample.target, record['future_waypoints'],
                                       record['trajectory_valid_mask'], reward)
            metrics = scores['trajectory_metrics']
            outputs.append({'seed': draw_seed, 'trajectory_text': text,
                            'invalid_reason': None if points is not None else
                            ('invalid_trajectory_text' if ended else 'unterminated_trajectory'),
                            'g': scores['trajectory'], 'ade_3s_m': metrics.get('ade_3s_m'),
                            'fde_3s_m': metrics.get('fde_3s_m')})
        branches[longitudinal] = {'A': outputs[:DRAWS], 'B': outputs[DRAWS:2 * DRAWS],
                                  'greedy': outputs[-1]}
    return branches


def pair_differences(branches: dict, alternative: str) -> dict:
    keep, other = branches['keep'], branches[alternative]
    diff = {batch: mean(row['g'] for row in other[batch]) - mean(row['g'] for row in keep[batch])
            for batch in ('A', 'B')}
    left, right = keep['greedy']['ade_3s_m'], other['greedy']['ade_3s_m']
    diff['greedy_ade_gain_m'] = left - right if left is not None and right is not None else None
    return diff


def _agreement(pairs: list[tuple[float, float | None]]) -> dict:
    counts = Counter()
    for x, y in pairs:
        if y is None:
            counts['unavailable'] += 1
        elif x == 0 or y == 0:
            counts['tie'] += 1
        else:
            counts['agree' if x * y > 0 else 'disagree'] += 1
    decisive = counts['agree'] + counts['disagree']
    return {'comparisons': len(pairs), **{k: counts[k] for k in ('agree', 'disagree', 'tie', 'unavailable')},
            'decisive': decisive, 'agreement_among_decisive': counts['agree'] / decisive if decisive else None}


def summarize_pairs(rows: list[dict], near_tie_g: float) -> dict:
    gaps = [r['differences'] for r in rows]
    batch_differences = [g[b] for g in gaps for b in ('A', 'B')]
    greedy = [g['greedy_ade_gain_m'] for g in gaps if g['greedy_ade_gain_m'] is not None]
    sampled = [d for r in rows for branch in r['branches'].values() for b in ('A', 'B') for d in branch[b]]
    return {
        'observations': len(rows), 'scenes': len({r['scene_token'] for r in rows}),
        'sampled_trajectories': len(sampled),
        'sampled_invalid': sum(d['invalid_reason'] is not None for d in sampled),
        'cross_batch_g': _agreement([(g['A'], g['B']) for g in gaps]),
        'batch_g_vs_greedy_ADE': {b: _agreement([(g[b], g['greedy_ade_gain_m']) for g in gaps])
                                  for b in ('A', 'B')},
        'pooled_g_vs_greedy_ADE': _agreement([(mean((g['A'], g['B'])), g['greedy_ade_gain_m']) for g in gaps]),
        'near_tie_g_margin': near_tie_g, 'batches': len(batch_differences),
        'near_tie_batch_fraction': mean(abs(x) <= near_tie_g for x in batch_differences),
        'greedy_valid_pairs': len(greedy), 'greedy_prefers_alternative': sum(x > 0 for x in greedy),
        'greedy_prefers_keep': sum(x < 0 for x in greedy), 'greedy_ties': sum(x == 0 for x in greedy),
        'mean_greedy_ade_gain_from_alternative_m': mean(greedy) if greedy else None,
    }

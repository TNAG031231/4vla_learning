from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.drivema_longitudinal_branches import evaluate_pair, pair_differences, select_pairs, summarize_pairs
from src.phase0.drivema_stage2_inference import generate_two_turn_batch
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_stage3_rewards import RewardConfig
from src.phase0.phase0_4b_protocol import ActionTarget
from test_drivema_stage2_sft import case, processor, records, runner, samples, tiny_qwen


def test_selection_precedes_generation_and_keeps_disjoint_controls():
    rows = [{'sample_token': str(i), 'scene_token': str(i // 2), 'split': 'train',
             'longitudinal_action': label, 'lateral_action': 'straight', 'factorized_action_joint_valid': True,
             'trajectory_valid_mask': [True] * 6}
            for i, label in enumerate(('keep', 'accelerate', 'keep', 'decelerate', 'stop', 'keep', 'accelerate'))]
    rows[5]['lateral_action'] = 'left'
    rows[6]['trajectory_valid_mask'][-1] = False
    by_token = {r['sample_token']: r for r in rows}
    plan = select_pairs(by_token, rows * 2, 1, 7)
    assert plan == select_pairs(by_token, list(reversed(rows * 2)), 1, 7)
    assert len({r['sample_token'] for r in plan['pairs']}) == plan['observations'] == 4
    assert plan['second_turn_generations'] == 72
    assert {(r['alternative'], r['target_longitudinal']) for r in plan['pairs']} == {
        ('accelerate', 'accelerate'), ('decelerate', 'decelerate'), ('accelerate', 'keep'), ('decelerate', 'keep')}
    assert all(r['lateral'] == 'straight' for r in plan['pairs'])
    by_token['0']['split'] = 'validation'
    with pytest.raises(ValueError, match='train schedule'):
        select_pairs(by_token, rows, 1, 7)


def test_branch_context_sampling_greedy_and_scores(samples, runner, processor, records, tiny_qwen, monkeypatch):
    sample = replace(samples['train'][0], target=ActionTarget('stop', 'right', True, True))
    record = {**records['train'][0], 'future_waypoints': [[0., 0.]] * 6, 'trajectory_valid_mask': [True] * 6}
    messages = runner.collator.messages(sample)
    sampling = GenerationConfig(do_sample=True, temperature=1., top_k=0, top_p=1., num_beams=1, max_new_tokens=192)
    tok = processor.tokenizer
    calls = []

    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        text = tok.decode(kwargs['input_ids'][0].tolist(), skip_special_tokens=False)
        assert sample.target.training_text() not in text
        index = len(calls)
        longitude = ('keep', 'accelerate')[index // 9]
        assert f'longitudinal={longitude}; lateral=straight' in text
        if index % 9 < 8:
            assert kwargs['use_model_defaults'] is False and kwargs['generation_config'] is sampling
            assert torch.initial_seed() == 42 + index % 9
        else:
            assert not kwargs['do_sample'] and kwargs['num_beams'] == 1 and kwargs['max_new_tokens'] == 192
        calls.append(kwargs)
        trajectory = '[0,2]' if index in (0, 17) else serialize_trajectory([[1., 0.]] * 6)
        ids = tok(trajectory, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
        return torch.cat((kwargs['input_ids'], kwargs['input_ids'].new_tensor([ids])), dim=1)

    monkeypatch.setattr(tiny_qwen, 'generate', generate)
    branches = evaluate_pair(tiny_qwen, processor, messages, sample, record, 'accelerate', 42,
                             'cpu', sampling, 192, RewardConfig(.7, .3, 1., .5, .0001))
    assert len(calls) == 18
    assert torch.equal(calls[0]['pixel_values'], calls[9]['pixel_values'])
    assert branches['keep']['A'][0]['g'] == 0
    assert branches['accelerate']['greedy']['ade_3s_m'] is None
    gaps = pair_differences(branches, 'accelerate')
    assert gaps['A'] > 0 and gaps['B'] == 0 and gaps['greedy_ade_gain_m'] is None
    report = summarize_pairs([{'scene_token': 's', 'differences': gaps, 'branches': branches}], .02)
    assert report['sampled_invalid'] == 1 and report['sampled_trajectories'] == 16
    assert report['cross_batch_g']['tie'] == report['pooled_g_vs_greedy_ADE']['unavailable'] == 1
    assert report['cross_batch_g']['agreement_among_decisive'] is None
    assert report['near_tie_batch_fraction'] == .5

    def formal_generate(**kwargs):
        if kwargs['max_new_tokens'] == 24:
            text = 'longitudinal=accelerate; lateral=straight'
        else:
            for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'):
                assert torch.equal(calls[-1][key], kwargs[key])
            text = serialize_trajectory([[1., 0.]] * 6)
        tokens = tok(text, add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
        return torch.cat((kwargs['input_ids'], kwargs['input_ids'].new_tensor([tokens])), dim=1)

    monkeypatch.setattr(tiny_qwen, 'generate', formal_generate)
    generate_two_turn_batch(model=tiny_qwen, processor=processor, observations=[sample.observation],
                            images=[runner.images(sample)], action_generation=runner.action_generation,
                            trajectory_max_new_tokens=192, device='cpu')


def test_rank_agreement_uses_lower_ade_and_higher_geometry_reward():
    def branch(a, b, ade):
        return {'A': [{'g': a, 'invalid_reason': None}] * 4, 'B': [{'g': b, 'invalid_reason': None}] * 4,
                'greedy': {'ade_3s_m': ade}}

    branches = {'keep': branch(.4, .4, .8), 'decelerate': branch(.8, .3, .5)}
    diffs = pair_differences(branches, 'decelerate')
    assert diffs == pytest.approx({'A': .4, 'B': -.1, 'greedy_ade_gain_m': .3})
    report = summarize_pairs([{'scene_token': 's', 'branches': branches, 'differences': diffs}], .02)
    assert report['cross_batch_g']['disagree'] == 1
    assert report['pooled_g_vs_greedy_ADE']['agree'] == report['greedy_prefers_alternative'] == 1

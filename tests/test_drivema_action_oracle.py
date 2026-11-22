from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.drivema_action_oracle import intervention_action_ids, replay_turn2
from src.phase0.drivema_action_oracle_metrics import summarize
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_stage2_sft import evaluate
from src.phase0.phase0_4b_protocol import ActionTarget, inference_messages
from test_drivema_stage2_sft import case, processor, records, runner, samples, tiny_qwen


def test_oracle_replays_optimized_prefix_and_changes_only_action(
    processor, runner, samples, records, tiny_qwen, monkeypatch,
):
    sample = replace(samples['validation'][0], target=ActionTarget('accelerate', 'left', True, True))
    predicted_text = 'longitudinal=stop; lateral=straight'
    tokenizer = processor.tokenizer
    predicted_ids = [token for char in predicted_text
                     for token in tokenizer(char, add_special_tokens=False)['input_ids']]
    predicted_ids.append(tokenizer.eos_token_id)
    points = [[float(i), 0.] for i in range(1, 7)]
    trajectory_text = serialize_trajectory(points)
    trajectory_ids = tokenizer(trajectory_text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    calls = []

    def generate(**kwargs):
        assert not torch.is_grad_enabled()
        calls.append(kwargs)
        ids = predicted_ids if kwargs['max_new_tokens'] == 24 else trajectory_ids
        return torch.cat((kwargs['input_ids'], kwargs['input_ids'].new_tensor([ids])), dim=1)

    monkeypatch.setattr(tiny_qwen, 'generate', generate)
    runner.optimized, runner.compute_log_probs = True, False
    targets = {r['sample_token']: r for r in records['validation']}
    rows, _ = evaluate(tiny_qwen, runner, [sample], targets)
    saved = rows[0]
    action, trajectory = saved['assistant_spans']
    assert action['token_ids'] == predicted_ids
    first = processor.apply_chat_template(
        inference_messages(sample.observation, runner.images(sample)), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt',
    )
    control = replay_turn2(tiny_qwen, processor, first, saved, predicted_ids, 192)
    assert control['trajectory_ids'] == trajectory['token_ids']
    assert control['waypoints'] == points
    for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'):
        assert torch.equal(calls[2][key], calls[1][key])
    for key in ('num_beams', 'max_new_tokens', 'do_sample', 'use_cache'):
        assert calls[2][key] == calls[1][key]

    gold_ids = intervention_action_ids(processor, sample)
    assert len(gold_ids) != len(predicted_ids)
    assert tokenizer.decode(gold_ids[:-1]) == 'longitudinal=accelerate; lateral=left'
    assert gold_ids[-1] == tokenizer.eos_token_id
    replay_turn2(tiny_qwen, processor, first, saved, gold_ids, 192)
    expected = saved['input_ids'][:action['start']] + gold_ids + saved['input_ids'][action['end']:trajectory['start']]
    assert calls[3]['input_ids'][0].tolist() == expected
    assert torch.equal(calls[3]['pixel_values'], calls[1]['pixel_values'])
    assert torch.equal(calls[3]['image_grid_thw'], calls[1]['image_grid_thw'])

    trajectory_ids[:] = tokenizer('[1, 2]', add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    malformed = replay_turn2(tiny_qwen, processor, first, saved, gold_ids, 192)
    assert malformed['waypoints'] is None and malformed['invalid_reason'] == 'invalid_trajectory_text'
    trajectory_ids.pop()
    truncated = replay_turn2(tiny_qwen, processor, first, saved, gold_ids, 192)
    assert truncated['waypoints'] is None and truncated['invalid_reason'] == 'unterminated_trajectory'


def test_oracle_report_from_persisted_pairs(processor, runner, samples, records, tiny_qwen, monkeypatch, tmp_path):
    from scripts.run_drivema_action_oracle import evaluate as evaluate_oracle, wrong_actions

    targets = {r['sample_token']: {**r, 'future_waypoints': [[0., 0.]] * 6} for r in records['validation']}
    selected_samples = [replace(sample, target=ActionTarget(action, 'straight', True, True))
                        for sample, action in zip(samples['validation'][:3], ('accelerate', 'keep', 'stop'), strict=True)]
    predicted = 'longitudinal=stop; lateral=straight'
    bad_trajectory = False

    def generate(**kwargs):
        if kwargs['max_new_tokens'] == 24:
            text = predicted
        else:
            context = processor.tokenizer.decode(kwargs['input_ids'][0].tolist(), skip_special_tokens=False)
            last_assistant = context.rsplit('assistant\n', 2)[-2]
            text = serialize_trajectory([[2. if predicted in last_assistant else 1., 0.]] * 6)
            if bad_trajectory and 'longitudinal=keep;' in last_assistant:
                text = '[1, 2]'
        tokens = processor.tokenizer(text, add_special_tokens=False)['input_ids'] + [processor.tokenizer.eos_token_id]
        return torch.cat((kwargs['input_ids'], kwargs['input_ids'].new_tensor([tokens])), dim=1)

    monkeypatch.setattr(tiny_qwen, 'generate', generate)
    runner.optimized, runner.compute_log_probs = True, False
    saved, _ = evaluate(tiny_qwen, runner, selected_samples, targets)
    wrong = wrong_actions(selected_samples, saved)
    assert [sample.sample_token for sample, _ in wrong] == [s.sample_token for s in selected_samples[:2]]
    for label, valid in [('valid', 2), ('invalid', 1)]:
        bad_trajectory = label == 'invalid'
        output = tmp_path / label
        output.mkdir()
        evaluate_oracle(wrong, runner.collator, tiny_qwen, 'cpu', 192, output)
        rows = [json.loads(line) for line in (output / 'pairs.jsonl').read_text().splitlines()]
        result = summarize(rows, len(selected_samples))
        assert [row['control_replayed'] for row in rows] == [True, False]
        assert all(row['conditioning_type'] == 'gt_action_diagnostic' for row in rows)
        assert result['population_count'] == 3 and result['wrong_action_count'] == 2
        assert result['paired_valid_count'] == valid and result['oracle_invalid_count'] == 2 - valid
        for key in ('ade_3s_m', 'fde_3s_m'):
            paired = result['paired'][key]
            assert paired['sample_count'] == valid and paired['excluded_count'] == 2 - valid
            assert paired['mean_delta_m'] == -1. and paired['improved'] == valid
            if valid == 2:
                assert result['full_population_mean_delta_m'][key] == pytest.approx(-2 / 3)
        if valid == 1:
            assert 'full_population_mean_delta_m' not in result

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_drivema_action_crossover import run_replays
from scripts.run_drivema_action_oracle import evaluate as oracle_evaluate, wrong_actions
from src.phase0.drivema_action_crossover import crossover_plan, decomposition, summarize_plan
from src.phase0.drivema_stage2_protocol import serialize_trajectory
from src.phase0.drivema_stage2_sft import evaluate
from src.phase0.phase0_4b_protocol import ActionTarget
from test_drivema_stage2_sft import case, processor, records, runner, samples, tiny_qwen


def test_crossover_uses_exact_saved_actions_and_three_way_common_population(
    processor, runner, samples, records, tiny_qwen, monkeypatch, tmp_path,
):
    selected = [replace(sample, target=ActionTarget(lon, 'straight', True, True))
                for sample, lon in zip(samples['validation'][:3], ('stop', 'keep', 'stop'), strict=True)]
    controls = {r['sample_token']: {**r, 'future_waypoints': [[0., 0.]] * 6} for r in records['validation']}
    actions = iter(['stop', 'stop', 'stop', 'stop', 'keep', 'accelerate'])
    contexts = []
    malformed = False
    alternate_encoding = False

    def generate(**kwargs):
        if kwargs['max_new_tokens'] == 24:
            text = f'longitudinal={next(actions)}; lateral=straight'
        else:
            context = processor.tokenizer.decode(kwargs['input_ids'][0].tolist(), skip_special_tokens=False)
            last_action = context.rsplit('assistant\n', 2)[-2]
            contexts.append(last_action)
            distance = 2. if 'longitudinal=stop;' in last_action else 1.
            text = '[1]' if malformed else serialize_trajectory([[distance, 0.]] * 6)
        if alternate_encoding and kwargs['max_new_tokens'] == 24:
            ids = [i for char in text for i in processor.tokenizer(char, add_special_tokens=False)['input_ids']]
        else:
            ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
        ids.append(processor.tokenizer.eos_token_id)
        return torch.cat((kwargs['input_ids'], kwargs['input_ids'].new_tensor([ids])), dim=1)

    monkeypatch.setattr(tiny_qwen, 'generate', generate)
    runner.optimized, runner.compute_log_probs = True, False
    old, _ = evaluate(tiny_qwen, runner, selected, controls)
    new, _ = evaluate(tiny_qwen, runner, selected, controls)
    cache = oracle_evaluate(wrong_actions(selected, old), runner.collator, tiny_qwen, 'cpu', 192, tmp_path)
    plan = crossover_plan(old, list(reversed(new)), cache)
    assert summarize_plan(plan)['sources'] == {'baseline': 1, 'oracle': 1, 'replay': 1}
    assert summarize_plan(plan)['known_contribution_m']['ade_3s_m']['action'] == pytest.approx(-1 / 3)
    saved = {r['sample_token']: r for r in old}
    feedback = {r['sample_token']: r for r in new}
    before = len(contexts)
    for invalid in (False, True):
        mixed = deepcopy(plan)
        malformed = invalid
        run_replays(mixed, saved, feedback, {s.sample_token: s for s in selected},
                    runner.collator, tiny_qwen, 'cpu', 192)
        result = decomposition(mixed)
        assert result['mixed_invalid_count'] == int(invalid)
        assert result['full_population_decomposition'] is (not invalid)
        for metric in ('ade_3s_m', 'fde_3s_m'):
            effects = result['paired']
            assert effects['action'][metric]['sample_count'] == 3 - int(invalid)
            assert effects['action'][metric]['mean_delta_m'] + effects['decoder'][metric]['mean_delta_m'] == pytest.approx(
                effects['total'][metric]['mean_delta_m'])
    assert len(contexts) == before + 2
    assert all('longitudinal=accelerate;' in c for c in contexts[before:])
    corrupted = deepcopy(new)
    corrupted[0]['input_ids'][0] += 1
    with pytest.raises(ValueError, match='prompt differs'):
        crossover_plan(old, corrupted, cache)
    malformed, alternate_encoding = False, True
    actions = iter(['keep'])
    alternate, _ = evaluate(tiny_qwen, runner, selected[1:2], controls)
    assert alternate[0]['action_context'] == new[1]['action_context']
    assert alternate[0]['assistant_spans'][0]['token_ids'] != new[1]['assistant_spans'][0]['token_ids']
    token = selected[1].sample_token
    changed = crossover_plan(old[1:2], alternate, cache)
    assert changed[0]['source'] == 'oracle'
    run_replays(changed, saved, {token: alternate[0]}, {token: selected[1]}, runner.collator, tiny_qwen, 'cpu', 192)
    assert changed[0]['source'] == 'replay'

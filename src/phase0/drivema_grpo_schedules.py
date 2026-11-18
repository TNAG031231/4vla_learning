from __future__ import annotations

from collections import Counter
from copy import deepcopy
import random

from src.phase0.phase0_4b_protocol import SFTSample


def coverage_schedules(replay: dict, train: list[SFTSample], weight: float,
                       selection: dict, seed: int) -> dict:
    by_token = {s.sample_token: s for s in train}
    original = replay['steps']
    original_tokens = {r['sample_token'] for r in original}
    if (len(original_tokens) != 256 or set(by_token) != set(replay['baseline_identity']['train_tokens'])
            or len(by_token) < 512):
        raise ValueError('coverage requires original 256 unique observations and the same eligible train pool')
    if any(by_token[r['sample_token']].scene_token != r['scene_token'] for r in original):
        raise ValueError('original observation scene differs from train manifest')

    def label(sample: SFTSample) -> str:
        t = sample.target
        return '/'.join(str(getattr(t, axis)) if getattr(t, f'{axis}_valid') else 'unavailable'
                        for axis in ('longitudinal', 'lateral'))

    pool = sorted(set(by_token) - original_tokens)
    random.Random(seed).shuffle(pool)
    extra, tiers = [], Counter()
    for row in original:
        source = by_token[row['sample_token']]
        def priority(token: str) -> tuple:
            candidate = by_token[token]
            same_action = label(candidate) == label(source)
            same_scene = candidate.scene_token == source.scene_token
            return (not (same_action and same_scene), not same_action, not same_scene)
        token = min(pool, key=priority)
        tier = priority(token)
        tiers['scene_and_gt_action' if not tier[0] else 'gt_action' if not tier[1] else
              'scene' if not tier[2] else 'unmatched'] += 1
        pool.remove(token)
        extra.append(token)
    schedules = {}
    for arm, second in (('repeat256', [r['sample_token'] for r in original]), ('unique512', extra)):
        order = [token for row, token2 in zip(original, second, strict=True) for token in (row['sample_token'], token2)]
        frequency = Counter(order)
        seen = Counter()
        steps = []
        for step, token in enumerate(order, 1):
            seen[token] += 1
            steps.append({'optimizer_step': step, 'sample_token': token, 'scene_token': by_token[token].scene_token,
                          'visit': seen[token], 'total_visits': frequency[token]})
        schedule = deepcopy(replay)
        schedule.update(steps=steps, experiment={'kind': 'coverage', 'arm': arm,
            'consistency_weight': weight, 'schedule_seed': seed, 'reward_selection': selection},
            coverage={'steps': 512, 'unique_observations': len(frequency),
                'observation_coverage_fraction': len(frequency) / len(train),
                'multiplicity': dict(Counter(frequency.values())),
                'scene_exposures': dict(Counter(by_token[t].scene_token for t in order)),
                'gt_action_exposures': dict(Counter(label(by_token[t]) for t in order)),
                'matching_tiers_for_new_observations': dict(tiers),
                'gt_source': 'original verified train-only temporal manifest; no generated actions'})
        schedules[arm] = schedule
    return schedules


def reward_selection(results: dict, weight: float, rationale: str) -> dict:
    if (results['round'] != 'reward' or results['status'] != 'full_validation_complete'
            or set(results['weights']) != {'0.0', '0.25', '0.5'} or not rationale.strip()):
        raise ValueError('coverage requires completed three-weight reward results and an explicit selection rationale')
    for result in results['workers'].values():
        if (result['verified_steps'] != 256
                or result['validation']['ablation_vs_sft']['comparison']['population_count'] != 3594):
            raise ValueError('reward results lack complete 256-step/full-validation evidence')
    if weight not in (0., .25, .5):
        raise ValueError('select one evaluated reward weight')
    return {'consistency_weight': weight, 'rationale': rationale,
            'source_round': 'reward', 'result_status': results['status'],
            'inference_scope': 'one matched run per weight; cross-seed stability unverified'}

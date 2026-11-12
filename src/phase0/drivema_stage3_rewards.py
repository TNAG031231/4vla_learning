from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from src.phase0.drivema_stage2_protocol import trajectory_to_meta_action
from src.phase0.phase0_4b_protocol import ActionTarget
from src.phase0.phase0_4c_evaluation import prediction_metrics


@dataclass(frozen=True)
class RewardConfig:
    ade_weight: float
    fde_weight: float
    distance_scale_m: float
    consistency_weight: float
    advantage_eps: float


def candidate_rewards(candidate: dict, target: ActionTarget, waypoints: list,
                      mask: list[bool], config: RewardConfig) -> dict:
    axes = [axis for axis in ('longitudinal', 'lateral') if getattr(target, f'{axis}_valid')]
    if not axes:
        raise ValueError('reward requires at least one supervised action dimension')
    action = candidate['parsed_action'] if candidate['turn_complete'][0] else None
    action_reward = (sum(action[axis] == getattr(target, axis) for axis in axes) / len(axes)
                     if action is not None else 0.0)
    points = candidate['waypoints']
    metrics = prediction_metrics(torch.tensor(points) if points is not None else None,
                                 torch.tensor(waypoints), torch.tensor(mask, dtype=torch.bool))
    trajectory_reward, consistency_reward = 0.0, 0.0
    error, rule = None, 'invalid_prediction'
    projection = None
    if metrics['prediction_valid']:
        ade, fde = metrics['ade_3s_m'], metrics['fde_3s_m']
        if fde is None:
            error, rule = ade, 'masked_ADE_only_missing_3s_endpoint'
        else:
            error = config.ade_weight * ade + config.fde_weight * fde
            rule = 'weighted_ADE3_FDE3'
        trajectory_reward = math.exp(-error / config.distance_scale_m)
        projection = trajectory_to_meta_action(points)
        if action is not None:
            consistency_reward = sum(
                projection[f'{axis}_action'] == action[axis]
                for axis in ('longitudinal', 'lateral')) / 2
    return {'action': action_reward, 'trajectory': trajectory_reward,
            'consistency': consistency_reward, 'error_m': error, 'trajectory_rule': rule,
            'trajectory_metrics': metrics, 'trajectory_projection': projection,
            'turn_rewards': [action_reward, trajectory_reward + config.consistency_weight * consistency_reward]}


def assign_advantages(candidates: list[dict], eps: float) -> dict:
    rewards = torch.tensor([c['rewards']['turn_rewards'] for c in candidates], dtype=torch.float64)
    mean = rewards.mean(dim=0)
    std = rewards.std(dim=0, correction=1)
    zero = rewards.eq(rewards[0]).all(dim=0)
    advantages = (rewards - mean) / (std + eps)
    advantages[:, zero] = 0
    for candidate, value in zip(candidates, advantages.tolist(), strict=True):
        candidate['turn_advantages'] = value
    return {'reward_mean': mean.tolist(), 'reward_std': std.tolist(),
            'zero_variance': zero.tolist(), 'std_correction': 1,
            'invalid_rate': [sum(not c['turn_valid'][turn] for c in candidates) / len(candidates)
                             for turn in range(2)],
            'missing_turn_rate': [sum(not c['turn_mask'][turn] for c in candidates) / len(candidates)
                                  for turn in range(2)]}


def summarize_groups(groups: list[dict]) -> dict:
    return {'group_count': len(groups), 'candidate_count': sum(len(g['candidates']) for g in groups),
            **{key: [sum(g['statistics'][source][turn] for g in groups) / len(groups)
                     for turn in range(2)]
               for key, source in (('reward_mean', 'reward_mean'), ('mean_group_reward_std', 'reward_std'),
                                   ('zero_variance_group_rate', 'zero_variance'),
                                   ('invalid_rate', 'invalid_rate'), ('missing_turn_rate', 'missing_turn_rate'))}}

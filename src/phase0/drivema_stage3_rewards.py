from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

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
    composition: str = 'additive'
    action_geometry_weight: float = 0.0

    def combine(self, trajectory: float, consistency: float) -> float:
        bonus = self.consistency_weight * consistency
        if self.composition == 'additive':
            return trajectory + bonus
        if self.composition == 'multiplicative':
            return trajectory * (1 + bonus)
        raise ValueError(f'unknown reward composition: {self.composition}')


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
            'turn_rewards': [action_reward, config.combine(trajectory_reward, consistency_reward)]}


def _action_geometry_feedback(candidates: list[dict], weight: float) -> list[float]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, candidate in enumerate(candidates):
        if candidate['turn_valid'][0]:
            action = candidate['parsed_action']
            groups.setdefault((action['longitudinal'], action['lateral']), []).append(index)
    feedback = [0.] * len(candidates)
    if len(groups) < 2:
        return feedback
    geometry = [c['rewards']['trajectory'] for c in candidates]
    baseline = statistics.mean(geometry[i] for indices in groups.values() for i in indices)
    for indices in groups.values():
        delta = weight * (statistics.mean(geometry[i] for i in indices) - baseline)
        for i in indices:
            feedback[i] = delta
    return feedback


def assign_advantages(candidates: list[dict], eps: float, *, action_geometry_weight: float = 0.0) -> dict:
    rewards = torch.tensor([c['rewards']['turn_rewards'] for c in candidates], dtype=torch.float64)
    mean = rewards.mean(dim=0)
    std = rewards.std(dim=0, correction=1)
    zero = rewards.eq(rewards[0]).all(dim=0)
    advantages = (rewards - mean) / (std + eps)
    advantages[:, zero] = 0
    stats = {'reward_mean': mean.tolist(), 'reward_std': std.tolist(),
             'zero_variance': zero.tolist(), 'std_correction': 1,
             'invalid_rate': [sum(not c['turn_valid'][turn] for c in candidates) / len(candidates)
                              for turn in range(2)],
             'missing_turn_rate': [sum(not c['turn_mask'][turn] for c in candidates) / len(candidates)
                                   for turn in range(2)]}
    if action_geometry_weight:
        feedback = _action_geometry_feedback(candidates, action_geometry_weight)
        advantages[:, 0] += advantages.new_tensor(feedback)
        stats['action_geometry_feedback'] = feedback
    for candidate, value in zip(candidates, advantages.tolist(), strict=True):
        candidate['turn_advantages'] = value
    return stats


def summarize_groups(groups: list[dict]) -> dict:
    return {'group_count': len(groups), 'candidate_count': sum(len(g['candidates']) for g in groups),
            **{key: [sum(g['statistics'][source][turn] for g in groups) / len(groups)
                     for turn in range(2)]
               for key, source in (('reward_mean', 'reward_mean'), ('mean_group_reward_std', 'reward_std'),
                                   ('zero_variance_group_rate', 'zero_variance'),
                                   ('invalid_rate', 'invalid_rate'), ('missing_turn_rate', 'missing_turn_rate'))}}

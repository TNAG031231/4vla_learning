from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
from typing import Final

from src.actions.schema import ACTION_SCHEMA


SPLIT_STRATEGY_VERSION: Final = "official_train_scene_label_stratified_v1"
HARD_CONSTRAINT_PENALTY: Final = 10.0
MAX_SWAP_REFINEMENTS: Final = 64
OBJECTIVE_TOLERANCE: Final = 1e-12


@dataclass(frozen=True)
class ActionConstraintStatus:
    action: str
    total_sample_count: int
    total_scene_support: int
    train_sample_count: int
    train_scene_support: int
    validation_sample_count: int
    validation_scene_support: int
    constraint_satisfied: bool
    unsatisfied_reason: str | None


@dataclass(frozen=True)
class SplitQuality:
    objective_score: float
    train_distribution_distance: float
    validation_distribution_distance: float
    validation_scene_support_distance: float
    train_sample_distribution: dict[str, int]
    validation_sample_distribution: dict[str, int]
    train_scene_support: dict[str, int]
    validation_scene_support: dict[str, int]
    constraints_satisfied: bool
    constraint_statuses: tuple[ActionConstraintStatus, ...]


@dataclass(frozen=True)
class StratifiedSplitResult:
    assignments: dict[str, str]
    quality: SplitQuality
    split_seed: int
    split_strategy_version: str
    refinement_count: int


def _stable_tie_break(seed: int, *tokens: str) -> bytes:
    payload = ":".join((str(seed), *tokens)).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _normalize_histograms(
    scene_histograms: Mapping[str, Mapping[str, int]],
    action_schema: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    if not action_schema or len(set(action_schema)) != len(action_schema):
        raise ValueError("action schema must contain unique actions")
    action_set = set(action_schema)
    normalized = {}
    for scene_token in sorted(scene_histograms):
        histogram = scene_histograms[scene_token]
        unknown_actions = set(histogram) - action_set
        if unknown_actions:
            raise ValueError(f"scene histogram has unknown actions: {unknown_actions}")
        counts = []
        for action in action_schema:
            count = histogram.get(action, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("scene histogram counts must be non-negative integers")
            counts.append(count)
        normalized[scene_token] = tuple(counts)
    if not normalized:
        raise ValueError("scene histograms must not be empty")
    return normalized


def _aggregate_counts(
    histograms: Mapping[str, tuple[int, ...]],
    scene_tokens: Sequence[str],
    action_count: int,
) -> tuple[list[int], list[int]]:
    sample_counts = [0] * action_count
    scene_support = [0] * action_count
    for scene_token in scene_tokens:
        histogram = histograms[scene_token]
        for index, count in enumerate(histogram):
            sample_counts[index] += count
            scene_support[index] += int(count > 0)
    return sample_counts, scene_support


def distribution_distance(
    sample_counts: Sequence[int],
    reference_counts: Sequence[int],
) -> float:
    sample_total = sum(sample_counts)
    reference_total = sum(reference_counts)
    if reference_total == 0:
        return 0.0 if sample_total == 0 else 1.0
    if sample_total == 0:
        return 1.0
    return 0.5 * sum(
        abs(
            sample_count / sample_total
            - reference_count / reference_total
        )
        for sample_count, reference_count in zip(
            sample_counts,
            reference_counts,
            strict=True,
        )
    )


def _constraint_statuses(
    action_schema: Sequence[str],
    total_counts: Sequence[int],
    total_support: Sequence[int],
    train_counts: Sequence[int],
    train_support: Sequence[int],
    validation_counts: Sequence[int],
    validation_support: Sequence[int],
) -> tuple[ActionConstraintStatus, ...]:
    statuses = []
    for index, action in enumerate(action_schema):
        reasons = []
        if total_counts[index] > 0 and train_counts[index] == 0:
            reasons.append("class_absent_from_train")
        if total_support[index] >= 2:
            if train_support[index] == 0:
                reasons.append("class_scene_support_absent_from_train")
            if validation_support[index] == 0:
                reasons.append("class_absent_from_validation")
        if total_support[index] >= 10 and validation_support[index] < 2:
            reasons.append("validation_scene_support_below_two")
        statuses.append(
            ActionConstraintStatus(
                action=action,
                total_sample_count=total_counts[index],
                total_scene_support=total_support[index],
                train_sample_count=train_counts[index],
                train_scene_support=train_support[index],
                validation_sample_count=validation_counts[index],
                validation_scene_support=validation_support[index],
                constraint_satisfied=not reasons,
                unsatisfied_reason=(
                    None
                    if not reasons
                    else (
                        ",".join(reasons)
                        + f";total_scene_support={total_support[index]}"
                    )
                ),
            )
        )
    return tuple(statuses)


def _quality_from_validation_totals(
    action_schema: Sequence[str],
    total_counts: Sequence[int],
    total_support: Sequence[int],
    validation_counts: Sequence[int],
    validation_support: Sequence[int],
    validation_fraction: float,
) -> SplitQuality:
    train_counts = [
        total - validation
        for total, validation in zip(
            total_counts,
            validation_counts,
            strict=True,
        )
    ]
    train_support = [
        total - validation
        for total, validation in zip(
            total_support,
            validation_support,
            strict=True,
        )
    ]
    train_distance = distribution_distance(train_counts, total_counts)
    validation_distance = distribution_distance(validation_counts, total_counts)
    supported_indices = tuple(
        index for index, support in enumerate(total_support) if support > 0
    )
    support_distance = (
        sum(
            abs(
                validation_support[index] / total_support[index]
                - validation_fraction
            )
            for index in supported_indices
        )
        / len(supported_indices)
        if supported_indices
        else 0.0
    )
    statuses = _constraint_statuses(
        action_schema=action_schema,
        total_counts=total_counts,
        total_support=total_support,
        train_counts=train_counts,
        train_support=train_support,
        validation_counts=validation_counts,
        validation_support=validation_support,
    )
    unsatisfied_count = sum(
        not status.constraint_satisfied for status in statuses
    )
    objective = (
        train_distance
        + validation_distance
        + support_distance
        + HARD_CONSTRAINT_PENALTY * unsatisfied_count
    )
    return SplitQuality(
        objective_score=objective,
        train_distribution_distance=train_distance,
        validation_distribution_distance=validation_distance,
        validation_scene_support_distance=support_distance,
        train_sample_distribution=dict(zip(action_schema, train_counts, strict=True)),
        validation_sample_distribution=dict(
            zip(action_schema, validation_counts, strict=True)
        ),
        train_scene_support=dict(zip(action_schema, train_support, strict=True)),
        validation_scene_support=dict(
            zip(action_schema, validation_support, strict=True)
        ),
        constraints_satisfied=unsatisfied_count == 0,
        constraint_statuses=statuses,
    )


def _objective_score_from_validation_totals(
    total_counts: Sequence[int],
    total_support: Sequence[int],
    validation_counts: Sequence[int],
    validation_support: Sequence[int],
    validation_fraction: float,
) -> float:
    train_counts = [
        total - validation
        for total, validation in zip(
            total_counts,
            validation_counts,
            strict=True,
        )
    ]
    train_support = [
        total - validation
        for total, validation in zip(
            total_support,
            validation_support,
            strict=True,
        )
    ]
    supported_indices = tuple(
        index for index, support in enumerate(total_support) if support > 0
    )
    support_distance = (
        sum(
            abs(
                validation_support[index] / total_support[index]
                - validation_fraction
            )
            for index in supported_indices
        )
        / len(supported_indices)
        if supported_indices
        else 0.0
    )
    unsatisfied_count = 0
    for index, total_count in enumerate(total_counts):
        unsatisfied = total_count > 0 and train_counts[index] == 0
        if total_support[index] >= 2:
            unsatisfied = unsatisfied or (
                train_support[index] == 0
                or validation_support[index] == 0
            )
        if total_support[index] >= 10:
            unsatisfied = unsatisfied or validation_support[index] < 2
        unsatisfied_count += int(unsatisfied)
    return (
        distribution_distance(train_counts, total_counts)
        + distribution_distance(validation_counts, total_counts)
        + support_distance
        + HARD_CONSTRAINT_PENALTY * unsatisfied_count
    )


def assign_fixed_random_scene_splits(
    scene_tokens: Sequence[str],
    seed: int,
    train_scene_count: int,
    validation_scene_count: int,
) -> dict[str, str]:
    ordered_tokens = sorted(scene_tokens)
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError("scene tokens must be unique")
    if train_scene_count + validation_scene_count != len(ordered_tokens):
        raise ValueError("scene split capacity must match input scene count")
    random.Random(seed).shuffle(ordered_tokens)
    return {
        scene_token: (
            "train" if index < train_scene_count else "validation"
        )
        for index, scene_token in enumerate(ordered_tokens)
    }


def evaluate_scene_split(
    scene_histograms: Mapping[str, Mapping[str, int]],
    assignments: Mapping[str, str],
    action_schema: Sequence[str] = ACTION_SCHEMA,
) -> SplitQuality:
    histograms = _normalize_histograms(scene_histograms, action_schema)
    if set(assignments) != set(histograms):
        raise ValueError("assignments must cover every scene histogram")
    if set(assignments.values()) - {"train", "validation"}:
        raise ValueError("assignments contain an unsupported split")
    validation_tokens = tuple(
        token for token in sorted(histograms)
        if assignments[token] == "validation"
    )
    total_counts, total_support = _aggregate_counts(
        histograms,
        tuple(sorted(histograms)),
        len(action_schema),
    )
    validation_counts, validation_support = _aggregate_counts(
        histograms,
        validation_tokens,
        len(action_schema),
    )
    return _quality_from_validation_totals(
        action_schema=action_schema,
        total_counts=total_counts,
        total_support=total_support,
        validation_counts=validation_counts,
        validation_support=validation_support,
        validation_fraction=len(validation_tokens) / len(histograms),
    )


def assign_stratified_scene_splits(
    scene_histograms: Mapping[str, Mapping[str, int]],
    seed: int,
    train_scene_count: int,
    validation_scene_count: int,
    action_schema: Sequence[str] = ACTION_SCHEMA,
    split_strategy_version: str = SPLIT_STRATEGY_VERSION,
) -> StratifiedSplitResult:
    histograms = _normalize_histograms(scene_histograms, action_schema)
    scene_tokens = tuple(sorted(histograms))
    assignments = assign_fixed_random_scene_splits(
        scene_tokens=scene_tokens,
        seed=seed,
        train_scene_count=train_scene_count,
        validation_scene_count=validation_scene_count,
    )
    validation_tokens = {
        token for token, split in assignments.items()
        if split == "validation"
    }
    train_tokens = set(scene_tokens) - validation_tokens
    total_counts, total_support = _aggregate_counts(
        histograms,
        scene_tokens,
        len(action_schema),
    )
    validation_counts, validation_support = _aggregate_counts(
        histograms,
        tuple(sorted(validation_tokens)),
        len(action_schema),
    )
    validation_fraction = validation_scene_count / len(scene_tokens)
    quality = _quality_from_validation_totals(
        action_schema=action_schema,
        total_counts=total_counts,
        total_support=total_support,
        validation_counts=validation_counts,
        validation_support=validation_support,
        validation_fraction=validation_fraction,
    )

    refinement_count = 0
    for _ in range(MAX_SWAP_REFINEMENTS):
        best_swap: tuple[str, str] | None = None
        best_score = quality.objective_score
        best_counts: list[int] | None = None
        best_support: list[int] | None = None
        best_tie_break: bytes | None = None
        for validation_token in sorted(validation_tokens):
            validation_histogram = histograms[validation_token]
            for train_token in sorted(train_tokens):
                train_histogram = histograms[train_token]
                candidate_counts = [
                    count - validation_histogram[index] + train_histogram[index]
                    for index, count in enumerate(validation_counts)
                ]
                candidate_support = [
                    support
                    - int(validation_histogram[index] > 0)
                    + int(train_histogram[index] > 0)
                    for index, support in enumerate(validation_support)
                ]
                candidate_score = _objective_score_from_validation_totals(
                    total_counts=total_counts,
                    total_support=total_support,
                    validation_counts=candidate_counts,
                    validation_support=candidate_support,
                    validation_fraction=validation_fraction,
                )
                improvement = best_score - candidate_score
                if improvement < -OBJECTIVE_TOLERANCE:
                    continue
                tie_break = _stable_tie_break(
                    seed,
                    validation_token,
                    train_token,
                )
                if improvement > OBJECTIVE_TOLERANCE or (
                    abs(improvement) <= OBJECTIVE_TOLERANCE
                    and best_swap is not None
                    and best_tie_break is not None
                    and tie_break < best_tie_break
                ):
                    best_swap = (validation_token, train_token)
                    best_score = candidate_score
                    best_counts = candidate_counts
                    best_support = candidate_support
                    best_tie_break = tie_break
        if (
            best_swap is None
            or best_counts is None
            or best_support is None
            or best_score >= quality.objective_score - OBJECTIVE_TOLERANCE
        ):
            break
        validation_token, train_token = best_swap
        validation_tokens.remove(validation_token)
        validation_tokens.add(train_token)
        train_tokens.remove(train_token)
        train_tokens.add(validation_token)
        validation_counts = best_counts
        validation_support = best_support
        quality = _quality_from_validation_totals(
            action_schema=action_schema,
            total_counts=total_counts,
            total_support=total_support,
            validation_counts=validation_counts,
            validation_support=validation_support,
            validation_fraction=validation_fraction,
        )
        refinement_count += 1

    final_assignments = {
        token: ("validation" if token in validation_tokens else "train")
        for token in scene_tokens
    }
    if (
        sum(split == "train" for split in final_assignments.values())
        != train_scene_count
    ):
        raise ValueError("stratified split changed train scene capacity")
    if (
        sum(split == "validation" for split in final_assignments.values())
        != validation_scene_count
    ):
        raise ValueError("stratified split changed validation scene capacity")
    return StratifiedSplitResult(
        assignments=final_assignments,
        quality=quality,
        split_seed=seed,
        split_strategy_version=split_strategy_version,
        refinement_count=refinement_count,
    )

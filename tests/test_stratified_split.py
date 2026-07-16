from collections import Counter
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.stratified_split import (
    HARD_CONSTRAINT_PENALTY,
    SPLIT_STRATEGY_VERSION,
    assign_fixed_random_scene_splits,
    assign_stratified_scene_splits,
    evaluate_scene_split,
)


def histogram(**counts: int) -> dict[str, int]:
    return {action: counts.get(action, 0) for action in ACTION_SCHEMA}


def representative_histograms(scene_count: int = 40) -> dict[str, dict[str, int]]:
    return {
        f"scene-{index:03d}": histogram(
            keep=20 + index % 5,
            accelerate=2 + index % 3,
            decelerate=3 + (index * 2) % 4,
            stop=1 + index % 4,
            left_lateral=int(index % 5 == 0) * (1 + index % 2),
            right_lateral=int(index % 7 == 0) * (1 + index % 3),
        )
        for index in range(scene_count)
    }


def test_exact_project_capacity_reproducibility_and_input_order_invariance() -> None:
    histograms = representative_histograms()

    first = assign_stratified_scene_splits(histograms, 20260710, 32, 8)
    second = assign_stratified_scene_splits(histograms, 20260710, 32, 8)
    reversed_input = assign_stratified_scene_splits(
        dict(reversed(tuple(histograms.items()))),
        20260710,
        32,
        8,
    )

    assert first.assignments == second.assignments == reversed_input.assignments
    assert Counter(first.assignments.values()) == {"train": 32, "validation": 8}
    assert first.split_seed == 20260710
    assert first.split_strategy_version == SPLIT_STRATEGY_VERSION


def test_exact_560_140_capacity() -> None:
    histograms = {
        f"scene-{index:03d}": histogram(keep=10, stop=1)
        for index in range(700)
    }

    result = assign_stratified_scene_splits(histograms, 20260710, 560, 140)

    assert Counter(result.assignments.values()) == {
        "train": 560,
        "validation": 140,
    }
    assert set(result.assignments) == set(histograms)


def test_quality_uses_secondary_counts_not_only_dominant_class() -> None:
    assignments = {
        "scene-a": "train",
        "scene-b": "train",
        "scene-c": "validation",
        "scene-d": "validation",
    }
    balanced = {
        "scene-a": histogram(keep=10, accelerate=4),
        "scene-b": histogram(keep=10, decelerate=4),
        "scene-c": histogram(keep=10, accelerate=4),
        "scene-d": histogram(keep=10, decelerate=4),
    }
    shifted = {
        **balanced,
        "scene-c": histogram(keep=10, accelerate=8),
        "scene-d": histogram(keep=10),
    }

    balanced_quality = evaluate_scene_split(balanced, assignments)
    shifted_quality = evaluate_scene_split(shifted, assignments)

    assert all(max(row, key=row.get) == "keep" for row in shifted.values())
    assert shifted_quality.objective_score != balanced_quality.objective_score


def test_splitter_is_not_a_dominant_class_grouping() -> None:
    grouped_secondary = {
        f"scene-{index}": histogram(
            keep=10,
            accelerate=4 if index < 4 else 0,
            decelerate=4 if index >= 4 else 0,
        )
        for index in range(8)
    }
    alternating_secondary = {
        f"scene-{index}": histogram(
            keep=10,
            accelerate=4 if index % 2 == 0 else 0,
            decelerate=4 if index % 2 else 0,
        )
        for index in range(8)
    }

    grouped = assign_stratified_scene_splits(
        grouped_secondary,
        seed=7,
        train_scene_count=6,
        validation_scene_count=2,
    )
    alternating = assign_stratified_scene_splits(
        alternating_secondary,
        seed=7,
        train_scene_count=6,
        validation_scene_count=2,
    )

    assert all(
        max(row, key=row.get) == "keep"
        for row in grouped_secondary.values()
    )
    assert grouped.assignments != alternating.assignments


def test_stratified_objective_is_not_worse_than_fixed_random() -> None:
    histograms = representative_histograms()
    random_assignments = assign_fixed_random_scene_splits(
        tuple(histograms),
        20260710,
        32,
        8,
    )
    random_quality = evaluate_scene_split(histograms, random_assignments)

    stratified = assign_stratified_scene_splits(histograms, 20260710, 32, 8)

    assert stratified.quality.objective_score <= random_quality.objective_score
    assert stratified.quality.train_distribution_distance < 0.05
    assert stratified.quality.validation_distribution_distance < 0.05


def test_validation_rare_class_absence_penalty_is_applied() -> None:
    histograms = {
        "scene-a": histogram(keep=10, left_lateral=2),
        "scene-b": histogram(keep=10, left_lateral=2),
        "scene-c": histogram(keep=10),
        "scene-d": histogram(keep=10),
    }
    absent = evaluate_scene_split(
        histograms,
        {
            "scene-a": "train",
            "scene-b": "train",
            "scene-c": "validation",
            "scene-d": "validation",
        },
    )
    covered = evaluate_scene_split(
        histograms,
        {
            "scene-a": "train",
            "scene-b": "validation",
            "scene-c": "train",
            "scene-d": "validation",
        },
    )

    assert absent.objective_score - covered.objective_score > (
        HARD_CONSTRAINT_PENALTY / 2
    )
    assert not absent.constraints_satisfied
    assert covered.constraints_satisfied


def test_rare_scene_coverage_constraint_is_satisfied_when_feasible() -> None:
    histograms = {
        f"scene-{index:02d}": histogram(
            keep=10,
            right_lateral=int(index < 10),
        )
        for index in range(20)
    }

    result = assign_stratified_scene_splits(histograms, 20260710, 16, 4)
    status = next(
        item for item in result.quality.constraint_statuses
        if item.action == "right_lateral"
    )

    assert result.quality.constraints_satisfied
    assert status.validation_scene_support >= 2


def test_unsatisfied_constraint_reports_reason_and_total_scene_support() -> None:
    histograms = {
        "only-scene": histogram(keep=4, left_lateral=1),
    }

    result = assign_stratified_scene_splits(histograms, 20260710, 0, 1)
    status = next(
        item for item in result.quality.constraint_statuses
        if item.action == "left_lateral"
    )

    assert not status.constraint_satisfied
    assert status.total_scene_support == 1
    assert status.unsatisfied_reason is not None
    assert "class_absent_from_train" in status.unsatisfied_reason
    assert "total_scene_support=1" in status.unsatisfied_reason


def test_capacity_must_cover_every_input_scene() -> None:
    with pytest.raises(ValueError, match="capacity"):
        assign_stratified_scene_splits(
            representative_histograms(5),
            20260710,
            3,
            1,
        )

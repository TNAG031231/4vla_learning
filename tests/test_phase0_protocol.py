from collections import Counter
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA
from src.phase0.protocol import (
    ManifestSample,
    assign_scene_splits,
    complete_action_distribution,
    evaluate_classification,
    validate_scene_split_isolation,
)


def test_scene_level_split_is_reproducible_and_has_six_two_two_scenes() -> None:
    scene_tokens = tuple(f"scene-{index}" for index in range(10))

    first = assign_scene_splits(
        scene_tokens=scene_tokens,
        seed=20260710,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    second = assign_scene_splits(
        scene_tokens=scene_tokens,
        seed=20260710,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert first == second
    assert Counter(first.values()) == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_scene_level_split_rejects_same_scene_across_splits() -> None:
    samples = (
        ManifestSample("sample-a", "scene-a", "keep", "train"),
        ManifestSample("sample-b", "scene-a", "keep", "test"),
    )

    with pytest.raises(ValueError, match="scene_token spans splits"):
        validate_scene_split_isolation(samples)


def test_metrics_keep_all_actions_when_split_has_missing_classes() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "stop"),
        predictions=("keep", "keep"),
    )

    assert metrics.sample_count == 2
    assert metrics.class_distribution == {
        "keep": 1,
        "accelerate": 0,
        "decelerate": 0,
        "stop": 1,
        "left_lateral": 0,
        "right_lateral": 0,
    }
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(1 / 9)
    assert tuple(metrics.per_class_f1) == ACTION_SCHEMA
    assert all(len(row) == 6 for row in metrics.confusion_matrix)
    assert len(metrics.confusion_matrix) == 6
    assert metrics.invalid_label_count == 0


def test_complete_action_distribution_retains_zero_count_classes() -> None:
    distribution = complete_action_distribution(("stop", "stop"))

    assert distribution == {
        "keep": 0,
        "accelerate": 0,
        "decelerate": 0,
        "stop": 2,
        "left_lateral": 0,
        "right_lateral": 0,
    }

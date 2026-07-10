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
    read_manifest_samples,
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


def test_invalid_prediction_counts_as_ground_truth_false_negative() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "keep"),
        predictions=("keep", "INVALID"),
    )

    assert metrics.sample_count == 2
    assert metrics.correct_count == 1
    assert metrics.valid_prediction_count == 1
    assert metrics.invalid_label_count == 1
    assert metrics.invalid_output_rate == pytest.approx(0.5)
    assert metrics.action_parsing_success_rate == pytest.approx(0.5)
    assert (
        metrics.invalid_output_rate + metrics.action_parsing_success_rate
    ) == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.per_class_precision["keep"] == pytest.approx(1.0)
    assert metrics.per_class_recall["keep"] == pytest.approx(0.5)
    assert metrics.per_class_f1["keep"] == pytest.approx(2 / 3)
    assert sum(sum(row) for row in metrics.confusion_matrix) == 1


def test_all_invalid_predictions_keep_ground_truth_support() -> None:
    metrics = evaluate_classification(
        ground_truth=("keep", "accelerate"),
        predictions=("INVALID", "INVALID"),
    )

    assert metrics.accuracy == 0.0
    assert metrics.valid_prediction_count == 0
    assert metrics.invalid_label_count == 2
    assert metrics.invalid_output_rate == 1.0
    assert metrics.action_parsing_success_rate == 0.0
    assert all(sum(row) == 0 for row in metrics.confusion_matrix)
    assert metrics.per_class_recall["keep"] == 0.0
    assert metrics.per_class_recall["accelerate"] == 0.0


def test_ground_truth_invalid_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported action"):
        evaluate_classification(
            ground_truth=("INVALID",),
            predictions=("keep",),
        )


def test_protocol_owns_manifest_reader_without_baseline_dependency(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"sample_token": "sample", "scene_token": "scene", "meta_action": "keep", "split": "train", "label_rule_version": "v0"}\n',
        encoding="utf-8",
    )

    samples = read_manifest_samples(manifest_path)

    assert samples == (ManifestSample("sample", "scene", "keep", "train", "v0"),)
    assert "src.baselines" not in (PROJECT_ROOT / "src/phase0/protocol.py").read_text()

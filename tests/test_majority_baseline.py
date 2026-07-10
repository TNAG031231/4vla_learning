from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.majority import (
    BASELINE_NAME,
    fit_majority_action,
    predict_split,
    write_predictions,
)
from src.phase0.protocol import ManifestSample, read_manifest_samples


def build_samples() -> tuple[ManifestSample, ...]:
    return (
        ManifestSample("train-1", "scene-train", "keep", "train"),
        ManifestSample("train-2", "scene-train", "keep", "train"),
        ManifestSample("train-3", "scene-train", "stop", "train"),
        ManifestSample("validation-1", "scene-val", "keep", "validation"),
        ManifestSample("validation-2", "scene-val", "stop", "validation"),
        ManifestSample("test-1", "scene-test", "accelerate", "test"),
    )


def test_majority_action_only_uses_train_split() -> None:
    samples = build_samples()
    changed_non_train = tuple(
        ManifestSample(
            sample.sample_token,
            sample.scene_token,
            "accelerate" if sample.split != "train" else sample.meta_action,
            sample.split,
        )
        for sample in samples
    )

    assert fit_majority_action(samples) == "keep"
    assert fit_majority_action(changed_non_train) == "keep"


def test_majority_predictions_are_traceable_and_have_expected_metrics(
    tmp_path: Path,
) -> None:
    majority_action = fit_majority_action(build_samples())
    predictions, metrics = predict_split(
        samples=build_samples(),
        split="validation",
        majority_action=majority_action,
        label_rule_version="phase-1.6-meta-action-v0.2",
    )

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.macro_f1 == pytest.approx(1 / 9)
    assert [prediction.sample_token for prediction in predictions] == [
        "validation-1",
        "validation-2",
    ]
    assert predictions[0].baseline_name == BASELINE_NAME
    assert predictions[0].predicted_action == "keep"
    assert predictions[0].is_correct is True

    output_path = tmp_path / "predictions.jsonl"
    write_predictions(predictions, output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        '{"sample_token": "validation-1", "scene_token": "scene-val", "split": "validation", "ground_truth_action": "keep", "predicted_action": "keep", "is_correct": true, "label_rule_version": "phase-1.6-meta-action-v0.2", "baseline_name": "majority"}',
        '{"sample_token": "validation-2", "scene_token": "scene-val", "split": "validation", "ground_truth_action": "stop", "predicted_action": "keep", "is_correct": false, "label_rule_version": "phase-1.6-meta-action-v0.2", "baseline_name": "majority"}',
    ]


def test_manifest_reader_rejects_scene_leakage(tmp_path: Path) -> None:
    manifest_path = tmp_path / "leaked_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            (
                '{"sample_token": "train", "scene_token": "scene-a", "meta_action": "keep", "split": "train", "label_rule_version": "v0"}',
                '{"sample_token": "test", "scene_token": "scene-a", "meta_action": "stop", "split": "test", "label_rule_version": "v0"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scene_token spans splits"):
        read_manifest_samples(manifest_path)


def test_majority_ignores_global_pose_fields_in_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            (
                '{"sample_token": "train", "scene_token": "scene-train", "meta_action": "keep", "split": "train", "label_rule_version": "v0", "current_ego_pose": {"translation_m": [1, 2, 3], "rotation_wxyz": [1, 0, 0, 0]}}',
                '{"sample_token": "validation", "scene_token": "scene-validation", "meta_action": "stop", "split": "validation", "label_rule_version": "v0", "current_ego_pose": {"translation_m": [999, 999, 999], "rotation_wxyz": [0, 1, 0, 0]}}',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    samples = read_manifest_samples(manifest_path)

    assert fit_majority_action(samples) == "keep"

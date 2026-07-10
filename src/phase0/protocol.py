from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Final, Sequence

from src.actions.schema import ACTION_SCHEMA, is_valid_action, normalize_action


SPLITS: Final = ("train", "validation", "test")


@dataclass(frozen=True)
class ManifestSample:
    sample_token: str
    scene_token: str
    meta_action: str
    split: str
    label_rule_version: str = ""


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    class_distribution: dict[str, int]
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    invalid_label_count: int


def assign_scene_splits(
    scene_tokens: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")

    unique_scenes = sorted(set(scene_tokens))
    if len(unique_scenes) != len(scene_tokens):
        raise ValueError("scene_tokens must be unique")

    raw_counts = tuple(len(unique_scenes) * ratio for ratio in ratios)
    split_counts = [int(count) for count in raw_counts]
    remaining = len(unique_scenes) - sum(split_counts)
    remainders = sorted(
        range(len(SPLITS)),
        key=lambda index: (raw_counts[index] - split_counts[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        split_counts[index] += 1

    shuffled_scenes = list(unique_scenes)
    random.Random(seed).shuffle(shuffled_scenes)
    assignments = {}
    start = 0
    for split, count in zip(SPLITS, split_counts, strict=True):
        for scene_token in shuffled_scenes[start : start + count]:
            assignments[scene_token] = split
        start += count
    return assignments


def validate_scene_split_isolation(samples: Sequence[ManifestSample]) -> None:
    scene_splits: dict[str, str] = {}
    for sample in samples:
        if sample.split not in SPLITS:
            raise ValueError(f"Unsupported split: {sample.split!r}")
        existing_split = scene_splits.setdefault(
            sample.scene_token,
            sample.split,
        )
        if existing_split != sample.split:
            raise ValueError(
                "scene_token spans splits: "
                f"{sample.scene_token} ({existing_split}, {sample.split})"
            )


def complete_action_distribution(actions: Sequence[str]) -> dict[str, int]:
    counts = Counter(normalize_action(action) for action in actions)
    return {action: counts[action] for action in ACTION_SCHEMA}


def evaluate_classification(
    ground_truth: Sequence[str],
    predictions: Sequence[str],
) -> ClassificationMetrics:
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have equal length")

    confusion = [[0 for _ in ACTION_SCHEMA] for _ in ACTION_SCHEMA]
    normalized_ground_truth = []
    invalid_label_count = 0
    for expected, predicted in zip(ground_truth, predictions, strict=True):
        normalized_expected = normalize_action(expected)
        normalized_ground_truth.append(normalized_expected)
        if not is_valid_action(predicted):
            invalid_label_count += 1
            continue
        expected_index = ACTION_SCHEMA.index(normalized_expected)
        predicted_index = ACTION_SCHEMA.index(predicted)
        confusion[expected_index][predicted_index] += 1

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    for index, action in enumerate(ACTION_SCHEMA):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_denominator = precision + recall
        per_class_precision[action] = precision
        per_class_recall[action] = recall
        per_class_f1[action] = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        )

    total_correct = sum(confusion[index][index] for index in range(len(ACTION_SCHEMA)))
    sample_count = len(ground_truth)
    return ClassificationMetrics(
        sample_count=sample_count,
        class_distribution=complete_action_distribution(normalized_ground_truth),
        accuracy=total_correct / sample_count if sample_count else 0.0,
        macro_f1=sum(per_class_f1.values()) / len(ACTION_SCHEMA),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=tuple(tuple(row) for row in confusion),
        invalid_label_count=invalid_label_count,
    )

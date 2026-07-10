from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from src.actions.schema import ACTION_SCHEMA, normalize_action
from src.phase0.protocol import (
    ClassificationMetrics,
    ManifestSample,
    evaluate_classification,
    read_manifest_samples,
)


BASELINE_NAME = "majority"


@dataclass(frozen=True)
class PredictionRecord:
    sample_token: str
    scene_token: str
    split: str
    ground_truth_action: str
    predicted_action: str
    is_correct: bool
    label_rule_version: str
    baseline_name: str


def fit_majority_action(samples: Sequence[ManifestSample]) -> str:
    counts = {action: 0 for action in ACTION_SCHEMA}
    for sample in samples:
        if sample.split == "train":
            counts[normalize_action(sample.meta_action)] += 1
    if not sum(counts.values()):
        raise ValueError("Majority Baseline requires at least one train sample")
    return max(ACTION_SCHEMA, key=lambda action: counts[action])


def predict_split(
    samples: Sequence[ManifestSample],
    split: str,
    majority_action: str,
    label_rule_version: str,
) -> tuple[tuple[PredictionRecord, ...], ClassificationMetrics]:
    predicted_action = normalize_action(majority_action)
    selected_samples = tuple(sample for sample in samples if sample.split == split)
    ground_truth = tuple(
        normalize_action(sample.meta_action) for sample in selected_samples
    )
    predictions = tuple(predicted_action for _ in selected_samples)
    records = tuple(
        PredictionRecord(
            sample_token=sample.sample_token,
            scene_token=sample.scene_token,
            split=sample.split,
            ground_truth_action=target,
            predicted_action=predicted_action,
            is_correct=target == predicted_action,
            label_rule_version=label_rule_version,
            baseline_name=BASELINE_NAME,
        )
        for sample, target in zip(selected_samples, ground_truth, strict=True)
    )
    return records, evaluate_classification(ground_truth, predictions)


def write_predictions(
    predictions: Sequence[PredictionRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(asdict(prediction)) + "\n")


def _load_config(config_path: Path) -> tuple[Path, Path]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    manifest_path = loaded.get("manifest_path")
    output_dir = loaded.get("output_dir")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("configuration missing manifest_path")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("configuration missing output_dir")
    return Path(manifest_path), Path(output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 0 Majority Baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    manifest_path, output_dir = _load_config(arguments.config)
    samples = read_manifest_samples(manifest_path)
    label_rule_versions = {sample.label_rule_version for sample in samples}
    if len(label_rule_versions) != 1:
        raise ValueError("manifest must contain one label_rule_version")
    majority_action = fit_majority_action(samples)
    predictions, metrics = predict_split(
        samples=samples,
        split=arguments.split,
        majority_action=majority_action,
        label_rule_version=label_rule_versions.pop(),
    )
    output_path = output_dir / f"majority_{arguments.split}_predictions.jsonl"
    write_predictions(predictions, output_path)
    print(f"majority_action: {majority_action}")
    print(json.dumps(asdict(metrics), indent=2))
    print(f"predictions: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

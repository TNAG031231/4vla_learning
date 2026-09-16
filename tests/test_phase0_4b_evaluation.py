from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.phase0_4b_evaluation import (
    checkpoint_score, factorized_metrics, failure_taxonomy, generalization_summary,
    majority_baselines, select_checkpoint,
)
from src.phase0.phase0_4b_protocol import ActionTarget, serialize_action
from test_phase0_4b_protocol import case, samples


def population(sample, counts, split="validation"):
    return [replace(sample, split=split, sample_token=f"{long}-{lat}-{i:04d}",
                    target=ActionTarget(long, lat, True, True))
            for (long, lat), count in counts.items() for i in range(count)]


def row(long, lat, prediction):
    return {"sample_token": f"{long}-{lat}", "split": "validation",
            "target": asdict(ActionTarget(long, lat, long is not None, lat is not None)),
            "raw_output": serialize_action(*prediction) if prediction else "invalid",
            "parsed_action": dict(zip(("longitudinal", "lateral"), prediction)) if prediction else None}


def test_metrics_independent_validity_invalids_and_absent_classes():
    predictions = [row("keep", "left", None), row(None, "left", ("stop", "left")),
                   row("stop", None, ("stop", "right")), row("keep", "right", ("keep", "straight")),
                   row(None, None, ("keep", "straight"))]
    result = factorized_metrics(predictions)
    assert [result[f"{axis}_count"] for axis in ("longitudinal", "lateral", "joint")] == [3, 3, 2]
    assert result["longitudinal_accuracy"] == pytest.approx(2 / 3)
    assert result["lateral_accuracy"] == pytest.approx(1 / 3)
    assert result["joint_accuracy"] == 0
    assert result["parser_success_rate"] == 0.8
    assert result["invalid_reason_distribution"] == {"noncanonical_structured_output": 1}
    longitudinal = result["longitudinal"]
    assert longitudinal["per_class"]["keep"] == {"precision": 1, "recall": 0.5, "f1": 2 / 3, "support": 2}
    assert longitudinal["macro_f1"] == pytest.approx((1 + 2 / 3) / 4)
    assert result["lateral_macro_f1"] == pytest.approx((2 / 3) / 3)
    assert longitudinal["confusion_matrix"] == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    assert len(result["lateral"]["confusion_matrix"]) == 3
    assert longitudinal["invalid_by_target"]["keep"] == 1
    assert longitudinal["prediction_distribution"]["invalid"] == 1
    assert result["joint"]["prediction_distribution"]["invalid"] == 1
    empty = factorized_metrics([row(None, None, None)])
    assert empty["longitudinal_macro_f1"] is None
    assert empty["joint_accuracy"] is None


def test_checkpoint_score_joint_then_parser_tiebreak_then_earliest():
    def candidate(step, longitudinal, lateral, joint, parser):
        return {"step": step, "metrics": {"longitudinal_macro_f1": longitudinal,
                "lateral_macro_f1": lateral, "joint_accuracy": joint, "parser_success_rate": parser}}
    best = candidate(50, 0.5, 0.5, 0.6, 0.9)
    candidates = [candidate(10, 0.9, 0, 0.95, 1.0), candidate(25, 0.5, 0.5, 0.4, 1.0),
                  candidate(40, 0.5, 0.5, 0.6, 0.8), best, candidate(75, 0.5, 0.5, 0.6, 0.9)]
    assert checkpoint_score(best["metrics"]) == 0.5
    assert select_checkpoint(candidates) == best
    assert select_checkpoint(list(reversed(candidates))) == best
    with pytest.raises(ValueError, match="finite"):
        checkpoint_score({"longitudinal_macro_f1": None, "lateral_macro_f1": 0.5})


def test_majorities_fit_train_only_and_joint_pair_can_differ(samples):
    train = population(samples[0], {("keep", "left"): 4, ("stop", "straight"): 3,
                                   ("accelerate", "straight"): 3}, "train")
    val = population(samples[0], {("stop", "right"): 6})
    result = majority_baselines(train, val)
    assert result["longitudinal_majority"] == "keep"
    assert result["lateral_majority"] == "straight"
    assert result["joint_pair_majority"] == ["keep", "left"]
    assert result["independent_majorities"]["longitudinal_count"] == 6
    assert result["independent_majorities"]["joint_accuracy"] == 0
    for bad_train, bad_val in ((val, val), (train, train), (train, [replace(val[0], split="test")])):
        with pytest.raises(ValueError, match="train"):
            majority_baselines(bad_train, bad_val)


def test_failure_categories_and_collapse_diagnostic():
    predictions = [row("keep", "left", ("keep", "left")),
                   row("keep", "left", ("stop", "left")),
                   row("keep", "left", ("keep", "right")),
                   row("keep", "left", ("stop", "right")), row("keep", "left", None),
                   row("keep", None, ("keep", "straight")), row(None, None, ("keep", "straight"))]
    failures = failure_taxonomy(predictions, 1)
    assert set(failures["counts"].values()) == {1}
    assert all(len(v) == 1 for v in failures["representative_cases"].values())
    collapsed = factorized_metrics([row("keep", "left", ("keep", "straight")),
                                   row("keep", "right", ("keep", "straight"))])
    baselines = {key: collapsed for key in ("independent_majorities", "joint_pair_majority_metrics")}
    summary = generalization_summary(collapsed, baselines, 0.95)
    assert summary["lateral_collapse_evidence"]
    assert summary["zero_f1_lateral_classes"] == ["left", "right"]
    assert summary["straight_prediction_fraction"] == 1

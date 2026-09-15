from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math

from src.phase0.phase0_4_factorized_targets import LATERAL_ACTIONS, LONGITUDINAL_ACTIONS
from src.phase0.phase0_4b_lora_smoke import metrics as accuracy_metrics
from src.phase0.phase0_4b_protocol import SFTSample, serialize_action

METRICS_VERSION = "phase0.4b-factorized-metrics-v0.1"


def joint_distribution(samples: list[SFTSample]) -> dict[str, int]:
    return dict(sorted(Counter(
        sample.target.training_text() for sample in samples
    ).items()))


def factorized_metrics(predictions: list[dict]) -> dict:
    if not predictions:
        raise ValueError("evaluation requires predictions")
    result = accuracy_metrics(predictions)
    result["metrics_version"] = METRICS_VERSION
    result["invalid_reason_distribution"] = dict(Counter(
        "noncanonical_structured_output" for row in predictions if row["parsed_action"] is None
    ))
    for axis, classes in (("longitudinal", LONGITUDINAL_ACTIONS), ("lateral", LATERAL_ACTIONS)):
        eligible = [r for r in predictions if r["target"][f"{axis}_valid"]]
        targets = Counter(r["target"][axis] for r in eligible)
        predicted = Counter(r["parsed_action"][axis] if r["parsed_action"] else "invalid"
                            for r in eligible)
        matrix = [[sum(r["target"][axis] == target and r["parsed_action"] is not None
                       and r["parsed_action"][axis] == prediction for r in eligible)
                   for prediction in classes] for target in classes]
        per_class = {}
        for index, label in enumerate(classes):
            tp = matrix[index][index]
            precision = tp / predicted[label] if predicted[label] else 0.0
            recall = tp / targets[label] if targets[label] else 0.0
            per_class[label] = {"precision": precision, "recall": recall,
                                "f1": 2 * tp / (targets[label] + predicted[label])
                                if targets[label] + predicted[label] else 0.0,
                                "support": targets[label]}
        macro = sum(m["f1"] for m in per_class.values()) / len(classes) if eligible else None
        result[f"{axis}_macro_f1"] = macro
        result[axis] = {
            "macro_f1": macro, "accuracy": result[f"{axis}_accuracy"],
            "effective_sample_count": len(eligible), "classes": list(classes),
            "per_class": per_class, "confusion_matrix": matrix,
            "confusion_matrix_axes": "rows=target, columns=prediction; invalids counted separately",
            "invalid_by_target": {label: sum(r["target"][axis] == label and r["parsed_action"] is None
                                              for r in eligible) for label in classes},
            "target_distribution": {label: targets[label] for label in classes},
            "prediction_distribution": {label: predicted[label] for label in (*classes, "invalid")},
        }
    joint = [r for r in predictions if r["target"]["longitudinal_valid"] and r["target"]["lateral_valid"]]
    result["joint"] = {
        "accuracy": result["joint_accuracy"], "effective_sample_count": len(joint),
        "target_distribution": dict(Counter(serialize_action(r["target"]["longitudinal"],
                                                             r["target"]["lateral"]) for r in joint)),
        "prediction_distribution": dict(Counter(
            serialize_action(r["parsed_action"]["longitudinal"], r["parsed_action"]["lateral"])
            if r["parsed_action"] else "invalid" for r in joint)),
    }
    return result


def checkpoint_score(metrics: dict) -> float:
    values = [metrics[f"{axis}_macro_f1"] for axis in ("longitudinal", "lateral")]
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("checkpoint selection requires finite metrics for both directions")
    return sum(values) / 2


def select_checkpoint(checkpoints: list[dict]) -> dict:
    return max(checkpoints, key=lambda c: (
        checkpoint_score(c["metrics"]), c["metrics"]["joint_accuracy"],
        c["metrics"]["parser_success_rate"], -c["step"],
    ))


def majority_baselines(train: list[SFTSample], validation: list[SFTSample]) -> dict:
    if any(s.split != "train" for s in train) or any(s.split != "validation" for s in validation):
        raise ValueError("majority fitting requires train; evaluation requires validation")
    chosen = {}
    for axis, classes in (("longitudinal", LONGITUDINAL_ACTIONS), ("lateral", LATERAL_ACTIONS)):
        counts = Counter(getattr(s.target, axis) for s in train if getattr(s.target, f"{axis}_valid"))
        if not counts:
            raise ValueError("majority fitting requires valid train targets")
        chosen[axis] = max(classes, key=lambda label: counts[label])
    pairs = Counter((s.target.longitudinal, s.target.lateral) for s in train
                    if s.target.longitudinal_valid and s.target.lateral_valid)
    if not pairs:
        raise ValueError("joint majority fitting requires joint-valid train targets")
    pair = min(pairs, key=lambda p: (-pairs[p], p))
    result = {"fit_split": "train", "evaluation_split": "validation", "train_count": len(train),
              "longitudinal_majority": chosen["longitudinal"], "lateral_majority": chosen["lateral"],
              "joint_pair_majority": list(pair)}
    for name, values in (("independent_majorities", chosen),
                         ("joint_pair_majority_metrics", dict(zip(("longitudinal", "lateral"), pair)))):
        rows = [{"target": asdict(s.target), "parsed_action": values} for s in validation]
        result[name] = factorized_metrics(rows)
    return result


def failure_taxonomy(predictions: list[dict], examples_per_category: int) -> dict:
    names = ("fully_correct", "longitudinal_wrong_only", "lateral_wrong_only", "both_wrong",
             "invalid_structured_output", "partial_valid_target", "no_valid_target")
    counts = dict.fromkeys(names, 0)
    examples = {name: [] for name in names if name != "fully_correct"}
    for row in predictions:
        target, parsed = row["target"], row["parsed_action"]
        if parsed is None:
            category = "invalid_structured_output"
        elif not (target["longitudinal_valid"] and target["lateral_valid"]):
            category = "partial_valid_target" if any((target["longitudinal_valid"], target["lateral_valid"])) else "no_valid_target"
        else:
            long_wrong = parsed["longitudinal"] != target["longitudinal"]
            lat_wrong = parsed["lateral"] != target["lateral"]
            category = ("both_wrong" if long_wrong and lat_wrong else
                        "longitudinal_wrong_only" if long_wrong else
                        "lateral_wrong_only" if lat_wrong else "fully_correct")
        row["failure_category"] = category
        row["invalid_reason"] = "noncanonical_structured_output" if parsed is None else None
        counts[category] += 1
        if category in examples and len(examples[category]) < examples_per_category:
            examples[category].append(row.copy())
    return {"counts": counts, "representative_cases": examples}


def generalization_summary(model_metrics: dict, baselines: dict, straight_threshold: float) -> dict:
    lateral = model_metrics["lateral"]
    count = lateral["effective_sample_count"]
    straight_fraction = lateral["prediction_distribution"]["straight"] / count if count else None
    zero_f1 = [label for label in ("left", "right") if lateral["per_class"][label]["f1"] == 0]
    return {
        "zero_f1_lateral_classes": zero_f1,
        "lateral_collapse_evidence": bool(zero_f1) or (straight_fraction is not None and straight_fraction >= straight_threshold),
        "straight_dominance_reporting_threshold": straight_threshold,
        "straight_prediction_fraction": straight_fraction,
        "lateral_prediction_distribution": lateral["prediction_distribution"],
        "versus_majority": {name: {
            metric: model_metrics[metric] - baselines[name][metric]
            if model_metrics[metric] is not None and baselines[name][metric] is not None else None
            for metric in ("longitudinal_macro_f1", "lateral_macro_f1", "joint_accuracy")
        } for name in ("independent_majorities", "joint_pair_majority_metrics")},
        "next_action": "stop_after_first_formal_result_for_user_review",
    }

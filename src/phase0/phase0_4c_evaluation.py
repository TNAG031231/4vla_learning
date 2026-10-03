from __future__ import annotations

from collections import Counter
import math

import torch


def prediction_metrics(prediction: torch.Tensor | None, target: torch.Tensor,
                       mask: torch.Tensor) -> dict:
    if target.shape != (6, 2) or mask.shape != (6,) or mask.dtype != torch.bool:
        raise ValueError("trajectory target/mask contract mismatch")
    if not torch.isfinite(target[mask]).all() or not mask.any():
        raise ValueError("trajectory evaluation requires finite valid GT points")
    reason = None
    if prediction is None:
        reason = "invalid_structured_action"
    elif prediction.shape != (6, 2):
        reason = "invalid_trajectory_shape"
    elif not torch.isfinite(prediction).all():
        reason = "nonfinite_trajectory"
    if reason:
        return {"prediction_valid": False, "invalid_reason": reason,
                "predicted_waypoints": None}
    distances = torch.linalg.vector_norm(prediction[mask] - target[mask], dim=-1)
    result = {"prediction_valid": True, "invalid_reason": None,
              "predicted_waypoints": prediction.tolist(),
              "ade_m": float(distances.mean()), "fde_m": float(distances[-1])}
    for seconds, endpoint in ((1, 1), (2, 3), (3, 5)):
        active = mask[:endpoint + 1]
        errors = torch.linalg.vector_norm(
            prediction[:endpoint + 1][active] - target[:endpoint + 1][active], dim=-1)
        result[f"ade_{seconds}s_m"] = float(errors.mean()) if active.any() else None
        result[f"fde_{seconds}s_m"] = (
            float(torch.linalg.vector_norm(prediction[endpoint] - target[endpoint]))
            if mask[endpoint] else None)
    return result


def aggregate_metrics(rows: list[dict], conditioning_type: str) -> dict:
    if any(row["conditioning_type"] != conditioning_type for row in rows):
        raise ValueError("mixed conditioning tracks")
    valid = [row for row in rows if row["prediction_valid"]]
    result = {
        "conditioning_type": conditioning_type, "sample_count": len(rows),
        "valid_prediction_count": len(valid), "invalid_prediction_count": len(rows) - len(valid),
        "trajectory_valid_rate": len(valid) / len(rows) if rows else None,
        "invalid_prediction_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        "invalid_reason_counts": dict(Counter(row["invalid_reason"] for row in rows
                                               if not row["prediction_valid"])),
        "metric_reduction": "mean_of_per_sample_masked_errors_on_valid_predictions",
        "fde_policy": "overall_last_valid_gt_point; horizon_FDE_requires_exact_endpoint",
    }
    for key in ("ade_m", "fde_m", "ade_1s_m", "ade_2s_m", "ade_3s_m",
                "fde_1s_m", "fde_2s_m", "fde_3s_m"):
        values = [row[key] for row in valid if row[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        result[f"{key}_sample_count"] = len(values)
    return result


def paired_gap(formal: list[dict], diagnostic: list[dict]) -> dict:
    reference = {row["sample_token"]: row for row in diagnostic if row["prediction_valid"]}
    pairs = [(row, reference[row["sample_token"]]) for row in formal
             if row["prediction_valid"] and row["sample_token"] in reference]
    return {"sample_count": len(pairs), "definition": "predicted_action_minus_gt_on_same_valid_samples",
            **{f"{key}_gap": sum(a[key] - b[key] for a, b in pairs) / len(pairs) if pairs else None
               for key in ("ade_m", "fde_m")}}


def compare_reload(before: list[dict], after: list[dict]) -> dict:
    tokens_match = [r["sample_token"] for r in before] == [r["sample_token"] for r in after]
    predictions_match = tokens_match
    for a, b in zip(before, after):
        predictions_match &= all(a[key] == b[key] for key in
                                 ("conditioning_type", "action_context", "prediction_valid", "invalid_reason"))
        if a["prediction_valid"] and b["prediction_valid"]:
            predictions_match &= torch.allclose(torch.tensor(a["predicted_waypoints"]),
                                                 torch.tensor(b["predicted_waypoints"]),
                                                 atol=1e-5, rtol=1e-5)
    first = aggregate_metrics(before, "predicted_action")
    second = aggregate_metrics(after, "predicted_action")
    metrics_match = all(
        math.isclose(value, second[key], abs_tol=1e-5, rel_tol=1e-5)
        if isinstance(value, float) and isinstance(second[key], float) else value == second[key]
        for key, value in first.items())
    return {"reload_consistency": bool(predictions_match and metrics_match),
            "predictions_match": bool(predictions_match), "metrics_match": metrics_match,
            "sample_count": len(before), "atol": 1e-5, "rtol": 1e-5,
            "reference_metrics": first, "reloaded_metrics": second,
            "reloaded_predictions": after}

from __future__ import annotations

from collections.abc import Sequence
import math

from data.inspect_nuscenes_sample import TrajectoryPoint


def extract_motion_features(
    trajectory: Sequence[TrajectoryPoint],
) -> dict[str, float]:
    """Extract statistics from a validated Phase 0.4a-1 raw trajectory.

    Point 0 is the current anchor; points 1..6 span nominal 0.5..3.0 s.
    Coordinates are current ego x-forward/y-left in meters, heading in radians.
    Speed proxies use planar segment distance and the actual producer times.
    No action labels or decision thresholds are applied.
    """
    if len(trajectory) != 7:
        raise ValueError("raw trajectory must contain 7 points including anchor")
    distances = [
        math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)
        for first, second in zip(trajectory, trajectory[1:])
    ]
    start_speed = distances[0] / (trajectory[1].t_sec - trajectory[0].t_sec)
    end_speed = distances[-1] / (trajectory[6].t_sec - trajectory[5].t_sec)
    lateral = [point.y_m for point in trajectory]
    headings = [point.heading_delta_rad for point in trajectory]
    return {
        "forward_displacement_m": trajectory[6].x_m - trajectory[0].x_m,
        "path_length_m": sum(distances),
        "start_speed_proxy_mps": start_speed,
        "end_speed_proxy_mps": end_speed,
        "delta_speed_proxy_mps": end_speed - start_speed,
        "final_lateral_displacement_m": lateral[-1],
        "max_left_displacement_m": max(lateral),
        "max_right_displacement_m": min(lateral),
        "max_abs_lateral_displacement_m": max(abs(value) for value in lateral),
        "final_heading_delta_rad": headings[-1],
        "max_abs_heading_delta_rad": max(abs(value) for value in headings),
    }


FACTORIZED_ACTION_RULE_VERSION = "phase0.4-factorized-action-v0.1"
LONGITUDINAL_ACTIONS = ("stop", "decelerate", "keep", "accelerate")
LATERAL_ACTIONS = ("left", "straight", "right")
STOP_PATH_LENGTH_M = 0.6
STOP_END_SPEED_MPS = 0.6
SPEED_CHANGE_MPS = 1.0
LATERAL_DISPLACEMENT_M = 1.0


def derive_feature_targets(features: dict[str, float]) -> dict[str, object]:
    """Classify each direction using only its required motion features."""
    required = {
        "longitudinal": (
            "path_length_m", "end_speed_proxy_mps", "delta_speed_proxy_mps",
        ),
        "lateral": ("final_lateral_displacement_m",),
    }
    result: dict[str, object] = {
        "factorized_action_rule_version": FACTORIZED_ACTION_RULE_VERSION,
    }
    for direction, names in required.items():
        invalid = next((
            name for name in names
            if name not in features or not math.isfinite(features[name])
        ), None)
        result[f"{direction}_action"] = None
        result[f"{direction}_action_valid"] = invalid is None
        result[f"{direction}_action_reason"] = (
            f"missing_or_nonfinite_feature:{invalid}" if invalid else "valid"
        )
    if result["longitudinal_action_valid"]:
        if (
            features["path_length_m"] <= STOP_PATH_LENGTH_M
            and features["end_speed_proxy_mps"] <= STOP_END_SPEED_MPS
        ):
            action = "stop"
        elif features["delta_speed_proxy_mps"] >= SPEED_CHANGE_MPS:
            action = "accelerate"
        elif features["delta_speed_proxy_mps"] <= -SPEED_CHANGE_MPS:
            action = "decelerate"
        else:
            action = "keep"
        result["longitudinal_action"] = action
        result["longitudinal_action_reason"] = f"v0.1:{action}"
    if result["lateral_action_valid"]:
        displacement = features["final_lateral_displacement_m"]
        action = (
            "left" if displacement >= LATERAL_DISPLACEMENT_M
            else "right" if displacement <= -LATERAL_DISPLACEMENT_M
            else "straight"
        )
        result["lateral_action"] = action
        result["lateral_action_reason"] = f"v0.1:{action}"
    result["factorized_action_joint_valid"] = (
        result["longitudinal_action_valid"] and result["lateral_action_valid"]
    )
    return result


def derive_trajectory_targets(
    trajectory: Sequence[TrajectoryPoint],
    *,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    anchor_absolute_tolerance: float,
) -> dict[str, object]:
    """Check the frozen timing/geometry boundary before feature extraction."""
    reason = None
    if len(trajectory) != 7:
        reason = "missing_or_incomplete_trajectory"
    elif any(
        not math.isfinite(value)
        for point in trajectory
        for value in (point.t_sec, point.x_m, point.y_m, point.heading_delta_rad)
    ):
        reason = "nonfinite_trajectory"
    elif trajectory[0].t_sec != 0 or any(
        second.t_sec <= first.t_sec
        for first, second in zip(trajectory, trajectory[1:])
    ) or any(
        abs(point.t_sec - index * sample_interval_sec) > time_tolerance_sec
        for index, point in enumerate(trajectory)
    ):
        reason = "invalid_trajectory_time"
    elif any(
        abs(value) > anchor_absolute_tolerance
        for value in (
            trajectory[0].x_m, trajectory[0].y_m,
            trajectory[0].heading_delta_rad,
        )
    ):
        reason = "invalid_current_anchor"
    if reason:
        result = derive_feature_targets({})
        for direction in ("longitudinal", "lateral"):
            result[f"{direction}_action_reason"] = reason
        return {**result, "motion_features": None}
    features = extract_motion_features(trajectory)
    return {**derive_feature_targets(features), "motion_features": features}

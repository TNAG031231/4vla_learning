from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path

import yaml

from data.inspect_nuscenes_sample import TrajectoryPoint
from src.actions.schema import ACCELERATE, DECELERATE, KEEP, STOP


LONGITUDINAL_CANDIDATE_ACTIONS = (STOP, KEEP, ACCELERATE, DECELERATE)
LATERAL_CANDIDATE_ACTIONS = ("left", "straight", "right")


@dataclass(frozen=True)
class CandidateRuleParameters:
    stop_distance_center_m: float
    speed_change_center_mps: float
    minimum_forward_displacement_m: float
    lateral_displacement_center_m: float
    heading_support_rad: float
    boundary_margin: float

    def bounds(self, center: float) -> tuple[float, float]:
        # Preserve decimal cutoffs such as 0.08 at strict comparison boundaries.
        value = Decimal(str(center))
        margin = Decimal(str(self.boundary_margin))
        return float(value * (1 - margin)), float(value * (1 + margin))


def load_candidate_parameters(path: Path) -> CandidateRuleParameters:
    config = yaml.safe_load(path.read_text())
    if config["status"] != "provisional":
        raise ValueError("candidate rules must be marked provisional")
    parameters = CandidateRuleParameters(**config["parameters"])
    for name, value in config["parameters"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if parameters.boundary_margin >= 1:
        raise ValueError("boundary_margin must be less than 1")
    return parameters


@dataclass(frozen=True)
class CandidateDirectionResult:
    candidate_action: str | None
    candidate_valid: bool
    candidate_reason: str


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_longitudinal: CandidateDirectionResult
    candidate_lateral: CandidateDirectionResult
    candidate_joint_valid: bool


def stop_motion_extent(features: Mapping[str, float]) -> float:
    return max(
        features["path_length_m"],
        abs(features["forward_displacement_m"]),
        abs(features["final_lateral_displacement_m"]),
    )


def evaluate_candidate_longitudinal(
    features: Mapping[str, float], parameters: CandidateRuleParameters,
) -> CandidateDirectionResult:
    stop_low, stop_high = parameters.bounds(parameters.stop_distance_center_m)
    extent = stop_motion_extent(features)
    if extent <= stop_low:
        return CandidateDirectionResult(STOP, True, "confident_stop")
    if extent <= stop_high:
        return CandidateDirectionResult(None, False, "stop_boundary")
    if features["forward_displacement_m"] < parameters.minimum_forward_displacement_m:
        return CandidateDirectionResult(None, False, "low_forward_displacement")
    speed_low, speed_high = parameters.bounds(parameters.speed_change_center_mps)
    delta = features["delta_speed_proxy_mps"]
    if delta > speed_high:
        return CandidateDirectionResult(ACCELERATE, True, "confident_accelerate")
    if delta < -speed_high:
        return CandidateDirectionResult(DECELERATE, True, "confident_decelerate")
    if abs(delta) < speed_low:
        return CandidateDirectionResult(KEEP, True, "confident_keep")
    return CandidateDirectionResult(None, False, "speed_boundary")


def evaluate_candidate_lateral(
    features: Mapping[str, float], parameters: CandidateRuleParameters,
) -> CandidateDirectionResult:
    lateral_low, lateral_high = parameters.bounds(
        parameters.lateral_displacement_center_m,
    )
    heading_low, heading_high = parameters.bounds(parameters.heading_support_rad)
    final_y = features["final_lateral_displacement_m"]
    heading = features["final_heading_delta_rad"]
    if abs(final_y) > lateral_high:
        if (final_y > 0 and heading < -heading_high) or (
            final_y < 0 and heading > heading_high
        ):
            return CandidateDirectionResult(
                None, False, "lateral_heading_sign_conflict",
            )
        action = "left" if final_y > 0 else "right"
        return CandidateDirectionResult(action, True, "confident_lateral_displacement")
    if abs(final_y) >= lateral_low:
        return CandidateDirectionResult(None, False, "lateral_displacement_boundary")
    excursion = features["max_abs_lateral_displacement_m"]
    if excursion < lateral_low and abs(heading) < heading_low:
        return CandidateDirectionResult("straight", True, "confident_straight")
    if excursion > lateral_high:
        reason = "lateral_excursion_without_final_displacement"
    elif abs(heading) > heading_high:
        reason = "heading_without_lateral_displacement"
    else:
        reason = "lateral_straight_boundary"
    return CandidateDirectionResult(None, False, reason)


def evaluate_candidate_rules(
    features: Mapping[str, float], parameters: CandidateRuleParameters,
) -> CandidateEvaluation:
    """Evaluate each direction independently; results are provisional only."""
    longitudinal = evaluate_candidate_longitudinal(features, parameters)
    lateral = evaluate_candidate_lateral(features, parameters)
    return CandidateEvaluation(
        longitudinal, lateral,
        longitudinal.candidate_valid and lateral.candidate_valid,
    )


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

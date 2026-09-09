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

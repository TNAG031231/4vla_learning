#!/usr/bin/env python3

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Protocol

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import yaml


CAMERA_CHANNEL = "CAM_FRONT"
MICROSECONDS_PER_SECOND = 1_000_000
TIMESTAMP_TOLERANCE_SEC = 1e-3
DEFAULT_TRAJECTORY_TIME_TOLERANCE_SEC = 0.075
AGENT_SIZE_ORDER = ("width_m", "length_m", "height_m")
VRU_CATEGORY_PREFIXES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.motorcycle",
)


class NuScenesReader(Protocol):
    scene: list[dict[str, object]]

    def get(self, table_name: str, token: str) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class TrajectoryConfig:
    nuscenes_root: Path
    version: str
    horizon_sec: float
    sample_interval_sec: float
    nearby_radius_m: float
    trajectory_time_tolerance_sec: float


@dataclass(frozen=True)
class EgoFramePose:
    x_m: float
    y_m: float
    heading_delta_rad: float


@dataclass(frozen=True)
class TrajectoryPoint:
    future_sample_token: str
    t_sec: float
    x_m: float
    y_m: float
    heading_delta_rad: float


@dataclass(frozen=True)
class FutureEgoTrajectory:
    sample_token: str
    scene_token: str
    current_timestamp: int
    points: tuple[TrajectoryPoint, ...]
    is_truncated: bool


@dataclass(frozen=True)
class NearbyAgent:
    annotation_token: str
    instance_token: str
    category_name: str
    is_vehicle: bool
    is_vru: bool
    translation_ego: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_ego_rad: float
    distance_xy_m: float
    num_lidar_pts: int
    num_radar_pts: int


@dataclass(frozen=True)
class NearbyAgents:
    sample_token: str
    scene_token: str
    agents: tuple[NearbyAgent, ...]


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _number_value(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _vector_value(
    mapping: Mapping[str, object],
    key: str,
    length: int,
) -> tuple[float, ...]:
    value = mapping[key]
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain {length} values")
    return tuple(float(component) for component in value)


def load_trajectory_config(config_path: Path) -> TrajectoryConfig:
    loaded: object = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration root must be a mapping")

    data_config = _mapping_value(loaded, "data")
    phase1_config = _mapping_value(loaded, "phase1")
    root_template = _string_value(data_config, "nuscenes_root")
    expanded_root = os.path.expandvars(root_template)
    if "$" in expanded_root:
        raise ValueError(
            "data.nuscenes_root contains an unresolved environment variable"
        )

    return TrajectoryConfig(
        nuscenes_root=Path(expanded_root).expanduser(),
        version=_string_value(data_config, "version"),
        horizon_sec=_number_value(phase1_config, "horizon_sec"),
        sample_interval_sec=_number_value(
            phase1_config,
            "sample_interval_sec",
        ),
        nearby_radius_m=_number_value(
            phase1_config,
            "max_agent_distance_m",
        ),
        trajectory_time_tolerance_sec=_number_value(
            phase1_config,
            "trajectory_time_tolerance_sec",
        ),
    )


def microseconds_to_seconds(timestamp_delta_us: int) -> float:
    return timestamp_delta_us / MICROSECONDS_PER_SECOND


def transform_global_point_to_ego(
    point_global: tuple[float, ...],
    ego_translation_global: tuple[float, ...],
    ego_rotation_global: tuple[float, ...],
) -> tuple[float, float, float]:
    """Transform a point from nuScenes global frame to current ego frame.

    Source and target units are meters. The target axes are x forward, y left,
    and z up. The ego pose stores global_from_ego rotation R and global
    translation t, so the required direction is:
    p_ego = R_global_from_ego.T @ (p_global - t_global).
    """
    global_displacement = tuple(
        point - ego
        for point, ego in zip(
            point_global,
            ego_translation_global,
            strict=True,
        )
    )
    point_ego = Quaternion(ego_rotation_global).inverse.rotate(
        global_displacement
    )
    return (
        float(point_ego[0]),
        float(point_ego[1]),
        float(point_ego[2]),
    )


def normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def transform_global_yaw_to_ego(
    yaw_global_rad: float,
    ego_yaw_global_rad: float,
) -> float:
    return normalize_angle(yaw_global_rad - ego_yaw_global_rad)


def classify_agent_category(category_name: str) -> tuple[bool, bool]:
    is_vehicle = category_name.startswith("vehicle.")
    is_vru = any(
        category_name == prefix or category_name.startswith(f"{prefix}.")
        for prefix in VRU_CATEGORY_PREFIXES
    )
    return is_vehicle, is_vru


def transform_pose_to_current_ego_frame(
    current_translation: tuple[float, ...],
    current_rotation: tuple[float, ...],
    future_translation: tuple[float, ...],
    future_rotation: tuple[float, ...],
) -> EgoFramePose:
    """Transform a future global ego pose into the current ego frame.

    Source is the nuScenes global Cartesian frame in meters, with its map x/y
    axes and z pointing up. Target is the current nuScenes ego frame in meters:
    x points forward, y left, and z up. The transform first subtracts the
    current global translation, then applies the inverse current ego-to-global
    rotation. Heading uses current_rotation.inverse * future_rotation and is
    reported as yaw about target +z in radians.
    """
    current_orientation = Quaternion(current_rotation)
    future_orientation = Quaternion(future_rotation)
    ego_displacement = transform_global_point_to_ego(
        point_global=future_translation,
        ego_translation_global=current_translation,
        ego_rotation_global=current_rotation,
    )
    relative_orientation = current_orientation.inverse * future_orientation
    heading_delta_rad = relative_orientation.yaw_pitch_roll[0]

    return EgoFramePose(
        x_m=float(ego_displacement[0]),
        y_m=float(ego_displacement[1]),
        heading_delta_rad=float(heading_delta_rad),
    )


def get_ego_pose(
    nuscenes: NuScenesReader,
    sample: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    sample_data = _mapping_value(sample, "data")
    camera_token = _string_value(sample_data, CAMERA_CHANNEL)
    camera_data = nuscenes.get("sample_data", camera_token)
    ego_pose_token = _string_value(camera_data, "ego_pose_token")
    ego_pose = nuscenes.get("ego_pose", ego_pose_token)
    return (
        _vector_value(ego_pose, "translation", 3),
        _vector_value(ego_pose, "rotation", 4),
    )


def get_nearby_agents(
    nuscenes: NuScenesReader,
    sample_token: str,
    radius_m: float,
) -> NearbyAgents:
    if radius_m < 0.0:
        raise ValueError("radius_m must be non-negative")

    sample = nuscenes.get("sample", sample_token)
    scene_token = _string_value(sample, "scene_token")
    ego_translation, ego_rotation = get_ego_pose(nuscenes, sample)
    ego_yaw_global_rad = Quaternion(ego_rotation).yaw_pitch_roll[0]
    annotation_tokens = sample.get("anns", [])
    if not isinstance(annotation_tokens, list):
        raise ValueError("anns must be a list")

    nearby_agents = []
    for annotation_token in annotation_tokens:
        if not isinstance(annotation_token, str):
            raise ValueError("annotation token must be a string")
        annotation = nuscenes.get("sample_annotation", annotation_token)
        translation_ego = transform_global_point_to_ego(
            point_global=_vector_value(annotation, "translation", 3),
            ego_translation_global=ego_translation,
            ego_rotation_global=ego_rotation,
        )
        distance_xy_m = math.hypot(
            translation_ego[0],
            translation_ego[1],
        )
        if distance_xy_m > radius_m:
            continue

        category_name = _string_value(annotation, "category_name")
        is_vehicle, is_vru = classify_agent_category(category_name)
        yaw_global_rad = Quaternion(
            _vector_value(annotation, "rotation", 4)
        ).yaw_pitch_roll[0]
        size = _vector_value(annotation, "size", 3)
        nearby_agents.append(
            NearbyAgent(
                annotation_token=_string_value(annotation, "token"),
                instance_token=_string_value(annotation, "instance_token"),
                category_name=category_name,
                is_vehicle=is_vehicle,
                is_vru=is_vru,
                translation_ego=(
                    translation_ego[0],
                    translation_ego[1],
                    translation_ego[2],
                ),
                size=(size[0], size[1], size[2]),
                yaw_ego_rad=transform_global_yaw_to_ego(
                    yaw_global_rad=yaw_global_rad,
                    ego_yaw_global_rad=ego_yaw_global_rad,
                ),
                distance_xy_m=distance_xy_m,
                num_lidar_pts=int(annotation["num_lidar_pts"]),
                num_radar_pts=int(annotation["num_radar_pts"]),
            )
        )

    return NearbyAgents(
        sample_token=sample_token,
        scene_token=scene_token,
        agents=tuple(nearby_agents),
    )


def extract_future_ego_trajectory(
    nuscenes: NuScenesReader,
    sample_token: str,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float = DEFAULT_TRAJECTORY_TIME_TOLERANCE_SEC,
) -> FutureEgoTrajectory:
    if horizon_sec < 0.0:
        raise ValueError("horizon_sec must be non-negative")
    if sample_interval_sec <= 0.0:
        raise ValueError("sample_interval_sec must be positive")
    if time_tolerance_sec < 0.0:
        raise ValueError("time_tolerance_sec must be non-negative")

    current_sample = nuscenes.get("sample", sample_token)
    current_timestamp = int(current_sample["timestamp"])
    scene_token = _string_value(current_sample, "scene_token")
    current_translation, current_rotation = get_ego_pose(
        nuscenes,
        current_sample,
    )
    current_pose = transform_pose_to_current_ego_frame(
        current_translation=current_translation,
        current_rotation=current_rotation,
        future_translation=current_translation,
        future_rotation=current_rotation,
    )
    points = [
        TrajectoryPoint(
            future_sample_token=sample_token,
            t_sec=0.0,
            x_m=current_pose.x_m,
            y_m=current_pose.y_m,
            heading_delta_rad=current_pose.heading_delta_rad,
        )
    ]

    target_times = []
    target_time_sec = sample_interval_sec
    while target_time_sec <= horizon_sec + TIMESTAMP_TOLERANCE_SEC:
        target_times.append(target_time_sec)
        target_time_sec += sample_interval_sec

    future_samples = []
    next_token = _string_value(current_sample, "next")
    while next_token:
        future_sample = nuscenes.get("sample", next_token)
        future_timestamp = int(future_sample["timestamp"])
        time_sec = microseconds_to_seconds(
            future_timestamp - current_timestamp
        )
        if time_sec > horizon_sec + time_tolerance_sec:
            break

        future_samples.append((future_sample, time_sec))
        next_token = _string_value(future_sample, "next")

    search_start_index = 0
    for target_time_sec in target_times:
        selected_index = None
        selected_error = None
        upper_bound_sec = target_time_sec + time_tolerance_sec
        for index in range(search_start_index, len(future_samples)):
            future_sample, time_sec = future_samples[index]
            if time_sec > upper_bound_sec:
                break
            error = abs(time_sec - target_time_sec)
            if error <= time_tolerance_sec and (
                selected_error is None or error < selected_error
            ):
                selected_index = index
                selected_error = error

        if selected_index is None:
            break

        future_sample, time_sec = future_samples[selected_index]
        future_translation, future_rotation = get_ego_pose(
            nuscenes,
            future_sample,
        )
        pose = transform_pose_to_current_ego_frame(
            current_translation=current_translation,
            current_rotation=current_rotation,
            future_translation=future_translation,
            future_rotation=future_rotation,
        )
        points.append(
            TrajectoryPoint(
                future_sample_token=_string_value(future_sample, "token"),
                t_sec=time_sec,
                x_m=pose.x_m,
                y_m=pose.y_m,
                heading_delta_rad=pose.heading_delta_rad,
            )
        )
        search_start_index = selected_index + 1

    horizon_covered = len(points) == len(target_times) + 1

    return FutureEgoTrajectory(
        sample_token=sample_token,
        scene_token=scene_token,
        current_timestamp=current_timestamp,
        points=tuple(points),
        is_truncated=not horizon_covered,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one nuScenes sample's future ego trajectory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Data configuration path (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--sample-token",
        help="Sample token to inspect (default: first sample of first scene).",
    )
    parser.add_argument("--trajectory-time-tolerance-sec", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = load_trajectory_config(arguments.config)
    nuscenes = NuScenes(
        version=config.version,
        dataroot=str(config.nuscenes_root),
        verbose=False,
    )
    sample_token = arguments.sample_token
    if sample_token is None:
        sample_token = str(nuscenes.scene[0]["first_sample_token"])
    time_tolerance_sec = (
        config.trajectory_time_tolerance_sec
        if arguments.trajectory_time_tolerance_sec is None
        else arguments.trajectory_time_tolerance_sec
    )

    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=config.horizon_sec,
        sample_interval_sec=config.sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
    )
    nearby_agents = get_nearby_agents(
        nuscenes=nuscenes,
        sample_token=sample_token,
        radius_m=config.nearby_radius_m,
    )
    payload = asdict(trajectory)
    payload["timestamp_unit"] = "microseconds"
    payload["nearby_radius_m"] = config.nearby_radius_m
    payload["agent_size_order"] = AGENT_SIZE_ORDER
    payload["nearby_agents"] = [
        asdict(agent) for agent in nearby_agents.agents
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Protocol

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import yaml


CAMERA_CHANNEL = "CAM_FRONT"
MICROSECONDS_PER_SECOND = 1_000_000
TIMESTAMP_TOLERANCE_SEC = 1e-3


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
    )


def microseconds_to_seconds(timestamp_delta_us: int) -> float:
    return timestamp_delta_us / MICROSECONDS_PER_SECOND


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
    global_displacement = tuple(
        future - current
        for current, future in zip(
            current_translation,
            future_translation,
            strict=True,
        )
    )
    ego_displacement = current_orientation.inverse.rotate(global_displacement)
    relative_orientation = current_orientation.inverse * future_orientation
    heading_delta_rad = relative_orientation.yaw_pitch_roll[0]

    return EgoFramePose(
        x_m=float(ego_displacement[0]),
        y_m=float(ego_displacement[1]),
        heading_delta_rad=float(heading_delta_rad),
    )


def _ego_pose(
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


def extract_future_ego_trajectory(
    nuscenes: NuScenesReader,
    sample_token: str,
    horizon_sec: float,
    sample_interval_sec: float,
) -> FutureEgoTrajectory:
    if horizon_sec < 0.0:
        raise ValueError("horizon_sec must be non-negative")
    if sample_interval_sec <= 0.0:
        raise ValueError("sample_interval_sec must be positive")

    current_sample = nuscenes.get("sample", sample_token)
    current_timestamp = int(current_sample["timestamp"])
    scene_token = _string_value(current_sample, "scene_token")
    current_translation, current_rotation = _ego_pose(
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

    next_target_sec = sample_interval_sec
    latest_time_sec = 0.0
    horizon_covered = horizon_sec == 0.0
    next_token = _string_value(current_sample, "next")
    while next_token:
        future_sample = nuscenes.get("sample", next_token)
        future_timestamp = int(future_sample["timestamp"])
        time_sec = microseconds_to_seconds(
            future_timestamp - current_timestamp
        )
        if time_sec > horizon_sec + TIMESTAMP_TOLERANCE_SEC:
            horizon_covered = True
            break

        latest_time_sec = time_sec
        if time_sec + TIMESTAMP_TOLERANCE_SEC >= next_target_sec:
            future_translation, future_rotation = _ego_pose(
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
                    future_sample_token=_string_value(
                        future_sample,
                        "token",
                    ),
                    t_sec=time_sec,
                    x_m=pose.x_m,
                    y_m=pose.y_m,
                    heading_delta_rad=pose.heading_delta_rad,
                )
            )
            next_target_sec += sample_interval_sec

        next_token = _string_value(future_sample, "next")

    if latest_time_sec + TIMESTAMP_TOLERANCE_SEC >= horizon_sec:
        horizon_covered = True

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

    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=config.horizon_sec,
        sample_interval_sec=config.sample_interval_sec,
    )
    payload = asdict(trajectory)
    payload["timestamp_unit"] = "microseconds"
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

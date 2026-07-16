from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Protocol, Sequence

from pyquaternion import Quaternion

from src.phase0.protocol import (
    ACCELERATION_UNIT,
    MOTION_SOURCE,
    POSE_TIMESTAMP_SOURCE,
    POSE_TIMESTAMP_UNIT,
    SPEED_UNIT,
    YAW_RATE_UNIT,
)


SAFETY_RULE_VERSION = "not_available"
COORDINATE_METADATA = {
    "current_ego_pose": {
        "frame": "nuScenes_global",
        "translation_unit": "meter",
        "rotation_order": "wxyz",
        "timestamp_unit": POSE_TIMESTAMP_UNIT,
        "timestamp_source": POSE_TIMESTAMP_SOURCE,
    },
    "current_ego_motion": {
        "speed_unit": SPEED_UNIT,
        "longitudinal_acceleration_unit": ACCELERATION_UNIT,
        "yaw_rate_unit": YAW_RATE_UNIT,
        "source": MOTION_SOURCE,
        "timestamp_source": POSE_TIMESTAMP_SOURCE,
    },
    "future_ego_trajectory": {
        "source_frame": "nuScenes_global",
        "target_frame": "current_ego",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
        "unit": "meter",
        "transform": (
            "subtract_current_ego_translation_then_apply_"
            "inverse_current_ego_rotation"
        ),
    },
    "nearby_agents": {
        "source_frame": "nuScenes_global",
        "target_frame": "current_ego",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
        "translation_unit": "meter",
        "yaw_unit": "radian",
        "transform": (
            "subtract_current_ego_translation_then_apply_"
            "inverse_current_ego_rotation"
        ),
    },
}


class NuScenesReader(Protocol):
    def get(self, table_name: str, token: str) -> dict[str, object]:
        ...


def json_compatible(value: object) -> object:
    if is_dataclass(value):
        return json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return json_compatible(attributes)
    raise TypeError(f"Cannot serialize manifest value: {type(value)!r}")


def json_record(record: object) -> dict[str, object]:
    payload = json_compatible(record)
    if not isinstance(payload, dict):
        raise TypeError("manifest record must serialize to an object")
    return payload


def write_jsonl_records(records: Sequence[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(json_record(record)) + "\n")


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _vector_value(
    mapping: Mapping[str, object],
    key: str,
    length: int,
) -> tuple[float, ...]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain {length} values")
    return tuple(float(component) for component in value)


def sample_pose(
    nuscenes: NuScenesReader,
    sample: Mapping[str, object],
    camera_channel: str,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    sample_data = _mapping_value(sample, "data")
    camera_token = _string_value(sample_data, camera_channel)
    camera_data = nuscenes.get("sample_data", camera_token)
    timestamp_us = camera_data.get("timestamp")
    if not isinstance(timestamp_us, int) or isinstance(timestamp_us, bool):
        raise ValueError(f"{camera_channel} sample_data timestamp must be an integer")
    ego_pose_token = _string_value(camera_data, "ego_pose_token")
    ego_pose = nuscenes.get("ego_pose", ego_pose_token)
    return (
        _vector_value(ego_pose, "translation", 3),
        _vector_value(ego_pose, "rotation", 4),
        timestamp_us,
    )


def current_ego_pose(
    nuscenes: NuScenesReader,
    sample_token: str,
    camera_channel: str = "CAM_FRONT",
) -> dict[str, object]:
    sample = nuscenes.get("sample", sample_token)
    translation, rotation, timestamp_us = sample_pose(
        nuscenes,
        sample,
        camera_channel,
    )
    return {
        "frame": "nuScenes_global",
        "translation_m": list(translation),
        "rotation_wxyz": list(rotation),
        "timestamp_us": timestamp_us,
        "timestamp_source": POSE_TIMESTAMP_SOURCE,
    }


def _unavailable_motion(reason: str) -> dict[str, object]:
    return {
        "speed_mps": None,
        "longitudinal_acceleration_mps2": None,
        "yaw_rate_radps": None,
        "source": MOTION_SOURCE,
        "timestamp_source": POSE_TIMESTAMP_SOURCE,
        "availability": "unavailable",
        "history_interval_sec": None,
        "acceleration_interval_sec": None,
        "unavailable_reason": reason,
    }


def _previous_sample(
    nuscenes: NuScenesReader,
    sample: Mapping[str, object],
) -> Mapping[str, object] | None:
    previous_token = sample.get("prev")
    if not isinstance(previous_token, str) or not previous_token:
        return None
    previous_sample = nuscenes.get("sample", previous_token)
    if previous_sample.get("scene_token") != sample.get("scene_token"):
        return None
    return previous_sample


def _history_interval_sec(
    newer_timestamp_us: int,
    older_timestamp_us: int,
) -> float | None:
    delta_us = newer_timestamp_us - older_timestamp_us
    if delta_us <= 0:
        return None
    return delta_us / 1_000_000.0


def _speed_mps(
    newer_translation: tuple[float, ...],
    older_translation: tuple[float, ...],
    interval_sec: float,
) -> float:
    return math.hypot(
        newer_translation[0] - older_translation[0],
        newer_translation[1] - older_translation[1],
    ) / interval_sec


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def current_ego_motion(
    nuscenes: NuScenesReader,
    sample_token: str,
    camera_channel: str = "CAM_FRONT",
) -> dict[str, object]:
    current_sample = nuscenes.get("sample", sample_token)
    current_translation, current_rotation, current_timestamp_us = sample_pose(
        nuscenes,
        current_sample,
        camera_channel,
    )
    previous_sample = _previous_sample(nuscenes, current_sample)
    if previous_sample is None:
        previous_token = current_sample.get("prev")
        if isinstance(previous_token, str) and previous_token:
            return _unavailable_motion("previous_sample_scene_mismatch")
        return _unavailable_motion("insufficient_past_history")

    try:
        previous_translation, previous_rotation, previous_timestamp_us = sample_pose(
            nuscenes,
            previous_sample,
            camera_channel,
        )
    except (KeyError, TypeError, ValueError):
        return _unavailable_motion("past_ego_pose_unavailable")
    interval_sec = _history_interval_sec(
        current_timestamp_us,
        previous_timestamp_us,
    )
    if interval_sec is None:
        return _unavailable_motion("non_monotonic_timestamp")

    speed_mps = _speed_mps(
        current_translation,
        previous_translation,
        interval_sec,
    )
    current_yaw = Quaternion(current_rotation).yaw_pitch_roll[0]
    previous_yaw = Quaternion(previous_rotation).yaw_pitch_roll[0]
    yaw_rate_radps = _normalize_angle(current_yaw - previous_yaw) / interval_sec

    previous_previous_sample = _previous_sample(nuscenes, previous_sample)
    if previous_previous_sample is None:
        return {
            "speed_mps": speed_mps,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": yaw_rate_radps,
            "source": MOTION_SOURCE,
            "timestamp_source": POSE_TIMESTAMP_SOURCE,
            "availability": "partial",
            "history_interval_sec": interval_sec,
            "acceleration_interval_sec": None,
            "unavailable_reason": "insufficient_past_history_for_acceleration",
        }

    try:
        (
            previous_previous_translation,
            _,
            previous_previous_timestamp_us,
        ) = sample_pose(
            nuscenes,
            previous_previous_sample,
            camera_channel,
        )
    except (KeyError, TypeError, ValueError):
        return {
            "speed_mps": speed_mps,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": yaw_rate_radps,
            "source": MOTION_SOURCE,
            "timestamp_source": POSE_TIMESTAMP_SOURCE,
            "availability": "partial",
            "history_interval_sec": interval_sec,
            "acceleration_interval_sec": None,
            "unavailable_reason": "past_ego_pose_unavailable_for_acceleration",
        }
    previous_interval_sec = _history_interval_sec(
        previous_timestamp_us,
        previous_previous_timestamp_us,
    )
    if previous_interval_sec is None:
        return {
            "speed_mps": speed_mps,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": yaw_rate_radps,
            "source": MOTION_SOURCE,
            "timestamp_source": POSE_TIMESTAMP_SOURCE,
            "availability": "partial",
            "history_interval_sec": interval_sec,
            "acceleration_interval_sec": None,
            "unavailable_reason": "insufficient_past_history_for_acceleration",
        }

    previous_speed_mps = _speed_mps(
        previous_translation,
        previous_previous_translation,
        previous_interval_sec,
    )
    return {
        "speed_mps": speed_mps,
        "longitudinal_acceleration_mps2": (
            speed_mps - previous_speed_mps
        ) / ((previous_interval_sec + interval_sec) / 2.0),
        "yaw_rate_radps": yaw_rate_radps,
        "source": MOTION_SOURCE,
        "timestamp_source": POSE_TIMESTAMP_SOURCE,
        "availability": "full",
        "history_interval_sec": interval_sec,
        "acceleration_interval_sec": (
            previous_interval_sec + interval_sec
        ) / 2.0,
        "unavailable_reason": None,
    }

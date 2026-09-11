from __future__ import annotations

from collections import Counter
import math
from pathlib import PurePosixPath

from src.phase0.manifest import COORDINATE_METADATA, NuScenesReader, current_ego_motion
from src.phase0.phase0_4_factorized_targets import FACTORIZED_ACTION_RULE_VERSION

SCHEMA_VERSION = "phase0.4-temporal-waypoint-v0.1"
FACTORIZED_FIELDS = (
    "longitudinal_action", "longitudinal_action_valid", "longitudinal_action_reason",
    "lateral_action", "lateral_action_valid", "lateral_action_reason",
    "factorized_action_joint_valid", "factorized_action_rule_version",
)
INPUT_FIELDS = (
    "historical_sample_tokens", "historical_cam_front_paths",
    "historical_sensor_timestamps", "historical_relative_times_sec",
    "ego_motion_history", "history_valid_mask",
)
TARGET_FIELDS = (*FACTORIZED_FIELDS, "future_waypoints", "trajectory_valid_mask")


def collect_history(reader: NuScenesReader, record: dict, length: int) -> list[dict]:
    """Read past keyframes; validate two extra frames used by the motion producer."""
    if record["split"] not in ("train", "validation"):
        raise ValueError("history only permits train and validation")
    if type(length) is not int or length < 1:
        raise ValueError("history length must be a positive integer")
    frames = []
    seen = set()
    token = record["sample_token"]
    newer_time = None
    while token and len(frames) < length + 2:
        if token in seen:
            raise ValueError("history tokens must be unique")
        seen.add(token)
        sample = reader.get("sample", token)
        if sample["scene_token"] != record["scene_token"]:
            raise ValueError("history crosses scene boundary")
        camera = reader.get("sample_data", sample["data"]["CAM_FRONT"])
        timestamp = camera["timestamp"]
        if camera["sample_token"] != token:
            raise ValueError("CAM_FRONT sample mismatch")
        if type(timestamp) is not int or (newer_time is not None and timestamp >= newer_time):
            raise ValueError("history timestamp must be strictly past to current")
        path = PurePosixPath(camera["filename"])
        if path.is_absolute() or ".." in path.parts or "\\" in camera["filename"]:
            raise ValueError("CAM_FRONT path must be relative")
        if not frames and (
            timestamp != record["current_ego_pose"]["timestamp_us"]
            or camera["filename"] != record["cam_front_path"]
            or sample["timestamp"] != record["timestamp"]
        ):
            raise ValueError("anchor timestamp/path differs from frozen source")
        frames.append({"token": token, "path": camera["filename"], "timestamp": timestamp})
        newer_time = timestamp
        token = sample["prev"]
    result = frames[:length]
    for frame in result:
        frame["motion"] = current_ego_motion(reader, frame["token"])
    if result[0]["motion"] != record["current_ego_motion"]:
        raise ValueError("anchor motion differs from frozen producer")
    return list(reversed(result))


def build_record(record: dict, history: list[dict], length: int) -> dict:
    if record["factorized_action_rule_version"] != FACTORIZED_ACTION_RULE_VERSION:
        raise ValueError("factorized rule version mismatch")
    points = record["future_ego_trajectory"]
    if len(points) != 7 or points[0]["future_sample_token"] != record["sample_token"]:
        raise ValueError("frozen trajectory must contain current anchor and six futures")
    frames = history[-length:]
    padding = length - len(frames)
    anchor_time = frames[-1]["timestamp"]
    preserved = (
        "sample_token", "scene_token", "split", "timestamp", "coordinate_metadata",
        "source_projection_version", "source_future_trajectory_version",
        "source_manifest_schema_version", "source_combined_manifest_sha256",
        "split_mapping_sha256", "split_seed", "split_strategy_version",
    )
    result = {key: record[key] for key in (*preserved, *FACTORIZED_FIELDS)}
    for field, key in (
        ("historical_sample_tokens", "token"), ("historical_cam_front_paths", "path"),
        ("historical_sensor_timestamps", "timestamp"), ("ego_motion_history", "motion"),
    ):
        result[field] = [None] * padding + [frame[key] for frame in frames]
    result.update(
        historical_relative_times_sec=[None] * padding + [
            (frame["timestamp"] - anchor_time) / 1_000_000 for frame in frames
        ],
        history_valid_mask=[False] * padding + [True] * len(frames),
        future_waypoints=[[point["x_m"], point["y_m"]] for point in points[1:7]],
        trajectory_valid_mask=[True] * 6,
        temporal_dataset_schema_version=SCHEMA_VERSION,
        history_length=length, history_policy="null_padding", history_order="oldest_to_current",
    )
    return result


def summarize(records: list[dict]) -> dict:
    return {
        "sample_count": len(records),
        "history_full": sum(all(row["history_valid_mask"]) for row in records),
        "history_padded": sum(not all(row["history_valid_mask"]) for row in records),
        "history_unavailable": sum(not any(row["history_valid_mask"]) for row in records),
        "real_frame_count_distribution": dict(Counter(sum(row["history_valid_mask"]) for row in records)),
        "ego_motion_availability": dict(Counter(
            motion["availability"] for row in records for motion in row["ego_motion_history"]
            if motion is not None
        )),
        "trajectory_all_valid": sum(all(row["trajectory_valid_mask"]) for row in records),
    }


def validate_record(record: dict) -> None:
    """Validate the serialized temporal dataset at its consumer boundary."""
    if (record["temporal_dataset_schema_version"] != SCHEMA_VERSION
            or record["split"] not in ("train", "validation")
            or record["history_policy"] != "null_padding"
            or record["history_order"] != "oldest_to_current"):
        raise ValueError("temporal schema, split or policy mismatch")
    if (record["factorized_action_rule_version"] != FACTORIZED_ACTION_RULE_VERSION
            or record["coordinate_metadata"]["future_ego_trajectory"]
            != COORDINATE_METADATA["future_ego_trajectory"]):
        raise ValueError("target version or coordinate contract mismatch")
    length = record["history_length"]
    if type(length) is not int or length < 1:
        raise ValueError("invalid history length")
    if any(len(record[key]) != length for key in INPUT_FIELDS):
        raise ValueError("history fields must have history_length entries")
    mask = record["history_valid_mask"]
    if any(type(value) is not bool for value in mask) or not mask[-1] or mask != sorted(mask):
        raise ValueError("history mask must be left padding followed by observations")
    real = []
    anchor_time = record["historical_sensor_timestamps"][-1]
    for index, valid in enumerate(mask):
        values = [record[key][index] for key in INPUT_FIELDS[:-1]]
        if not valid:
            if any(value is not None for value in values):
                raise ValueError("padding fields must be null")
            continue
        token, path, timestamp, relative_time, motion = values
        if not isinstance(token, str) or not token or not isinstance(path, str) or not path:
            raise ValueError("real observations require token and image path")
        if (type(timestamp) is not int
                or relative_time != (timestamp - anchor_time) / 1_000_000
                or (real and timestamp <= real[-1][1])):
            raise ValueError("history timestamp alignment mismatch")
        if (motion["timestamp_source"] != "CAM_FRONT_sample_data"
                or motion["availability"] not in ("full", "partial", "unavailable")):
            raise ValueError("ego motion contract mismatch")
        real.append((token, timestamp))
    if len({token for token, _ in real}) != len(real) or real[-1][0] != record["sample_token"]:
        raise ValueError("history tokens must be unique and end at anchor")
    waypoints = record["future_waypoints"]
    if len(waypoints) != 6 or any(len(point) != 2 for point in waypoints):
        raise ValueError("future waypoints must have shape [6,2]")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for point in waypoints for value in point):
        raise ValueError("waypoints must be finite")
    if (len(record["trajectory_valid_mask"]) != 6
            or any(value is not True for value in record["trajectory_valid_mask"])):
        raise ValueError("v0.1 requires six valid future points")

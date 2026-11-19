from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from data.inspect_nuscenes_sample import NuScenesReader, transform_global_point_to_ego


PROTOCOL = "phase0.5-temporal-geometry-v0.1"
CATEGORY_NAMES = {0: "unavailable", 1: "vehicle", 2: "VRU"}
MOTION_SOURCES = {0: "unavailable", 1: "observed"}
ARRAY_FIELDS = (
    "agent_numeric_state", "agent_category_id", "agent_valid_mask", "agent_motion_source",
    "temporal_occupancy", "temporal_occupancy_valid_mask",
)


@dataclass(frozen=True)
class Grid:
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    resolution_m: float

    def __post_init__(self) -> None:
        values = (*self.x_range_m, *self.y_range_m, self.resolution_m)
        if not np.isfinite(values).all() or self.resolution_m <= 0:
            raise ValueError("grid values must be finite with positive resolution")
        for low, high in (self.x_range_m, self.y_range_m):
            cells = (high - low) / self.resolution_m
            if cells <= 0 or not np.isclose(cells, round(cells)):
                raise ValueError("grid extent must contain a positive integer number of cells")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(round((high - low) / self.resolution_m)
                     for low, high in (self.x_range_m, self.y_range_m))

    def metadata(self) -> dict:
        return {
            "x_range_m": self.x_range_m, "y_range_m": self.y_range_m,
            "resolution_m": self.resolution_m, "shape": self.shape,
            "array_axes": ["x_forward", "y_left"],
            "cell_rule": "positive_area_box_cell_intersection",
            "valid_mask_semantics": "annotation-backed grid; not observed free-space coverage",
        }


def category_id(name: str) -> int:
    if name.startswith("human.pedestrian.") or name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return 2
    return 1 if name.startswith("vehicle.") else 0


def box_corners(state: np.ndarray) -> np.ndarray:
    x, y, length, width, yaw = state
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2, width / 2]
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def box_cells(state: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray, str]:
    """Intersect a planar oriented box with cells using the four separating axes."""
    corners = box_corners(state)
    origin = np.array([grid.x_range_m[0], grid.y_range_m[0]])
    upper = np.array([grid.x_range_m[1], grid.y_range_m[1]])
    low, high = corners.min(axis=0), corners.max(axis=0)
    first = np.maximum(0, np.floor((low - origin) / grid.resolution_m).astype(int))
    last = np.minimum(grid.shape, np.ceil((high - origin) / grid.resolution_m).astype(int))
    if np.any(first >= last):
        empty = np.array([], dtype=np.int64)
        return empty, empty, "outside"
    ix, iy = np.meshgrid(np.arange(first[0], last[0]), np.arange(first[1], last[1]), indexing="ij")
    dx = origin[0] + (ix + 0.5) * grid.resolution_m - state[0]
    dy = origin[1] + (iy + 0.5) * grid.resolution_m - state[1]
    half_cell = grid.resolution_m / 2
    half_length, half_width = state[2:4] / 2
    c, s = np.cos(state[4]), np.sin(state[4])
    cell_projection = half_cell * (abs(c) + abs(s))
    overlap = (
        (np.abs(dx) < half_cell + abs(c) * half_length + abs(s) * half_width)
        & (np.abs(dy) < half_cell + abs(s) * half_length + abs(c) * half_width)
        & (np.abs(c * dx + s * dy) < half_length + cell_projection)
        & (np.abs(-s * dx + c * dy) < half_width + cell_projection)
    )
    status = "partial" if np.any(low < origin) or np.any(high > upper) else "inside"
    return ix[overlap], iy[overlap], status if overlap.any() else "outside"


def build_geometry(reader: NuScenesReader, source: dict, grid: Grid, *,
                   max_agents: int, annotation_camera_tolerance_sec: float) -> tuple[dict, dict]:
    if source["split"] not in ("train", "validation"):
        raise ValueError("geometry permits train/validation only")
    anchor = reader.get("sample", source["sample_token"])
    if anchor["scene_token"] != source["scene_token"]:
        raise ValueError("anchor scene differs from source")
    pose = source["current_ego_pose"]
    anchor_camera = reader.get("sample_data", anchor["data"]["CAM_FRONT"])
    if anchor_camera["timestamp"] != pose["timestamp_us"]:
        raise ValueError("anchor camera timestamp differs from source")
    anchor_rotation = Quaternion(pose["rotation_wxyz"])
    frames, categories, sample_tokens, timestamps, camera_offsets = [], {}, [], [], []
    ignored = 0
    for point in source["future_ego_trajectory"][1:7]:
        sample = reader.get("sample", point["future_sample_token"])
        if sample["scene_token"] != source["scene_token"]:
            raise ValueError("future geometry crosses scene boundary")
        if not np.isclose((sample["timestamp"] - anchor["timestamp"]) / 1e6, point["t_sec"]):
            raise ValueError("future sample time differs from frozen trajectory")
        camera = reader.get("sample_data", sample["data"]["CAM_FRONT"])
        offset = (sample["timestamp"] - camera["timestamp"]) / 1e6
        if abs(offset) > annotation_camera_tolerance_sec:
            raise ValueError(f"annotation/camera time mismatch: {sample['token']} ({offset}s)")
        frame = {}
        for token in sample["anns"]:
            annotation = reader.get("sample_annotation", token)
            if annotation["sample_token"] != sample["token"]:
                raise ValueError("annotation belongs to a different sample")
            name = annotation["category_name"]
            category = category_id(name)
            if not category:
                ignored += 1
                continue
            instance = annotation["instance_token"]
            if instance in frame:
                raise ValueError("duplicate instance in one timestamp")
            center = transform_global_point_to_ego(
                tuple(annotation["translation"]), tuple(pose["translation_m"]),
                tuple(pose["rotation_wxyz"]),
            )
            yaw = (anchor_rotation.inverse * Quaternion(annotation["rotation"])).yaw_pitch_roll[0]
            width, length, height = annotation["size"]
            state = [center[0], center[1], length, width, yaw]
            if not np.isfinite(state).all() or min(width, length, height) <= 0:
                raise ValueError(f"invalid annotation geometry: {token}")
            categories[instance] = name
            frame[instance] = (state, category, token)
        frames.append(frame)
        sample_tokens.append(sample["token"])
        timestamps.append(sample["timestamp"])
        camera_offsets.append(offset)
    instances = sorted(categories)
    if len(instances) > max_agents:
        raise ValueError(f"{len(instances)} instances exceed max_agents={max_agents}; no objects dropped")
    slots = {token: i for i, token in enumerate(instances)}
    arrays = {
        "agent_numeric_state": np.zeros((6, max_agents, 5), dtype=np.float32),
        "agent_category_id": np.zeros((6, max_agents), dtype=np.uint8),
        "agent_valid_mask": np.zeros((6, max_agents), dtype=bool),
        "agent_motion_source": np.zeros((6, max_agents), dtype=np.uint8),
        "temporal_occupancy": np.zeros((6, 2, *grid.shape), dtype=bool),
        "temporal_occupancy_valid_mask": np.ones((6, *grid.shape), dtype=bool),
    }
    annotation_tokens = [[None] * max_agents for _ in range(6)]
    counts = {"inside": 0, "partial": 0, "outside": 0, "ignored": ignored}
    for t, frame in enumerate(frames):
        for instance, (state, category, token) in frame.items():
            i = slots[instance]
            arrays["agent_numeric_state"][t, i] = state
            arrays["agent_category_id"][t, i] = category
            arrays["agent_valid_mask"][t, i] = True
            arrays["agent_motion_source"][t, i] = 1
            annotation_tokens[t][i] = token
            x, y, status = box_cells(arrays["agent_numeric_state"][t, i], grid)
            arrays["temporal_occupancy"][t, category - 1, x, y] = True
            counts[status] += 1
    metadata = {
        "protocol_version": PROTOCOL, "information_role": "evaluator_only_gt",
        "sample_token": source["sample_token"], "scene_token": source["scene_token"],
        "split": source["split"], "anchor_timestamp_us": pose["timestamp_us"],
        "anchor_pose": pose, "coordinate_metadata": source["coordinate_metadata"]["future_ego_trajectory"],
        "source_sample_tokens": sample_tokens, "source_timestamps_us": timestamps,
        "future_relative_times_sec": [p["t_sec"] for p in source["future_ego_trajectory"][1:7]],
        "time_policy": "frozen sample grid; observed boxes; no interpolation",
        "annotation_camera_offsets_sec": camera_offsets,
        "annotation_camera_tolerance_sec": annotation_camera_tolerance_sec,
        "instance_tokens": instances + [None] * (max_agents - len(instances)),
        "category_names": [categories[token] for token in instances] + [None] * (max_agents - len(instances)),
        "annotation_tokens": annotation_tokens, "category_codes": CATEGORY_NAMES,
        "motion_source_codes": MOTION_SOURCES, "instance_count": len(instances),
        "numeric_fields": ["center_x_m", "center_y_m", "length_m", "width_m", "yaw_rad"],
        "box_projection": "planar length/width rectangle with relative quaternion yaw",
        "grid": grid.metadata(), "object_counts": counts,
        "future_waypoints": [[p["x_m"], p["y_m"]] for p in source["future_ego_trajectory"][1:7]],
        "trajectory_valid_mask": [True] * 6,
        "source_manifest_sha256": source["source_combined_manifest_sha256"],
        "split_mapping_sha256": source["split_mapping_sha256"],
    }
    return arrays, metadata


def save_geometry(path: Path, arrays: dict, metadata: dict) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata)))


def load_geometry(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ARRAY_FIELDS}
        metadata = json.loads(str(saved["metadata"]))
    if metadata["protocol_version"] != PROTOCOL or metadata["information_role"] != "evaluator_only_gt":
        raise ValueError("unsupported geometry artifact")
    return arrays, metadata

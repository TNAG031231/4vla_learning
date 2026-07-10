#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import sys

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DATA_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DATA_DIRECTORY))

from derive_meta_action import (
    _load_data_config,
    derive_sample_record,
    load_meta_action_rules,
)
from inspect_nuscenes_sample import (
    CAMERA_CHANNEL,
    FutureEgoTrajectory,
    NearbyAgents,
    get_ego_pose,
    get_nearby_agents,
    extract_future_ego_trajectory,
    normalize_angle,
)
from validate_label_freeze import (
    EXPECTED_LABEL_RULE_VERSION,
    HistoricalAuditRow,
    read_and_merge_audits,
    validate_cam_front_path,
    validate_historical_audit_integrity,
)
from src.actions.schema import normalize_action
from src.phase0.protocol import (
    ManifestSample,
    ACCELERATION_UNIT,
    MANIFEST_SCHEMA_VERSION,
    MOTION_SOURCE,
    POSE_TIMESTAMP_SOURCE,
    POSE_TIMESTAMP_UNIT,
    SPEED_UNIT,
    YAW_RATE_UNIT,
    assign_scene_splits,
    complete_action_distribution,
    validate_manifest,
    validate_scene_split_isolation,
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


@dataclass(frozen=True)
class SourceAuditRecord:
    source_audit: str
    sample_token: str
    historical_derived_action: str
    reviewed_action: str
    label_correct: str
    historical_label_rule_version: str


@dataclass(frozen=True)
class Phase0ManifestRecord:
    sample_token: str
    scene_token: str
    timestamp: int
    cam_front_path: str
    current_ego_pose: dict[str, object]
    current_ego_motion: dict[str, object]
    coordinate_metadata: dict[str, dict[str, str]]
    future_ego_trajectory: tuple[object, ...]
    nearby_agents: tuple[object, ...]
    meta_action: str
    label_rule_version: str
    safety_rule_version: str
    manifest_schema_version: str
    split: str
    source_audit_record: SourceAuditRecord


def _json_compatible(value: object) -> object:
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _json_compatible(attributes)
    raise TypeError(f"Cannot serialize manifest value: {type(value)!r}")


def build_manifest_record(
    audit_row: HistoricalAuditRow,
    derived_record: object,
    current_ego_pose: dict[str, object],
    current_ego_motion: dict[str, object],
    trajectory: FutureEgoTrajectory,
    nearby_agents: NearbyAgents,
    split: str,
) -> Phase0ManifestRecord:
    sample_token = str(getattr(derived_record, "sample_token"))
    scene_token = str(getattr(derived_record, "scene_token"))
    timestamp = int(getattr(derived_record, "timestamp_us"))
    cam_front_path = str(getattr(derived_record, "cam_front_path"))
    meta_action = normalize_action(str(getattr(derived_record, "derived_action")))
    label_rule_version = str(getattr(derived_record, "label_rule_version"))
    if sample_token != audit_row.sample_token:
        raise ValueError(f"{audit_row.sample_token}: sample_token mismatch")
    if scene_token != audit_row.scene_token:
        raise ValueError(f"{audit_row.sample_token}: scene_token mismatch")
    if timestamp != int(audit_row.timestamp):
        raise ValueError(f"{audit_row.sample_token}: timestamp mismatch")
    if cam_front_path != audit_row.cam_front_path:
        raise ValueError(f"{audit_row.sample_token}: cam_front_path mismatch")
    if meta_action != audit_row.reviewed_action:
        raise ValueError(f"{audit_row.sample_token}: frozen action mismatch")
    if label_rule_version != EXPECTED_LABEL_RULE_VERSION:
        raise ValueError(f"{audit_row.sample_token}: label_rule_version mismatch")
    return Phase0ManifestRecord(
        sample_token=sample_token,
        scene_token=scene_token,
        timestamp=timestamp,
        cam_front_path=cam_front_path,
        current_ego_pose=current_ego_pose,
        current_ego_motion=current_ego_motion,
        coordinate_metadata=COORDINATE_METADATA,
        future_ego_trajectory=tuple(trajectory.points),
        nearby_agents=tuple(nearby_agents.agents),
        meta_action=meta_action,
        label_rule_version=label_rule_version,
        safety_rule_version=SAFETY_RULE_VERSION,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        split=split,
        source_audit_record=SourceAuditRecord(
            source_audit=audit_row.source_audit,
            sample_token=audit_row.sample_token,
            historical_derived_action=audit_row.historical_derived_action,
            reviewed_action=audit_row.reviewed_action,
            label_correct=audit_row.label_correct,
            historical_label_rule_version=audit_row.label_rule_version,
        ),
    )


def to_json_record(record: Phase0ManifestRecord) -> dict[str, object]:
    payload = _json_compatible(record)
    if not isinstance(payload, dict):
        raise TypeError("manifest record must serialize to an object")
    return payload


def current_ego_pose(nuscenes: NuScenes, sample_token: str) -> dict[str, object]:
    sample = nuscenes.get("sample", sample_token)
    translation, rotation, timestamp_us = _sample_pose(nuscenes, sample)
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


def _sample_pose(
    nuscenes: NuScenes,
    sample: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    sample_data = sample.get("data")
    if not isinstance(sample_data, Mapping):
        raise ValueError("sample data must be a mapping")
    camera_token = sample_data.get(CAMERA_CHANNEL)
    if not isinstance(camera_token, str):
        raise ValueError("sample is missing CAM_FRONT data")
    camera_data = nuscenes.get("sample_data", camera_token)
    timestamp_us = camera_data.get("timestamp")
    if not isinstance(timestamp_us, int):
        raise ValueError("CAM_FRONT sample_data timestamp must be an integer")
    translation, rotation = get_ego_pose(nuscenes, sample)
    return translation, rotation, timestamp_us


def _previous_sample(
    nuscenes: NuScenes,
    sample: Mapping[str, object],
) -> Mapping[str, object] | None:
    previous_token = sample.get("prev")
    if not isinstance(previous_token, str) or not previous_token:
        return None
    previous_sample = nuscenes.get("sample", previous_token)
    scene_token = sample.get("scene_token")
    if previous_sample.get("scene_token") != scene_token:
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


def current_ego_motion(
    nuscenes: NuScenes,
    sample_token: str,
) -> dict[str, object]:
    current_sample = nuscenes.get("sample", sample_token)
    current_translation, current_rotation, current_timestamp_us = _sample_pose(
        nuscenes,
        current_sample,
    )
    previous_sample = _previous_sample(nuscenes, current_sample)
    if previous_sample is None:
        previous_token = current_sample.get("prev")
        if isinstance(previous_token, str) and previous_token:
            return _unavailable_motion("previous_sample_scene_mismatch")
        return _unavailable_motion("insufficient_past_history")

    previous_translation, previous_rotation, previous_timestamp_us = _sample_pose(
        nuscenes,
        previous_sample,
    )
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
    yaw_rate_radps = normalize_angle(current_yaw - previous_yaw) / interval_sec

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
    previous_previous_translation, _, previous_previous_timestamp_us = _sample_pose(
        nuscenes,
        previous_previous_sample,
    )
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


def build_manifest_records(
    nuscenes: NuScenes,
    audit_rows: Sequence[HistoricalAuditRow],
    scene_splits: Mapping[str, str],
    dataroot: Path,
    action_config_path: Path,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> tuple[Phase0ManifestRecord, ...]:
    rules = load_meta_action_rules(action_config_path)
    records = []
    for audit_row in audit_rows:
        derived_record = derive_sample_record(
            nuscenes=nuscenes,
            sample_token=audit_row.sample_token,
            camera=CAMERA_CHANNEL,
            rules=rules,
            time_tolerance_sec=time_tolerance_sec,
        )
        if not validate_cam_front_path(derived_record.cam_front_path, dataroot):
            raise ValueError(f"{audit_row.sample_token}: CAM_FRONT path is invalid")
        trajectory = extract_future_ego_trajectory(
            nuscenes=nuscenes,
            sample_token=audit_row.sample_token,
            horizon_sec=rules.horizon_sec,
            sample_interval_sec=rules.sample_interval_sec,
            time_tolerance_sec=time_tolerance_sec,
        )
        if trajectory.is_truncated:
            raise ValueError(f"{audit_row.sample_token}: future trajectory is truncated")
        nearby_agents = get_nearby_agents(
            nuscenes=nuscenes,
            sample_token=audit_row.sample_token,
            radius_m=agent_radius_m,
        )
        records.append(
            build_manifest_record(
                audit_row=audit_row,
                derived_record=derived_record,
                current_ego_pose=current_ego_pose(
                    nuscenes,
                    audit_row.sample_token,
                ),
                current_ego_motion=current_ego_motion(
                    nuscenes,
                    audit_row.sample_token,
                ),
                trajectory=trajectory,
                nearby_agents=nearby_agents,
                split=scene_splits[audit_row.scene_token],
            )
        )
    return tuple(records)


def write_manifest(
    records: Sequence[Phase0ManifestRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(to_json_record(record)) + "\n")


def _required_config_string(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _required_config_float(config: Mapping[str, object], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"configuration missing {key}")
    return float(value)


def _load_config(config_path: Path) -> Mapping[str, object]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    return loaded


def _print_audit(records: Sequence[Phase0ManifestRecord]) -> None:
    samples = tuple(
        ManifestSample(
            sample_token=record.sample_token,
            scene_token=record.scene_token,
            meta_action=record.meta_action,
            split=record.split,
            label_rule_version=record.label_rule_version,
        )
        for record in records
    )
    validate_scene_split_isolation(samples)
    print(f"audited Phase 0 seed subset samples: {len(samples)}")
    print(
        "motion availability: "
        f"{dict(Counter(record.current_ego_motion['availability'] for record in records))}"
    )
    print(
        "label_rule_versions: "
        f"{sorted({record.label_rule_version for record in records})}"
    )
    print(
        "manifest_schema_versions: "
        f"{sorted({record.manifest_schema_version for record in records})}"
    )
    for split in ("train", "validation", "test"):
        split_samples = tuple(sample for sample in samples if sample.split == split)
        print(f"{split} scenes: {len({sample.scene_token for sample in split_samples})}")
        print(f"{split} samples: {len(split_samples)}")
        print(
            f"{split} class_distribution: "
            f"{complete_action_distribution(tuple(sample.meta_action for sample in split_samples))}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited Phase 0 seed subset manifest."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    config = _load_config(arguments.config)
    data_config_path = Path(_required_config_string(config, "data_config_path"))
    dataroot = Path(_required_config_string(config, "dataroot"))
    data_config = _load_data_config(data_config_path, dataroot)
    audit_rows = read_and_merge_audits(
        Path(_required_config_string(config, "base_audit_path")),
        Path(_required_config_string(config, "supplement_audit_path")),
    )
    validate_historical_audit_integrity(audit_rows)
    scene_splits = assign_scene_splits(
        scene_tokens=tuple(sorted({row.scene_token for row in audit_rows})),
        seed=int(_required_config_float(config, "seed")),
        train_ratio=_required_config_float(config, "train_ratio"),
        val_ratio=_required_config_float(config, "val_ratio"),
        test_ratio=_required_config_float(config, "test_ratio"),
    )
    nuscenes = NuScenes(
        version=data_config.version,
        dataroot=str(data_config.nuscenes_root),
        verbose=False,
    )
    records = build_manifest_records(
        nuscenes=nuscenes,
        audit_rows=audit_rows,
        scene_splits=scene_splits,
        dataroot=data_config.nuscenes_root,
        action_config_path=Path(
            _required_config_string(config, "action_config_path")
        ),
        time_tolerance_sec=data_config.trajectory_time_tolerance_sec,
        agent_radius_m=data_config.nearby_radius_m,
    )
    _print_audit(records)
    output_path = Path(_required_config_string(config, "manifest_path"))
    write_manifest(records, output_path)
    validation = validate_manifest(output_path)
    if validation.sample_count != len(records):
        raise ValueError("written manifest sample count does not match records")
    print(f"manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

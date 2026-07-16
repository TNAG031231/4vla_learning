import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import build_phase0_manifest as manifest_builder
from src.phase0.protocol import read_manifest_samples


class FakeNuScenes:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, dict[str, object]]] = {
            "sample": {},
            "sample_data": {},
            "ego_pose": {},
        }

    def get(self, table_name: str, token: str) -> dict[str, object]:
        return self.tables[table_name][token]


def rotation_from_yaw(yaw_rad: float) -> list[float]:
    return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]


def build_motion_reader(
    previous_scene_token: str = "scene-a",
    previous_yaw_rad: float = 0.0,
    current_yaw_rad: float = 0.0,
    sample_timestamps: tuple[int, ...] = (
        0,
        1_000_000,
        2_000_000,
        3_000_000,
    ),
    camera_timestamps: tuple[int, ...] = (
        0,
        1_000_000,
        2_000_000,
        3_000_000,
    ),
    x_positions: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
) -> FakeNuScenes:
    reader = FakeNuScenes()
    sample_specs = (
        ("start", "", "scene-a", 0, 0.0, 0.0),
        ("previous", "start", previous_scene_token, 1, 1.0, previous_yaw_rad),
        ("current", "previous", "scene-a", 2, 2.0, current_yaw_rad),
        ("future", "current", "scene-a", 3, 3.0, 0.0),
    )
    for token, previous_token, scene_token, index, _, yaw_rad in sample_specs:
        camera_token = f"camera-{token}"
        pose_token = f"pose-{token}"
        reader.tables["sample"][token] = {
            "token": token,
            "scene_token": scene_token,
            "timestamp": sample_timestamps[index],
            "prev": previous_token,
            "next": "future" if token == "current" else "",
            "data": {"CAM_FRONT": camera_token},
        }
        reader.tables["sample_data"][camera_token] = {
            "ego_pose_token": pose_token,
            "timestamp": camera_timestamps[index],
        }
        reader.tables["ego_pose"][pose_token] = {
            "translation": [x_positions[index], 0.0, 0.0],
            "rotation": rotation_from_yaw(yaw_rad),
            "timestamp": camera_timestamps[index],
        }
    return reader


def test_manifest_record_preserves_phase_minus_one_source_and_required_fields() -> None:
    audit_row = SimpleNamespace(
        source_audit="base",
        sample_token="sample-a",
        scene_token="scene-a",
        timestamp="10",
        cam_front_path="samples/CAM_FRONT/a.jpg",
        historical_derived_action="keep",
        reviewed_action="keep",
        label_correct="yes",
        trajectory_alignment_correct="yes",
        agent_alignment_correct="yes",
        label_rule_version="phase-1.6-meta-action-v0.1",
    )
    derived = SimpleNamespace(
        sample_token="sample-a",
        scene_token="scene-a",
        timestamp_us=10,
        cam_front_path="samples/CAM_FRONT/a.jpg",
        derived_action="keep",
        label_rule_version="phase-1.6-meta-action-v0.2",
    )
    trajectory = SimpleNamespace(points=(SimpleNamespace(t_sec=0.0),))
    agents = SimpleNamespace(agents=(SimpleNamespace(instance_token="agent-a"),))

    record = manifest_builder.build_manifest_record(
        audit_row=audit_row,
        derived_record=derived,
        current_ego_pose={
            "frame": "nuScenes_global",
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "timestamp_us": 10,
            "timestamp_source": "CAM_FRONT_sample_data",
        },
        current_ego_motion={
            "speed_mps": None,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": None,
            "source": "ego_pose_past_difference",
            "timestamp_source": "CAM_FRONT_sample_data",
            "availability": "unavailable",
            "history_interval_sec": None,
            "acceleration_interval_sec": None,
            "unavailable_reason": "insufficient_past_history",
        },
        trajectory=trajectory,
        nearby_agents=agents,
        split="train",
    )

    payload = manifest_builder.to_json_record(record)

    assert set(payload) >= {
        "sample_token",
        "scene_token",
        "timestamp",
        "cam_front_path",
        "current_ego_pose",
        "current_ego_motion",
        "future_ego_trajectory",
        "nearby_agents",
        "meta_action",
        "label_rule_version",
        "safety_rule_version",
        "split",
        "source_audit_record",
        "coordinate_metadata",
        "manifest_schema_version",
    }
    assert "current_ego_state" not in payload
    assert payload["meta_action"] == "keep"
    assert payload["source_audit_record"]["source_audit"] == "base"
    assert payload["source_audit_record"]["sample_token"] == "sample-a"
    assert payload["future_ego_trajectory"] == [{"t_sec": 0.0}]
    assert payload["nearby_agents"] == [{"instance_token": "agent-a"}]
    assert payload["coordinate_metadata"]["future_ego_trajectory"] == {
        "source_frame": "nuScenes_global",
        "target_frame": "current_ego",
        "x_axis": "forward",
        "y_axis": "left",
        "z_axis": "up",
        "unit": "meter",
        "transform": "subtract_current_ego_translation_then_apply_inverse_current_ego_rotation",
    }
    assert payload["manifest_schema_version"] == "phase0_audited_seed_subset_v1"


def test_manifest_reader_rejects_unknown_action(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"sample_token": "sample", "scene_token": "scene", "meta_action": "turn_left", "split": "train", "label_rule_version": "v0"}\n',
        encoding="utf-8",
    )

    try:
        read_manifest_samples(manifest_path)
    except ValueError as error:
        assert "Unsupported action" in str(error)
    else:
        raise AssertionError("unknown meta_action must be rejected")


def test_current_ego_pose_and_motion_use_past_samples_only() -> None:
    reader = build_motion_reader()

    pose = manifest_builder.current_ego_pose(reader, "current")
    motion = manifest_builder.current_ego_motion(reader, "current")

    assert pose == {
        "frame": "nuScenes_global",
        "translation_m": [2.0, 0.0, 0.0],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "timestamp_us": 2_000_000,
        "timestamp_source": "CAM_FRONT_sample_data",
    }
    assert motion == {
        "speed_mps": pytest.approx(1.0),
        "longitudinal_acceleration_mps2": pytest.approx(0.0),
        "yaw_rate_radps": pytest.approx(0.0),
        "source": "ego_pose_past_difference",
        "timestamp_source": "CAM_FRONT_sample_data",
        "availability": "full",
        "history_interval_sec": pytest.approx(1.0),
        "acceleration_interval_sec": pytest.approx(1.0),
        "unavailable_reason": None,
    }


def test_current_ego_motion_normalizes_yaw_wraparound() -> None:
    reader = build_motion_reader(
        previous_yaw_rad=3.13,
        current_yaw_rad=-3.13,
    )

    motion = manifest_builder.current_ego_motion(reader, "current")

    assert motion["yaw_rate_radps"] == pytest.approx(math.tau - 6.26)


def test_scene_start_motion_is_unavailable_without_future_pose() -> None:
    reader = build_motion_reader()

    motion = manifest_builder.current_ego_motion(reader, "start")

    assert motion == {
        "speed_mps": None,
        "longitudinal_acceleration_mps2": None,
        "yaw_rate_radps": None,
        "source": "ego_pose_past_difference",
        "timestamp_source": "CAM_FRONT_sample_data",
        "availability": "unavailable",
        "history_interval_sec": None,
        "acceleration_interval_sec": None,
        "unavailable_reason": "insufficient_past_history",
    }


def test_motion_without_two_past_intervals_is_partial() -> None:
    reader = build_motion_reader()

    motion = manifest_builder.current_ego_motion(reader, "previous")

    assert motion["speed_mps"] == pytest.approx(1.0)
    assert motion["longitudinal_acceleration_mps2"] is None
    assert motion["yaw_rate_radps"] == pytest.approx(0.0)
    assert motion["availability"] == "partial"
    assert motion["acceleration_interval_sec"] is None
    assert motion["unavailable_reason"] == "insufficient_past_history_for_acceleration"


def test_motion_with_missing_past_pose_is_unavailable() -> None:
    reader = build_motion_reader()
    del reader.tables["ego_pose"]["pose-previous"]

    motion = manifest_builder.current_ego_motion(reader, "current")

    assert motion["availability"] == "unavailable"
    assert motion["unavailable_reason"] == "past_ego_pose_unavailable"


def test_motion_does_not_cross_scenes() -> None:
    reader = build_motion_reader(previous_scene_token="scene-b")

    motion = manifest_builder.current_ego_motion(reader, "current")

    assert motion["availability"] == "unavailable"
    assert motion["unavailable_reason"] == "previous_sample_scene_mismatch"


def test_future_pose_change_does_not_affect_current_motion() -> None:
    reader = build_motion_reader()

    before_future_change = manifest_builder.current_ego_motion(reader, "current")
    reader.tables["ego_pose"]["pose-future"]["translation"] = [999.0, 0.0, 0.0]
    after_future_change = manifest_builder.current_ego_motion(reader, "current")

    assert before_future_change == after_future_change
    assert before_future_change["availability"] == "full"


def test_motion_uses_cam_front_timestamp_instead_of_sample_timestamp() -> None:
    reader = build_motion_reader(
        sample_timestamps=(0, 10_000_000, 30_000_000, 60_000_000),
    )

    pose = manifest_builder.current_ego_pose(reader, "current")
    motion = manifest_builder.current_ego_motion(reader, "current")

    assert pose["timestamp_us"] == 2_000_000
    assert pose["timestamp_source"] == "CAM_FRONT_sample_data"
    assert motion["history_interval_sec"] == pytest.approx(1.0)
    assert motion["speed_mps"] == pytest.approx(1.0)


def test_acceleration_uses_midpoint_interval_for_nonuniform_sampling() -> None:
    reader = build_motion_reader(
        camera_timestamps=(0, 1_000_000, 3_000_000, 6_000_000),
        x_positions=(0.0, 1.0, 5.0, 11.0),
    )

    motion = manifest_builder.current_ego_motion(reader, "current")

    assert motion["speed_mps"] == pytest.approx(2.0)
    assert motion["acceleration_interval_sec"] == pytest.approx(1.5)
    assert motion["longitudinal_acceleration_mps2"] == pytest.approx(2 / 3)

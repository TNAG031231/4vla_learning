import importlib.util
import math
from pathlib import Path
import sys

import pytest
from pyquaternion import Quaternion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSPECTION_SCRIPT = PROJECT_ROOT / "data" / "inspect_nuscenes_sample.py"


def load_inspection_module():
    assert INSPECTION_SCRIPT.is_file(), "data/inspect_nuscenes_sample.py is missing"
    specification = importlib.util.spec_from_file_location(
        "inspect_nuscenes_sample",
        INSPECTION_SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class FakeNuScenes:
    def __init__(self) -> None:
        self.scene = [{"first_sample_token": "current"}]
        self.records = {
            ("sample", "current"): {
                "token": "current",
                "scene_token": "scene",
                "timestamp": 1_000_000,
                "next": "future",
                "data": {"CAM_FRONT": "current_camera"},
            },
            ("sample", "future"): {
                "token": "future",
                "scene_token": "scene",
                "timestamp": 1_500_000,
                "next": "",
                "data": {"CAM_FRONT": "future_camera"},
            },
            ("sample_data", "current_camera"): {
                "ego_pose_token": "current_pose",
            },
            ("sample_data", "future_camera"): {
                "ego_pose_token": "future_pose",
            },
            ("ego_pose", "current_pose"): {
                "translation": [10.0, 20.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            ("ego_pose", "future_pose"): {
                "translation": [11.0, 20.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        }

    def get(self, table_name: str, token: str) -> dict[str, object]:
        return self.records[(table_name, token)]


class FakeNearbyNuScenes:
    def __init__(self, annotation_tokens: list[str]) -> None:
        self.scene = [{"first_sample_token": "agents"}]
        self.records = {
            ("sample", "agents"): {
                "token": "agents",
                "scene_token": "scene",
                "timestamp": 1_000_000,
                "next": "",
                "data": {"CAM_FRONT": "agents_camera"},
                "anns": annotation_tokens,
            },
            ("sample_data", "agents_camera"): {
                "ego_pose_token": "agents_pose",
            },
            ("ego_pose", "agents_pose"): {
                "translation": [10.0, 20.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            ("sample_annotation", "near"): {
                "token": "near",
                "instance_token": "near_instance",
                "category_name": "vehicle.car",
                "translation": [20.0, 20.0, 0.0],
                "size": [2.0, 4.5, 1.7],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "num_lidar_pts": 12,
                "num_radar_pts": 3,
            },
            ("sample_annotation", "far"): {
                "token": "far",
                "instance_token": "far_instance",
                "category_name": "human.pedestrian.adult",
                "translation": [70.0, 20.0, 0.0],
                "size": [0.7, 0.8, 1.8],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "num_lidar_pts": 4,
                "num_radar_pts": 0,
            },
        }

    def get(self, table_name: str, token: str) -> dict[str, object]:
        return self.records[(table_name, token)]


def test_global_displacement_is_rotated_into_current_ego_frame() -> None:
    inspection = load_inspection_module()
    current_rotation = Quaternion(
        axis=[0.0, 0.0, 1.0],
        angle=math.pi / 2.0,
    )

    pose = inspection.transform_pose_to_current_ego_frame(
        current_translation=(10.0, 20.0, 0.0),
        current_rotation=tuple(current_rotation.elements),
        future_translation=(10.0, 21.0, 0.0),
        future_rotation=tuple(current_rotation.elements),
    )

    assert pose.x_m == pytest.approx(1.0)
    assert pose.y_m == pytest.approx(0.0, abs=1e-9)
    assert pose.heading_delta_rad == pytest.approx(0.0, abs=1e-9)


def test_current_pose_is_origin_in_current_ego_frame() -> None:
    inspection = load_inspection_module()
    pose = inspection.transform_pose_to_current_ego_frame(
        current_translation=(10.0, 20.0, 0.0),
        current_rotation=(1.0, 0.0, 0.0, 0.0),
        future_translation=(10.0, 20.0, 0.0),
        future_rotation=(1.0, 0.0, 0.0, 0.0),
    )

    assert pose.x_m == pytest.approx(0.0)
    assert pose.y_m == pytest.approx(0.0)
    assert pose.heading_delta_rad == pytest.approx(0.0)


def test_scene_tail_returns_available_points_and_marks_truncated() -> None:
    inspection = load_inspection_module()

    trajectory = inspection.extract_future_ego_trajectory(
        nuscenes=FakeNuScenes(),
        sample_token="current",
        horizon_sec=1.0,
        sample_interval_sec=0.5,
    )

    assert [point.future_sample_token for point in trajectory.points] == [
        "current",
        "future",
    ]
    assert trajectory.points[0].x_m == pytest.approx(0.0)
    assert trajectory.points[0].y_m == pytest.approx(0.0)
    assert trajectory.is_truncated is True


def test_nuscenes_microseconds_are_converted_to_seconds() -> None:
    inspection = load_inspection_module()

    assert inspection.microseconds_to_seconds(500_000) == pytest.approx(0.5)


def test_config_expands_nuscenes_root_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = load_inspection_module()
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "\n".join(
            (
                "data:",
                '  nuscenes_root: "${NUSCENES_ROOT}"',
                '  version: "v1.0-mini"',
                "phase1:",
                "  horizon_sec: 3.0",
                "  sample_interval_sec: 0.5",
                "  max_agent_distance_m: 50.0",
            )
        )
    )
    monkeypatch.setenv("NUSCENES_ROOT", "data/nuscenes")

    config = inspection.load_trajectory_config(config_path)

    assert config.nuscenes_root == Path("data/nuscenes")
    assert config.version == "v1.0-mini"
    assert config.horizon_sec == pytest.approx(3.0)
    assert config.sample_interval_sec == pytest.approx(0.5)
    assert config.nearby_radius_m == pytest.approx(50.0)


def test_global_point_is_transformed_into_current_ego_frame() -> None:
    inspection = load_inspection_module()
    current_rotation = Quaternion(
        axis=[0.0, 0.0, 1.0],
        angle=math.pi / 2.0,
    )

    point = inspection.transform_global_point_to_ego(
        point_global=(10.0, 21.0, 0.0),
        ego_translation_global=(10.0, 20.0, 0.0),
        ego_rotation_global=tuple(current_rotation.elements),
    )

    assert point == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_point_near_ego_origin_preserves_expected_offset() -> None:
    inspection = load_inspection_module()

    point = inspection.transform_global_point_to_ego(
        point_global=(10.2, 19.8, 0.1),
        ego_translation_global=(10.0, 20.0, 0.0),
        ego_rotation_global=(1.0, 0.0, 0.0, 0.0),
    )

    assert point == pytest.approx((0.2, -0.2, 0.1))


def test_relative_yaw_is_normalized_to_pi_interval() -> None:
    inspection = load_inspection_module()

    yaw_ego_rad = inspection.transform_global_yaw_to_ego(
        yaw_global_rad=math.radians(-170.0),
        ego_yaw_global_rad=math.radians(170.0),
    )

    assert yaw_ego_rad == pytest.approx(math.radians(20.0))
    assert -math.pi <= yaw_ego_rad <= math.pi


def test_radius_filter_keeps_near_agent_and_excludes_far_agent() -> None:
    inspection = load_inspection_module()

    result = inspection.get_nearby_agents(
        nuscenes=FakeNearbyNuScenes(["near", "far"]),
        sample_token="agents",
        radius_m=50.0,
    )

    assert [agent.annotation_token for agent in result.agents] == ["near"]
    assert result.agents[0].translation_ego == pytest.approx((10.0, 0.0, 0.0))
    assert result.agents[0].distance_xy_m == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("category_name", "expected"),
    (
        ("vehicle.car", (True, False)),
        ("vehicle.bicycle", (True, True)),
        ("vehicle.motorcycle", (True, True)),
        ("human.pedestrian.adult", (False, True)),
        ("movable_object.barrier", (False, False)),
    ),
)
def test_agent_category_classification(
    category_name: str,
    expected: tuple[bool, bool],
) -> None:
    inspection = load_inspection_module()

    assert inspection.classify_agent_category(category_name) == expected


def test_empty_annotations_return_empty_nearby_agents() -> None:
    inspection = load_inspection_module()

    result = inspection.get_nearby_agents(
        nuscenes=FakeNearbyNuScenes([]),
        sample_token="agents",
        radius_m=50.0,
    )

    assert result.agents == ()

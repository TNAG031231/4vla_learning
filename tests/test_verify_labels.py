import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
VISUALIZATION_SCRIPT = DATA_DIRECTORY / "verify_labels.py"


def load_visualization_module() -> ModuleType:
    assert VISUALIZATION_SCRIPT.is_file(), "data/verify_labels.py is missing"
    specification = importlib.util.spec_from_file_location(
        "verify_labels",
        VISUALIZATION_SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(DATA_DIRECTORY))
    try:
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    finally:
        sys.path.remove(str(DATA_DIRECTORY))
    return module


def test_resolve_output_path_uses_sample_token_and_explicit_override() -> None:
    visualization = load_visualization_module()

    default_path = visualization.resolve_output_path(
        sample_token="sample-123",
        output=None,
        output_dir=Path("outputs/phase1_visualizations"),
    )
    explicit_path = visualization.resolve_output_path(
        sample_token="sample-123",
        output=Path("custom/result.png"),
        output_dir=Path("outputs/phase1_visualizations"),
    )

    assert default_path == Path(
        "outputs/phase1_visualizations/sample-123_one_page.png"
    )
    assert explicit_path == Path("custom/result.png")


@pytest.mark.parametrize(
    ("category_name", "expected"),
    (
        ("vehicle.car", "vehicle"),
        ("human.pedestrian.adult", "pedestrian"),
        ("vehicle.bicycle", "bicycle"),
        ("vehicle.motorcycle", "motorcycle"),
        ("movable_object.barrier", "other"),
    ),
)
def test_agent_display_category_mapping(
    category_name: str,
    expected: str,
) -> None:
    visualization = load_visualization_module()

    assert visualization.agent_display_category(category_name) == expected


def test_bev_limits_cover_radius_and_all_points() -> None:
    visualization = load_visualization_module()

    limits = visualization.calculate_bev_limits(
        trajectory_xy=((0.0, 0.0), (12.0, -3.0)),
        agent_xy=((-4.0, 9.0),),
        radius_m=10.0,
    )

    assert limits.x_min <= -10.0
    assert limits.x_max >= 12.0
    assert limits.y_min <= -10.0
    assert limits.y_max >= 10.0


def test_empty_summary_and_render_do_not_crash() -> None:
    visualization = load_visualization_module()
    payload = visualization.VisualizationPayload(
        sample_token="sample",
        scene_token="scene",
        scene_name="scene-name",
        current_timestamp=1_000_000,
        camera="CAM_FRONT",
        cam_front_path="samples/CAM_FRONT/image.jpg",
        trajectory=(),
        agents=(),
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        max_agent_distance_m=50.0,
        meta_action="unavailable",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )

    summary = visualization.build_sanity_summary(
        payload.trajectory,
        payload.agents,
    )
    figure = visualization.render_one_page_visualization(
        payload=payload,
        image=Image.new("RGB", (64, 48), color="gray"),
    )

    assert summary.trajectory_first is None
    assert summary.trajectory_last is None
    assert summary.nearest_agent_distance_m is None
    assert summary.trajectory_empty is True
    assert summary.agents_empty is True
    assert len(figure.axes) == 4


def test_summary_contains_trajectory_ranges_and_nearest_agent() -> None:
    visualization = load_visualization_module()
    trajectory = (
        visualization.TrajectoryPoint(
            future_sample_token="start",
            t_sec=0.0,
            x_m=0.0,
            y_m=0.0,
            heading_delta_rad=0.0,
        ),
        visualization.TrajectoryPoint(
            future_sample_token="end",
            t_sec=1.0,
            x_m=5.0,
            y_m=-2.0,
            heading_delta_rad=0.1,
        ),
    )
    agents = (
        visualization.NearbyAgent(
            annotation_token="near",
            instance_token="instance",
            category_name="vehicle.car",
            is_vehicle=True,
            is_vru=False,
            translation_ego=(3.0, 4.0, 0.0),
            size=(2.0, 4.0, 1.5),
            yaw_ego_rad=0.0,
            distance_xy_m=5.0,
            num_lidar_pts=1,
            num_radar_pts=0,
        ),
    )

    summary = visualization.build_sanity_summary(trajectory, agents)

    assert summary.trajectory_first == pytest.approx((0.0, 0.0))
    assert summary.trajectory_last == pytest.approx((5.0, -2.0))
    assert summary.min_x_m == pytest.approx(0.0)
    assert summary.max_x_m == pytest.approx(5.0)
    assert summary.min_y_m == pytest.approx(-2.0)
    assert summary.max_y_m == pytest.approx(0.0)
    assert summary.nearest_agent_distance_m == pytest.approx(5.0)
    assert summary.nearest_agent_category == "vehicle.car"

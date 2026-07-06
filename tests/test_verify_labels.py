import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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


def test_review_record_contains_stable_phase_1_5_schema() -> None:
    visualization = load_visualization_module()

    record = visualization.create_review_record(
        sample_token="sample",
        scene_token="scene",
        timestamp=1_000_000,
        cam_front_path="samples/CAM_FRONT/image.jpg",
        visualization_path="visualizations/sample_one_page.png",
        selection_reason="has_nearby_agents",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )

    assert tuple(asdict(record)) == visualization.REVIEW_RECORD_FIELDS
    assert record.derived_action == "not_available"
    assert record.reviewed_action == ""
    assert record.label_correct == "uncertain"
    assert record.trajectory_alignment_correct == "uncertain"
    assert record.agent_alignment_correct == "uncertain"
    assert record.visualization_sufficient == "uncertain"
    assert record.safety_score_reasonable == "not_available"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("label_correct", "maybe"),
        ("trajectory_alignment_correct", "not_available"),
        ("agent_alignment_correct", "not_applicable"),
        ("visualization_sufficient", "not_available"),
        ("safety_score_reasonable", "not_applicable"),
    ),
)
def test_review_record_rejects_invalid_enum_values(
    field_name: str,
    invalid_value: str,
) -> None:
    visualization = load_visualization_module()
    record = visualization.create_review_record(
        sample_token="sample",
        scene_token="scene",
        timestamp=1_000_000,
        cam_front_path="samples/CAM_FRONT/image.jpg",
        visualization_path="visualizations/sample_one_page.png",
        selection_reason="ordinary_motion_candidate",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )

    with pytest.raises(ValueError, match=field_name):
        visualization.validate_review_record(
            replace(record, **{field_name: invalid_value})
        )


def test_uncertain_and_not_available_are_not_counted_as_correct() -> None:
    visualization = load_visualization_module()
    unreviewed = visualization.create_review_record(
        sample_token="unreviewed",
        scene_token="scene-a",
        timestamp=1_000_000,
        cam_front_path="samples/CAM_FRONT/a.jpg",
        visualization_path="visualizations/a.png",
        selection_reason="low_displacement_candidate",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )
    confirmed = replace(
        unreviewed,
        sample_token="confirmed",
        label_correct="yes",
        trajectory_alignment_correct="yes",
        agent_alignment_correct="yes",
        visualization_sufficient="yes",
        safety_score_reasonable="yes",
    )

    summary = visualization.summarize_review_records(
        (unreviewed, confirmed)
    )

    assert summary.total_records == 2
    assert summary.label_correct == 1
    assert summary.trajectory_alignment_correct == 1
    assert summary.agent_alignment_correct == 1
    assert summary.visualization_sufficient == 1
    assert summary.safety_score_reasonable == 1


def test_review_outputs_share_the_same_complete_schema(
    tmp_path: Path,
) -> None:
    visualization = load_visualization_module()
    record = visualization.create_review_record(
        sample_token="sample",
        scene_token="scene",
        timestamp=1_000_000,
        cam_front_path="samples/CAM_FRONT/image.jpg",
        visualization_path="visualizations/sample_one_page.png",
        selection_reason="lateral_displacement_candidate",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )
    manifest_path = tmp_path / "review_manifest.jsonl"
    template_path = tmp_path / "review_template.csv"

    visualization.write_review_outputs(
        records=(record,),
        manifest_path=manifest_path,
        template_path=template_path,
    )

    manifest_row = json.loads(manifest_path.read_text().strip())
    header = template_path.read_text().splitlines()[0].split(",")
    assert tuple(manifest_row) == visualization.REVIEW_RECORD_FIELDS
    assert tuple(header) == visualization.REVIEW_RECORD_FIELDS


def test_review_cli_arguments_parse_without_dataset_access(
    tmp_path: Path,
) -> None:
    visualization = load_visualization_module()

    arguments = visualization.parse_args(
        [
            "--init-review",
            "--dataroot",
            str(tmp_path / "nuscenes"),
            "--output-dir",
            str(tmp_path / "review"),
            "--sample-count",
            "12",
            "--label-rule-version",
            "planned-label-rules",
            "--safety-rule-version",
            "unavailable",
            "--preview",
        ]
    )

    assert arguments.init_review is True
    assert arguments.dataroot == tmp_path / "nuscenes"
    assert arguments.output_dir == tmp_path / "review"
    assert arguments.sample_count == 12
    assert arguments.label_rule_version == "planned-label-rules"
    assert arguments.safety_rule_version == "unavailable"
    assert arguments.preview is True


def test_review_selection_covers_auditable_candidate_reasons() -> None:
    visualization = load_visualization_module()
    candidates = (
        visualization.ReviewCandidate(
            "forward",
            "scene-forward",
            1,
            "forward.jpg",
            20.0,
            0.1,
            20.0,
            1,
        ),
        visualization.ReviewCandidate(
            "stopped",
            "scene-stopped",
            2,
            "stopped.jpg",
            0.1,
            0.0,
            0.1,
            0,
        ),
        visualization.ReviewCandidate(
            "agents",
            "scene-agents",
            3,
            "agents.jpg",
            5.0,
            0.2,
            5.0,
            8,
        ),
        visualization.ReviewCandidate(
            "empty",
            "scene-empty",
            4,
            "empty.jpg",
            4.0,
            0.3,
            4.0,
            0,
        ),
        visualization.ReviewCandidate(
            "lateral",
            "scene-lateral",
            5,
            "lateral.jpg",
            6.0,
            4.0,
            7.2,
            2,
        ),
        visualization.ReviewCandidate(
            "ordinary",
            "scene-ordinary",
            6,
            "ordinary.jpg",
            7.0,
            0.05,
            7.0,
            1,
        ),
    )

    selections = visualization.select_review_candidates(
        candidates=candidates,
        sample_count=6,
    )

    assert len(selections) == 6
    selected_scenes = {
        selection.candidate.scene_token for selection in selections
    }
    assert len(selected_scenes) == 6
    assert {selection.reason for selection in selections} == {
        "high_forward_displacement_candidate",
        "low_displacement_candidate",
        "has_nearby_agents",
        "no_nearby_agents",
        "lateral_displacement_candidate",
        "ordinary_motion_candidate",
    }


def test_review_records_keep_selection_reason_and_relative_visualization_path(
) -> None:
    visualization = load_visualization_module()
    selection = visualization.ReviewSelection(
        candidate=visualization.ReviewCandidate(
            "sample",
            "scene",
            1_000_000,
            "samples/CAM_FRONT/image.jpg",
            5.0,
            0.1,
            5.0,
            2,
        ),
        reason="has_nearby_agents",
    )

    records = visualization.create_review_records(
        selections=(selection,),
        label_rule_version="planned-label-rules",
        safety_rule_version="unavailable",
    )

    assert records[0].visualization_path == (
        "visualizations/sample_one_page.png"
    )
    assert records[0].selection_reason == "has_nearby_agents"
    assert records[0].label_rule_version == "planned-label-rules"
    assert records[0].safety_rule_version == "unavailable"


def test_review_pool_traverses_scene_samples_before_validity_filtering() -> None:
    visualization = load_visualization_module()

    class FakeNuScenes:
        scene = (
            {"first_sample_token": "a-0"},
            {"first_sample_token": "b-0"},
        )
        samples = {
            "a-0": {"next": "a-1"},
            "a-1": {"next": "a-2"},
            "a-2": {"next": ""},
            "b-0": {"next": "b-1"},
            "b-1": {"next": "b-2"},
            "b-2": {"next": ""},
        }

        def get(self, table_name: str, token: str) -> dict[str, str]:
            assert table_name == "sample"
            return self.samples[token]

    tokens = visualization.collect_review_sample_tokens(
        nuscenes=FakeNuScenes(),
    )

    assert set(tokens) == {"a-0", "a-1", "a-2", "b-0", "b-1", "b-2"}


def test_review_candidate_uses_existing_trajectory_and_agent_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualization = load_visualization_module()

    class FakeNuScenes:
        records = {
            ("sample", "sample"): {
                "scene_token": "scene",
                "timestamp": 1_000_000,
                "data": {"CAM_FRONT": "camera"},
            },
            ("sample_data", "camera"): {
                "filename": "samples/CAM_FRONT/image.jpg",
            },
        }

        def get(self, table_name: str, token: str) -> dict[str, object]:
            return self.records[(table_name, token)]

    monkeypatch.setattr(
        visualization,
        "extract_future_ego_trajectory",
        lambda **_: SimpleNamespace(
            points=(
                visualization.TrajectoryPoint(
                    "current",
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                visualization.TrajectoryPoint(
                    "future",
                    1.0,
                    6.0,
                    -2.0,
                    0.1,
                ),
            ),
            is_truncated=False,
        ),
    )
    monkeypatch.setattr(
        visualization,
        "get_nearby_agents",
        lambda **_: SimpleNamespace(agents=("agent-a", "agent-b")),
    )

    candidate = visualization.build_review_candidate(
        nuscenes=FakeNuScenes(),
        sample_token="sample",
        camera="CAM_FRONT",
        horizon_sec=1.0,
        sample_interval_sec=1.0,
        max_agent_distance_m=50.0,
    )

    assert candidate.scene_token == "scene"
    assert candidate.cam_front_path == "samples/CAM_FRONT/image.jpg"
    assert candidate.forward_displacement_m == pytest.approx(6.0)
    assert candidate.lateral_displacement_m == pytest.approx(-2.0)
    assert candidate.total_displacement_m == pytest.approx(6.3249, rel=1e-4)
    assert candidate.trajectory_points == 2
    assert candidate.expected_min_trajectory_points == 2
    assert candidate.trajectory_x_range_m == pytest.approx(6.0)
    assert candidate.trajectory_y_range_m == pytest.approx(2.0)
    assert candidate.trajectory_is_valid is True
    assert candidate.trajectory_invalid_reason == ""
    assert candidate.nearby_agent_count == 2


def test_single_point_zero_displacement_trajectory_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualization = load_visualization_module()

    class FakeNuScenes:
        records = {
            ("sample", "tail"): {
                "scene_token": "scene",
                "timestamp": 1_000_000,
                "data": {"CAM_FRONT": "camera"},
            },
            ("sample_data", "camera"): {"filename": "tail.jpg"},
        }

        def get(self, table_name: str, token: str) -> dict[str, object]:
            return self.records[(table_name, token)]

    monkeypatch.setattr(
        visualization,
        "extract_future_ego_trajectory",
        lambda **_: SimpleNamespace(
            points=(
                visualization.TrajectoryPoint(
                    "tail", 0.0, 0.0, 0.0, 0.0
                ),
            ),
            is_truncated=True,
        ),
    )
    monkeypatch.setattr(
        visualization,
        "get_nearby_agents",
        lambda **_: SimpleNamespace(agents=()),
    )

    candidate = visualization.build_review_candidate(
        nuscenes=FakeNuScenes(),
        sample_token="tail",
        camera="CAM_FRONT",
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        max_agent_distance_m=50.0,
    )

    assert candidate.trajectory_points == 1
    assert candidate.expected_min_trajectory_points == 7
    assert candidate.trajectory_displacement_m == pytest.approx(0.0)
    assert candidate.trajectory_is_valid is False
    assert candidate.trajectory_invalid_reason == (
        "insufficient_future_trajectory"
    )


def test_candidate_pool_filters_scene_tail_with_insufficient_horizon(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    visualization = load_visualization_module()
    valid = visualization.ReviewCandidate(
        "valid",
        "scene",
        1,
        "valid.jpg",
        0.1,
        0.0,
        0.1,
        0,
        trajectory_points=7,
        expected_min_trajectory_points=7,
        trajectory_is_valid=True,
    )
    invalid = visualization.ReviewCandidate(
        "tail",
        "scene",
        2,
        "tail.jpg",
        0.0,
        0.0,
        0.0,
        0,
        trajectory_points=1,
        expected_min_trajectory_points=7,
        trajectory_is_valid=False,
        trajectory_invalid_reason="insufficient_future_trajectory",
    )
    monkeypatch.setattr(
        visualization,
        "collect_review_sample_tokens",
        lambda **_: ("valid", "tail"),
    )
    monkeypatch.setattr(
        visualization,
        "build_review_candidate",
        lambda sample_token, **_: valid if sample_token == "valid" else invalid,
    )

    candidates = visualization.build_review_candidate_pool(
        nuscenes=object(),
        sample_count=2,
        camera="CAM_FRONT",
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        max_agent_distance_m=50.0,
    )

    assert candidates == (valid,)
    output = capsys.readouterr().out
    assert "total candidate samples: 2" in output
    assert "valid trajectory samples: 1" in output
    assert "invalid insufficient_future_trajectory samples: 1" in output
    assert "warning: only 1 valid trajectory samples" in output


def test_low_displacement_requires_complete_future_trajectory() -> None:
    visualization = load_visualization_module()
    invalid_tail = visualization.ReviewCandidate(
        "tail",
        "scene-tail",
        1,
        "tail.jpg",
        0.0,
        0.0,
        0.0,
        0,
        trajectory_points=1,
        expected_min_trajectory_points=7,
        trajectory_is_valid=False,
        trajectory_invalid_reason="insufficient_future_trajectory",
    )
    valid_stopped = visualization.ReviewCandidate(
        "stopped",
        "scene-stopped",
        2,
        "stopped.jpg",
        0.01,
        0.0,
        0.01,
        0,
        trajectory_points=7,
        expected_min_trajectory_points=7,
        trajectory_is_valid=True,
    )
    valid_moving = visualization.ReviewCandidate(
        "moving",
        "scene-moving",
        3,
        "moving.jpg",
        8.0,
        0.0,
        8.0,
        0,
        trajectory_points=7,
        expected_min_trajectory_points=7,
        trajectory_is_valid=True,
    )

    selections = visualization.select_review_candidates(
        candidates=(invalid_tail, valid_stopped, valid_moving),
        sample_count=2,
    )

    low_displacement = next(
        selection
        for selection in selections
        if selection.reason == "low_displacement_candidate"
    )
    assert low_displacement.candidate.sample_token == "stopped"
    assert all(
        selection.candidate.trajectory_is_valid for selection in selections
    )


def test_review_preview_returns_records_without_writing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    visualization = load_visualization_module()
    candidate = visualization.ReviewCandidate(
        "sample",
        "scene",
        1_000_000,
        "samples/CAM_FRONT/image.jpg",
        5.0,
        0.1,
        5.0,
        2,
    )
    monkeypatch.setattr(
        visualization,
        "build_review_candidate_pool",
        lambda **_: (candidate,),
    )

    records = visualization.initialize_review_batch(
        nuscenes=object(),
        dataroot=tmp_path / "nuscenes",
        output_dir=tmp_path / "review",
        sample_count=12,
        camera="CAM_FRONT",
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        max_agent_distance_m=50.0,
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
        preview=True,
    )

    assert len(records) == 1
    assert records[0].sample_token == "sample"
    assert not (tmp_path / "review").exists()
    output = capsys.readouterr().out
    assert "selected samples: 1" in output
    assert "selected trajectory_points distribution: {7: 1}" in output
    assert '"sample_token": "sample"' in output

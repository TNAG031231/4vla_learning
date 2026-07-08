import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DERIVATION_SCRIPT = DATA_DIRECTORY / "derive_meta_action.py"


def load_derivation_module() -> ModuleType:
    assert DERIVATION_SCRIPT.is_file(), "data/derive_meta_action.py is missing"
    specification = importlib.util.spec_from_file_location(
        "derive_meta_action",
        DERIVATION_SCRIPT,
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


def build_rules(module: ModuleType):
    return module.MetaActionRules(
        label_rule_version="phase-1.6-meta-action-v0",
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        stop_distance_threshold_m=0.5,
        lateral_displacement_threshold_m=1.0,
        forward_displacement_threshold_m=3.0,
        speed_change_threshold_mps=1.0,
        boundary_margin=0.2,
        all_zero_tolerance_m=0.001,
        coordinate_frame="ego",
        x_axis="forward",
        y_axis="left",
        unit="meter",
    )


def build_trajectory(
    module: ModuleType,
    coordinates: tuple[tuple[float, float], ...],
):
    return tuple(
        module.TrajectoryPoint(
            future_sample_token=f"sample-{index}",
            t_sec=index * 0.5,
            x_m=x_m,
            y_m=y_m,
            heading_delta_rad=0.0,
        )
        for index, (x_m, y_m) in enumerate(coordinates)
    )


def load_schema_module() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        return importlib.import_module("src.actions.schema")
    finally:
        sys.path.remove(str(PROJECT_ROOT))


def test_canonical_schema_imports_from_src_actions() -> None:
    schema = load_schema_module()

    assert Path(schema.__file__).resolve() == (
        PROJECT_ROOT / "src" / "actions" / "schema.py"
    )


def test_action_schema_contains_exactly_six_actions() -> None:
    schema = load_schema_module()

    assert schema.ACTION_SCHEMA == (
        "keep",
        "accelerate",
        "decelerate",
        "stop",
        "left_lateral",
        "right_lateral",
    )
    assert schema.ACTION_SET == frozenset(schema.ACTION_SCHEMA)
    assert schema.is_valid_action("keep") is True
    assert schema.is_valid_action("turn_left") is False
    assert schema.normalize_action(" LEFT_LATERAL ") == "left_lateral"


def test_positive_lateral_displacement_is_left_lateral() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        (
            (0.0, 0.0),
            (0.5, 0.25),
            (1.0, 0.5),
            (1.5, 0.75),
            (2.0, 1.0),
            (2.5, 1.25),
            (3.0, 1.5),
        ),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "left_lateral"


def test_negative_lateral_displacement_is_right_lateral() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        (
            (0.0, 0.0),
            (0.5, -0.25),
            (1.0, -0.5),
            (1.5, -0.75),
            (2.0, -1.0),
            (2.5, -1.25),
            (3.0, -1.5),
        ),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "right_lateral"


def test_all_zero_trajectory_is_stop_candidate() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        tuple((0.0, 0.0) for _ in range(7)),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "stop"
    assert result.rule_features.is_all_zero_trajectory is True
    assert result.rule_features.is_stop_candidate is True
    assert "all_zero_trajectory" in result.boundary_flags


def test_constant_forward_trajectory_is_keep() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        tuple((float(index), 0.0) for index in range(7)),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "keep"
    assert result.action_confidence == "high"
    assert result.rule_features.approx_delta_speed_mps == 0.0


def test_lateral_threshold_sample_is_marked_boundary() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        (
            (0.0, 0.0),
            (0.5, 0.2),
            (1.0, 0.4),
            (1.5, 0.6),
            (2.0, 0.8),
            (2.5, 0.9),
            (3.0, 1.0),
        ),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "keep"
    assert result.rule_features.is_lateral_boundary is True
    assert "lateral_threshold_boundary" in result.boundary_flags
    assert result.action_confidence == "low"


def test_missing_speed_proxy_does_not_force_speed_action() -> None:
    derivation = load_derivation_module()
    trajectory = build_trajectory(
        derivation,
        ((0.0, 0.0), (3.0, 0.0)),
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.derived_action == "keep"
    assert result.derived_action not in {"accelerate", "decelerate"}
    assert result.rule_features.approx_speed_start_mps == "not_available"
    assert result.rule_features.approx_speed_end_mps == "not_available"
    assert result.rule_features.approx_delta_speed_mps == "not_available"
    assert result.uncertainty_reason == "speed_proxy_unavailable"
    assert "trajectory_too_short" in result.boundary_flags


def test_speed_proxy_accepts_complete_horizon_with_timestamp_tolerance() -> None:
    derivation = load_derivation_module()
    trajectory = tuple(
        derivation.TrajectoryPoint(
            future_sample_token=f"sample-{index}",
            t_sec=2.9995 if index == 6 else index * 0.5,
            x_m=float(index),
            y_m=0.0,
            heading_delta_rad=0.0,
        )
        for index in range(7)
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
    )

    assert result.rule_features.approx_speed_start_mps != "not_available"
    assert result.rule_features.approx_speed_end_mps != "not_available"
    assert result.rule_features.approx_delta_speed_mps != "not_available"
    assert "speed_proxy_unavailable" not in result.boundary_flags


def test_speed_proxy_uses_configured_trajectory_time_tolerance() -> None:
    derivation = load_derivation_module()
    trajectory = tuple(
        derivation.TrajectoryPoint(
            future_sample_token=f"sample-{index}",
            t_sec=2.9505 if index == 6 else index * 0.5,
            x_m=float(index),
            y_m=0.0,
            heading_delta_rad=0.0,
        )
        for index in range(7)
    )

    result = derivation.derive_meta_action(
        trajectory,
        build_rules(derivation),
        time_tolerance_sec=0.075,
    )

    assert result.rule_features.approx_speed_start_mps != "not_available"
    assert result.rule_features.approx_speed_end_mps != "not_available"
    assert result.rule_features.approx_delta_speed_mps != "not_available"
    assert "speed_proxy_unavailable" not in result.boundary_flags


def _write_review_manifest(
    path: Path,
    records: tuple[dict[str, str], ...],
) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return path


def test_review_manifest_reads_only_overall_pass_yes(tmp_path: Path) -> None:
    derivation = load_derivation_module()
    manifest = _write_review_manifest(
        tmp_path / "review_manifest.jsonl",
        (
            {"sample_token": "yes-lower", "overall_pass": "yes"},
            {"sample_token": "yes-normalized", "overall_pass": " YES "},
            {"sample_token": "no", "overall_pass": "no"},
            {"sample_token": "uncertain", "overall_pass": "uncertain"},
            {"sample_token": "missing"},
        ),
    )

    assert derivation.read_review_sample_tokens(manifest) == (
        "yes-lower",
        "yes-normalized",
    )


def test_review_manifest_without_overall_pass_reads_all_and_warns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    derivation = load_derivation_module()
    manifest = _write_review_manifest(
        tmp_path / "review_manifest.jsonl",
        (
            {"sample_token": "first"},
            {"sample_token": "second"},
        ),
    )

    sample_tokens = derivation.read_review_sample_tokens(manifest)

    assert sample_tokens == ("first", "second")
    assert (
        "review manifest has no overall_pass field; using all sample tokens"
        in capsys.readouterr().out
    )


def test_review_manifest_with_no_passing_samples_raises(
    tmp_path: Path,
) -> None:
    derivation = load_derivation_module()
    manifest = _write_review_manifest(
        tmp_path / "review_manifest.jsonl",
        (
            {"sample_token": "no", "overall_pass": "no"},
            {"sample_token": "uncertain", "overall_pass": "uncertain"},
        ),
    )

    with pytest.raises(
        ValueError,
        match="review manifest has no overall_pass=yes samples",
    ):
        derivation.read_review_sample_tokens(manifest)


def test_collect_valid_future_tokens_filters_before_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derivation = load_derivation_module()
    rules = build_rules(derivation)

    class FakeNuScenes:
        scene = (
            {"first_sample_token": "a"},
            {"first_sample_token": "c"},
        )
        samples = {
            "a": {"next": "b"},
            "b": {"next": ""},
            "c": {"next": ""},
        }

        def get(self, table_name: str, token: str) -> dict[str, str]:
            assert table_name == "sample"
            return self.samples[token]

    complete = SimpleNamespace(
        points=build_trajectory(
            derivation,
            tuple((float(index), 0.0) for index in range(7)),
        ),
        is_truncated=False,
    )
    short = SimpleNamespace(
        points=build_trajectory(derivation, ((0.0, 0.0), (1.0, 0.0))),
        is_truncated=False,
    )
    truncated = SimpleNamespace(points=complete.points, is_truncated=True)
    trajectories = {"a": complete, "b": short, "c": truncated}
    monkeypatch.setattr(
        derivation,
        "extract_future_ego_trajectory",
        lambda sample_token, **_: trajectories[sample_token],
    )

    sample_tokens = derivation.collect_valid_future_sample_tokens(
        nuscenes=FakeNuScenes(),
        rules=rules,
        time_tolerance_sec=0.075,
    )

    assert sample_tokens == ("a",)

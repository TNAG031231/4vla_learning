import importlib.util
from pathlib import Path
import sys
from types import ModuleType


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

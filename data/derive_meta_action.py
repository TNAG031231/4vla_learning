#!/usr/bin/env python3

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys

from nuscenes.nuscenes import NuScenes
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import (
    ACCELERATE,
    DECELERATE,
    KEEP,
    LEFT_LATERAL,
    RIGHT_LATERAL,
    STOP,
)
from inspect_nuscenes_sample import (
    CAMERA_CHANNEL,
    TIMESTAMP_TOLERANCE_SEC,
    TrajectoryConfig,
    TrajectoryPoint,
    extract_future_ego_trajectory,
    load_trajectory_config,
)


NOT_AVAILABLE = "not_available"
NO_UNCERTAINTY = "none"
DEFAULT_REVIEW_MANIFEST = Path(
    "data/outputs/phase_1_5_manual_review_smoke_v2/review_manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/outputs/phase_1_6_meta_action_v0/derived_meta_action.jsonl"
)


@dataclass(frozen=True)
class MetaActionRules:
    label_rule_version: str
    horizon_sec: float
    sample_interval_sec: float
    stop_distance_threshold_m: float
    lateral_displacement_threshold_m: float
    forward_displacement_threshold_m: float
    speed_change_threshold_mps: float
    boundary_margin: float
    all_zero_tolerance_m: float
    coordinate_frame: str
    x_axis: str
    y_axis: str
    unit: str


@dataclass(frozen=True)
class RuleFeatures:
    trajectory_points: int
    first_x_m: float | str
    first_y_m: float | str
    last_x_m: float | str
    last_y_m: float | str
    delta_x_m: float
    delta_y_m: float
    x_range_m: float
    y_range_m: float
    path_length_m: float
    approx_speed_start_mps: float | str
    approx_speed_end_mps: float | str
    approx_delta_speed_mps: float | str
    is_all_zero_trajectory: bool
    is_lateral_boundary: bool
    is_speed_boundary: bool
    is_stop_candidate: bool
    uncertainty_reason: str


@dataclass(frozen=True)
class MetaActionResult:
    derived_action: str
    action_confidence: str
    rule_features: RuleFeatures
    boundary_flags: tuple[str, ...]
    uncertainty_reason: str


@dataclass(frozen=True)
class SampleLabelRecord:
    sample_token: str
    scene_token: str
    scene_name: str
    timestamp_us: int
    cam_front_path: str
    trajectory_points: int
    trajectory_last_x_m: float | str
    trajectory_last_y_m: float | str
    trajectory_x_range_m: float
    trajectory_y_range_m: float
    derived_action: str
    action_confidence: str
    label_rule_version: str
    rule_features: RuleFeatures
    boundary_flags: tuple[str, ...]
    uncertainty_reason: str


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _number_value(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def load_meta_action_rules(config_path: Path) -> MetaActionRules:
    loaded: object = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration root must be a mapping")

    rules = _mapping_value(loaded, "meta_action_v0")
    coordinates = _mapping_value(rules, "coordinates")
    return MetaActionRules(
        label_rule_version=_string_value(rules, "label_rule_version"),
        horizon_sec=_number_value(rules, "horizon_sec"),
        sample_interval_sec=_number_value(rules, "sample_interval_sec"),
        stop_distance_threshold_m=_number_value(
            rules,
            "stop_distance_threshold_m",
        ),
        lateral_displacement_threshold_m=_number_value(
            rules,
            "lateral_displacement_threshold_m",
        ),
        forward_displacement_threshold_m=_number_value(
            rules,
            "forward_displacement_threshold_m",
        ),
        speed_change_threshold_mps=_number_value(
            rules,
            "speed_change_threshold_mps",
        ),
        boundary_margin=_number_value(rules, "boundary_margin"),
        all_zero_tolerance_m=_number_value(
            rules,
            "all_zero_tolerance_m",
        ),
        coordinate_frame=_string_value(coordinates, "frame"),
        x_axis=_string_value(coordinates, "x_axis"),
        y_axis=_string_value(coordinates, "y_axis"),
        unit=_string_value(coordinates, "unit"),
    )


def _path_length(trajectory: tuple[TrajectoryPoint, ...]) -> float:
    return sum(
        math.hypot(
            current.x_m - previous.x_m,
            current.y_m - previous.y_m,
        )
        for previous, current in zip(
            trajectory,
            trajectory[1:],
            strict=False,
        )
    )


def _segment_speed(
    first: TrajectoryPoint,
    second: TrajectoryPoint,
) -> float:
    return math.hypot(
        second.x_m - first.x_m,
        second.y_m - first.y_m,
    ) / (second.t_sec - first.t_sec)


def _expected_trajectory_points(rules: MetaActionRules) -> int:
    return round(rules.horizon_sec / rules.sample_interval_sec) + 1


def _speed_proxy(
    trajectory: tuple[TrajectoryPoint, ...],
    rules: MetaActionRules,
    time_tolerance_sec: float,
) -> tuple[float | str, float | str, float | str]:
    complete_horizon = (
        len(trajectory) >= _expected_trajectory_points(rules)
        and trajectory[-1].t_sec + time_tolerance_sec
        >= rules.horizon_sec
    )
    if not complete_horizon:
        return NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE

    start_speed = _segment_speed(trajectory[0], trajectory[1])
    end_speed = _segment_speed(trajectory[-2], trajectory[-1])
    return start_speed, end_speed, end_speed - start_speed


def _is_threshold_boundary(
    value: float,
    threshold: float,
    boundary_margin: float,
) -> bool:
    return abs(abs(value) - threshold) <= threshold * boundary_margin


def derive_meta_action(
    trajectory: tuple[TrajectoryPoint, ...],
    rules: MetaActionRules,
    time_tolerance_sec: float = TIMESTAMP_TOLERANCE_SEC,
) -> MetaActionResult:
    if trajectory:
        first_x_m = trajectory[0].x_m
        first_y_m = trajectory[0].y_m
        last_x_m = trajectory[-1].x_m
        last_y_m = trajectory[-1].y_m
        delta_x_m = last_x_m - first_x_m
        delta_y_m = last_y_m - first_y_m
        x_values = tuple(point.x_m for point in trajectory)
        y_values = tuple(point.y_m for point in trajectory)
        x_range_m = max(x_values) - min(x_values)
        y_range_m = max(y_values) - min(y_values)
    else:
        first_x_m = NOT_AVAILABLE
        first_y_m = NOT_AVAILABLE
        last_x_m = NOT_AVAILABLE
        last_y_m = NOT_AVAILABLE
        delta_x_m = 0.0
        delta_y_m = 0.0
        x_range_m = 0.0
        y_range_m = 0.0

    path_length_m = _path_length(trajectory)
    is_all_zero_trajectory = bool(trajectory) and all(
        math.hypot(point.x_m, point.y_m) <= rules.all_zero_tolerance_m
        for point in trajectory
    )
    is_stop_candidate = is_all_zero_trajectory or (
        bool(trajectory)
        and path_length_m <= rules.stop_distance_threshold_m
        and abs(delta_x_m) <= rules.stop_distance_threshold_m
    )
    is_lateral_boundary = _is_threshold_boundary(
        delta_y_m,
        rules.lateral_displacement_threshold_m,
        rules.boundary_margin,
    )
    (
        approx_speed_start_mps,
        approx_speed_end_mps,
        approx_delta_speed_mps,
    ) = _speed_proxy(trajectory, rules, time_tolerance_sec)
    speed_proxy_available = isinstance(approx_delta_speed_mps, float)
    is_speed_boundary = (
        speed_proxy_available
        and _is_threshold_boundary(
            approx_delta_speed_mps,
            rules.speed_change_threshold_mps,
            rules.boundary_margin,
        )
    )

    boundary_flags = []
    if is_all_zero_trajectory:
        boundary_flags.append("all_zero_trajectory")
    if len(trajectory) < _expected_trajectory_points(rules):
        boundary_flags.append("trajectory_too_short")
    if is_lateral_boundary:
        boundary_flags.append("lateral_threshold_boundary")
    if not speed_proxy_available:
        boundary_flags.append("speed_proxy_unavailable")
    if is_speed_boundary:
        boundary_flags.append("speed_threshold_boundary")

    uncertainty_reason = NO_UNCERTAINTY
    action_confidence = "high"
    if not trajectory:
        derived_action = KEEP
        action_confidence = "low"
        uncertainty_reason = "trajectory_too_short"
    elif is_stop_candidate:
        derived_action = STOP
        if not is_all_zero_trajectory:
            action_confidence = "medium"
            uncertainty_reason = "stop_threshold_candidate"
    elif abs(delta_y_m) > rules.lateral_displacement_threshold_m:
        derived_action = (
            LEFT_LATERAL if delta_y_m > 0.0 else RIGHT_LATERAL
        )
    elif is_lateral_boundary:
        derived_action = KEEP
        action_confidence = "low"
        uncertainty_reason = "lateral_threshold_boundary"
    elif not speed_proxy_available:
        derived_action = KEEP
        action_confidence = "low"
        uncertainty_reason = "speed_proxy_unavailable"
    elif abs(delta_x_m) < rules.forward_displacement_threshold_m:
        derived_action = KEEP
        action_confidence = "low"
        if approx_delta_speed_mps > rules.speed_change_threshold_mps:
            uncertainty_reason = "keep_vs_accelerate_ambiguity"
        elif approx_delta_speed_mps < -rules.speed_change_threshold_mps:
            uncertainty_reason = "keep_vs_decelerate_ambiguity"
        else:
            uncertainty_reason = "low_forward_displacement_speed_guard"
        boundary_flags.append("insufficient_forward_displacement")
    elif is_speed_boundary:
        derived_action = KEEP
        action_confidence = "low"
        uncertainty_reason = (
            "keep_vs_accelerate_ambiguity"
            if approx_delta_speed_mps >= 0.0
            else "keep_vs_decelerate_ambiguity"
        )
    elif approx_delta_speed_mps > rules.speed_change_threshold_mps:
        derived_action = ACCELERATE
        action_confidence = "medium"
        uncertainty_reason = "trajectory_speed_proxy_only"
    elif approx_delta_speed_mps < -rules.speed_change_threshold_mps:
        derived_action = DECELERATE
        action_confidence = "medium"
        uncertainty_reason = "trajectory_speed_proxy_only"
    else:
        derived_action = KEEP

    features = RuleFeatures(
        trajectory_points=len(trajectory),
        first_x_m=first_x_m,
        first_y_m=first_y_m,
        last_x_m=last_x_m,
        last_y_m=last_y_m,
        delta_x_m=delta_x_m,
        delta_y_m=delta_y_m,
        x_range_m=x_range_m,
        y_range_m=y_range_m,
        path_length_m=path_length_m,
        approx_speed_start_mps=approx_speed_start_mps,
        approx_speed_end_mps=approx_speed_end_mps,
        approx_delta_speed_mps=approx_delta_speed_mps,
        is_all_zero_trajectory=is_all_zero_trajectory,
        is_lateral_boundary=is_lateral_boundary,
        is_speed_boundary=is_speed_boundary,
        is_stop_candidate=is_stop_candidate,
        uncertainty_reason=uncertainty_reason,
    )
    return MetaActionResult(
        derived_action=derived_action,
        action_confidence=action_confidence,
        rule_features=features,
        boundary_flags=tuple(boundary_flags),
        uncertainty_reason=uncertainty_reason,
    )


def _load_data_config(
    config_path: Path,
    dataroot: Path | None,
) -> TrajectoryConfig:
    if dataroot is None:
        return load_trajectory_config(config_path)

    previous_root = os.environ.get("NUSCENES_ROOT")
    os.environ["NUSCENES_ROOT"] = str(dataroot)
    try:
        return load_trajectory_config(config_path)
    finally:
        if previous_root is None:
            os.environ.pop("NUSCENES_ROOT", None)
        else:
            os.environ["NUSCENES_ROOT"] = previous_root


def read_review_sample_tokens(review_manifest: Path) -> tuple[str, ...]:
    records = []
    for line in review_manifest.read_text().splitlines():
        record: object = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError("Each review manifest row must be a mapping")
        records.append(record)

    if not any("overall_pass" in record for record in records):
        print(
            "review manifest has no overall_pass field; "
            "using all sample tokens"
        )
        return tuple(
            _string_value(record, "sample_token") for record in records
        )

    sample_tokens = tuple(
        _string_value(record, "sample_token")
        for record in records
        if "overall_pass" in record
        and _string_value(record, "overall_pass").strip().lower() == "yes"
    )
    if not sample_tokens:
        raise ValueError("review manifest has no overall_pass=yes samples")
    return sample_tokens


def collect_sample_tokens(nuscenes: NuScenes) -> tuple[str, ...]:
    sample_tokens = []
    for scene in nuscenes.scene:
        token = str(scene["first_sample_token"])
        while token:
            sample_tokens.append(token)
            sample = nuscenes.get("sample", token)
            token = str(sample["next"])
    return tuple(sample_tokens)


def has_valid_future_trajectory(
    nuscenes: NuScenes,
    sample_token: str,
    rules: MetaActionRules,
    time_tolerance_sec: float,
) -> bool:
    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=rules.horizon_sec,
        sample_interval_sec=rules.sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
    )
    return (
        len(trajectory.points) >= _expected_trajectory_points(rules)
        and not trajectory.is_truncated
        and trajectory.points[-1].t_sec + time_tolerance_sec
        >= rules.horizon_sec
    )


def collect_valid_future_sample_tokens(
    nuscenes: NuScenes,
    rules: MetaActionRules,
    time_tolerance_sec: float,
) -> tuple[str, ...]:
    return tuple(
        sample_token
        for sample_token in collect_sample_tokens(nuscenes)
        if has_valid_future_trajectory(
            nuscenes=nuscenes,
            sample_token=sample_token,
            rules=rules,
            time_tolerance_sec=time_tolerance_sec,
        )
    )


def derive_sample_record(
    nuscenes: NuScenes,
    sample_token: str,
    camera: str,
    rules: MetaActionRules,
    time_tolerance_sec: float,
) -> SampleLabelRecord:
    sample = nuscenes.get("sample", sample_token)
    scene = nuscenes.get("scene", sample["scene_token"])
    camera_data = nuscenes.get(
        "sample_data",
        sample["data"][camera],
    )
    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=rules.horizon_sec,
        sample_interval_sec=rules.sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
    )
    result = derive_meta_action(
        trajectory.points,
        rules,
        time_tolerance_sec=time_tolerance_sec,
    )
    features = result.rule_features
    return SampleLabelRecord(
        sample_token=sample_token,
        scene_token=str(sample["scene_token"]),
        scene_name=str(scene["name"]),
        timestamp_us=int(sample["timestamp"]),
        cam_front_path=Path(camera_data["filename"]).as_posix(),
        trajectory_points=features.trajectory_points,
        trajectory_last_x_m=features.last_x_m,
        trajectory_last_y_m=features.last_y_m,
        trajectory_x_range_m=features.x_range_m,
        trajectory_y_range_m=features.y_range_m,
        derived_action=result.derived_action,
        action_confidence=result.action_confidence,
        label_rule_version=rules.label_rule_version,
        rule_features=features,
        boundary_flags=result.boundary_flags,
        uncertainty_reason=result.uncertainty_reason,
    )


def write_jsonl(
    records: tuple[SampleLabelRecord, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(asdict(record), ensure_ascii=False) + "\n"
            )


def print_diagnostics(records: tuple[SampleLabelRecord, ...]) -> None:
    action_distribution = Counter(
        record.derived_action for record in records
    )
    confidence_distribution = Counter(
        record.action_confidence for record in records
    )
    uncertainty_distribution = Counter(
        record.uncertainty_reason for record in records
    )
    print(
        "action distribution: "
        f"{dict(sorted(action_distribution.items()))}"
    )
    print(
        "confidence distribution: "
        f"{dict(sorted(confidence_distribution.items()))}"
    )
    print(
        "uncertainty_reason distribution: "
        f"{dict(sorted(uncertainty_distribution.items()))}"
    )
    print("boundary sample list:")
    for record in records:
        if record.boundary_flags:
            print(
                f"- {record.sample_token}: "
                f"{', '.join(record.boundary_flags)}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive auditable Phase -1.6 meta-action v0 labels."
    )
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=DEFAULT_REVIEW_MANIFEST,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--action-config",
        type=Path,
        default=Path("configs/action_rules.yaml"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
    )
    parser.add_argument("--dataroot", type=Path)
    parser.add_argument("--camera", default=CAMERA_CHANNEL)
    parser.add_argument(
        "--all-valid-samples",
        action="store_true",
        help=(
            "Derive labels for every sample with a complete configured "
            "future trajectory instead of reading --review-manifest."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    rules = load_meta_action_rules(arguments.action_config)
    data_config = _load_data_config(
        arguments.data_config,
        arguments.dataroot,
    )
    nuscenes = NuScenes(
        version=data_config.version,
        dataroot=str(data_config.nuscenes_root),
        verbose=False,
    )
    if arguments.all_valid_samples:
        sample_tokens = collect_valid_future_sample_tokens(
            nuscenes=nuscenes,
            rules=rules,
            time_tolerance_sec=data_config.trajectory_time_tolerance_sec,
        )
        print(
            "valid_3s_future_trajectory_samples: "
            f"{len(sample_tokens)}"
        )
    else:
        sample_tokens = read_review_sample_tokens(arguments.review_manifest)

    records = tuple(
        derive_sample_record(
            nuscenes=nuscenes,
            sample_token=sample_token,
            camera=arguments.camera,
            rules=rules,
            time_tolerance_sec=data_config.trajectory_time_tolerance_sec,
        )
        for sample_token in sample_tokens
    )
    write_jsonl(records, arguments.output)
    print(f"output: {arguments.output}")
    print_diagnostics(records)


if __name__ == "__main__":
    main()

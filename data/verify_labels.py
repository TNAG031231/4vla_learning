#!/usr/bin/env python3

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from nuscenes.nuscenes import NuScenes
from PIL import Image

from inspect_nuscenes_sample import (
    CAMERA_CHANNEL,
    NearbyAgent,
    TrajectoryConfig,
    TrajectoryPoint,
    extract_future_ego_trajectory,
    get_nearby_agents,
    load_trajectory_config,
)
from manual_review import (
    REVIEW_RECORD_FIELDS,
    ReviewCandidate,
    ReviewRecord,
    ReviewSelection,
    create_review_records,
    create_review_record,
    select_review_candidates,
    summarize_review_records,
    validate_review_record,
    write_review_outputs,
)


DEFAULT_OUTPUT_DIRECTORY = Path("outputs/phase1_visualizations")
CATEGORY_COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#dc2626",
    "bicycle": "#16a34a",
    "motorcycle": "#ea580c",
    "other": "#6b7280",
}


@dataclass(frozen=True)
class VisualizationPayload:
    sample_token: str
    scene_token: str
    scene_name: str
    current_timestamp: int
    camera: str
    cam_front_path: str
    trajectory: tuple[TrajectoryPoint, ...]
    agents: tuple[NearbyAgent, ...]
    horizon_sec: float
    sample_interval_sec: float
    max_agent_distance_m: float
    meta_action: str
    label_rule_version: str
    safety_rule_version: str


@dataclass(frozen=True)
class BevLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class SanitySummary:
    trajectory_first: tuple[float, float] | None
    trajectory_last: tuple[float, float] | None
    min_x_m: float | None
    max_x_m: float | None
    min_y_m: float | None
    max_y_m: float | None
    nearest_agent_distance_m: float | None
    nearest_agent_category: str | None
    trajectory_empty: bool
    agents_empty: bool


def resolve_output_path(
    sample_token: str,
    output: Path | None,
    output_dir: Path,
) -> Path:
    if output is not None:
        return output
    return output_dir / f"{sample_token}_one_page.png"


def agent_display_category(category_name: str) -> str:
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle."):
        return "vehicle"
    return "other"


def calculate_bev_limits(
    trajectory_xy: tuple[tuple[float, float], ...],
    agent_xy: tuple[tuple[float, float], ...],
    radius_m: float,
) -> BevLimits:
    points = (
        *trajectory_xy,
        *agent_xy,
        (-radius_m, -radius_m),
        (radius_m, radius_m),
    )
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    span = max(
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        1.0,
    )
    padding = max(2.0, span * 0.05)
    return BevLimits(
        x_min=min(x_values) - padding,
        x_max=max(x_values) + padding,
        y_min=min(y_values) - padding,
        y_max=max(y_values) + padding,
    )


def build_sanity_summary(
    trajectory: tuple[TrajectoryPoint, ...],
    agents: tuple[NearbyAgent, ...],
) -> SanitySummary:
    if trajectory:
        x_values = [point.x_m for point in trajectory]
        y_values = [point.y_m for point in trajectory]
        trajectory_first = (trajectory[0].x_m, trajectory[0].y_m)
        trajectory_last = (trajectory[-1].x_m, trajectory[-1].y_m)
        min_x_m = min(x_values)
        max_x_m = max(x_values)
        min_y_m = min(y_values)
        max_y_m = max(y_values)
    else:
        trajectory_first = None
        trajectory_last = None
        min_x_m = None
        max_x_m = None
        min_y_m = None
        max_y_m = None

    nearest_agent = min(
        agents,
        key=lambda agent: agent.distance_xy_m,
        default=None,
    )
    return SanitySummary(
        trajectory_first=trajectory_first,
        trajectory_last=trajectory_last,
        min_x_m=min_x_m,
        max_x_m=max_x_m,
        min_y_m=min_y_m,
        max_y_m=max_y_m,
        nearest_agent_distance_m=(
            nearest_agent.distance_xy_m if nearest_agent else None
        ),
        nearest_agent_category=(
            nearest_agent.category_name if nearest_agent else None
        ),
        trajectory_empty=not trajectory,
        agents_empty=not agents,
    )


def _agent_footprint(agent: NearbyAgent) -> tuple[tuple[float, float], ...]:
    width_m, length_m, _ = agent.size
    half_length = length_m / 2.0
    half_width = width_m / 2.0
    center_x, center_y, _ = agent.translation_ego
    cosine = math.cos(agent.yaw_ego_rad)
    sine = math.sin(agent.yaw_ego_rad)
    corners = (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    )
    return tuple(
        (
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in corners
    )


def _draw_bev(
    axis,
    payload: VisualizationPayload,
) -> None:
    trajectory_xy = tuple(
        (point.x_m, point.y_m) for point in payload.trajectory
    )
    agent_xy = tuple(
        (agent.translation_ego[0], agent.translation_ego[1])
        for agent in payload.agents
    )
    limits = calculate_bev_limits(
        trajectory_xy=trajectory_xy,
        agent_xy=agent_xy,
        radius_m=payload.max_agent_distance_m,
    )
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            payload.max_agent_distance_m,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            color="#9ca3af",
        )
    )
    axis.scatter(
        [0.0],
        [0.0],
        marker="^",
        s=100,
        color="black",
        label="ego",
        zorder=5,
    )
    axis.annotate(
        "ego",
        (0.0, 0.0),
        xytext=(-8, -14),
        textcoords="offset points",
        ha="right",
    )

    if trajectory_xy:
        x_values, y_values = zip(*trajectory_xy, strict=True)
        axis.plot(
            x_values,
            y_values,
            color="#7c3aed",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="future trajectory",
            zorder=4,
        )
        axis.scatter(
            [x_values[0]],
            [y_values[0]],
            color="#16a34a",
            s=55,
            zorder=6,
        )
        axis.scatter(
            [x_values[-1]],
            [y_values[-1]],
            color="#dc2626",
            s=55,
            zorder=6,
        )
        axis.annotate(
            "start",
            trajectory_xy[0],
            xytext=(8, 8),
            textcoords="offset points",
        )
        axis.annotate(
            "end",
            trajectory_xy[-1],
            xytext=(5, 5),
            textcoords="offset points",
        )
    else:
        axis.text(
            0.5,
            0.95,
            "no future trajectory available",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#dc2626",
        )

    present_categories = set()
    for agent in payload.agents:
        display_category = agent_display_category(agent.category_name)
        present_categories.add(display_category)
        color = CATEGORY_COLORS[display_category]
        axis.add_patch(
            Polygon(
                _agent_footprint(agent),
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=0.35,
                linewidth=1.0,
                zorder=2,
            )
        )
        axis.scatter(
            [agent.translation_ego[0]],
            [agent.translation_ego[1]],
            s=12,
            color=color,
            zorder=3,
        )
    if not payload.agents:
        axis.text(
            0.5,
            0.88,
            "no nearby agents within threshold",
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#dc2626",
        )

    category_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CATEGORY_COLORS[category],
            label=category,
        )
        for category in CATEGORY_COLORS
        if category in present_categories
    ]
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles + category_handles,
        labels + [handle.get_label() for handle in category_handles],
        loc="upper right",
        fontsize=8,
    )
    axis.set_xlim(limits.x_min, limits.x_max)
    axis.set_ylim(limits.y_min, limits.y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x = forward (meter)")
    axis.set_ylabel("y = left (meter)")
    axis.set_title(
        f"BEV ego frame | radius={payload.max_agent_distance_m:.1f}m"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)


def _format_point(point: tuple[float, float] | None) -> str:
    if point is None:
        return "unavailable"
    return f"({point[0]:.3f}, {point[1]:.3f})"


def _format_range(
    minimum: float | None,
    maximum: float | None,
) -> str:
    if minimum is None or maximum is None:
        return "unavailable"
    return f"[{minimum:.3f}, {maximum:.3f}]"


def _metadata_text(payload: VisualizationPayload) -> str:
    return "\n".join(
        (
            "Sample metadata",
            "",
            f"sample_token: {payload.sample_token}",
            f"scene_token: {payload.scene_token}",
            f"scene_name: {payload.scene_name}",
            f"timestamp_us: {payload.current_timestamp}",
            f"camera: {payload.camera}",
            f"cam_front_path: {payload.cam_front_path}",
            "",
            f"trajectory_points: {len(payload.trajectory)}",
            f"horizon_sec: {payload.horizon_sec}",
            f"sample_interval_sec: {payload.sample_interval_sec}",
            f"nearby_agents: {len(payload.agents)}",
            f"max_agent_distance_m: {payload.max_agent_distance_m}",
            "",
            f"meta_action: {payload.meta_action}",
            f"label_rule_version: {payload.label_rule_version}",
            f"safety_rule_version: {payload.safety_rule_version}",
            "",
            "Frames: trajectory + agents in current ego frame",
            "Axes: x forward, y left, z up; unit: meter",
        )
    )


def _summary_text(summary: SanitySummary) -> str:
    lines = [
        "Sanity summary",
        "",
        f"trajectory_first_xy: {_format_point(summary.trajectory_first)}",
        f"trajectory_last_xy: {_format_point(summary.trajectory_last)}",
        f"trajectory_x_range_m: {_format_range(summary.min_x_m, summary.max_x_m)}",
        f"trajectory_y_range_m: {_format_range(summary.min_y_m, summary.max_y_m)}",
    ]
    if summary.trajectory_empty:
        lines.append("no future trajectory available")
    if summary.agents_empty:
        lines.append("no nearby agents within threshold")
    else:
        lines.extend(
            (
                f"nearest_agent_distance_m: "
                f"{summary.nearest_agent_distance_m:.3f}",
                f"nearest_agent_category: {summary.nearest_agent_category}",
            )
        )
    return "\n".join(lines)


def render_one_page_visualization(
    payload: VisualizationPayload,
    image: Image.Image,
) -> Figure:
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.35, 1.15, 0.9),
        height_ratios=(1.0, 1.0),
    )
    image_axis = figure.add_subplot(grid[:, 0])
    bev_axis = figure.add_subplot(grid[:, 1])
    metadata_axis = figure.add_subplot(grid[0, 2])
    summary_axis = figure.add_subplot(grid[1, 2])

    image_axis.imshow(image)
    image_axis.set_title(f"camera={payload.camera}")
    image_axis.axis("off")

    _draw_bev(bev_axis, payload)

    metadata_axis.axis("off")
    metadata_axis.set_facecolor("#f8fafc")
    metadata_axis.text(
        0.02,
        0.98,
        _metadata_text(payload),
        transform=metadata_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        wrap=True,
    )

    summary = build_sanity_summary(payload.trajectory, payload.agents)
    summary_axis.axis("off")
    summary_axis.set_facecolor("#f8fafc")
    summary_axis.text(
        0.02,
        0.98,
        _summary_text(summary),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )
    figure.suptitle(
        "Phase -1.4 | One-page sample alignment verification",
        fontsize=15,
    )
    return figure


def _load_config_with_optional_dataroot(
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


def load_sample_visualization_payload(
    nuscenes: NuScenes,
    dataroot: Path,
    sample_token: str,
    camera: str,
    horizon_sec: float,
    sample_interval_sec: float,
    max_agent_distance_m: float,
) -> tuple[VisualizationPayload, Image.Image]:
    sample = nuscenes.get("sample", sample_token)
    scene = nuscenes.get("scene", sample["scene_token"])
    camera_token = sample["data"][camera]
    camera_data = nuscenes.get("sample_data", camera_token)
    relative_image_path = Path(camera_data["filename"])
    image_path = dataroot / relative_image_path
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
    )
    nearby_agents = get_nearby_agents(
        nuscenes=nuscenes,
        sample_token=sample_token,
        radius_m=max_agent_distance_m,
    )
    payload = VisualizationPayload(
        sample_token=sample_token,
        scene_token=str(sample["scene_token"]),
        scene_name=str(scene["name"]),
        current_timestamp=int(sample["timestamp"]),
        camera=camera,
        cam_front_path=relative_image_path.as_posix(),
        trajectory=trajectory.points,
        agents=nearby_agents.agents,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        max_agent_distance_m=max_agent_distance_m,
        meta_action="unavailable",
        label_rule_version="unavailable",
        safety_rule_version="unavailable",
    )
    return payload, image


def save_one_page_visualization(
    figure: Figure,
    output_path: Path,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    if show:
        plt.show()
    plt.close(figure)


def _print_summary(
    output_path: Path,
    summary: SanitySummary,
) -> None:
    print(f"output: {output_path}")
    print(f"future trajectory first: {_format_point(summary.trajectory_first)}")
    print(f"future trajectory last: {_format_point(summary.trajectory_last)}")
    print(
        "future trajectory x range: "
        f"{_format_range(summary.min_x_m, summary.max_x_m)}"
    )
    print(
        "future trajectory y range: "
        f"{_format_range(summary.min_y_m, summary.max_y_m)}"
    )
    if summary.trajectory_empty:
        print("no future trajectory available")
    if summary.agents_empty:
        print("no nearby agents within threshold")
    else:
        print(
            "nearest agent: "
            f"{summary.nearest_agent_category} at "
            f"{summary.nearest_agent_distance_m:.3f}m"
        )


def collect_review_sample_tokens(
    nuscenes: NuScenes,
) -> tuple[str, ...]:
    sample_tokens = []
    for scene in nuscenes.scene:
        token = str(scene["first_sample_token"])
        while token:
            sample_tokens.append(token)
            sample = nuscenes.get("sample", token)
            token = str(sample["next"])
    return tuple(sample_tokens)


def expected_trajectory_points(
    horizon_sec: float,
    sample_interval_sec: float,
) -> int:
    return math.floor(horizon_sec / sample_interval_sec + 1e-9) + 1


def build_review_candidate(
    nuscenes: NuScenes,
    sample_token: str,
    camera: str,
    horizon_sec: float,
    sample_interval_sec: float,
    max_agent_distance_m: float,
) -> ReviewCandidate:
    sample = nuscenes.get("sample", sample_token)
    camera_token = sample["data"][camera]
    camera_data = nuscenes.get("sample_data", camera_token)
    trajectory = extract_future_ego_trajectory(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
    )
    nearby_agents = get_nearby_agents(
        nuscenes=nuscenes,
        sample_token=sample_token,
        radius_m=max_agent_distance_m,
    )
    trajectory_points = len(trajectory.points)
    expected_min_trajectory_points = expected_trajectory_points(
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
    )
    if trajectory.points:
        x_values = tuple(point.x_m for point in trajectory.points)
        y_values = tuple(point.y_m for point in trajectory.points)
        first_point = trajectory.points[0]
        final_point = trajectory.points[-1]
        forward_displacement_m = final_point.x_m
        lateral_displacement_m = final_point.y_m
        trajectory_x_range_m = max(x_values) - min(x_values)
        trajectory_y_range_m = max(y_values) - min(y_values)
        trajectory_displacement_m = math.hypot(
            final_point.x_m - first_point.x_m,
            final_point.y_m - first_point.y_m,
        )
    else:
        forward_displacement_m = 0.0
        lateral_displacement_m = 0.0
        trajectory_x_range_m = 0.0
        trajectory_y_range_m = 0.0
        trajectory_displacement_m = 0.0
    trajectory_is_valid = (
        trajectory_points >= expected_min_trajectory_points
        and not trajectory.is_truncated
    )
    trajectory_invalid_reason = (
        "" if trajectory_is_valid else "insufficient_future_trajectory"
    )

    return ReviewCandidate(
        sample_token=sample_token,
        scene_token=str(sample["scene_token"]),
        timestamp=int(sample["timestamp"]),
        cam_front_path=Path(camera_data["filename"]).as_posix(),
        forward_displacement_m=forward_displacement_m,
        lateral_displacement_m=lateral_displacement_m,
        total_displacement_m=math.hypot(
            forward_displacement_m,
            lateral_displacement_m,
        ),
        nearby_agent_count=len(nearby_agents.agents),
        trajectory_points=trajectory_points,
        expected_min_trajectory_points=expected_min_trajectory_points,
        trajectory_x_range_m=trajectory_x_range_m,
        trajectory_y_range_m=trajectory_y_range_m,
        trajectory_displacement_m=trajectory_displacement_m,
        trajectory_is_valid=trajectory_is_valid,
        trajectory_invalid_reason=trajectory_invalid_reason,
    )


def build_review_candidate_pool(
    nuscenes: NuScenes,
    sample_count: int,
    camera: str,
    horizon_sec: float,
    sample_interval_sec: float,
    max_agent_distance_m: float,
) -> tuple[ReviewCandidate, ...]:
    all_candidates = tuple(
        build_review_candidate(
            nuscenes=nuscenes,
            sample_token=sample_token,
            camera=camera,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            max_agent_distance_m=max_agent_distance_m,
        )
        for sample_token in collect_review_sample_tokens(
            nuscenes=nuscenes,
        )
    )
    valid_candidates = tuple(
        candidate
        for candidate in all_candidates
        if candidate.trajectory_is_valid
    )
    invalid_reasons = Counter(
        candidate.trajectory_invalid_reason
        for candidate in all_candidates
        if not candidate.trajectory_is_valid
    )
    print(f"total candidate samples: {len(all_candidates)}")
    print(f"valid trajectory samples: {len(valid_candidates)}")
    print(
        "invalid insufficient_future_trajectory samples: "
        f"{invalid_reasons['insufficient_future_trajectory']}"
    )
    print(
        "invalid reason distribution: "
        f"{dict(sorted(invalid_reasons.items()))}"
    )
    if len(valid_candidates) < sample_count:
        print(
            f"warning: only {len(valid_candidates)} valid trajectory samples "
            f"available for requested {sample_count}"
        )
    return valid_candidates


def initialize_review_batch(
    nuscenes: NuScenes,
    dataroot: Path,
    output_dir: Path,
    sample_count: int,
    camera: str,
    horizon_sec: float,
    sample_interval_sec: float,
    max_agent_distance_m: float,
    label_rule_version: str,
    safety_rule_version: str,
    preview: bool,
) -> tuple[ReviewRecord, ...]:
    candidates = build_review_candidate_pool(
        nuscenes=nuscenes,
        sample_count=sample_count,
        camera=camera,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        max_agent_distance_m=max_agent_distance_m,
    )
    selections = select_review_candidates(
        candidates=candidates,
        sample_count=sample_count,
    )
    records = create_review_records(
        selections=selections,
        label_rule_version=label_rule_version,
        safety_rule_version=safety_rule_version,
    )
    selected_point_distribution = Counter(
        record.trajectory_points for record in records
    )
    print(f"selected samples: {len(records)}")
    print(
        "selected trajectory_points distribution: "
        f"{dict(sorted(selected_point_distribution.items()))}"
    )
    if preview:
        for record in records:
            print(json.dumps(asdict(record), ensure_ascii=False))
        return records

    for record in records:
        payload, image = load_sample_visualization_payload(
            nuscenes=nuscenes,
            dataroot=dataroot,
            sample_token=record.sample_token,
            camera=camera,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            max_agent_distance_m=max_agent_distance_m,
        )
        payload = replace(
            payload,
            meta_action=record.derived_action,
            label_rule_version=label_rule_version,
            safety_rule_version=safety_rule_version,
        )
        figure = render_one_page_visualization(payload, image)
        save_one_page_visualization(
            figure=figure,
            output_path=output_dir / record.visualization_path,
            show=False,
        )

    manifest_path = output_dir / "review_manifest.jsonl"
    template_path = output_dir / "review_template.csv"
    write_review_outputs(
        records=records,
        manifest_path=manifest_path,
        template_path=template_path,
    )
    print(f"review manifest: {manifest_path}")
    print(f"review template: {template_path}")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one-page alignment visualization or initialize a "
            "Phase -1.5 review batch."
        )
    )
    parser.add_argument("--sample-token")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
    )
    parser.add_argument("--dataroot", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--camera", default=CAMERA_CHANNEL)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-agent-distance-m", type=float)
    parser.add_argument("--horizon-sec", type=float)
    parser.add_argument("--sample-interval-sec", type=float)
    parser.add_argument(
        "--init-review",
        action="store_true",
        help="Initialize a Phase -1.5 review batch.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="Number of samples in a review batch (default: 12).",
    )
    parser.add_argument(
        "--label-rule-version",
        default="unavailable",
        help="Label rule version recorded in review outputs.",
    )
    parser.add_argument(
        "--safety-rule-version",
        default="unavailable",
        help="Safety rule version recorded in review outputs.",
    )
    parser.add_argument(
        "--preview",
        "--dry-run",
        dest="preview",
        action="store_true",
        help="Preview review selection without writing outputs.",
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    config = _load_config_with_optional_dataroot(
        config_path=arguments.config,
        dataroot=arguments.dataroot,
    )
    dataroot = arguments.dataroot or config.nuscenes_root
    version = arguments.version or config.version
    horizon_sec = (
        config.horizon_sec
        if arguments.horizon_sec is None
        else arguments.horizon_sec
    )
    sample_interval_sec = (
        config.sample_interval_sec
        if arguments.sample_interval_sec is None
        else arguments.sample_interval_sec
    )
    max_agent_distance_m = (
        config.nearby_radius_m
        if arguments.max_agent_distance_m is None
        else arguments.max_agent_distance_m
    )
    nuscenes = NuScenes(
        version=version,
        dataroot=str(dataroot),
        verbose=False,
    )
    if arguments.init_review or arguments.preview:
        initialize_review_batch(
            nuscenes=nuscenes,
            dataroot=dataroot,
            output_dir=arguments.output_dir,
            sample_count=arguments.sample_count,
            camera=arguments.camera,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            max_agent_distance_m=max_agent_distance_m,
            label_rule_version=arguments.label_rule_version,
            safety_rule_version=arguments.safety_rule_version,
            preview=arguments.preview,
        )
        return

    sample_token = arguments.sample_token
    if sample_token is None:
        sample_token = str(nuscenes.scene[0]["first_sample_token"])

    payload, image = load_sample_visualization_payload(
        nuscenes=nuscenes,
        dataroot=dataroot,
        sample_token=sample_token,
        camera=arguments.camera,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        max_agent_distance_m=max_agent_distance_m,
    )
    output_path = resolve_output_path(
        sample_token=sample_token,
        output=arguments.output,
        output_dir=arguments.output_dir,
    )
    figure = render_one_page_visualization(payload, image)
    save_one_page_visualization(
        figure=figure,
        output_path=output_path,
        show=arguments.show,
    )
    _print_summary(
        output_path=output_path,
        summary=build_sanity_summary(payload.trajectory, payload.agents),
    )


if __name__ == "__main__":
    main()

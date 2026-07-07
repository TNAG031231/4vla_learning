#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.schema import ACTION_SCHEMA


DEFAULT_DERIVED_LABELS = Path(
    "data/outputs/phase_1_6_meta_action_v0/derived_meta_action.jsonl"
)
DEFAULT_REVIEW_MANIFEST = Path(
    "data/outputs/phase_1_5_manual_review_smoke_v2/review_manifest.jsonl"
)
DEFAULT_OUTPUT = Path("data/phase_1_7_manual_audit.csv")
NOT_AVAILABLE = "not_available"
NO_UNCERTAINTY = "none"

REVIEW_FIELDS = (
    "sample_token",
    "scene_token",
    "timestamp",
    "cam_front_path",
    "visualization_path",
    "derived_action",
    "reviewed_action",
    "label_correct",
    "trajectory_alignment_correct",
    "agent_alignment_correct",
    "safety_score_reasonable",
    "error_type",
    "review_note",
    "label_rule_version",
    "safety_rule_version",
    "split",
    "action_confidence",
    "selection_reason",
    "has_vru",
    "safety_status",
    "safety_score",
    "safety_penalty",
    "boundary_flags",
    "uncertainty_reason",
    "nearby_agent_count",
    "trajectory_points",
    "trajectory_delta_x_m",
    "trajectory_delta_y_m",
    "trajectory_path_length_m",
    "approx_delta_speed_mps",
)


@dataclass(frozen=True)
class AuditCandidate:
    sample_token: str
    scene_token: str
    timestamp: str
    cam_front_path: str
    visualization_path: str
    derived_action: str
    label_rule_version: str
    safety_rule_version: str
    split: str
    action_confidence: str
    has_vru: str
    safety_status: str
    safety_score: str
    safety_penalty: str
    boundary_flags: tuple[str, ...]
    uncertainty_reason: str
    nearby_agent_count: str
    trajectory_points: str
    trajectory_delta_x_m: str
    trajectory_delta_y_m: str
    trajectory_path_length_m: str
    approx_delta_speed_mps: str


@dataclass(frozen=True)
class AuditRecord:
    sample_token: str
    scene_token: str
    timestamp: str
    cam_front_path: str
    visualization_path: str
    derived_action: str
    reviewed_action: str
    label_correct: str
    trajectory_alignment_correct: str
    agent_alignment_correct: str
    safety_score_reasonable: str
    error_type: str
    review_note: str
    label_rule_version: str
    safety_rule_version: str
    split: str
    action_confidence: str
    selection_reason: str
    has_vru: str
    safety_status: str
    safety_score: str
    safety_penalty: str
    boundary_flags: str
    uncertainty_reason: str
    nearby_agent_count: str
    trajectory_points: str
    trajectory_delta_x_m: str
    trajectory_delta_y_m: str
    trajectory_path_length_m: str
    approx_delta_speed_mps: str


@dataclass(frozen=True)
class SelectionResult:
    records: tuple[AuditRecord, ...]
    warnings: tuple[str, ...]


def read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded: object = json.loads(line)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append(loaded)
    return tuple(rows)


def _value_as_string(
    mapping: Mapping[str, object],
    keys: tuple[str, ...],
    default: str = NOT_AVAILABLE,
) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value)
    return default


def _mapping_value(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _tuple_string_value(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return ()


def _manifest_index(
    review_manifest_path: Path | None,
) -> dict[str, Mapping[str, object]]:
    if review_manifest_path is None or not review_manifest_path.exists():
        return {}
    return {
        _value_as_string(row, ("sample_token",)): row
        for row in read_jsonl(review_manifest_path)
    }


def _relative_manifest_visualization_path(
    manifest_path: Path | None,
    manifest_row: Mapping[str, object],
) -> str:
    raw_path = _value_as_string(manifest_row, ("visualization_path",), "")
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.name
    if manifest_path is None:
        return path.as_posix()
    manifest_base = (
        manifest_path.parent
        if manifest_path.is_absolute()
        else PROJECT_ROOT / manifest_path.parent
    )
    repo_relative_path = manifest_base / path
    try:
        return repo_relative_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _default_visualization_path(
    sample_token: str,
    visualization_dir: Path | None,
) -> str:
    if visualization_dir is None:
        return NOT_AVAILABLE
    path = visualization_dir / f"{sample_token}_one_page.png"
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def _infer_has_vru(
    derived_row: Mapping[str, object],
    manifest_row: Mapping[str, object],
) -> str:
    for row in (derived_row, manifest_row):
        for key in ("has_vru", "contains_vru"):
            value = row.get(key)
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, str) and value.lower() in {"yes", "no"}:
                return value.lower()
        vru_count = row.get("nearby_vru_count")
        if isinstance(vru_count, (int, float)):
            return "yes" if vru_count > 0 else "no"
        nearby_agents = row.get("nearby_agents")
        if isinstance(nearby_agents, Sequence) and not isinstance(
            nearby_agents,
            str,
        ):
            for agent in nearby_agents:
                if isinstance(agent, Mapping) and agent.get("is_vru") is True:
                    return "yes"
            if nearby_agents:
                return "no"
    return NOT_AVAILABLE


def _infer_safety_status(row: Mapping[str, object]) -> str:
    status = _value_as_string(
        row,
        ("safety_status", "safety_label"),
        default="",
    ).strip().lower()
    if status in {"safe", "unsafe"}:
        return status
    is_safe = row.get("is_safe")
    if isinstance(is_safe, bool):
        return "safe" if is_safe else "unsafe"
    for key in ("safety_penalty", "total_penalty"):
        penalty = row.get(key)
        if isinstance(penalty, (int, float)):
            return "unsafe" if penalty > 0.0 else "safe"
    return NOT_AVAILABLE


def _is_boundary_sample(candidate: AuditCandidate) -> bool:
    return bool(candidate.boundary_flags) or candidate.uncertainty_reason not in {
        "",
        NO_UNCERTAINTY,
        NOT_AVAILABLE,
    }


def _candidate_from_rows(
    derived_row: Mapping[str, object],
    manifest_row: Mapping[str, object],
    manifest_path: Path | None,
    visualization_dir: Path | None,
) -> AuditCandidate:
    sample_token = _value_as_string(derived_row, ("sample_token",))
    rule_features = _mapping_value(derived_row, "rule_features")
    manifest_visualization = _relative_manifest_visualization_path(
        manifest_path,
        manifest_row,
    )
    visualization_path = manifest_visualization or _default_visualization_path(
        sample_token,
        visualization_dir,
    )
    boundary_flags = _tuple_string_value(derived_row.get("boundary_flags"))
    return AuditCandidate(
        sample_token=sample_token,
        scene_token=_value_as_string(
            manifest_row,
            ("scene_token",),
            _value_as_string(derived_row, ("scene_token",)),
        ),
        timestamp=_value_as_string(
            manifest_row,
            ("timestamp",),
            _value_as_string(derived_row, ("timestamp", "timestamp_us")),
        ),
        cam_front_path=_value_as_string(
            manifest_row,
            ("cam_front_path",),
            _value_as_string(derived_row, ("cam_front_path",)),
        ),
        visualization_path=visualization_path,
        derived_action=_value_as_string(derived_row, ("derived_action",)),
        label_rule_version=_value_as_string(
            derived_row,
            ("label_rule_version",),
            _value_as_string(manifest_row, ("label_rule_version",)),
        ),
        safety_rule_version=_value_as_string(
            derived_row,
            ("safety_rule_version",),
            _value_as_string(manifest_row, ("safety_rule_version",)),
        ),
        split=_value_as_string(
            derived_row,
            ("split",),
            _value_as_string(manifest_row, ("split",)),
        ),
        action_confidence=_value_as_string(
            derived_row,
            ("action_confidence",),
        ),
        has_vru=_infer_has_vru(derived_row, manifest_row),
        safety_status=_infer_safety_status(derived_row),
        safety_score=_value_as_string(
            derived_row,
            ("safety_score", "total_safety_score"),
        ),
        safety_penalty=_value_as_string(
            derived_row,
            ("safety_penalty", "total_penalty"),
        ),
        boundary_flags=boundary_flags,
        uncertainty_reason=_value_as_string(
            derived_row,
            ("uncertainty_reason",),
        ),
        nearby_agent_count=_value_as_string(
            manifest_row,
            ("nearby_agent_count",),
            _value_as_string(derived_row, ("nearby_agent_count",)),
        ),
        trajectory_points=_value_as_string(
            derived_row,
            ("trajectory_points",),
            _value_as_string(rule_features, ("trajectory_points",)),
        ),
        trajectory_delta_x_m=_value_as_string(
            rule_features,
            ("delta_x_m",),
            _value_as_string(derived_row, ("trajectory_last_x_m",)),
        ),
        trajectory_delta_y_m=_value_as_string(
            rule_features,
            ("delta_y_m",),
            _value_as_string(derived_row, ("trajectory_last_y_m",)),
        ),
        trajectory_path_length_m=_value_as_string(
            rule_features,
            ("path_length_m",),
        ),
        approx_delta_speed_mps=_value_as_string(
            rule_features,
            ("approx_delta_speed_mps",),
        ),
    )


def load_candidates(
    derived_path: Path,
    review_manifest_path: Path | None,
    visualization_dir: Path | None,
) -> tuple[AuditCandidate, ...]:
    manifest_rows = _manifest_index(review_manifest_path)
    return tuple(
        _candidate_from_rows(
            derived_row=row,
            manifest_row=manifest_rows.get(
                _value_as_string(row, ("sample_token",)),
                {},
            ),
            manifest_path=review_manifest_path,
            visualization_dir=visualization_dir,
        )
        for row in read_jsonl(derived_path)
    )


def _record_from_candidate(
    candidate: AuditCandidate,
    selection_reason: str,
) -> AuditRecord:
    return AuditRecord(
        sample_token=candidate.sample_token,
        scene_token=candidate.scene_token,
        timestamp=candidate.timestamp,
        cam_front_path=candidate.cam_front_path,
        visualization_path=candidate.visualization_path,
        derived_action=candidate.derived_action,
        reviewed_action="uncertain",
        label_correct="uncertain",
        trajectory_alignment_correct="uncertain",
        agent_alignment_correct="uncertain",
        safety_score_reasonable=(
            "not_available"
            if candidate.safety_status == NOT_AVAILABLE
            else "uncertain"
        ),
        error_type="",
        review_note="",
        label_rule_version=candidate.label_rule_version,
        safety_rule_version=candidate.safety_rule_version,
        split=candidate.split,
        action_confidence=candidate.action_confidence,
        selection_reason=selection_reason,
        has_vru=candidate.has_vru,
        safety_status=candidate.safety_status,
        safety_score=candidate.safety_score,
        safety_penalty=candidate.safety_penalty,
        boundary_flags=";".join(candidate.boundary_flags),
        uncertainty_reason=candidate.uncertainty_reason,
        nearby_agent_count=candidate.nearby_agent_count,
        trajectory_points=candidate.trajectory_points,
        trajectory_delta_x_m=candidate.trajectory_delta_x_m,
        trajectory_delta_y_m=candidate.trajectory_delta_y_m,
        trajectory_path_length_m=candidate.trajectory_path_length_m,
        approx_delta_speed_mps=candidate.approx_delta_speed_mps,
    )


def _add_candidate(
    selected: dict[str, AuditRecord],
    candidate: AuditCandidate | None,
    reason: str,
) -> None:
    if candidate is None or candidate.sample_token in selected:
        return
    selected[candidate.sample_token] = _record_from_candidate(
        candidate,
        "rule_boundary_sample" if _is_boundary_sample(candidate) else reason,
    )


def _first_candidate(
    candidates: Sequence[AuditCandidate],
    predicate,
) -> AuditCandidate | None:
    return next((candidate for candidate in candidates if predicate(candidate)), None)


def _coverage_warnings(
    candidates: tuple[AuditCandidate, ...],
    records: tuple[AuditRecord, ...],
    target_count: int,
) -> tuple[str, ...]:
    warnings = []
    if len(candidates) < target_count:
        warnings.append(
            f"requested {target_count} samples but only "
            f"{len(candidates)} candidates are available"
        )
    candidate_actions = {candidate.derived_action for candidate in candidates}
    missing_actions = tuple(
        action for action in ACTION_SCHEMA if action not in candidate_actions
    )
    if missing_actions:
        warnings.append(
            "missing derived_action coverage: "
            f"{', '.join(missing_actions)}"
        )
    selected_vru = {record.has_vru for record in records}
    if "yes" not in selected_vru or "no" not in selected_vru:
        warnings.append("selected set does not cover both has_vru=yes and has_vru=no")
    selected_safety = {record.safety_status for record in records}
    if "safe" not in selected_safety or "unsafe" not in selected_safety:
        warnings.append("selected set does not cover both safe and unsafe samples")
    if not any(record.selection_reason == "rule_boundary_sample" for record in records):
        warnings.append("selected set has no rule boundary samples")
    return tuple(warnings)


def select_manual_review_samples(
    candidates: tuple[AuditCandidate, ...],
    target_count: int,
) -> SelectionResult:
    if target_count <= 0:
        raise ValueError("target_count must be positive")

    ordered_candidates = tuple(
        sorted(candidates, key=lambda candidate: candidate.sample_token)
    )
    selected: dict[str, AuditRecord] = {}
    for action in ACTION_SCHEMA:
        _add_candidate(
            selected,
            _first_candidate(
                ordered_candidates,
                lambda candidate, action=action: candidate.derived_action
                == action,
            ),
            f"action_coverage_{action}",
        )
        if len(selected) >= target_count:
            break

    for has_vru, reason in (("yes", "vru_present_sample"), ("no", "vru_absent_sample")):
        if len(selected) >= target_count:
            break
        _add_candidate(
            selected,
            _first_candidate(
                ordered_candidates,
                lambda candidate, has_vru=has_vru: candidate.has_vru
                == has_vru,
            ),
            reason,
        )

    for status, reason in (("unsafe", "unsafe_sample"), ("safe", "safe_sample")):
        if len(selected) >= target_count:
            break
        _add_candidate(
            selected,
            _first_candidate(
                ordered_candidates,
                lambda candidate, status=status: candidate.safety_status
                == status,
            ),
            reason,
        )

    if len(selected) < target_count:
        _add_candidate(
            selected,
            _first_candidate(ordered_candidates, _is_boundary_sample),
            "rule_boundary_sample",
        )

    for candidate in ordered_candidates:
        if len(selected) >= target_count:
            break
        _add_candidate(selected, candidate, "coverage_fill_sample")

    records = tuple(selected.values())
    return SelectionResult(
        records=records,
        warnings=_coverage_warnings(candidates, records, target_count),
    )


def write_review_csv(records: tuple[AuditRecord, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def print_summary(result: SelectionResult, output_path: Path) -> None:
    action_counts = Counter(record.derived_action for record in result.records)
    vru_counts = Counter(record.has_vru for record in result.records)
    safety_counts = Counter(record.safety_status for record in result.records)
    reason_counts = Counter(record.selection_reason for record in result.records)
    print(f"output: {output_path}")
    print(f"selected samples: {len(result.records)}")
    print(f"derived_action coverage: {dict(sorted(action_counts.items()))}")
    print(f"has_vru coverage: {dict(sorted(vru_counts.items()))}")
    print(f"safety_status coverage: {dict(sorted(safety_counts.items()))}")
    print(f"selection_reason coverage: {dict(sorted(reason_counts.items()))}")
    for warning in result.warnings:
        print(f"warning: {warning}")


def _existing_optional_path(path: Path | None) -> Path | None:
    if path is not None and path.exists():
        return path
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Phase -1.7 samples for manual meta-action audit."
    )
    parser.add_argument(
        "--derived-labels",
        type=Path,
        default=DEFAULT_DERIVED_LABELS,
    )
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=DEFAULT_REVIEW_MANIFEST,
    )
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-count", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if not arguments.derived_labels.exists():
        print(
            f"missing derived label input: {arguments.derived_labels}",
            file=sys.stderr,
        )
        return 1

    review_manifest = _existing_optional_path(arguments.review_manifest)
    if review_manifest is None:
        print(f"warning: review manifest not found: {arguments.review_manifest}")
    visualization_dir = arguments.visualization_dir
    if visualization_dir is None and review_manifest is not None:
        visualization_dir = review_manifest.parent / "visualizations"

    candidates = load_candidates(
        derived_path=arguments.derived_labels,
        review_manifest_path=review_manifest,
        visualization_dir=visualization_dir,
    )
    result = select_manual_review_samples(
        candidates=candidates,
        target_count=arguments.target_count,
    )
    write_review_csv(result.records, arguments.output)
    print_summary(result, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from nuscenes.nuscenes import NuScenes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from derive_meta_action import (
    MetaActionRules,
    _load_data_config,
    derive_sample_record,
    load_meta_action_rules,
)
from inspect_nuscenes_sample import (
    CAMERA_CHANNEL,
    extract_future_ego_trajectory,
    get_nearby_agents,
)
from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION


EXPECTED_TOTAL_SAMPLES = 108
EXPECTED_LABEL_RULE_VERSION = LABEL_RULE_VERSION
EXPECTED_ACTION_DISTRIBUTION = Counter(
    {
        "accelerate": 6,
        "decelerate": 16,
        "keep": 55,
        "left_lateral": 5,
        "right_lateral": 5,
        "stop": 21,
    }
)
EXPECTED_TRANSITION_DISTRIBUTION = Counter(
    {
        ("accelerate", "accelerate"): 5,
        ("decelerate", "decelerate"): 13,
        ("keep", "keep"): 55,
        ("keep", "accelerate"): 1,
        ("keep", "decelerate"): 3,
        ("keep", "stop"): 1,
        ("left_lateral", "left_lateral"): 5,
        ("right_lateral", "right_lateral"): 5,
        ("stop", "stop"): 20,
    }
)
EXPECTED_HISTORICAL_LABEL_RULE_VERSION = "phase-1.6-meta-action-v0.1"
EXPECTED_HISTORICAL_LABEL_CORRECT_DISTRIBUTION = Counter(
    {"yes": 103, "no": 5}
)
NO_UNCERTAINTY_VALUES = {"", "none", "not_available"}
AUDIT_OUTPUT_FIELDS = (
    "source_audit",
    "sample_token",
    "scene_token",
    "timestamp",
    "cam_front_path",
    "historical_derived_action",
    "reviewed_action",
    "frozen_action",
    "action_match",
    "label_rule_version",
    "action_confidence",
    "boundary_flags",
    "uncertainty_reason",
    "trajectory_points",
    "trajectory_last_t_sec",
    "trajectory_complete",
    "nearby_agent_count",
    "nearby_vru_count",
    "has_vru",
    "cam_front_exists",
)


@dataclass(frozen=True)
class HistoricalAuditRow:
    source_audit: str
    sample_token: str
    scene_token: str
    timestamp: str
    cam_front_path: str
    historical_derived_action: str
    reviewed_action: str
    label_correct: str
    trajectory_alignment_correct: str
    agent_alignment_correct: str
    label_rule_version: str


@dataclass(frozen=True)
class FreezeAuditRecord:
    source_audit: str
    sample_token: str
    scene_token: str
    timestamp: int
    cam_front_path: str
    historical_derived_action: str
    reviewed_action: str
    frozen_action: str
    action_match: str
    label_rule_version: str
    action_confidence: str
    boundary_flags: tuple[str, ...]
    uncertainty_reason: str
    trajectory_points: int
    trajectory_last_t_sec: float
    trajectory_complete: bool
    nearby_agent_count: int
    nearby_vru_count: int
    has_vru: str
    cam_front_exists: bool


@dataclass(frozen=True)
class FreezeSummary:
    total_samples: int
    unique_sample_tokens: int
    action_match_count: int
    action_distribution: Counter[str]
    vru_distribution: Counter[str]
    boundary_flag_case_count: int
    diagnostic_case_count: int
    boundary_flag_distribution: Counter[str]
    uncertainty_reason_distribution: Counter[str]
    trajectory_complete_count: int
    cam_front_exists_count: int
    transition_distribution: Counter[tuple[str, str]]
    failures: tuple[str, ...]


def _required_value(row: dict[str, str], key: str, path: Path) -> str:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{path}: missing {key}")
    return value


def _read_audit(path: Path, source_audit: str) -> tuple[HistoricalAuditRow, ...]:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = tuple(csv.DictReader(input_file))

    audit_rows = []
    for row in rows:
        reviewed_action = _required_value(row, "reviewed_action", path)
        historical_action = _required_value(row, "derived_action", path)
        if reviewed_action not in ACTION_SCHEMA:
            raise ValueError(
                f"{path}: unsupported reviewed_action {reviewed_action!r}"
            )
        if historical_action not in ACTION_SCHEMA:
            raise ValueError(
                f"{path}: unsupported derived_action {historical_action!r}"
            )
        label_correct = _required_value(row, "label_correct", path).lower()
        if label_correct not in {"yes", "no"}:
            raise ValueError(
                f"{path}: label_correct must be yes or no, got "
                f"{label_correct!r}"
            )
        trajectory_alignment_correct = _required_value(
            row,
            "trajectory_alignment_correct",
            path,
        ).lower()
        if trajectory_alignment_correct != "yes":
            raise ValueError(
                f"{path}: trajectory_alignment_correct must be yes"
            )
        agent_alignment_correct = _required_value(
            row,
            "agent_alignment_correct",
            path,
        ).lower()
        if agent_alignment_correct != "yes":
            raise ValueError(
                f"{path}: agent_alignment_correct must be yes"
            )
        label_rule_version = _required_value(
            row,
            "label_rule_version",
            path,
        )
        if label_rule_version != EXPECTED_HISTORICAL_LABEL_RULE_VERSION:
            raise ValueError(
                f"{path}: historical label_rule_version must be "
                f"{EXPECTED_HISTORICAL_LABEL_RULE_VERSION}"
            )
        audit_rows.append(
            HistoricalAuditRow(
                source_audit=source_audit,
                sample_token=_required_value(row, "sample_token", path),
                scene_token=_required_value(row, "scene_token", path),
                timestamp=_required_value(row, "timestamp", path),
                cam_front_path=_required_value(row, "cam_front_path", path),
                historical_derived_action=historical_action,
                reviewed_action=reviewed_action,
                label_correct=label_correct,
                trajectory_alignment_correct=trajectory_alignment_correct,
                agent_alignment_correct=agent_alignment_correct,
                label_rule_version=label_rule_version,
            )
        )
    return tuple(audit_rows)


def read_and_merge_audits(
    base_audit: Path,
    supplement_audit: Path,
) -> tuple[HistoricalAuditRow, ...]:
    rows = _read_audit(base_audit, "base") + _read_audit(
        supplement_audit,
        "supplement",
    )
    sample_tokens = tuple(row.sample_token for row in rows)
    if len(sample_tokens) != len(set(sample_tokens)):
        duplicates = sorted(
            token
            for token, count in Counter(sample_tokens).items()
            if count > 1
        )
        raise ValueError(
            "duplicate sample_token across audit CSVs: "
            + ", ".join(duplicates)
        )
    return rows


def validate_historical_audit_integrity(
    rows: tuple[HistoricalAuditRow, ...],
) -> None:
    label_correct_distribution = Counter(
        row.label_correct for row in rows
    )
    if label_correct_distribution != EXPECTED_HISTORICAL_LABEL_CORRECT_DISTRIBUTION:
        raise ValueError(
            "historical label_correct distribution must be "
            f"{dict(sorted(EXPECTED_HISTORICAL_LABEL_CORRECT_DISTRIBUTION.items()))}: "
            f"{dict(sorted(label_correct_distribution.items()))}"
        )
    inconsistent_rows = tuple(
        row.sample_token
        for row in rows
        if row.label_correct
        != (
            "yes"
            if row.historical_derived_action == row.reviewed_action
            else "no"
        )
    )
    if inconsistent_rows:
        raise ValueError(
            "historical label_correct does not match derived/reviewed action: "
            + ", ".join(inconsistent_rows)
        )


def validate_cam_front_path(cam_front_path: str, dataroot: Path) -> bool:
    path = Path(cam_front_path)
    if path.is_absolute():
        return False
    resolved_root = dataroot.resolve()
    resolved_candidate = (dataroot / path).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        return False
    return resolved_candidate.is_file()


def _trajectory_complete(
    trajectory_points: int,
    trajectory_is_truncated: bool,
    trajectory_last_t_sec: float,
    rules: MetaActionRules,
    time_tolerance_sec: float,
) -> bool:
    expected_points = round(rules.horizon_sec / rules.sample_interval_sec) + 1
    return (
        trajectory_points >= expected_points
        and not trajectory_is_truncated
        and trajectory_last_t_sec + time_tolerance_sec >= rules.horizon_sec
    )


def build_freeze_records(
    nuscenes: NuScenes,
    historical_rows: tuple[HistoricalAuditRow, ...],
    rules: MetaActionRules,
    dataroot: Path,
    agent_radius_m: float,
    time_tolerance_sec: float,
) -> tuple[tuple[FreezeAuditRecord, ...], tuple[str, ...]]:
    records = []
    failures = []
    for row in historical_rows:
        try:
            derived_record = derive_sample_record(
                nuscenes=nuscenes,
                sample_token=row.sample_token,
                camera=CAMERA_CHANNEL,
                rules=rules,
                time_tolerance_sec=time_tolerance_sec,
            )
            trajectory = extract_future_ego_trajectory(
                nuscenes=nuscenes,
                sample_token=row.sample_token,
                horizon_sec=rules.horizon_sec,
                sample_interval_sec=rules.sample_interval_sec,
                time_tolerance_sec=time_tolerance_sec,
            )
            nearby_agents = get_nearby_agents(
                nuscenes=nuscenes,
                sample_token=row.sample_token,
                radius_m=agent_radius_m,
            )
            sample = nuscenes.get("sample", row.sample_token)
        except (KeyError, ValueError) as error:
            failures.append(f"{row.sample_token}: data derivation failed: {error}")
            continue

        if derived_record.scene_token != row.scene_token:
            failures.append(
                f"{row.sample_token}: scene_token mismatch "
                f"{row.scene_token} != {derived_record.scene_token}"
            )
        if str(derived_record.timestamp_us) != row.timestamp:
            failures.append(
                f"{row.sample_token}: timestamp mismatch "
                f"{row.timestamp} != {derived_record.timestamp_us}"
            )
        if str(sample["scene_token"]) != row.scene_token:
            failures.append(
                f"{row.sample_token}: nuScenes sample scene_token mismatch"
            )
        if int(sample["timestamp"]) != int(row.timestamp):
            failures.append(
                f"{row.sample_token}: nuScenes sample timestamp mismatch"
            )
        if row.cam_front_path != derived_record.cam_front_path:
            failures.append(
                f"{row.sample_token}: historical CAM_FRONT path mismatch\n"
                f"  historical path: {row.cam_front_path}\n"
                f"  current derived path: {derived_record.cam_front_path}"
            )

        last_t_sec = trajectory.points[-1].t_sec if trajectory.points else 0.0
        trajectory_complete = _trajectory_complete(
            trajectory_points=len(trajectory.points),
            trajectory_is_truncated=trajectory.is_truncated,
            trajectory_last_t_sec=last_t_sec,
            rules=rules,
            time_tolerance_sec=time_tolerance_sec,
        )
        if not trajectory_complete:
            failures.append(f"{row.sample_token}: incomplete 3s trajectory")

        cam_front_exists = validate_cam_front_path(
            derived_record.cam_front_path,
            dataroot,
        )
        if not cam_front_exists:
            failures.append(
                f"{row.sample_token}: CAM_FRONT path is absolute or missing"
            )

        nearby_vru_count = sum(
            agent.is_vru for agent in nearby_agents.agents
        )
        records.append(
            FreezeAuditRecord(
                source_audit=row.source_audit,
                sample_token=row.sample_token,
                scene_token=derived_record.scene_token,
                timestamp=derived_record.timestamp_us,
                cam_front_path=derived_record.cam_front_path,
                historical_derived_action=row.historical_derived_action,
                reviewed_action=row.reviewed_action,
                frozen_action=derived_record.derived_action,
                action_match=(
                    "yes"
                    if derived_record.derived_action == row.reviewed_action
                    else "no"
                ),
                label_rule_version=derived_record.label_rule_version,
                action_confidence=derived_record.action_confidence,
                boundary_flags=derived_record.boundary_flags,
                uncertainty_reason=derived_record.uncertainty_reason,
                trajectory_points=len(trajectory.points),
                trajectory_last_t_sec=last_t_sec,
                trajectory_complete=trajectory_complete,
                nearby_agent_count=len(nearby_agents.agents),
                nearby_vru_count=nearby_vru_count,
                has_vru="yes" if nearby_vru_count else "no",
                cam_front_exists=cam_front_exists,
            )
        )
    return tuple(records), tuple(failures)


def summarize_freeze_records(
    records: tuple[FreezeAuditRecord, ...],
) -> FreezeSummary:
    action_distribution = Counter(record.frozen_action for record in records)
    vru_distribution = Counter(record.has_vru for record in records)
    boundary_flag_records = tuple(
        record for record in records if record.boundary_flags
    )
    diagnostic_records = tuple(
        record
        for record in records
        if record.boundary_flags
        or record.uncertainty_reason.strip().lower()
        not in NO_UNCERTAINTY_VALUES
    )
    boundary_flag_distribution = Counter(
        flag
        for record in boundary_flag_records
        for flag in record.boundary_flags
    )
    uncertainty_reason_distribution = Counter(
        record.uncertainty_reason for record in records
    )
    transition_distribution = Counter(
        (record.historical_derived_action, record.frozen_action)
        for record in records
    )
    action_match_count = sum(
        record.action_match == "yes" for record in records
    )
    trajectory_complete_count = sum(
        record.trajectory_complete for record in records
    )
    cam_front_exists_count = sum(
        record.cam_front_exists for record in records
    )
    failures = []
    if len(records) != EXPECTED_TOTAL_SAMPLES:
        failures.append(
            f"total_samples={len(records)}, expected={EXPECTED_TOTAL_SAMPLES}"
        )
    unique_sample_tokens = len({record.sample_token for record in records})
    if unique_sample_tokens != len(records):
        failures.append("duplicate sample_token in freeze records")
    audit_source_counts = Counter(record.source_audit for record in records)
    if audit_source_counts != Counter({"base": 100, "supplement": 8}):
        failures.append(
            "audit source counts must be base=100 and supplement=8"
        )
    if action_match_count != EXPECTED_TOTAL_SAMPLES:
        failures.append(
            f"action_match={action_match_count}/{EXPECTED_TOTAL_SAMPLES}"
        )
    if any(
        record.action_match
        != ("yes" if record.frozen_action == record.reviewed_action else "no")
        for record in records
    ):
        failures.append("action_match value is inconsistent with frozen_action")
    if action_distribution != EXPECTED_ACTION_DISTRIBUTION:
        failures.append(
            "frozen action distribution does not match expected: "
            f"{dict(sorted(action_distribution.items()))}"
        )
    if transition_distribution != EXPECTED_TRANSITION_DISTRIBUTION:
        failures.append(
            "transition distribution does not match expected: "
            f"{dict(sorted(transition_distribution.items()))}"
        )
    label_versions = {record.label_rule_version for record in records}
    if label_versions != {EXPECTED_LABEL_RULE_VERSION}:
        failures.append(
            "label_rule_version must be "
            f"{EXPECTED_LABEL_RULE_VERSION}: {sorted(label_versions)}"
        )
    if set(vru_distribution) - {"yes", "no"}:
        failures.append("has_vru contains unsupported values")
    if not vru_distribution.get("yes") or not vru_distribution.get("no"):
        failures.append("has_vru requires both yes and no coverage")
    if not boundary_flag_records:
        failures.append("boundary_flag_case_count=0")
    elif not any(
        any(
            keyword in flag
            for keyword in ("speed", "stop", "lateral")
        )
        for record in boundary_flag_records
        for flag in record.boundary_flags
    ):
        failures.append("boundary coverage has no speed, stop, or lateral flag")
    if trajectory_complete_count != EXPECTED_TOTAL_SAMPLES:
        failures.append(
            "trajectory_complete="
            f"{trajectory_complete_count}/{EXPECTED_TOTAL_SAMPLES}"
        )
    if cam_front_exists_count != EXPECTED_TOTAL_SAMPLES:
        failures.append(
            "cam_front_exists="
            f"{cam_front_exists_count}/{EXPECTED_TOTAL_SAMPLES}"
        )
    return FreezeSummary(
        total_samples=len(records),
        unique_sample_tokens=unique_sample_tokens,
        action_match_count=action_match_count,
        action_distribution=action_distribution,
        vru_distribution=vru_distribution,
        boundary_flag_case_count=len(boundary_flag_records),
        diagnostic_case_count=len(diagnostic_records),
        boundary_flag_distribution=boundary_flag_distribution,
        uncertainty_reason_distribution=uncertainty_reason_distribution,
        trajectory_complete_count=trajectory_complete_count,
        cam_front_exists_count=cam_front_exists_count,
        transition_distribution=transition_distribution,
        failures=tuple(failures),
    )


def _write_audit(
    records: tuple[FreezeAuditRecord, ...],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=AUDIT_OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["boundary_flags"] = "|".join(record.boundary_flags)
            row["trajectory_complete"] = "yes" if record.trajectory_complete else "no"
            row["cam_front_exists"] = "yes" if record.cam_front_exists else "no"
            writer.writerow(row)


def _print_summary(summary: FreezeSummary) -> None:
    print(f"total samples: {summary.total_samples}")
    print(f"unique sample tokens: {summary.unique_sample_tokens}")
    print(
        "action_match: "
        f"{summary.action_match_count}/{EXPECTED_TOTAL_SAMPLES}"
    )
    print(
        "frozen action distribution: "
        f"{dict(sorted(summary.action_distribution.items()))}"
    )
    print(f"has_vru distribution: {dict(sorted(summary.vru_distribution.items()))}")
    print(
        "strict boundary-flag case count: "
        f"{summary.boundary_flag_case_count}"
    )
    print(f"diagnostic case count: {summary.diagnostic_case_count}")
    print(
        "boundary flag distribution: "
        f"{dict(sorted(summary.boundary_flag_distribution.items()))}"
    )
    print(
        "uncertainty reason distribution: "
        f"{dict(sorted(summary.uncertainty_reason_distribution.items()))}"
    )
    print(
        "trajectory_complete: "
        f"{summary.trajectory_complete_count}/{EXPECTED_TOTAL_SAMPLES}"
    )
    print(
        "cam_front_exists: "
        f"{summary.cam_front_exists_count}/{EXPECTED_TOTAL_SAMPLES}"
    )
    print("historical -> frozen transition summary:")
    for (historical_action, frozen_action), count in sorted(
        summary.transition_distribution.items()
    ):
        print(f"  {historical_action} -> {frozen_action}: {count}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Phase -1 meta-action v0.2 label freeze gate."
    )
    parser.add_argument(
        "--base-audit",
        type=Path,
        default=Path("data/phase_1_7_manual_audit.csv"),
    )
    parser.add_argument(
        "--supplement-audit",
        type=Path,
        default=Path("data/phase_1_7_lateral_supplement_audit.csv"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
    )
    parser.add_argument(
        "--action-config",
        type=Path,
        default=Path("configs/action_rules.yaml"),
    )
    parser.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase_1_9_label_freeze_audit.csv"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        historical_rows = read_and_merge_audits(
            arguments.base_audit,
            arguments.supplement_audit,
        )
        validate_historical_audit_integrity(historical_rows)
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
        records, record_failures = build_freeze_records(
            nuscenes=nuscenes,
            historical_rows=historical_rows,
            rules=rules,
            dataroot=data_config.nuscenes_root,
            agent_radius_m=data_config.nearby_radius_m,
            time_tolerance_sec=data_config.trajectory_time_tolerance_sec,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"PHASE -1 LABEL FREEZE GATE: FAIL\n- {error}", file=sys.stderr)
        return 1

    summary = summarize_freeze_records(records)
    _print_summary(summary)
    failures = record_failures + summary.failures
    if failures:
        print("PHASE -1 LABEL FREEZE GATE: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    _write_audit(records, arguments.output)
    print(f"output: {arguments.output}")
    print("PHASE -1 LABEL FREEZE GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

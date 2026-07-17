#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import sys

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DATA_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DATA_DIRECTORY))

from derive_meta_action import (
    MetaActionRules,
    derive_meta_action,
    load_meta_action_rules,
)
from inspect_nuscenes_sample import (
    CAMERA_CHANNEL,
    extract_future_ego_trajectory,
    get_nearby_agents,
    load_trajectory_config,
    select_future_sample_grid,
)
from validate_label_freeze import (
    HistoricalAuditRow,
    read_and_merge_audits,
    validate_historical_audit_integrity,
)
from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION
from src.phase0.manifest import (
    COORDINATE_METADATA,
    SAFETY_RULE_VERSION,
    SourceAuditRecord,
    current_ego_motion,
    current_ego_pose,
    json_record,
    write_jsonl_records,
)
from src.phase0.protocol import (
    ManifestValidationSummary,
    OFFICIAL_TRAIN_SCENE_COUNT,
    OFFICIAL_VAL_SCENE_COUNT,
    PHASE0_SPLIT_SEED,
    PROJECT_TRAIN_SCENE_COUNT,
    PROJECT_VALIDATION_SCENE_COUNT,
    SPLITS,
    TRAINVAL_MANIFEST_SCHEMA_VERSION,
    complete_action_distribution,
    is_valid_cam_front_path,
    select_pilot_scene_tokens,
    validate_manifest,
)
from src.phase0.scene_mapping import (
    build_scene_mapping_payload,
    ensure_scene_mapping,
    hash_scene_histograms,
)
from src.phase0.stratified_split import (
    SPLIT_STRATEGY_VERSION,
    SplitQuality,
    assign_fixed_random_scene_splits,
    assign_stratified_scene_splits,
    evaluate_scene_split,
)


EXCLUSION_REASONS = (
    "insufficient_remaining_horizon",
    "timestamp_out_of_tolerance",
    "broken_next_chain",
    "scene_mismatch",
    "missing_cam_front",
    "missing_cam_front_file",
    "missing_ego_pose",
    "label_derivation_error",
    "other_error",
)
UNFROZEN_SPLIT_MAPPING_SHA256 = "unfrozen_scene_mapping"


@dataclass(frozen=True)
class TrainvalManifestConfig:
    version: str
    split_seed: int
    split_strategy_version: str
    pilot_seed: int
    pilot_scene_count: int
    data_config_path: Path
    action_config_path: Path
    base_audit_path: Path
    supplement_audit_path: Path
    manifest_relative_path: Path
    pilot_manifest_relative_path: Path
    scene_split_mapping_relative_path: Path


@dataclass(frozen=True)
class TrainvalManifestRecord:
    sample_token: str
    scene_token: str
    timestamp: int
    cam_front_path: str
    current_ego_pose: dict[str, object]
    current_ego_motion: dict[str, object]
    coordinate_metadata: dict[str, dict[str, str]]
    future_ego_trajectory: tuple[object, ...]
    nearby_agents: tuple[object, ...]
    meta_action: str
    label_rule_version: str
    safety_rule_version: str
    manifest_schema_version: str
    split: str
    official_split: str
    split_seed: int
    split_strategy_version: str
    split_mapping_sha256: str
    audit_status: str
    source_audit_record: SourceAuditRecord | None


@dataclass(frozen=True)
class SampleDecision:
    record: TrainvalManifestRecord | None
    exclusion_reason: str | None


@dataclass(frozen=True)
class BuildResult:
    records: tuple[TrainvalManifestRecord, ...]
    scanned_sample_count: int
    exclusion_counts: dict[str, int]
    audit_outcomes: dict[str, str]


@dataclass(frozen=True)
class OfficialSceneTokens:
    train: tuple[str, ...]
    val: tuple[str, ...]


@dataclass(frozen=True)
class SceneLabelStatistics:
    scene_histograms: dict[str, dict[str, int]]
    sample_distribution: dict[str, int]
    scene_support: dict[str, int]
    scanned_sample_count: int
    included_sample_count: int
    exclusion_counts: dict[str, int]
    audit_outcomes: dict[str, str]


@dataclass(frozen=True)
class FullSplitResult:
    scene_splits: dict[str, str]
    official_splits: dict[str, str]
    stratified_quality: SplitQuality
    random_quality: SplitQuality
    train_statistics: SceneLabelStatistics
    test_statistics: SceneLabelStatistics
    refinement_count: int


def _string_value(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _int_value(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"configuration missing integer {key}")
    return value


def load_config(config_path: Path) -> TrainvalManifestConfig:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    return TrainvalManifestConfig(
        version=_string_value(loaded, "version"),
        split_seed=_int_value(loaded, "split_seed"),
        split_strategy_version=_string_value(
            loaded,
            "split_strategy_version",
        ),
        pilot_seed=_int_value(loaded, "pilot_seed"),
        pilot_scene_count=_int_value(loaded, "pilot_scene_count"),
        data_config_path=Path(_string_value(loaded, "data_config_path")),
        action_config_path=Path(_string_value(loaded, "action_config_path")),
        base_audit_path=Path(_string_value(loaded, "base_audit_path")),
        supplement_audit_path=Path(
            _string_value(loaded, "supplement_audit_path")
        ),
        manifest_relative_path=Path(
            _string_value(loaded, "manifest_relative_path")
        ),
        pilot_manifest_relative_path=Path(
            _string_value(loaded, "pilot_manifest_relative_path")
        ),
        scene_split_mapping_relative_path=Path(
            _string_value(loaded, "scene_split_mapping_relative_path")
        ),
    )


def load_audit_index(
    base_audit_path: Path,
    supplement_audit_path: Path,
) -> dict[str, HistoricalAuditRow]:
    rows = read_and_merge_audits(base_audit_path, supplement_audit_path)
    validate_historical_audit_integrity(rows)
    return {row.sample_token: row for row in rows}


def audit_token_summary(
    result: FullSplitResult,
    audit_index: Mapping[str, HistoricalAuditRow],
) -> dict[str, object]:
    outcomes = dict(result.train_statistics.audit_outcomes)
    duplicate_tokens = set(outcomes) & set(result.test_statistics.audit_outcomes)
    if duplicate_tokens:
        raise ValueError("audit token appears in official train and val scans")
    outcomes.update(result.test_statistics.audit_outcomes)
    missing_tokens = sorted(set(audit_index) - set(outcomes))
    filtered_reasons = Counter(
        outcome for outcome in outcomes.values() if outcome != "included"
    )
    return {
        "audit_token_count": len(audit_index),
        "successfully_matched_count": sum(
            outcome == "included" for outcome in outcomes.values()
        ),
        "filtered_count": sum(
            outcome != "included" for outcome in outcomes.values()
        ),
        "filtered_reason_counts": dict(sorted(filtered_reasons.items())),
        "missing_count": len(missing_tokens),
        "missing_sample_tokens": missing_tokens,
    }


def require_complete_audit_coverage(
    result: FullSplitResult,
    audit_index: Mapping[str, HistoricalAuditRow],
) -> dict[str, object]:
    summary = audit_token_summary(result, audit_index)
    expected_count = len(audit_index)
    if (
        summary["successfully_matched_count"] != expected_count
        or summary["filtered_count"] != 0
        or summary["missing_count"] != 0
    ):
        print(
            json.dumps(
                {"audit_coverage_gate": summary},
                indent=2,
                sort_keys=True,
            )
        )
        raise ValueError(
            "audit coverage gate failed: expected all audit tokens to be "
            "matched with no filtered or missing tokens"
        )
    return summary


def _environment_path(name: str, *, require_absolute: bool = False) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser()
    if require_absolute and not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path.resolve()


def output_path(derived_root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("manifest output path must be relative to VLA_DERIVED_ROOT")
    return derived_root / relative_path


def resolve_official_scene_tokens(
    nuscenes: NuScenes,
    official_scene_names: Mapping[str, Sequence[str]] | None = None,
) -> OfficialSceneTokens:
    scene_name_to_token = {
        str(scene["name"]): str(scene["token"])
        for scene in nuscenes.scene
    }
    if len(scene_name_to_token) != len(nuscenes.scene):
        raise ValueError("nuScenes scene names must be unique")
    split_names = official_scene_names or create_splits_scenes()
    official_train = tuple(sorted(set(split_names["train"])))
    official_val = tuple(sorted(set(split_names["val"])))
    if len(official_train) != OFFICIAL_TRAIN_SCENE_COUNT:
        raise ValueError("official train split must contain 700 unique scenes")
    if len(official_val) != OFFICIAL_VAL_SCENE_COUNT:
        raise ValueError("official val split must contain 150 unique scenes")
    if set(official_train) & set(official_val):
        raise ValueError("official train and val scenes overlap")
    expected_names = set(official_train) | set(official_val)
    missing_names = expected_names - set(scene_name_to_token)
    if missing_names:
        raise ValueError(
            f"trainval data is missing {len(missing_names)} official scenes"
        )
    unexpected_names = set(scene_name_to_token) - expected_names
    if unexpected_names:
        raise ValueError(
            f"trainval data contains {len(unexpected_names)} unexpected scenes"
        )
    return OfficialSceneTokens(
        train=tuple(scene_name_to_token[name] for name in official_train),
        val=tuple(scene_name_to_token[name] for name in official_val),
    )


def compose_project_scene_splits(
    train_assignments: Mapping[str, str],
    official_val_tokens: Sequence[str],
) -> dict[str, str]:
    if Counter(train_assignments.values()) != {
        "train": PROJECT_TRAIN_SCENE_COUNT,
        "validation": PROJECT_VALIDATION_SCENE_COUNT,
    }:
        raise ValueError("unexpected official-train project split sizes")
    val_tokens = tuple(sorted(official_val_tokens))
    if len(val_tokens) != OFFICIAL_VAL_SCENE_COUNT:
        raise ValueError("official val split must contain 150 scenes")
    if len(set(val_tokens)) != len(val_tokens):
        raise ValueError("official val scene tokens must be unique")
    if set(train_assignments) & set(val_tokens):
        raise ValueError("official train and val scene tokens overlap")
    scene_splits = dict(train_assignments)
    scene_splits.update((token, "test") for token in val_tokens)
    return scene_splits


def _cam_front_data(
    nuscenes: NuScenes,
    sample: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, str | None]:
    sample_data = sample.get("data")
    if not isinstance(sample_data, Mapping):
        return None, "missing_cam_front"
    camera_token = sample_data.get(CAMERA_CHANNEL)
    if not isinstance(camera_token, str) or not camera_token:
        return None, "missing_cam_front"
    try:
        camera_data = nuscenes.get("sample_data", camera_token)
    except KeyError:
        return None, "missing_cam_front"
    if camera_data.get("sample_token") != sample.get("token"):
        return None, "scene_mismatch"
    return camera_data, None


def evaluate_sample(
    nuscenes: NuScenes,
    sample_token: str,
    expected_scene_token: str,
    split: str,
    official_split: str,
    split_seed: int,
    split_strategy_version: str,
    split_mapping_sha256: str,
    audit_row: HistoricalAuditRow | None,
    dataroot: Path,
    rules: MetaActionRules,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> SampleDecision:
    try:
        sample = nuscenes.get("sample", sample_token)
    except KeyError:
        return SampleDecision(None, "other_error")
    if sample.get("scene_token") != expected_scene_token:
        return SampleDecision(None, "scene_mismatch")

    camera_data, camera_error = _cam_front_data(nuscenes, sample)
    if camera_error is not None:
        return SampleDecision(None, camera_error)
    if camera_data is None:
        return SampleDecision(None, "missing_cam_front")
    filename = camera_data.get("filename")
    if not isinstance(filename, str) or not filename:
        return SampleDecision(None, "missing_cam_front")
    if not is_valid_cam_front_path(filename):
        return SampleDecision(None, "missing_cam_front_file")
    portable_path = PurePosixPath(filename)
    if not (dataroot / Path(filename)).is_file():
        return SampleDecision(None, "missing_cam_front_file")

    try:
        pose = current_ego_pose(nuscenes, sample_token, CAMERA_CHANNEL)
    except (KeyError, TypeError, ValueError):
        return SampleDecision(None, "missing_ego_pose")

    future_grid = select_future_sample_grid(
        nuscenes=nuscenes,
        sample_token=sample_token,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
    )
    if future_grid.exclusion_reason is not None:
        return SampleDecision(None, future_grid.exclusion_reason)

    try:
        trajectory = extract_future_ego_trajectory(
            nuscenes=nuscenes,
            sample_token=sample_token,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            time_tolerance_sec=time_tolerance_sec,
            future_grid=future_grid,
        )
        result = derive_meta_action(
            trajectory.points,
            rules,
            time_tolerance_sec=time_tolerance_sec,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return SampleDecision(None, "label_derivation_error")
    if trajectory.is_truncated != (not future_grid.horizon_covered):
        raise ValueError("future grid and trajectory truncation disagree")

    try:
        motion = current_ego_motion(nuscenes, sample_token, CAMERA_CHANNEL)
        nearby_agents = get_nearby_agents(
            nuscenes=nuscenes,
            sample_token=sample_token,
            radius_m=agent_radius_m,
        )
    except (KeyError, TypeError, ValueError):
        return SampleDecision(None, "other_error")

    label_rule_version = getattr(rules, "label_rule_version", None)
    if label_rule_version != LABEL_RULE_VERSION:
        return SampleDecision(None, "label_derivation_error")
    timestamp = sample.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        return SampleDecision(None, "other_error")
    source_audit_record = None
    audit_status = "unaudited"
    if audit_row is not None:
        if audit_row.scene_token != expected_scene_token:
            raise ValueError(f"{sample_token}: audit scene_token mismatch")
        if int(audit_row.timestamp) != timestamp:
            raise ValueError(f"{sample_token}: audit timestamp mismatch")
        if audit_row.cam_front_path != portable_path.as_posix():
            raise ValueError(f"{sample_token}: audit CAM_FRONT path mismatch")
        if audit_row.reviewed_action != result.derived_action:
            raise ValueError(
                f"{sample_token}: audit action does not match frozen action"
            )
        audit_status = "audited"
        source_audit_record = SourceAuditRecord(
            source_audit=audit_row.source_audit,
            sample_token=audit_row.sample_token,
            historical_derived_action=audit_row.historical_derived_action,
            reviewed_action=audit_row.reviewed_action,
            label_correct=audit_row.label_correct,
            historical_label_rule_version=audit_row.label_rule_version,
        )
    record = TrainvalManifestRecord(
        sample_token=sample_token,
        scene_token=expected_scene_token,
        timestamp=timestamp,
        cam_front_path=portable_path.as_posix(),
        current_ego_pose=pose,
        current_ego_motion=motion,
        coordinate_metadata=COORDINATE_METADATA,
        future_ego_trajectory=tuple(trajectory.points),
        nearby_agents=tuple(nearby_agents.agents),
        meta_action=result.derived_action,
        label_rule_version=LABEL_RULE_VERSION,
        safety_rule_version=SAFETY_RULE_VERSION,
        manifest_schema_version=TRAINVAL_MANIFEST_SCHEMA_VERSION,
        split=split,
        official_split=official_split,
        split_seed=split_seed,
        split_strategy_version=split_strategy_version,
        split_mapping_sha256=split_mapping_sha256,
        audit_status=audit_status,
        source_audit_record=source_audit_record,
    )
    return SampleDecision(record, None)


def scene_sample_tokens(
    nuscenes: NuScenes,
    scene_token: str,
) -> tuple[str, ...]:
    scene = nuscenes.get("scene", scene_token)
    token = scene.get("first_sample_token")
    if not isinstance(token, str) or not token:
        raise ValueError("scene is missing first_sample_token")
    tokens = []
    seen = set()
    while token:
        if token in seen:
            raise ValueError("scene sample chain contains a cycle")
        seen.add(token)
        tokens.append(token)
        sample = nuscenes.get("sample", token)
        next_token = sample.get("next")
        if not isinstance(next_token, str):
            raise ValueError("scene sample chain has an invalid next token")
        token = next_token
    return tuple(tokens)


def build_records(
    nuscenes: NuScenes,
    scene_tokens: Sequence[str],
    scene_splits: Mapping[str, str],
    official_splits: Mapping[str, str],
    split_seed: int,
    split_strategy_version: str,
    split_mapping_sha256: str,
    audit_index: Mapping[str, HistoricalAuditRow],
    dataroot: Path,
    rules: MetaActionRules,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> BuildResult:
    records = []
    exclusions = Counter[str]()
    audit_outcomes = {}
    scanned_sample_count = 0
    for scene_token in scene_tokens:
        split = scene_splits[scene_token]
        for sample_token in scene_sample_tokens(nuscenes, scene_token):
            scanned_sample_count += 1
            decision = evaluate_sample(
                nuscenes=nuscenes,
                sample_token=sample_token,
                expected_scene_token=scene_token,
                split=split,
                official_split=official_splits[scene_token],
                split_seed=split_seed,
                split_strategy_version=split_strategy_version,
                split_mapping_sha256=split_mapping_sha256,
                audit_row=audit_index.get(sample_token),
                dataroot=dataroot,
                rules=rules,
                horizon_sec=horizon_sec,
                sample_interval_sec=sample_interval_sec,
                time_tolerance_sec=time_tolerance_sec,
                agent_radius_m=agent_radius_m,
            )
            if decision.record is not None:
                records.append(decision.record)
            elif decision.exclusion_reason is not None:
                exclusions[decision.exclusion_reason] += 1
            if sample_token in audit_index:
                audit_outcomes[sample_token] = (
                    "included"
                    if decision.record is not None
                    else str(decision.exclusion_reason)
                )
    return BuildResult(
        records=tuple(records),
        scanned_sample_count=scanned_sample_count,
        exclusion_counts={reason: exclusions[reason] for reason in EXCLUSION_REASONS},
        audit_outcomes=audit_outcomes,
    )


def build_scene_label_statistics(
    nuscenes: NuScenes,
    scene_tokens: Sequence[str],
    scene_splits: Mapping[str, str],
    official_splits: Mapping[str, str],
    split_seed: int,
    split_strategy_version: str,
    audit_index: Mapping[str, HistoricalAuditRow],
    dataroot: Path,
    rules: MetaActionRules,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> SceneLabelStatistics:
    histograms = {}
    sample_distribution = Counter[str]()
    exclusions = Counter[str]()
    scanned_sample_count = 0
    included_sample_count = 0
    audit_outcomes = {}
    for scene_token in sorted(scene_tokens):
        result = build_records(
            nuscenes=nuscenes,
            scene_tokens=(scene_token,),
            scene_splits=scene_splits,
            official_splits=official_splits,
            split_seed=split_seed,
            split_strategy_version=split_strategy_version,
            split_mapping_sha256=UNFROZEN_SPLIT_MAPPING_SHA256,
            audit_index=audit_index,
            dataroot=dataroot,
            rules=rules,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            time_tolerance_sec=time_tolerance_sec,
            agent_radius_m=agent_radius_m,
        )
        histogram = complete_action_distribution(
            tuple(record.meta_action for record in result.records)
        )
        histograms[scene_token] = histogram
        sample_distribution.update(histogram)
        exclusions.update(result.exclusion_counts)
        scanned_sample_count += result.scanned_sample_count
        included_sample_count += len(result.records)
        duplicate_audit_tokens = set(audit_outcomes) & set(result.audit_outcomes)
        if duplicate_audit_tokens:
            raise ValueError("audit token was scanned more than once")
        audit_outcomes.update(result.audit_outcomes)
    scene_support = {
        action: sum(
            histogram[action] > 0 for histogram in histograms.values()
        )
        for action in ACTION_SCHEMA
    }
    return SceneLabelStatistics(
        scene_histograms=histograms,
        sample_distribution={
            action: sample_distribution[action] for action in ACTION_SCHEMA
        },
        scene_support=scene_support,
        scanned_sample_count=scanned_sample_count,
        included_sample_count=included_sample_count,
        exclusion_counts={reason: exclusions[reason] for reason in EXCLUSION_REASONS},
        audit_outcomes=audit_outcomes,
    )


def build_full_scene_splits(
    nuscenes: NuScenes,
    split_seed: int,
    split_strategy_version: str,
    audit_index: Mapping[str, HistoricalAuditRow],
    dataroot: Path,
    rules: MetaActionRules,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> FullSplitResult:
    official_tokens = resolve_official_scene_tokens(nuscenes)
    official_splits = {
        **{token: "train" for token in official_tokens.train},
        **{token: "val" for token in official_tokens.val},
    }
    provisional_splits = {token: "train" for token in official_tokens.train}
    train_statistics = build_scene_label_statistics(
        nuscenes=nuscenes,
        scene_tokens=official_tokens.train,
        scene_splits=provisional_splits,
        official_splits=official_splits,
        split_seed=split_seed,
        split_strategy_version=split_strategy_version,
        audit_index=audit_index,
        dataroot=dataroot,
        rules=rules,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
        agent_radius_m=agent_radius_m,
    )
    stratified = assign_stratified_scene_splits(
        scene_histograms=train_statistics.scene_histograms,
        seed=split_seed,
        train_scene_count=PROJECT_TRAIN_SCENE_COUNT,
        validation_scene_count=PROJECT_VALIDATION_SCENE_COUNT,
        action_schema=ACTION_SCHEMA,
        split_strategy_version=split_strategy_version,
    )
    if not stratified.quality.constraints_satisfied:
        unsatisfied = [
            asdict(status)
            for status in stratified.quality.constraint_statuses
            if not status.constraint_satisfied
        ]
        raise ValueError(
            "stratified split constraints are not satisfied: "
            + json.dumps(unsatisfied, sort_keys=True)
        )
    random_assignments = assign_fixed_random_scene_splits(
        scene_tokens=official_tokens.train,
        seed=split_seed,
        train_scene_count=PROJECT_TRAIN_SCENE_COUNT,
        validation_scene_count=PROJECT_VALIDATION_SCENE_COUNT,
    )
    random_quality = evaluate_scene_split(
        train_statistics.scene_histograms,
        random_assignments,
        ACTION_SCHEMA,
    )
    if stratified.quality.objective_score > random_quality.objective_score + 1e-12:
        raise ValueError("stratified split objective is worse than fixed random")
    scene_splits = compose_project_scene_splits(
        stratified.assignments,
        official_tokens.val,
    )
    test_statistics = build_scene_label_statistics(
        nuscenes=nuscenes,
        scene_tokens=official_tokens.val,
        scene_splits=scene_splits,
        official_splits=official_splits,
        split_seed=split_seed,
        split_strategy_version=split_strategy_version,
        audit_index=audit_index,
        dataroot=dataroot,
        rules=rules,
        horizon_sec=horizon_sec,
        sample_interval_sec=sample_interval_sec,
        time_tolerance_sec=time_tolerance_sec,
        agent_radius_m=agent_radius_m,
    )
    return FullSplitResult(
        scene_splits=scene_splits,
        official_splits=official_splits,
        stratified_quality=stratified.quality,
        random_quality=random_quality,
        train_statistics=train_statistics,
        test_statistics=test_statistics,
        refinement_count=stratified.refinement_count,
    )


def full_split_summary(result: FullSplitResult, split_seed: int) -> dict[str, object]:
    stratified = result.stratified_quality
    random_quality = result.random_quality
    return {
        "official_train_scene_count": sum(
            split == "train" for split in result.official_splits.values()
        ),
        "official_val_scene_count": sum(
            split == "val" for split in result.official_splits.values()
        ),
        "project_split_scene_counts": dict(
            Counter(result.scene_splits.values())
        ),
        "scene_split_overlap_count": _scene_overlap_count(result.scene_splits),
        "split_seed": split_seed,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "official_train_scanned_sample_count": (
            result.train_statistics.scanned_sample_count
        ),
        "official_train_included_sample_count": (
            result.train_statistics.included_sample_count
        ),
        "official_train_exclusion_reason_counts": (
            result.train_statistics.exclusion_counts
        ),
        "official_train_sample_distribution": (
            result.train_statistics.sample_distribution
        ),
        "official_train_scene_support": result.train_statistics.scene_support,
        "project_train_sample_distribution": (
            stratified.train_sample_distribution
        ),
        "project_validation_sample_distribution": (
            stratified.validation_sample_distribution
        ),
        "project_train_scene_support": stratified.train_scene_support,
        "project_validation_scene_support": stratified.validation_scene_support,
        "official_val_project_test_scanned_sample_count": (
            result.test_statistics.scanned_sample_count
        ),
        "official_val_project_test_included_sample_count": (
            result.test_statistics.included_sample_count
        ),
        "official_val_project_test_exclusion_reason_counts": (
            result.test_statistics.exclusion_counts
        ),
        "official_val_project_test_sample_distribution": (
            result.test_statistics.sample_distribution
        ),
        "official_val_project_test_scene_support": (
            result.test_statistics.scene_support
        ),
        "stratified_train_distribution_distance": (
            stratified.train_distribution_distance
        ),
        "stratified_validation_distribution_distance": (
            stratified.validation_distribution_distance
        ),
        "stratified_distribution_distance_sum": (
            stratified.train_distribution_distance
            + stratified.validation_distribution_distance
        ),
        "stratified_validation_scene_support_distance": (
            stratified.validation_scene_support_distance
        ),
        "stratified_objective_score": stratified.objective_score,
        "fixed_random_train_distribution_distance": (
            random_quality.train_distribution_distance
        ),
        "fixed_random_validation_distribution_distance": (
            random_quality.validation_distribution_distance
        ),
        "fixed_random_distribution_distance_sum": (
            random_quality.train_distribution_distance
            + random_quality.validation_distribution_distance
        ),
        "fixed_random_objective_score": random_quality.objective_score,
        "stratified_not_worse_than_fixed_random": (
            stratified.objective_score <= random_quality.objective_score + 1e-12
        ),
        "constraints_satisfied": stratified.constraints_satisfied,
        "constraint_statuses": [
            asdict(status) for status in stratified.constraint_statuses
        ],
        "swap_refinement_count": result.refinement_count,
    }


def _scene_overlap_count(scene_splits: Mapping[str, str]) -> int:
    split_scenes = {
        split: {
            token for token, assigned_split in scene_splits.items()
            if assigned_split == split
        }
        for split in SPLITS
    }
    return sum(
        len(split_scenes[first] & split_scenes[second])
        for index, first in enumerate(SPLITS)
        for second in SPLITS[index + 1 :]
    )


def _record_has_absolute_path_leak(
    record: TrainvalManifestRecord,
    dataroot: Path,
    derived_root: Path,
) -> bool:
    roots = (str(dataroot), str(derived_root))

    def contains_root(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(contains_root(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_root(item) for item in value)
        return isinstance(value, str) and any(root in value for root in roots)

    return (
        not is_valid_cam_front_path(record.cam_front_path)
        or contains_root(json_record(record))
    )


def pilot_summary(
    result: BuildResult,
    selected_scene_tokens: Sequence[str],
    scene_splits: Mapping[str, str],
    split_seed: int,
    split_strategy_version: str,
    dataroot: Path,
    derived_root: Path,
) -> dict[str, object]:
    records_by_split = {
        split: tuple(record for record in result.records if record.split == split)
        for split in SPLITS
    }
    absolute_path_leaks = sum(
        _record_has_absolute_path_leak(
            record,
            dataroot,
            derived_root,
        )
        for record in result.records
    )
    return {
        "full_scene_count": len(scene_splits),
        "full_split_scene_counts": dict(Counter(scene_splits.values())),
        "scene_count": len(selected_scene_tokens),
        "split_scene_counts": dict(
            Counter(scene_splits[token] for token in selected_scene_tokens)
        ),
        "scanned_sample_count": result.scanned_sample_count,
        "included_sample_count": len(result.records),
        "excluded_sample_count": (
            result.scanned_sample_count - len(result.records)
        ),
        "exclusion_reason_counts": result.exclusion_counts,
        "split_sample_counts": {
            split: len(records) for split, records in records_by_split.items()
        },
        "split_action_distribution": {
            split: complete_action_distribution(
                tuple(record.meta_action for record in records)
            )
            for split, records in records_by_split.items()
        },
        "motion_availability_distribution": dict(
            Counter(
                str(record.current_ego_motion["availability"])
                for record in result.records
            )
        ),
        "manifest_schema_version": TRAINVAL_MANIFEST_SCHEMA_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "split_seed": split_seed,
        "split_strategy_version": split_strategy_version,
        "split_mapping_sha256": (
            result.records[0].split_mapping_sha256 if result.records else None
        ),
        "audited_record_count": sum(
            record.audit_status == "audited" for record in result.records
        ),
        "unaudited_record_count": sum(
            record.audit_status == "unaudited" for record in result.records
        ),
        "absolute_path_leak_count": absolute_path_leaks,
        "absolute_path_leak_check": (
            "pass" if absolute_path_leaks == 0 else "fail"
        ),
        "scene_split_overlap_count": _scene_overlap_count(scene_splits),
        "scene_split_overlap_check": (
            "pass" if _scene_overlap_count(scene_splits) == 0 else "fail"
        ),
    }


def diagnostic_results(
    nuscenes: NuScenes,
    sample_tokens: Sequence[str],
    scene_splits: Mapping[str, str],
    official_splits: Mapping[str, str],
    split_seed: int,
    split_strategy_version: str,
    split_mapping_sha256: str,
    audit_index: Mapping[str, HistoricalAuditRow],
    dataroot: Path,
    rules: MetaActionRules,
    horizon_sec: float,
    sample_interval_sec: float,
    time_tolerance_sec: float,
    agent_radius_m: float,
) -> dict[str, str]:
    diagnostics = {}
    for sample_token in sample_tokens:
        sample = nuscenes.get("sample", sample_token)
        scene_token = str(sample["scene_token"])
        decision = evaluate_sample(
            nuscenes=nuscenes,
            sample_token=sample_token,
            expected_scene_token=scene_token,
            split=scene_splits[scene_token],
            official_split=official_splits[scene_token],
            split_seed=split_seed,
            split_strategy_version=split_strategy_version,
            split_mapping_sha256=split_mapping_sha256,
            audit_row=audit_index.get(sample_token),
            dataroot=dataroot,
            rules=rules,
            horizon_sec=horizon_sec,
            sample_interval_sec=sample_interval_sec,
            time_tolerance_sec=time_tolerance_sec,
            agent_radius_m=agent_radius_m,
        )
        diagnostics[sample_token] = decision.exclusion_reason or "included"
    return diagnostics


def write_trainval_manifest(
    records: Sequence[TrainvalManifestRecord],
    manifest_path: Path,
    split_mapping_sha256: str,
) -> ManifestValidationSummary:
    validation = write_jsonl_records(
        records,
        manifest_path,
        validator=validate_manifest,
    )
    if validation.sample_count != len(records):
        raise ValueError("written manifest sample count does not match records")
    if validation.split_mapping_sha256 != split_mapping_sha256:
        raise ValueError("written manifest split mapping hash does not match sidecar")
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 0.1b nuScenes trainval manifest v1."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trainval_manifest.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--diagnostic-sample-token",
        action="append",
        default=[],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    config = load_config(arguments.config)
    dataroot = _environment_path("NUSCENES_ROOT")
    derived_root = _environment_path(
        "VLA_DERIVED_ROOT",
        require_absolute=True,
    )
    trajectory_config = load_trajectory_config(config.data_config_path)
    rules = load_meta_action_rules(config.action_config_path)
    audit_index = load_audit_index(
        config.base_audit_path,
        config.supplement_audit_path,
    )
    if config.version != "v1.0-trainval":
        raise ValueError("trainval builder requires version v1.0-trainval")
    if config.split_seed != PHASE0_SPLIT_SEED:
        raise ValueError("trainval split_seed must match the frozen Phase 0 seed")
    if config.split_strategy_version != SPLIT_STRATEGY_VERSION:
        raise ValueError("unexpected trainval split_strategy_version")
    if rules.label_rule_version != LABEL_RULE_VERSION:
        raise ValueError("unexpected label_rule_version")
    if (
        rules.horizon_sec != trajectory_config.horizon_sec
        or rules.sample_interval_sec != trajectory_config.sample_interval_sec
    ):
        raise ValueError("trajectory and action rule timing must match")

    nuscenes = NuScenes(
        version=config.version,
        dataroot=str(dataroot),
        verbose=False,
    )
    full_split = build_full_scene_splits(
        nuscenes=nuscenes,
        split_seed=config.split_seed,
        split_strategy_version=config.split_strategy_version,
        audit_index=audit_index,
        dataroot=dataroot,
        rules=rules,
        horizon_sec=trajectory_config.horizon_sec,
        sample_interval_sec=trajectory_config.sample_interval_sec,
        time_tolerance_sec=trajectory_config.trajectory_time_tolerance_sec,
        agent_radius_m=trajectory_config.nearby_radius_m,
    )
    audit_summary = require_complete_audit_coverage(
        full_split,
        audit_index,
    )
    scene_histogram_sha256 = hash_scene_histograms(
        full_split.train_statistics.scene_histograms
    )
    mapping_payload = build_scene_mapping_payload(
        nuscenes_version=config.version,
        official_splits=full_split.official_splits,
        project_splits=full_split.scene_splits,
        split_seed=config.split_seed,
        split_strategy_version=config.split_strategy_version,
        label_rule_version=LABEL_RULE_VERSION,
        scene_histogram_sha256=scene_histogram_sha256,
    )
    mapping_path = output_path(
        derived_root,
        config.scene_split_mapping_relative_path,
    )
    split_mapping_sha256 = ensure_scene_mapping(
        mapping_path,
        mapping_payload,
        allow_create=arguments.pilot,
    )
    split_summary = full_split_summary(full_split, config.split_seed)
    split_summary.update(
        {
            "scene_histogram_sha256": scene_histogram_sha256,
            "scene_split_mapping_sha256": split_mapping_sha256,
            "audit_tokens": audit_summary,
        }
    )
    print(
        json.dumps(
            {"full_split_summary": split_summary},
            indent=2,
            sort_keys=True,
        )
    )
    scene_splits = full_split.scene_splits
    if arguments.pilot:
        selected_scene_tokens = select_pilot_scene_tokens(
            scene_splits=scene_splits,
            scene_count=config.pilot_scene_count,
            seed=config.pilot_seed,
        )
        relative_output = config.pilot_manifest_relative_path
    else:
        selected_scene_tokens = tuple(sorted(scene_splits))
        relative_output = config.manifest_relative_path

    result = build_records(
        nuscenes=nuscenes,
        scene_tokens=selected_scene_tokens,
        scene_splits=scene_splits,
        official_splits=full_split.official_splits,
        split_seed=config.split_seed,
        split_strategy_version=config.split_strategy_version,
        split_mapping_sha256=split_mapping_sha256,
        audit_index=audit_index,
        dataroot=dataroot,
        rules=rules,
        horizon_sec=trajectory_config.horizon_sec,
        sample_interval_sec=trajectory_config.sample_interval_sec,
        time_tolerance_sec=trajectory_config.trajectory_time_tolerance_sec,
        agent_radius_m=trajectory_config.nearby_radius_m,
    )
    manifest_path = output_path(derived_root, relative_output)
    write_trainval_manifest(
        result.records,
        manifest_path,
        split_mapping_sha256,
    )

    summary = pilot_summary(
        result=result,
        selected_scene_tokens=selected_scene_tokens,
        scene_splits=scene_splits,
        split_seed=config.split_seed,
        split_strategy_version=config.split_strategy_version,
        dataroot=dataroot,
        derived_root=derived_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if arguments.diagnostic_sample_token:
        diagnostics = diagnostic_results(
            nuscenes=nuscenes,
            sample_tokens=arguments.diagnostic_sample_token,
            scene_splits=scene_splits,
            official_splits=full_split.official_splits,
            split_seed=config.split_seed,
            split_strategy_version=config.split_strategy_version,
            split_mapping_sha256=split_mapping_sha256,
            audit_index=audit_index,
            dataroot=dataroot,
            rules=rules,
            horizon_sec=trajectory_config.horizon_sec,
            sample_interval_sec=trajectory_config.sample_interval_sec,
            time_tolerance_sec=trajectory_config.trajectory_time_tolerance_sec,
            agent_radius_m=trajectory_config.nearby_radius_m,
        )
        print(json.dumps({"diagnostics": diagnostics}, indent=2, sort_keys=True))
    print(f"manifest_relative_path: {relative_output.as_posix()}")
    print(
        "scene_split_mapping_relative_path: "
        f"{config.scene_split_mapping_relative_path.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

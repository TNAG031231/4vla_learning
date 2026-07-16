from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data import build_trainval_manifest as trainval_builder
from data.derive_meta_action import MetaActionRules, load_meta_action_rules
from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION
from src.phase0.manifest import write_jsonl_records
from src.phase0.protocol import read_manifest_samples, validate_manifest
from src.phase0.stratified_split import assign_stratified_scene_splits


class FakeNuScenes:
    def __init__(self) -> None:
        self.scene: list[dict[str, object]] = []
        self.tables: dict[str, dict[str, dict[str, object]]] = {
            "scene": {},
            "sample": {},
            "sample_data": {},
            "ego_pose": {},
        }

    def get(self, table_name: str, token: str) -> dict[str, object]:
        return self.tables[table_name][token]


def build_scene(
    dataroot: Path,
    timestamps: tuple[int, ...],
    scene_token: str = "scene-a",
) -> FakeNuScenes:
    reader = FakeNuScenes()
    scene = {
        "token": scene_token,
        "name": f"name-{scene_token}",
        "first_sample_token": "sample-0",
        "last_sample_token": f"sample-{len(timestamps) - 1}",
    }
    reader.scene.append(scene)
    reader.tables["scene"][scene_token] = scene
    image_directory = dataroot / "samples" / "CAM_FRONT"
    image_directory.mkdir(parents=True)
    for index, timestamp in enumerate(timestamps):
        sample_token = f"sample-{index}"
        camera_token = f"camera-{index}"
        pose_token = f"pose-{index}"
        filename = f"samples/CAM_FRONT/{sample_token}.jpg"
        (dataroot / filename).write_bytes(b"jpg")
        reader.tables["sample"][sample_token] = {
            "token": sample_token,
            "scene_token": scene_token,
            "timestamp": timestamp,
            "prev": f"sample-{index - 1}" if index else "",
            "next": (
                f"sample-{index + 1}"
                if index + 1 < len(timestamps)
                else ""
            ),
            "data": {"CAM_FRONT": camera_token},
            "anns": [],
        }
        reader.tables["sample_data"][camera_token] = {
            "token": camera_token,
            "sample_token": sample_token,
            "ego_pose_token": pose_token,
            "timestamp": timestamp,
            "filename": filename,
        }
        reader.tables["ego_pose"][pose_token] = {
            "token": pose_token,
            "translation": [float(index), 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        }
    return reader


def rules() -> MetaActionRules:
    return load_meta_action_rules(PROJECT_ROOT / "configs/action_rules.yaml")


def evaluate(
    reader: FakeNuScenes,
    dataroot: Path,
    sample_token: str = "sample-0",
) -> trainval_builder.SampleDecision:
    return trainval_builder.evaluate_sample(
        nuscenes=reader,
        sample_token=sample_token,
        expected_scene_token="scene-a",
        split="train",
        official_split="train",
        split_seed=20260710,
        split_strategy_version=(
            "official_train_scene_label_stratified_v1"
        ),
        dataroot=dataroot,
        rules=rules(),
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        time_tolerance_sec=0.075,
        agent_radius_m=50.0,
    )


def test_complete_sample_builds_unaudited_trainval_record(
    tmp_path: Path,
) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)

    decision = evaluate(reader, tmp_path)

    assert decision.exclusion_reason is None
    assert decision.record is not None
    assert decision.record.audit_status == "unaudited"
    assert decision.record.source_audit_record is None
    assert decision.record.cam_front_path == "samples/CAM_FRONT/sample-0.jpg"
    assert decision.record.manifest_schema_version == (
        "phase0_trainval_dataset_manifest_v1"
    )
    assert decision.record.label_rule_version == LABEL_RULE_VERSION
    assert decision.record.official_split == "train"
    assert decision.record.split_seed == 20260710
    assert decision.record.split_strategy_version == (
        "official_train_scene_label_stratified_v1"
    )
    assert decision.record.current_ego_motion["availability"] == "unavailable"
    assert tuple(ACTION_SCHEMA) == (
        "keep",
        "accelerate",
        "decelerate",
        "stop",
        "left_lateral",
        "right_lateral",
    )


@pytest.mark.parametrize(
    ("timestamps", "expected_reason"),
    (
        (
            tuple(index * 500_000 for index in range(6)),
            "insufficient_remaining_horizon",
        ),
        (
            (0, 500_000, 1_000_000, 1_500_000, 2_500_000, 3_000_000),
            "timestamp_out_of_tolerance",
        ),
    ),
)
def test_truncated_reasons_are_distinguished(
    tmp_path: Path,
    timestamps: tuple[int, ...],
    expected_reason: str,
) -> None:
    reader = build_scene(tmp_path, timestamps)

    decision = evaluate(reader, tmp_path)

    assert decision.record is None
    assert decision.exclusion_reason == expected_reason


def test_broken_next_chain_is_excluded(tmp_path: Path) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)
    reader.tables["sample"]["sample-1"]["prev"] = "wrong-sample"

    decision = evaluate(reader, tmp_path)

    assert decision.exclusion_reason == "broken_next_chain"


def test_cross_scene_future_chain_is_excluded(tmp_path: Path) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)
    reader.tables["sample"]["sample-1"]["scene_token"] = "scene-b"

    decision = evaluate(reader, tmp_path)

    assert decision.exclusion_reason == "scene_mismatch"


def test_missing_cam_front_file_is_excluded(tmp_path: Path) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)
    (tmp_path / "samples/CAM_FRONT/sample-0.jpg").unlink()

    decision = evaluate(reader, tmp_path)

    assert decision.exclusion_reason == "missing_cam_front_file"


def test_split_mapping_is_fixed_before_sample_filtering(tmp_path: Path) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)
    scene_splits = {"scene-a": "validation"}
    official_splits = {"scene-a": "train"}

    result = trainval_builder.build_records(
        nuscenes=reader,
        scene_tokens=("scene-a",),
        scene_splits=scene_splits,
        official_splits=official_splits,
        split_seed=20260710,
        split_strategy_version=(
            "official_train_scene_label_stratified_v1"
        ),
        dataroot=tmp_path,
        rules=rules(),
        horizon_sec=3.0,
        sample_interval_sec=0.5,
        time_tolerance_sec=0.075,
        agent_radius_m=50.0,
    )

    assert scene_splits == {"scene-a": "validation"}
    assert result.scanned_sample_count == 8
    assert {record.split for record in result.records} == {"validation"}
    assert result.exclusion_counts["insufficient_remaining_horizon"] == 6


def test_trainval_manifest_can_be_validated_and_reloaded(
    tmp_path: Path,
) -> None:
    timestamps = tuple(index * 500_000 for index in range(8))
    reader = build_scene(tmp_path, timestamps)
    decision = evaluate(reader, tmp_path)
    assert decision.record is not None
    manifest_path = tmp_path / "derived" / "manifest.jsonl"

    write_jsonl_records((decision.record,), manifest_path)
    summary = validate_manifest(manifest_path)
    samples = read_manifest_samples(manifest_path)

    assert summary.sample_count == 1
    assert samples[0].sample_token == "sample-0"
    assert samples[0].split == "train"


def test_output_path_must_remain_under_derived_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative to VLA_DERIVED_ROOT"):
        trainval_builder.output_path(tmp_path, Path("../outside.jsonl"))


def test_official_scene_resolution_and_project_mapping_have_exact_counts() -> None:
    reader = FakeNuScenes()
    official_train = tuple(f"train-{index:03d}" for index in range(700))
    official_val = tuple(f"val-{index:03d}" for index in range(150))
    reader.scene = [
        {"name": name, "token": f"token-{name}"}
        for name in official_train + official_val
    ]

    resolved = trainval_builder.resolve_official_scene_tokens(
        reader,
        {"train": official_train, "val": official_val},
    )
    train_assignments = {
        token: ("train" if index < 560 else "validation")
        for index, token in enumerate(resolved.train)
    }
    mapping = trainval_builder.compose_project_scene_splits(
        train_assignments,
        resolved.val,
    )

    assert len(resolved.train) == 700
    assert len(resolved.val) == 150
    assert Counter(mapping.values()) == {
        "train": 560,
        "validation": 140,
        "test": 150,
    }
    assert all(mapping[token] == "test" for token in resolved.val)
    assert not set(resolved.train) & set(resolved.val)


def test_official_val_tokens_do_not_enter_stratified_optimizer() -> None:
    train_histograms = {
        f"train-{index:03d}": {
            action: (10 if action == "keep" else int(index % 7 == 0))
            for action in ACTION_SCHEMA
        }
        for index in range(700)
    }

    optimized = assign_stratified_scene_splits(
        train_histograms,
        20260710,
        560,
        140,
    )
    first = trainval_builder.compose_project_scene_splits(
        optimized.assignments,
        tuple(f"val-a-{index:03d}" for index in range(150)),
    )
    second = trainval_builder.compose_project_scene_splits(
        optimized.assignments,
        tuple(f"val-b-{index:03d}" for index in range(150)),
    )

    assert {
        token: split for token, split in first.items() if token.startswith("train-")
    } == {
        token: split for token, split in second.items() if token.startswith("train-")
    }
    assert all(
        split == "test"
        for token, split in first.items()
        if token.startswith("val-a-")
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("official_split", "val", "official val scenes"),
        ("split_seed", 1, "frozen seed"),
        ("split_strategy_version", "old", "split_strategy_version"),
    ),
)
def test_trainval_validator_enforces_split_traceability(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    reader = build_scene(tmp_path, tuple(index * 500_000 for index in range(8)))
    decision = evaluate(reader, tmp_path)
    assert decision.record is not None
    invalid_record = replace(decision.record, **{field: value})
    manifest_path = tmp_path / "invalid.jsonl"
    write_jsonl_records((invalid_record,), manifest_path)

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest_path)

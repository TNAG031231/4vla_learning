from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.actions.schema import ACTION_SCHEMA, LABEL_RULE_VERSION
import src.phase0.qwen3vl_dataset_adapter as adapter_module
from src.phase0.qwen3vl_dataset_adapter import (
    ADAPTER_RECORD_FIELDS,
    GitProvenance,
    SourceFileConfig,
    SourceValidationState,
    adapter_record,
    build_messages,
    build_qwen3vl_dataset_adapter,
    canonical_json_bytes,
    collect_git_provenance,
    load_config,
    serialize_ego_state,
    sha256_file,
    validate_source_record,
)
from src.phase0.protocol import TRAINVAL_MANIFEST_SCHEMA_VERSION


CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_dataset_adapter.yaml"
CONFIG_RELATIVE_PATH = "configs/phase0_3_dataset_adapter.yaml"


def _motion(availability: str = "full") -> dict[str, object]:
    if availability == "full":
        return {
            "speed_mps": 2,
            "longitudinal_acceleration_mps2": -0.0,
            "yaw_rate_radps": 0.125,
            "source": "ego_pose_past_difference",
            "timestamp_source": "CAM_FRONT_sample_data",
            "availability": "full",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": 1,
            "unavailable_reason": None,
        }
    if availability == "partial":
        return {
            "speed_mps": 1.0,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": 0.0,
            "source": "ego_pose_past_difference",
            "timestamp_source": "CAM_FRONT_sample_data",
            "availability": "partial",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": None,
            "unavailable_reason": "insufficient_history",
        }
    return {
        "speed_mps": None,
        "longitudinal_acceleration_mps2": None,
        "yaw_rate_radps": None,
        "source": "ego_pose_past_difference",
        "timestamp_source": "CAM_FRONT_sample_data",
        "availability": "unavailable",
        "history_interval_sec": None,
        "acceleration_interval_sec": None,
        "unavailable_reason": "past_pose_unavailable",
    }


def _source_record(
    split: str,
    index: int,
    *,
    availability: str = "full",
    action: str = "keep",
) -> dict[str, object]:
    material: dict[str, object] = {
        "projection_schema_version": (
            "phase0_3_development_projection_v0.1"
        ),
        "sample_token": f"sample-{split}-{index}",
        "scene_token": f"scene-{split}-{index}",
        "timestamp": index,
        "split": split,
        "cam_front_path": f"samples/CAM_FRONT/{split}-{index}.jpg",
        "current_ego_motion": _motion(availability),
        "target_action": action,
        "source_manifest_schema_version": TRAINVAL_MANIFEST_SCHEMA_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "split_mapping_sha256": "a" * 64,
        "source_combined_manifest_sha256": "b" * 64,
    }
    material["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    return material


def _write_jsonl(path: Path, records: list[Mapping[str, object]]) -> None:
    with path.open("wb") as output:
        for record in records:
            output.write(canonical_json_bytes(record) + b"\n")


def _source_receipt(
    train_sha: str,
    validation_sha: str,
) -> dict[str, object]:
    return {
        "projection_version": "phase0.3b-development-projection-v0.1",
        "projection_schema_version": "phase0_3_development_projection_v0.1",
        "nuscenes_version": "v1.0-trainval",
        "git": {
            "commit": "ee50005caaa2ba3fa28074a70bbe9178b089b6a8",
            "branch": "feat-phase-0.3b-dev-projection",
            "detached_head": False,
            "worktree_clean": True,
        },
        "outputs": {
            "train": {
                "relative_path": "train.jsonl",
                "sha256": train_sha,
                "record_count": 2,
            },
            "validation": {
                "relative_path": "validation.jsonl",
                "sha256": validation_sha,
                "record_count": 1,
            },
        },
        "combined_manifest": {"sha256": "b" * 64},
        "scene_mapping": {"split_mapping_sha256": "a" * 64},
        "action_distribution_by_split": {
            "train": {
                action: int(action == "keep") + int(action == "accelerate")
                for action in ACTION_SCHEMA
            },
            "validation": {
                action: int(action == "stop") for action in ACTION_SCHEMA
            },
        },
        "motion_availability_distribution_by_split": {
            "train": {"full": 1, "partial": 1, "unavailable": 0},
            "validation": {"full": 0, "partial": 0, "unavailable": 1},
        },
        "combined_manifest_records_parsed": 0,
        "test_scene_traversal_attempts": 0,
        "test_sample_records_read": 0,
        "test_images_opened": 0,
        "test_labels_read": 0,
        "test_evaluation_performed": False,
        "model_load_performed": False,
        "processor_load_performed": False,
    }


@dataclass
class SyntheticArtifact:
    config: object
    derived_root: Path
    repository_root: Path
    source_dir: Path
    train_records: list[dict[str, object]]
    validation_records: list[dict[str, object]]


@pytest.fixture
def artifact(tmp_path: Path, monkeypatch) -> SyntheticArtifact:
    monkeypatch.setattr(
        adapter_module,
        "FROZEN_MOTION_AVAILABILITY",
        {
            "train": {"full": 1, "partial": 1, "unavailable": 0},
            "validation": {"full": 0, "partial": 0, "unavailable": 1},
        },
    )
    config = load_config(CONFIG_PATH)
    derived_root = tmp_path / "derived"
    repository_root = tmp_path / "repository"
    source_dir = derived_root / config.source_projection_relative_dir
    source_dir.mkdir(parents=True)
    repository_root.mkdir()
    train_records = [
        _source_record("train", 0, action="keep"),
        _source_record("train", 1, availability="partial", action="accelerate"),
    ]
    validation_records = [
        _source_record(
            "validation", 0, availability="unavailable", action="stop"
        )
    ]
    train_path = source_dir / "train.jsonl"
    validation_path = source_dir / "validation.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(validation_path, validation_records)
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    receipt_path = source_dir / "projection_receipt.json"
    receipt_path.write_bytes(
        canonical_json_bytes(_source_receipt(train_sha, validation_sha)) + b"\n"
    )
    config = replace(
        config,
        expected_source_receipt_sha256=sha256_file(receipt_path),
        source_files={
            "train": SourceFileConfig("train.jsonl", train_sha, 2),
            "validation": SourceFileConfig(
                "validation.jsonl", validation_sha, 1
            ),
        },
        config_sha256="c" * 64,
    )
    return SyntheticArtifact(
        config=config,
        derived_root=derived_root,
        repository_root=repository_root,
        source_dir=source_dir,
        train_records=train_records,
        validation_records=validation_records,
    )


def _git(*, clean: bool = True) -> GitProvenance:
    return GitProvenance(
        commit="d" * 40,
        branch="adapter-test",
        detached_head=False,
        worktree_clean=clean,
    )


def _build(artifact: SyntheticArtifact, **overrides: object):
    arguments: dict[str, object] = {
        "config": artifact.config,
        "config_relative_path": CONFIG_RELATIVE_PATH,
        "repository_root": artifact.repository_root,
        "derived_root": artifact.derived_root,
        "git_provenance": _git(),
        "now_utc": lambda: "2026-08-10T00:00:00Z",
    }
    arguments.update(overrides)
    return build_qwen3vl_dataset_adapter(**arguments)


def _rewrite_config(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "adapter.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _rehash_source(artifact: SyntheticArtifact) -> None:
    train_path = artifact.source_dir / "train.jsonl"
    validation_path = artifact.source_dir / "validation.jsonl"
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt_path.write_bytes(
        canonical_json_bytes(_source_receipt(train_sha, validation_sha)) + b"\n"
    )
    artifact.config = replace(
        artifact.config,
        expected_source_receipt_sha256=sha256_file(receipt_path),
        source_files={
            "train": SourceFileConfig("train.jsonl", train_sha, 2),
            "validation": SourceFileConfig("validation.jsonl", validation_sha, 1),
        },
    )


def test_frozen_config_loads_exact_source_hashes() -> None:
    config = load_config(CONFIG_PATH)
    assert config.expected_source_receipt_sha256 == (
        "16fd73b069b61061fdf3f81e0b31ee7367003e13fcfe3196fd2482d50150673b"
    )
    assert config.source_files["train"].sha256 == (
        "32dd520c87921804e6640273bb1a4ad663acc72c82a43e0f6f092759389dfb5a"
    )
    assert config.source_files["validation"].sha256 == (
        "0cbd4cf94bde422d6810caa5a302d079e4770a49dd6c3f05548d9b8c12fac94a"
    )


def test_source_receipt_sha_constant_matches_frozen_config() -> None:
    expected = (
        "16fd73b069b61061fdf3f81e0b31ee7367003e13fcfe3196fd2482d50150673b"
    )
    assert (
        load_config(CONFIG_PATH).expected_source_receipt_sha256
        == adapter_module.SOURCE_RECEIPT_SHA256
        == expected
    )


def test_config_missing_required_field(tmp_path: Path) -> None:
    path = _rewrite_config(tmp_path, lambda raw: raw.pop("adapter_version"))
    with pytest.raises(ValueError, match="adapter_version"):
        load_config(path)


def test_config_rejects_absolute_source_path(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw.__setitem__("source_projection_relative_dir", "/tmp/x"),
    )
    with pytest.raises(ValueError, match="relative POSIX path"):
        load_config(path)


def test_config_rejects_parent_traversal(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw.__setitem__("output_relative_dir", "../output"),
    )
    with pytest.raises(ValueError, match="relative POSIX path"):
        load_config(path)


def test_config_rejects_receipt_sha_change(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw.__setitem__("expected_source_receipt_sha256", "f" * 64),
    )
    with pytest.raises(ValueError, match="frozen adapter contract"):
        load_config(path)


def test_config_rejects_train_sha_change(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw["source_files"]["train"].__setitem__("sha256", "f" * 64),
    )
    with pytest.raises(ValueError, match="source_files"):
        load_config(path)


def test_config_rejects_validation_count_change(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw["source_files"]["validation"].__setitem__(
            "record_count", 1
        ),
    )
    with pytest.raises(ValueError, match="source_files"):
        load_config(path)


def test_source_receipt_sha_mismatch_blocks_before_parse(artifact) -> None:
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256="f" * 64
    )
    with pytest.raises(ValueError, match="receipt SHA-256"):
        _build(artifact)


def test_source_train_sha_mismatch_blocks_before_parse(artifact) -> None:
    with (artifact.source_dir / "train.jsonl").open("ab") as source_file:
        source_file.write(b" ")
    with pytest.raises(ValueError, match="train projection SHA-256"):
        _build(artifact)


def test_source_validation_sha_mismatch_blocks_before_parse(artifact) -> None:
    with (artifact.source_dir / "validation.jsonl").open("ab") as source_file:
        source_file.write(b" ")
    with pytest.raises(ValueError, match="validation projection SHA-256"):
        _build(artifact)


def test_missing_source_directory_blocks(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(FileNotFoundError, match="directory"):
        build_qwen3vl_dataset_adapter(
            config=config,
            config_relative_path=CONFIG_RELATIVE_PATH,
            repository_root=tmp_path / "repo",
            derived_root=tmp_path,
            git_provenance=_git(),
        )


def test_missing_source_receipt_blocks(artifact) -> None:
    (artifact.source_dir / "projection_receipt.json").unlink()
    with pytest.raises(FileNotFoundError, match="receipt"):
        _build(artifact)


def test_source_test_file_presence_blocks(artifact) -> None:
    (artifact.source_dir / "test.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="test.jsonl"):
        _build(artifact)


def test_source_receipt_git_commit_mismatch_blocks(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["git"]["commit"] = "f" * 40
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256=sha256_file(receipt_path)
    )
    with pytest.raises(ValueError, match="git commit"):
        _build(artifact)


def test_source_receipt_dirty_worktree_blocks(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["git"]["worktree_clean"] = False
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256=sha256_file(receipt_path)
    )
    with pytest.raises(ValueError, match="worktree_clean"):
        _build(artifact)


def test_source_receipt_nonzero_isolation_counter_blocks(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_labels_read"] = 1
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256=sha256_file(receipt_path)
    )
    with pytest.raises(ValueError, match="test_labels_read"):
        _build(artifact)


class SplitProbe(Mapping[str, object]):
    def __init__(self, split: str):
        self.split = split
        self.accessed: list[str] = []

    def __getitem__(self, key: str) -> object:
        self.accessed.append(key)
        if key == "split":
            return self.split
        raise AssertionError(f"sensitive field accessed: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("record fields inspected before split guard")

    def __len__(self) -> int:
        raise AssertionError("record length inspected before split guard")

    def get(self, key: str, default: object = None) -> object:
        return self[key] if key == "split" else default


def test_test_split_fails_before_sensitive_field_access() -> None:
    record = SplitProbe("test")
    state = SourceValidationState()
    with pytest.raises(ValueError, match="forbidden"):
        validate_source_record(
            record, expected_split="train", state=state
        )
    assert record.accessed == ["split"]
    assert state.forbidden_split_records_seen == 1


def test_unknown_split_fails_before_sensitive_field_access() -> None:
    record = SplitProbe("shadow")
    with pytest.raises(ValueError, match="forbidden"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )
    assert record.accessed == ["split"]


def test_cross_split_record_fails_before_sensitive_field_access() -> None:
    record = SplitProbe("validation")
    with pytest.raises(ValueError, match="mismatched"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )
    assert record.accessed == ["split"]


def test_train_record_in_validation_file_fails_before_sensitive_access() -> None:
    record = SplitProbe("train")
    with pytest.raises(ValueError, match="mismatched"):
        validate_source_record(
            record,
            expected_split="validation",
            state=SourceValidationState(),
        )
    assert record.accessed == ["split"]


def test_source_record_rejects_extra_field() -> None:
    record = _source_record("train", 0)
    record["future_ego_trajectory"] = []
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )


def test_source_record_rejects_invalid_projection_hash() -> None:
    record = _source_record("train", 0)
    record["projection_record_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="record SHA-256"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )


def test_source_record_rejects_duplicate_sample_token() -> None:
    state = SourceValidationState()
    record = _source_record("train", 0)
    validate_source_record(record, expected_split="train", state=state)
    with pytest.raises(ValueError, match="duplicate"):
        validate_source_record(record, expected_split="train", state=state)


def test_source_record_rejects_invalid_action() -> None:
    record = _source_record("train", 0, action="turn_left")
    with pytest.raises(ValueError, match="Unsupported action"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )


def test_source_record_rejects_invalid_cam_front_path() -> None:
    record = _source_record("train", 0)
    record["cam_front_path"] = "samples/CAM_BACK/image.jpg"
    material = {
        key: value
        for key, value in record.items()
        if key != "projection_record_sha256"
    }
    record["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(ValueError, match="CAM_FRONT"):
        validate_source_record(
            record, expected_split="train", state=SourceValidationState()
        )


@pytest.mark.parametrize(
    "invalid_path",
    (
        "/absolute/image.jpg",
        "C:\\samples\\CAM_FRONT\\image.jpg",
        "samples/CAM_FRONT/../secret.jpg",
    ),
)
def test_source_record_rejects_unsafe_cam_front_path(invalid_path: str) -> None:
    record = _source_record("train", 0)
    record["cam_front_path"] = invalid_path
    material = {
        key: value
        for key, value in record.items()
        if key != "projection_record_sha256"
    }
    record["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(ValueError, match="CAM_FRONT"):
        validate_source_record(
            record,
            expected_split="train",
            state=SourceValidationState(),
        )


def test_source_record_rejects_boolean_timestamp() -> None:
    record = _source_record("train", 0)
    record["timestamp"] = True
    material = {
        key: value
        for key, value in record.items()
        if key != "projection_record_sha256"
    }
    record["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(ValueError, match="timestamp"):
        validate_source_record(
            record,
            expected_split="train",
            state=SourceValidationState(),
        )


def test_source_record_rejects_inconsistent_split_mapping_hash() -> None:
    state = SourceValidationState()
    validate_source_record(
        _source_record("train", 0), expected_split="train", state=state
    )
    record = _source_record("train", 1)
    record["split_mapping_sha256"] = "f" * 64
    material = {
        key: value
        for key, value in record.items()
        if key != "projection_record_sha256"
    }
    record["projection_record_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(ValueError, match="split mapping"):
        validate_source_record(record, expected_split="train", state=state)


def test_ego_state_serialization_is_exact_and_ordered() -> None:
    text = serialize_ego_state(_motion(), load_config(CONFIG_PATH))
    assert text == (
        "Current ego state:\n"
        "speed_mps=2.000000; longitudinal_acceleration_mps2=0.000000; "
        "yaw_rate_radps=0.125000; history_interval_sec=0.500000; "
        "acceleration_interval_sec=1.000000; availability=full"
    )


def test_ego_state_serializes_null_as_unavailable() -> None:
    text = serialize_ego_state(_motion("unavailable"), load_config(CONFIG_PATH))
    assert text.count("=unavailable") == 6
    assert text.endswith("availability=unavailable")


def test_ego_state_omits_unavailable_reason() -> None:
    text = serialize_ego_state(_motion("partial"), load_config(CONFIG_PATH))
    assert "unavailable_reason" not in text


def test_ego_state_rejects_boolean_number() -> None:
    motion = _motion()
    motion["speed_mps"] = True
    with pytest.raises(ValueError, match="finite"):
        serialize_ego_state(motion, load_config(CONFIG_PATH))


def test_ego_state_rejects_nan() -> None:
    motion = _motion()
    motion["speed_mps"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        serialize_ego_state(motion, load_config(CONFIG_PATH))


def test_ego_state_rejects_infinity() -> None:
    motion = _motion()
    motion["yaw_rate_radps"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        serialize_ego_state(motion, load_config(CONFIG_PATH))


def test_image_only_messages_have_no_ego_state() -> None:
    source = validate_source_record(
        _source_record("train", 0),
        expected_split="train",
        state=SourceValidationState(),
    )
    messages = build_messages(source, "image_only", load_config(CONFIG_PATH))
    assert messages[0]["content"][1]["text"] == adapter_module.TASK_PROMPT
    assert "Current ego state" not in messages[0]["content"][1]["text"]


def test_image_ego_state_messages_use_fixed_separator() -> None:
    source = validate_source_record(
        _source_record("train", 0),
        expected_split="train",
        state=SourceValidationState(),
    )
    messages = build_messages(source, "image_ego_state", load_config(CONFIG_PATH))
    assert "availability=full\n\nObserve" in messages[0]["content"][1]["text"]


def test_assistant_message_contains_only_target_text() -> None:
    source = validate_source_record(
        _source_record("train", 0, action="decelerate"),
        expected_split="train",
        state=SourceValidationState(),
    )
    messages = build_messages(source, "image_only", load_config(CONFIG_PATH))
    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "decelerate"}],
    }


def test_user_message_does_not_contain_target() -> None:
    first_source = validate_source_record(
        _source_record("train", 0, action="decelerate"),
        expected_split="train",
        state=SourceValidationState(),
    )
    second_source = validate_source_record(
        _source_record("train", 0, action="stop"),
        expected_split="train",
        state=SourceValidationState(),
    )
    config = load_config(CONFIG_PATH)
    first_user = build_messages(first_source, "image_only", config)[0]
    second_user = build_messages(second_source, "image_only", config)[0]
    assert first_user == second_user


def test_adapter_record_has_exact_fields_and_recomputable_hash() -> None:
    source = validate_source_record(
        _source_record("train", 0),
        expected_split="train",
        state=SourceValidationState(),
    )
    record = adapter_record(
        source,
        variant="image_only",
        config=load_config(CONFIG_PATH),
        source_file_sha256="e" * 64,
    )
    assert set(record) == ADAPTER_RECORD_FIELDS
    digest = record.pop("adapter_record_sha256")
    assert hashlib.sha256(canonical_json_bytes(record)).hexdigest() == digest


def test_adapter_record_omits_raw_motion_and_reason() -> None:
    source = validate_source_record(
        _source_record("train", 0),
        expected_split="train",
        state=SourceValidationState(),
    )
    record = adapter_record(
        source,
        variant="image_ego_state",
        config=load_config(CONFIG_PATH),
        source_file_sha256="e" * 64,
    )
    assert "current_ego_motion" not in record
    assert "unavailable_reason" not in json.dumps(record)


def test_build_creates_four_outputs_and_receipt(artifact) -> None:
    result = _build(artifact)
    output_dir = artifact.derived_root / artifact.config.output_relative_dir
    assert result["status"] == "created"
    assert {path.name for path in output_dir.iterdir()} == set(
        adapter_module.OUTPUT_FILENAMES
    )


def test_build_receipt_records_zero_test_access(artifact) -> None:
    receipt = _build(artifact)["receipt"]
    assert receipt["combined_manifest_accessed"] is False
    assert receipt["test_files_opened"] == 0
    assert receipt["test_records_read"] == 0
    assert receipt["test_images_opened"] == 0
    assert receipt["test_labels_read"] == 0


def test_build_never_opens_images(artifact, monkeypatch) -> None:
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            raise AssertionError("image opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert _build(artifact)["receipt"]["image_files_opened"] == 0


def test_build_preserves_source_order(artifact) -> None:
    _build(artifact)
    path = (
        artifact.derived_root
        / artifact.config.output_relative_dir
        / "train_image_only.jsonl"
    )
    tokens = [
        json.loads(line)["sample_token"]
        for line in path.read_text().splitlines()
    ]
    assert tokens == ["sample-train-0", "sample-train-1"]


def test_build_outputs_are_byte_deterministic(tmp_path: Path, monkeypatch) -> None:
    first = artifact.__wrapped__(tmp_path / "first", monkeypatch)
    second = artifact.__wrapped__(tmp_path / "second", monkeypatch)
    _build(first)
    _build(second)
    for filename in adapter_module.OUTPUT_FILENAMES[:-1]:
        first_path = first.derived_root / first.config.output_relative_dir / filename
        second_path = second.derived_root / second.config.output_relative_dir / filename
        assert first_path.read_bytes() == second_path.read_bytes()


def test_failure_does_not_publish_formal_directory(artifact) -> None:
    artifact.train_records[1]["split"] = "test"
    _write_jsonl(artifact.source_dir / "train.jsonl", artifact.train_records)
    _rehash_source(artifact)
    with pytest.raises(ValueError, match="forbidden"):
        _build(artifact)
    assert not (
        artifact.derived_root / artifact.config.output_relative_dir
    ).exists()


def test_source_record_count_mismatch_blocks_publish(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["outputs"]["train"]["record_count"] = 3
    receipt["action_distribution_by_split"]["train"]["stop"] = 1
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config,
        expected_source_receipt_sha256=sha256_file(receipt_path),
        source_files={
            "train": replace(artifact.config.source_files["train"], record_count=3),
            "validation": artifact.config.source_files["validation"],
        },
    )
    with pytest.raises(ValueError, match="record count"):
        _build(artifact)
    assert not (
        artifact.derived_root / artifact.config.output_relative_dir
    ).exists()


def test_source_action_distribution_mismatch_blocks_publish(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["action_distribution_by_split"]["train"]["keep"] = 0
    receipt["action_distribution_by_split"]["train"]["accelerate"] = 2
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256=sha256_file(receipt_path)
    )
    with pytest.raises(ValueError, match="action distribution"):
        _build(artifact)


def test_source_motion_distribution_mismatch_blocks_publish(artifact) -> None:
    receipt_path = artifact.source_dir / "projection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["motion_availability_distribution_by_split"]["train"] = {
        "full": 2,
        "partial": 0,
        "unavailable": 0,
    }
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    artifact.config = replace(
        artifact.config, expected_source_receipt_sha256=sha256_file(receipt_path)
    )
    with pytest.raises(ValueError, match="motion distribution"):
        _build(artifact)


def test_existing_exact_artifact_is_fully_revalidated(artifact) -> None:
    assert _build(artifact)["status"] == "created"
    assert _build(artifact)["status"] == "already_exists"


def test_existing_tampered_output_sha_blocks(artifact) -> None:
    _build(artifact)
    output_path = (
        artifact.derived_root
        / artifact.config.output_relative_dir
        / "train_image_only.jsonl"
    )
    with output_path.open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="output SHA-256"):
        _build(artifact)


def test_existing_tampered_receipt_isolation_flag_blocks(artifact) -> None:
    _build(artifact)
    receipt_path = (
        artifact.derived_root
        / artifact.config.output_relative_dir
        / "adapter_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_records_read"] = 1
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(ValueError, match="test_records_read"):
        _build(artifact)


def test_dirty_adapter_git_blocks_before_source_access(artifact) -> None:
    with pytest.raises(ValueError, match="clean worktree"):
        _build(artifact, git_provenance=_git(clean=False))


def test_collect_git_provenance_detects_untracked_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ("git", "init", "-b", "adapter-test"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Adapter Test"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "adapter@example.com"),
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-m", "test: initialize"),
        cwd=repository,
        check=True,
    )
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean worktree"):
        collect_git_provenance(repository)


def test_adapter_modules_do_not_import_forbidden_runtime_packages() -> None:
    source = (REPOSITORY_ROOT / "src/phase0/qwen3vl_dataset_adapter.py").read_text()
    script = (REPOSITORY_ROOT / "scripts/run_qwen3vl_dataset_adapter.py").read_text()
    forbidden = ("torch", "transformers", "peft", "PIL", "nuscenes")
    for package in forbidden:
        assert f"import {package}" not in source
        assert f"import {package}" not in script
        assert f"from {package}" not in source
        assert f"from {package}" not in script


def test_adapter_source_contains_no_model_or_processor_loader() -> None:
    source = (REPOSITORY_ROOT / "src/phase0/qwen3vl_dataset_adapter.py").read_text()
    assert "from_pretrained" not in source
    assert "apply_chat_template" not in source
    assert "Image.open" not in source

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import data.build_trainval_manifest as trainval_builder
from src.actions.schema import LABEL_RULE_VERSION
import src.phase0.development_projection as projection_module
from src.phase0.development_projection import (
    GitProvenance,
    GuardedNuScenesReader,
    IsolationCounters,
    ProducerInputs,
    build_development_projection,
    collect_projection_git_provenance,
    load_config,
    project_record,
    select_development_scenes,
)
from src.phase0.protocol import PHASE0_SPLIT_SEED
import src.phase0.protocol as protocol
from src.phase0.scene_mapping import (
    build_scene_mapping_payload,
    canonical_json_bytes,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


CONFIG_PATH = REPOSITORY_ROOT / "configs/phase0_3_development_projection.yaml"


class FakeNuScenes:
    def get(self, table_name: str, token: str) -> dict[str, object]:
        if table_name == "scene":
            return {"token": token, "first_sample_token": ""}
        if table_name == "sample":
            return {"token": token, "scene_token": token.removeprefix("sample-")}
        return {"token": token}


class FakeProducer:
    def __init__(self, records_by_split: dict[str, tuple[dict[str, object], ...]]):
        self.records_by_split = records_by_split
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        scene_tokens = tuple(kwargs["scene_tokens"])
        scene_splits = kwargs["scene_splits"]
        split = scene_splits[scene_tokens[0]]
        self.calls.append(dict(kwargs))
        records = tuple(
            {
                **record,
                "split_mapping_sha256": kwargs["split_mapping_sha256"],
            }
            for record in self.records_by_split[split]
        )
        return SimpleNamespace(records=records)


def _mapping_payload() -> dict[str, object]:
    official_splits = {}
    project_splits = {}
    for index in range(850):
        token = f"scene-{index:03d}"
        if index < 560:
            official_splits[token] = "train"
            project_splits[token] = "train"
        elif index < 700:
            official_splits[token] = "train"
            project_splits[token] = "validation"
        else:
            official_splits[token] = "val"
            project_splits[token] = "test"
    return build_scene_mapping_payload(
        nuscenes_version="v1.0-trainval",
        official_splits=official_splits,
        project_splits=project_splits,
        split_seed=PHASE0_SPLIT_SEED,
        split_strategy_version=SPLIT_STRATEGY_VERSION,
        label_rule_version=LABEL_RULE_VERSION,
        scene_histogram_sha256="a" * 64,
    )


def _motion(availability: str) -> dict[str, object]:
    if availability == "full":
        return {
            "speed_mps": 2.0,
            "longitudinal_acceleration_mps2": 0.25,
            "yaw_rate_radps": 0.01,
            "source": "ego_pose_past_difference",
            "timestamp_source": "CAM_FRONT_sample_data",
            "availability": "full",
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": 0.5,
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
            "unavailable_reason": "insufficient_past_history_for_acceleration",
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
        "unavailable_reason": "past_ego_pose_unavailable",
    }


def _source_record(
    split: str,
    scene_token: str,
    index: int,
    availability: str,
    *,
    cam_front_path: str | None = None,
    action: str = "keep",
) -> dict[str, object]:
    return {
        "sample_token": f"sample-{split}-{index}",
        "scene_token": scene_token,
        "timestamp": index,
        "cam_front_path": (
            cam_front_path or f"samples/CAM_FRONT/{split}-{index}.jpg"
        ),
        "current_ego_pose": {"private": "omitted"},
        "current_ego_motion": _motion(availability),
        "coordinate_metadata": {"private": "omitted"},
        "future_ego_trajectory": [{"private": "omitted"}],
        "nearby_agents": [{"private": "omitted"}],
        "meta_action": action,
        "label_rule_version": LABEL_RULE_VERSION,
        "safety_rule_version": "not_available",
        "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
        "split": split,
        "official_split": "train",
        "split_seed": PHASE0_SPLIT_SEED,
        "split_strategy_version": SPLIT_STRATEGY_VERSION,
        "split_mapping_sha256": "b" * 64,
        "audit_status": "unaudited",
        "source_audit_record": None,
    }


@pytest.fixture
def synthetic_config():
    return replace(
        load_config(CONFIG_PATH),
        expected_combined_manifest_sha256="c" * 64,
        expected_sample_counts={"train": 2, "validation": 1},
        expected_motion_availability={
            "train": {"full": 1, "partial": 1, "unavailable": 0},
            "validation": {"full": 0, "partial": 0, "unavailable": 1},
        },
        config_sha256="d" * 64,
    )


@pytest.fixture
def source_records():
    return {
        "train": (
            _source_record("train", "scene-000", 0, "full"),
            _source_record("train", "scene-001", 1, "partial"),
        ),
        "validation": (
            _source_record("validation", "scene-560", 0, "unavailable"),
        ),
    }


def _write_inputs(
    derived_root: Path,
    config,
    *,
    mapping: dict[str, object] | None = None,
    combined_bytes: bytes = b"not-json\x00still-binary",
    selected_rule_overrides: dict[str, object] | None = None,
):
    combined_path = derived_root / config.combined_manifest_relative_path
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_bytes(combined_bytes)
    combined_sha = hashlib.sha256(combined_bytes).hexdigest()
    config = replace(config, expected_combined_manifest_sha256=combined_sha)
    mapping_payload = mapping or _mapping_payload()
    mapping_path = derived_root / config.scene_mapping_relative_path
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping_payload), encoding="utf-8")
    selected_rule = {
        "manifest_sha256": combined_sha,
        "split_mapping_sha256": mapping_payload[
            "scene_split_mapping_sha256"
        ],
        "test_evaluation_performed": False,
    }
    if selected_rule_overrides:
        selected_rule.update(selected_rule_overrides)
    rule_path = derived_root / config.selected_rule_relative_path
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(json.dumps(selected_rule), encoding="utf-8")
    return config


def _clean_git(
    *,
    commit: str = "e" * 40,
    branch: str | None = "feature",
) -> GitProvenance:
    return GitProvenance(
        commit=commit,
        branch=branch,
        detached_head=branch is None,
        worktree_clean=True,
    )


def _build(
    tmp_path: Path,
    config,
    source_records,
    *,
    producer=None,
    mapping=None,
    selected_rule_overrides=None,
    git_provenance=None,
    now_utc=lambda: "2026-08-06T00:00:00Z",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository_root = tmp_path / "repository"
    nuscenes_root = tmp_path / "nuscenes"
    derived_root = tmp_path / "derived"
    repository_root.mkdir()
    nuscenes_root.mkdir()
    derived_root.mkdir()
    config = _write_inputs(
        derived_root,
        config,
        mapping=mapping,
        selected_rule_overrides=selected_rule_overrides,
    )
    active_producer = producer or FakeProducer(source_records)
    result = build_development_projection(
        config=config,
        config_relative_path="configs/phase0_3_development_projection.yaml",
        repository_root=repository_root,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        nuscenes=FakeNuScenes(),
        producer=active_producer,
        producer_inputs=ProducerInputs(
            rules=object(),
            horizon_sec=3.0,
            sample_interval_sec=0.5,
            time_tolerance_sec=0.075,
            agent_radius_m=30.0,
        ),
        git_provenance=git_provenance or _clean_git(),
        now_utc=now_utc,
    )
    return result, derived_root, config, active_producer


def _rewrite_config(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialized_git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "git-repository"
    repository.mkdir()
    _git(repository, "init", "-b", "projection-test")
    _git(repository, "config", "user.name", "Projection Test")
    _git(repository, "config", "user.email", "projection@example.com")
    (repository / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "test: initialize repository")
    return repository


def _rerun_existing(
    tmp_path: Path,
    derived_root: Path,
    config,
    *,
    git_provenance: GitProvenance | None = None,
):
    class ForbiddenProducer:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            raise AssertionError("existing artifact must bypass producer")

    return build_development_projection(
        config=config,
        config_relative_path="configs/phase0_3_development_projection.yaml",
        repository_root=tmp_path / "repository",
        nuscenes_root=tmp_path / "nuscenes",
        derived_root=derived_root,
        nuscenes=FakeNuScenes(),
        producer=ForbiddenProducer(),
        producer_inputs=ProducerInputs(object(), 3.0, 0.5, 0.075, 30.0),
        git_provenance=git_provenance or _clean_git(commit="f" * 40),
    )


def _tamper_receipt(
    derived_root: Path,
    config,
    mutate,
) -> None:
    receipt_path = (
        derived_root / config.output_relative_dir / "projection_receipt.json"
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(payload)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")


def test_dirty_tracked_modification_blocks_git_provenance(tmp_path: Path) -> None:
    repository = _initialized_git_repository(tmp_path)
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean worktree"):
        collect_projection_git_provenance(repository)


def test_staged_modification_blocks_git_provenance(tmp_path: Path) -> None:
    repository = _initialized_git_repository(tmp_path)
    (repository / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")

    with pytest.raises(ValueError, match="clean worktree"):
        collect_projection_git_provenance(repository)


def test_untracked_file_blocks_git_provenance(tmp_path: Path) -> None:
    repository = _initialized_git_repository(tmp_path)
    (repository / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean worktree"):
        collect_projection_git_provenance(repository)


def test_clean_branch_git_provenance_is_collected(tmp_path: Path) -> None:
    repository = _initialized_git_repository(tmp_path)
    provenance = collect_projection_git_provenance(repository)

    assert provenance.commit == _git(repository, "rev-parse", "HEAD")
    assert provenance.branch == "projection-test"
    assert provenance.detached_head is False
    assert provenance.worktree_clean is True


def test_detached_head_git_provenance_is_recorded(tmp_path: Path) -> None:
    repository = _initialized_git_repository(tmp_path)
    _git(repository, "checkout", "--detach", "HEAD")

    provenance = collect_projection_git_provenance(repository)

    assert provenance.branch is None
    assert provenance.detached_head is True
    assert provenance.worktree_clean is True


def test_config_missing_required_field(tmp_path: Path) -> None:
    path = _rewrite_config(tmp_path, lambda raw: raw.pop("projection_version"))
    with pytest.raises(ValueError, match="projection_version"):
        load_config(path)


@pytest.mark.parametrize(
    "field,value",
    (
        ("combined_manifest_relative_path", "/absolute/manifest.jsonl"),
        ("scene_mapping_relative_path", "../mapping.json"),
        ("selected_rule_relative_path", "C:\\rule.json"),
        ("output_relative_dir", "phase_0_3\\projection"),
    ),
)
def test_config_rejects_invalid_relative_paths(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    path = _rewrite_config(tmp_path, lambda raw: raw.__setitem__(field, value))
    with pytest.raises(ValueError, match="relative POSIX path"):
        load_config(path)


def test_config_rejects_invalid_sha(tmp_path: Path) -> None:
    path = _rewrite_config(
        tmp_path,
        lambda raw: raw.__setitem__("expected_combined_manifest_sha256", "bad"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_config(path)


def test_combined_manifest_sha_mismatch(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    derived_root = tmp_path / "derived"
    derived_root.mkdir()
    config = _write_inputs(derived_root, synthetic_config)
    config = replace(config, expected_combined_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="manifest_sha256_mismatch"):
        build_development_projection(
            config=config,
            config_relative_path="configs/projection.yaml",
            repository_root=tmp_path / "repo",
            nuscenes_root=tmp_path,
            derived_root=derived_root,
            nuscenes=FakeNuScenes(),
            producer=FakeProducer(source_records),
            producer_inputs=ProducerInputs(object(), 3.0, 0.5, 0.075, 30.0),
            git_provenance=_clean_git(),
        )


def test_library_rejects_invalid_git_commit(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    with pytest.raises(ValueError, match="40-character SHA"):
        _build(
            tmp_path,
            synthetic_config,
            source_records,
            git_provenance=_clean_git(commit="invalid"),
        )


def test_library_rejects_dirty_git_provenance_before_producer_or_output(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    class ForbiddenProducer:
        called = False

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            self.called = True
            raise AssertionError("dirty Git provenance must block producer")

    producer = ForbiddenProducer()
    dirty = GitProvenance(
        commit="e" * 40,
        branch="feature",
        detached_head=False,
        worktree_clean=False,
    )
    with pytest.raises(ValueError, match="clean worktree"):
        _build(
            tmp_path,
            synthetic_config,
            source_records,
            producer=producer,
            git_provenance=dirty,
        )
    output_parent = tmp_path / "derived" / "phase_0_3"
    assert producer.called is False
    assert not (output_parent / "development_projection_v0_1").exists()
    assert not output_parent.exists()


def test_invalid_json_combined_manifest_is_hashed_without_parsing(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    result, _, _, _ = _build(tmp_path, synthetic_config, source_records)
    assert result["status"] == "created"
    assert result["receipt"]["combined_manifest"]["records_parsed"] == 0


@pytest.mark.parametrize(
    "module,name",
    (
        (protocol, "iter_manifest_rows"),
        (protocol, "validate_manifest"),
        (trainval_builder, "build_full_scene_splits"),
        (trainval_builder, "load_audit_index"),
    ),
)
def test_forbidden_functions_are_never_called(
    tmp_path,
    monkeypatch,
    synthetic_config,
    source_records,
    module,
    name,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"forbidden function called: {name}")

    monkeypatch.setattr(module, name, forbidden)
    result, _, _, _ = _build(tmp_path, synthetic_config, source_records)
    assert result["status"] == "created"


def test_scene_mapping_internal_sha_mismatch(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    mapping = _mapping_payload()
    mapping["scene_split_mapping_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="scene split mapping hash mismatch"):
        _build(tmp_path, synthetic_config, source_records, mapping=mapping)


def test_selected_rule_manifest_sha_mismatch(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    with pytest.raises(ValueError, match="manifest SHA-256"):
        _build(
            tmp_path,
            synthetic_config,
            source_records,
            selected_rule_overrides={"manifest_sha256": "f" * 64},
        )


def test_selected_rule_mapping_sha_mismatch(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    with pytest.raises(ValueError, match="selected rule and scene mapping"):
        _build(
            tmp_path,
            synthetic_config,
            source_records,
            selected_rule_overrides={"split_mapping_sha256": "f" * 64},
        )


def test_scene_count_mismatch(synthetic_config) -> None:
    config = replace(
        synthetic_config,
        expected_scene_counts={"train": 559, "validation": 141},
    )
    with pytest.raises(ValueError, match="scene counts"):
        select_development_scenes(_mapping_payload(), config)


def test_only_sorted_train_and_validation_scenes_reach_producer(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, _, _, producer = _build(tmp_path, synthetic_config, source_records)
    assert len(producer.calls) == 2
    for call in producer.calls:
        scene_tokens = tuple(call["scene_tokens"])
        assert scene_tokens == tuple(sorted(scene_tokens))
        assert all(int(token[-3:]) < 700 for token in scene_tokens)
        assert call["audit_index"] == {}
        assert set(call["official_splits"].values()) == {"train"}


def test_test_scene_traversal_attempt_fails_before_delegate(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    class ForbiddenProducer(FakeProducer):
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            kwargs["nuscenes"].get("scene", "scene-700")
            return super().__call__(**kwargs)

    producer = ForbiddenProducer(source_records)
    with pytest.raises(ValueError, match="project test scene traversal"):
        _build(tmp_path, synthetic_config, source_records, producer=producer)
    output_dir = tmp_path / "derived" / synthetic_config.output_relative_dir
    assert not output_dir.exists()


def test_guard_rejects_test_sample_after_detection() -> None:
    class TestSampleReader:
        def get(self, table_name: str, token: str) -> dict[str, object]:
            return {"scene_token": "scene-700"}

    counters = IsolationCounters()
    reader = GuardedNuScenesReader(
        TestSampleReader(),
        frozenset({"scene-000"}),
        frozenset({"scene-700"}),
        counters,
    )
    with pytest.raises(ValueError, match="test sample access"):
        reader.get("sample", "sample-test")
    assert counters.test_sample_records_read == 1


def test_projection_omits_future_pose_agent_and_audit_fields(
    synthetic_config,
) -> None:
    record = project_record(
        _source_record("train", "scene-000", 0, "full"),
        split="train",
        config=synthetic_config,
        split_mapping_sha256="b" * 64,
    )
    forbidden = {
        "future_ego_trajectory",
        "nearby_agents",
        "current_ego_pose",
        "coordinate_metadata",
        "audit_status",
        "source_audit_record",
    }
    assert forbidden.isdisjoint(record)
    assert record["target_action"] == "keep"


def test_projection_rejects_invalid_action(synthetic_config) -> None:
    source = _source_record(
        "train",
        "scene-000",
        0,
        "full",
        action="turn_left",
    )
    with pytest.raises(ValueError, match="Unsupported action"):
        project_record(
            source,
            split="train",
            config=synthetic_config,
            split_mapping_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "path",
    (
        "/absolute.jpg",
        "C:\\samples\\CAM_FRONT\\image.jpg",
        "samples/CAM_FRONT/../secret.jpg",
        "samples/CAM_BACK/image.jpg",
    ),
)
def test_projection_rejects_invalid_cam_front_paths(
    synthetic_config,
    path: str,
) -> None:
    source = _source_record(
        "train",
        "scene-000",
        0,
        "full",
        cam_front_path=path,
    )
    with pytest.raises(ValueError, match="invalid CAM_FRONT"):
        project_record(
            source,
            split="train",
            config=synthetic_config,
            split_mapping_sha256="b" * 64,
        )


def test_projection_record_sha_is_recomputable(synthetic_config) -> None:
    record = project_record(
        _source_record("train", "scene-000", 0, "full"),
        split="train",
        config=synthetic_config,
        split_mapping_sha256="b" * 64,
    )
    expected = record.pop("projection_record_sha256")
    assert hashlib.sha256(canonical_json_bytes(record)).hexdigest() == expected


def test_same_inputs_write_identical_jsonl(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, first_root, _, _ = _build(
        tmp_path / "first",
        synthetic_config,
        source_records,
    )
    _, second_root, _, _ = _build(
        tmp_path / "second",
        synthetic_config,
        source_records,
    )
    for split in ("train", "validation"):
        relative = Path(synthetic_config.output_relative_dir) / f"{split}.jsonl"
        assert (first_root / relative).read_bytes() == (
            second_root / relative
        ).read_bytes()


def test_sample_count_mismatch_hard_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    config = replace(
        synthetic_config,
        expected_sample_counts={"train": 3, "validation": 1},
    )
    with pytest.raises(ValueError, match="sample count"):
        _build(tmp_path, config, source_records)


def test_motion_availability_mismatch_hard_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    config = replace(
        synthetic_config,
        expected_motion_availability={
            "train": {"full": 2, "partial": 0, "unavailable": 0},
            "validation": {"full": 0, "partial": 0, "unavailable": 1},
        },
    )
    with pytest.raises(ValueError, match="motion availability"):
        _build(tmp_path, config, source_records)


@pytest.mark.parametrize(
    "provenance,expected",
    (
        (
            GitProvenance("e" * 40, "projection-test", False, True),
            {
                "commit": "e" * 40,
                "branch": "projection-test",
                "detached_head": False,
                "worktree_clean": True,
            },
        ),
        (
            GitProvenance("f" * 40, None, True, True),
            {
                "commit": "f" * 40,
                "branch": None,
                "detached_head": True,
                "worktree_clean": True,
            },
        ),
    ),
)
def test_validated_git_provenance_is_written_to_receipt(
    tmp_path,
    synthetic_config,
    source_records,
    provenance,
    expected,
) -> None:
    result, _, _, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
        git_provenance=provenance,
    )
    assert result["receipt"]["git"] == expected


def test_existing_artifact_is_not_overwritten(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    first_result, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )
    receipt_path = (
        derived_root / config.output_relative_dir / "projection_receipt.json"
    )
    original_receipt = receipt_path.read_bytes()
    second_result = _rerun_existing(
        tmp_path,
        derived_root,
        config,
        git_provenance=_clean_git(commit="f" * 40),
    )
    assert first_result["status"] == "created"
    assert second_result["status"] == "already_exists"
    assert first_result["receipt"]["git"]["commit"] == "e" * 40
    assert second_result["receipt"]["git"]["commit"] == "e" * 40
    assert receipt_path.read_bytes() == original_receipt


def test_existing_receipt_missing_isolation_field_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )
    _tamper_receipt(
        derived_root,
        config,
        lambda payload: payload.pop("test_labels_read"),
    )

    with pytest.raises(ValueError, match="test_labels_read mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_nonzero_isolation_counter_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )
    _tamper_receipt(
        derived_root,
        config,
        lambda payload: payload.__setitem__("test_sample_records_read", 1),
    )

    with pytest.raises(ValueError, match="test_sample_records_read mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_motion_distribution_mismatch_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["motion_availability_distribution_by_split"]["train"][
            "full"
        ] = 2

    _tamper_receipt(derived_root, config, mutate)
    with pytest.raises(ValueError, match="receipt train mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_action_distribution_total_mismatch_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["action_distribution_by_split"]["train"]["keep"] = 1

    _tamper_receipt(derived_root, config, mutate)
    with pytest.raises(ValueError, match="action distribution total mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_selected_rule_provenance_mismatch_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["selected_rule"]["split_mapping_sha256"] = "a" * 64

    _tamper_receipt(derived_root, config, mutate)
    with pytest.raises(ValueError, match="split_mapping_sha256 mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_combined_manifest_file_size_mismatch_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["combined_manifest"]["file_size_bytes"] += 1

    _tamper_receipt(derived_root, config, mutate)
    with pytest.raises(ValueError, match="file_size_bytes mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_existing_receipt_output_relative_path_mismatch_fails(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    _, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["outputs"]["train"]["relative_path"] = "tampered.jsonl"

    _tamper_receipt(derived_root, config, mutate)
    with pytest.raises(ValueError, match="relative_path mismatch"):
        _rerun_existing(tmp_path, derived_root, config)


def test_staging_failure_leaves_no_formal_artifact(
    tmp_path,
    monkeypatch,
    synthetic_config,
    source_records,
) -> None:
    def fail_receipt(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic receipt failure")

    monkeypatch.setattr(projection_module, "write_canonical_json", fail_receipt)
    with pytest.raises(OSError, match="synthetic receipt failure"):
        _build(tmp_path, synthetic_config, source_records)
    output_parent = tmp_path / "derived" / "phase_0_3"
    assert not (output_parent / "development_projection_v0_1").exists()
    assert list(output_parent.glob(".*.staging")) == []


def test_receipt_records_isolation_and_projection_contract(
    tmp_path,
    synthetic_config,
    source_records,
) -> None:
    result, derived_root, config, _ = _build(
        tmp_path,
        synthetic_config,
        source_records,
    )
    receipt = result["receipt"]
    for field in (
        "combined_manifest_records_parsed",
        "test_scene_traversal_attempts",
        "test_sample_records_read",
        "test_images_opened",
        "test_labels_read",
        "audit_records_read",
    ):
        assert receipt[field] == 0
    assert receipt["test_evaluation_performed"] is False
    assert receipt["model_load_performed"] is False
    assert receipt["processor_load_performed"] is False
    assert receipt["future_trajectory_used_for_target_derivation"] is True
    assert receipt["future_trajectory_written_to_projection"] is False
    assert receipt["nearby_agents_written_to_projection"] is False
    assert receipt["current_ego_pose_written_to_projection"] is False
    output_dir = derived_root / config.output_relative_dir
    assert not (output_dir / "test.jsonl").exists()
    assert receipt["outputs"]["train"]["record_count"] == 2
    assert receipt["outputs"]["validation"]["record_count"] == 1


def test_projection_modules_do_not_import_model_libraries() -> None:
    sources = (
        (REPOSITORY_ROOT / "src/phase0/development_projection.py").read_text(
            encoding="utf-8"
        ),
        (
            REPOSITORY_ROOT
            / "scripts/build_phase0_3_development_projection.py"
        ).read_text(encoding="utf-8"),
    )
    for source in sources:
        assert "transformers" not in source.lower()
        assert "peft" not in source.lower()

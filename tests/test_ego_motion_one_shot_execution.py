from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import BinaryIO

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_ego_motion_one_shot_test import (
    ManifestPreflightSummary,
    _protocol,
    build_preflight_receipt,
)
from scripts.run_ego_motion_rule_baseline import (
    build_validation_metrics_artifact_payload,
)
from scripts.run_ego_motion_one_shot_test import (
    CONFIRMATION,
    EXECUTION_CLAIM_FILENAME,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    EXPECTED_SPLIT_SCENE_COUNTS,
    ExecutionClaim,
    ExecutionPaths,
    ExecutionProvenance,
    FormalOutputs,
    _canonical_json_bytes,
    _source_hashes,
    _validate_no_outputs_or_claim,
    _write_atomic_once,
    build_formal_outputs,
    create_execution_claim,
    formal_temporary_path,
    load_test_samples_after_claim,
    load_train_samples_without_test_access,
    parse_args,
    run_one_shot_execution,
    validate_cli_config_path,
    validate_execution_manifest_metadata,
    validate_execution_preconditions,
    write_formal_outputs_once,
)
from src.actions.schema import ACTION_SCHEMA
from src.baselines.ego_motion import (
    EgoMotionCandidateEvaluation,
    EgoMotionFeatures,
    EgoMotionRuleThresholds,
)
from src.baselines.ego_motion_analysis import DiagnosticMargins
from src.baselines.ego_motion_test import (
    DECLARED_TEST_OUTPUTS,
    EVALUATOR_SOURCE_PATHS,
    FORMAL_RESULT_SHA_FILENAMES,
    FORBIDDEN_TEST_FIELDS,
    TEST_PREDICTION_FIELDS,
    build_validation_to_test_comparison,
    evaluate_frozen_rule_test_samples,
)
from src.phase0.protocol import evaluate_classification


MAPPING_SHA = "a96e04aaf068e75b0aa3ecb8412dc5b35fea2412d7090bbee0a6661132923b12"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion(
    availability: str = "full",
) -> dict[str, object]:
    if availability == "unavailable":
        return {
            "availability": availability,
            "speed_mps": None,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": None,
            "history_interval_sec": None,
            "acceleration_interval_sec": None,
        }
    if availability == "partial":
        return {
            "availability": availability,
            "speed_mps": 3.0,
            "longitudinal_acceleration_mps2": None,
            "yaw_rate_radps": 0.0,
            "history_interval_sec": 0.5,
            "acceleration_interval_sec": None,
        }
    return {
        "availability": availability,
        "speed_mps": 3.0,
        "longitudinal_acceleration_mps2": 0.0,
        "yaw_rate_radps": 0.0,
        "history_interval_sec": 0.5,
        "acceleration_interval_sec": 0.5,
    }


def manifest_row(
    token: str,
    *,
    scene: str,
    split: str,
    action: str = "keep",
    availability: str = "full",
) -> dict[str, object]:
    return {
        "sample_token": token,
        "scene_token": scene,
        "split": split,
        "official_split": "val" if split == "test" else "train",
        "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
        "label_rule_version": "phase-1.6-meta-action-v0.2",
        "split_seed": 20260710,
        "split_strategy_version": "official_train_scene_label_stratified_v1",
        "split_mapping_sha256": MAPPING_SHA,
        "meta_action": action,
        "current_ego_motion": motion(availability),
        "future_ego_trajectory": [{"x": 1.0}],
        "nearby_agents": [],
        "current_ego_pose": {"translation": [0.0, 0.0, 0.0]},
        "cam_front_path": "samples/CAM_FRONT/example.jpg",
    }


def synthetic_rows() -> list[dict[str, object]]:
    rows = [
        manifest_row("train-0", scene="train-scene-0", split="train"),
        manifest_row(
            "train-1",
            scene="train-scene-1",
            split="train",
            action="stop",
        ),
        manifest_row(
            "validation-0", scene="validation-scene-0", split="validation"
        ),
    ]
    for index in range(3799):
        availability = "full"
        if index == 0:
            availability = "unavailable"
        elif index == 1:
            availability = "partial"
        rows.append(
            manifest_row(
                f"test-{index:04d}",
                scene=f"test-scene-{index % 150:03d}",
                split="test",
                action=ACTION_SCHEMA[index % len(ACTION_SCHEMA)],
                availability=availability,
            )
        )
    return rows


def canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validation_metrics_artifact() -> dict[str, object]:
    sample_count = EXPECTED_SPLIT_SAMPLE_COUNTS["validation"]
    actions = tuple(
        ACTION_SCHEMA[index % len(ACTION_SCHEMA)]
        for index in range(sample_count)
    )
    metrics = evaluate_classification(actions, actions)
    thresholds = EgoMotionRuleThresholds(0.2, 0.05, 0.5, 0.3)
    selected = EgoMotionCandidateEvaluation(
        candidate_id="candidate-0293",
        thresholds=thresholds,
        thresholds_sha256=thresholds.sha256(),
        metrics=metrics,
        minimum_per_class_f1=min(metrics.per_class_f1.values()),
        predicted_class_distribution={
            action: sample_count // len(ACTION_SCHEMA)
            for action in ACTION_SCHEMA
        },
    )
    return build_validation_metrics_artifact_payload(
        "phase0.2b-ego-motion-rule-v0.1",
        selected,
        {"default_keep": sample_count},
    )


def build_tree(tmp_path: Path) -> tuple[ExecutionPaths, ExecutionProvenance]:
    derived_root = tmp_path / "derived"
    output_dir = derived_root / "phase_0_2/ego_motion_one_shot_test_v0_1"
    output_dir.mkdir(parents=True)
    manifest_path = (
        derived_root / "phase_0_1b/trainval_manifest_v1/manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in synthetic_rows()),
        encoding="utf-8",
    )
    freeze_path = (
        derived_root
        / "phase_0_2/ego_motion_rule_freeze_v0_1/rule_freeze.json"
    )
    freeze_path.parent.mkdir(parents=True)
    config = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/phase0_2_one_shot_test_v0_1.yaml"
        ).read_text(encoding="utf-8")
    )
    config["expected_manifest_sha256"] = sha256(manifest_path)
    protocol = _protocol(config)
    freeze = {
        "freeze_status": "frozen",
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "selected_candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "next_gate": "phase0.2d_one_shot_test",
        "test_evaluation_performed": False,
        "manifest_sha256": sha256(manifest_path),
        "manifest_schema_version": protocol.manifest_schema_version,
        "label_rule_version": protocol.label_rule_version,
        "split_mapping_sha256": protocol.split_mapping_sha256,
    }
    freeze_path.write_text(canonical_json(freeze), encoding="utf-8")
    config["expected_freeze_sha256"] = sha256(freeze_path)
    config_path = tmp_path / "synthetic_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    validation_path = (
        derived_root / "phase_0_2/ego_motion_rule_v0_1/validation_metrics.json"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        canonical_json(validation_metrics_artifact()), encoding="utf-8"
    )
    evaluator_hashes = _source_hashes(config_path)
    preflight = build_preflight_receipt(
        protocol=protocol,
        manifest_sha256=sha256(manifest_path),
        freeze_sha256=sha256(freeze_path),
        freeze_source_sha256={
            "failure_analysis.json": "a" * 64,
            "validation_failures.jsonl": "b" * 64,
        },
        manifest_summary=ManifestPreflightSummary(3799, 150),
        evaluator_source_sha256=evaluator_hashes,
    )
    preflight_path = output_dir / "one_shot_preflight_receipt.json"
    preflight_path.write_text(canonical_json(preflight), encoding="utf-8")
    paths = ExecutionPaths(
        config_path=config_path,
        manifest_path=manifest_path,
        freeze_path=freeze_path,
        validation_metrics_path=validation_path,
        preflight_receipt_path=preflight_path,
        output_dir=output_dir,
        claim_path=output_dir / EXECUTION_CLAIM_FILENAME,
        execution_source_path=(
            PROJECT_ROOT / "scripts/run_ego_motion_one_shot_test.py"
        ),
    )
    provenance = ExecutionProvenance(
        protocol=protocol,
        expected_config_sha256=sha256(config_path),
        expected_manifest_sha256=sha256(manifest_path),
        expected_freeze_sha256=sha256(freeze_path),
        expected_validation_metrics_sha256=sha256(validation_path),
        expected_preflight_receipt_sha256=sha256(preflight_path),
        diagnostic_margins=DiagnosticMargins(0.05, 0.01, 0.05),
        expected_split_seed=20260710,
        expected_split_strategy_version=(
            "official_train_scene_label_stratified_v1"
        ),
        expected_split_sample_counts={
            "train": 2,
            "validation": 1,
            "test": 3799,
        },
        expected_split_scene_counts={
            "train": 2,
            "validation": 1,
            "test": 150,
        },
    )
    return paths, provenance


@pytest.fixture
def execution_tree(
    tmp_path: Path,
) -> tuple[ExecutionPaths, ExecutionProvenance]:
    return build_tree(tmp_path)


@pytest.mark.parametrize(
    "argv",
    (
        ("--config", "configs/phase0_2_one_shot_test_v0_1.yaml"),
        (
            "--config",
            "configs/phase0_2_one_shot_test_v0_1.yaml",
            "--confirm-one-shot",
            "wrong",
        ),
        (
            "--config",
            "configs/phase0_2_one_shot_test_v0_1.yaml",
            "--confirm-one-shot",
            CONFIRMATION,
            "--unknown",
        ),
    ),
)
def test_cli_rejects_missing_wrong_or_unknown_arguments(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_cli_accepts_only_complete_confirmation() -> None:
    arguments = parse_args(
        (
            "--config",
            "configs/phase0_2_one_shot_test_v0_1.yaml",
            "--confirm-one-shot",
            CONFIRMATION,
        )
    )

    assert arguments.confirm_one_shot == CONFIRMATION


def test_cli_rejects_non_frozen_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--config must be"):
        validate_cli_config_path(tmp_path / "config.yaml")


@pytest.mark.parametrize(
    ("path_field", "expected_field", "expected_message"),
    (
        ("manifest_path", "expected_manifest_sha256", "manifest SHA-256"),
        ("freeze_path", "expected_freeze_sha256", "freeze artifact SHA-256"),
        (
            "validation_metrics_path",
            "expected_validation_metrics_sha256",
            "validation metrics SHA-256",
        ),
        (
            "preflight_receipt_path",
            "expected_preflight_receipt_sha256",
            "preflight receipt SHA-256",
        ),
    ),
)
def test_provenance_hash_mismatch_fails_before_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    path_field: str,
    expected_field: str,
    expected_message: str,
) -> None:
    paths, provenance = execution_tree
    path = getattr(paths, path_field)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=expected_message):
        validate_execution_preconditions(paths, provenance)

    assert not paths.claim_path.exists()


def test_validation_adapter_failure_precedes_manifest_scan_and_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    flat_metrics = {
        "sample_count": EXPECTED_SPLIT_SAMPLE_COUNTS["validation"],
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "per_class_f1": {action: 1.0 for action in ACTION_SCHEMA},
        "prediction_class_distribution": {
            action: EXPECTED_SPLIT_SAMPLE_COUNTS["validation"]
            // len(ACTION_SCHEMA)
            for action in ACTION_SCHEMA
        },
    }
    paths.validation_metrics_path.write_text(
        canonical_json(flat_metrics), encoding="utf-8"
    )
    provenance = replace(
        provenance,
        expected_validation_metrics_sha256=sha256(
            paths.validation_metrics_path
        ),
    )
    manifest_scan_called = False

    def forbidden_manifest_scan(path: Path) -> tuple[Mapping[str, object], ...]:
        nonlocal manifest_scan_called
        manifest_scan_called = True
        raise AssertionError(f"manifest scan reached after adapter failure: {path}")

    monkeypatch.setattr(
        "scripts.run_ego_motion_one_shot_test.iter_manifest_rows",
        forbidden_manifest_scan,
    )

    with pytest.raises(ValueError, match="producer schema"):
        validate_execution_preconditions(paths, provenance)

    assert not manifest_scan_called
    assert not paths.claim_path.exists()


def test_preflight_mapping_mismatch_fails_before_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    changed = json.loads(paths.preflight_receipt_path.read_text(encoding="utf-8"))
    changed["test_scene_count"] = 149

    with pytest.raises(ValueError, match="mapping does not match receipt bytes"):
        validate_execution_preconditions(
            paths, provenance, preflight_receipt=changed
        )

    assert not paths.claim_path.exists()


def test_evaluator_source_hash_mismatch_fails_before_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    receipt = json.loads(paths.preflight_receipt_path.read_text(encoding="utf-8"))
    receipt["evaluator_source_sha256"][EVALUATOR_SOURCE_PATHS[1]] = "e" * 64
    paths.preflight_receipt_path.write_text(
        canonical_json(receipt), encoding="utf-8"
    )
    provenance = replace(
        provenance,
        expected_preflight_receipt_sha256=sha256(paths.preflight_receipt_path),
    )

    with pytest.raises(ValueError, match="differ from preflight"):
        validate_execution_preconditions(paths, provenance)

    assert not paths.claim_path.exists()


@pytest.mark.parametrize("existing_name", DECLARED_TEST_OUTPUTS)
def test_any_formal_output_blocks_execution_before_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    existing_name: str,
) -> None:
    paths, provenance = execution_tree
    (paths.output_dir / existing_name).write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="formal one-shot test output"):
        validate_execution_preconditions(paths, provenance)

    assert not paths.claim_path.exists()


@pytest.mark.parametrize("formal_filename", DECLARED_TEST_OUTPUTS)
def test_stale_temporary_output_blocks_execution_before_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    formal_filename: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    temporary_path = formal_temporary_path(paths.output_dir, formal_filename)
    temporary_path.write_bytes(b"stale-temporary-output")
    manifest_scan_called = False

    def forbidden_manifest_scan(path: Path) -> tuple[Mapping[str, object], ...]:
        nonlocal manifest_scan_called
        manifest_scan_called = True
        raise AssertionError(f"manifest scan reached after stale temp: {path}")

    monkeypatch.setattr(
        "scripts.run_ego_motion_one_shot_test.iter_manifest_rows",
        forbidden_manifest_scan,
    )

    with pytest.raises(FileExistsError, match="stale temporary output"):
        validate_execution_preconditions(paths, provenance)

    assert not manifest_scan_called
    assert not paths.claim_path.exists()
    assert all(
        not (paths.output_dir / filename).exists()
        for filename in DECLARED_TEST_OUTPUTS
    )
    assert temporary_path.read_bytes() == b"stale-temporary-output"


def test_existing_claim_blocks_execution(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    paths.claim_path.write_text("claimed", encoding="utf-8")

    with pytest.raises(FileExistsError, match="retry is permanently forbidden"):
        validate_execution_preconditions(paths, provenance)


def test_claim_is_exclusive_and_has_fixed_canonical_payload(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    expected_fields = {
        "claim_schema_version",
        "protocol_status",
        "preflight_receipt_sha256",
        "manifest_sha256",
        "freeze_sha256",
        "validation_metrics_sha256",
        "execution_source_sha256",
        "candidate_id",
        "thresholds_sha256",
        "execution_count",
        "rerun_permitted",
        "test_label_access_allowed_after_claim",
        "test_motion_access_allowed_after_claim",
    }

    assert set(claim.payload) == expected_fields
    assert claim.payload["claim_schema_version"] == (
        "phase0.2_one_shot_execution_claim_v0.1"
    )
    assert claim.payload["execution_count"] == 1
    assert claim.payload["rerun_permitted"] is False
    assert claim.path.read_bytes() == _canonical_json_bytes(claim.payload)
    assert "timestamp" not in claim.payload
    with pytest.raises(FileExistsError):
        create_execution_claim(paths, provenance, preconditions)


def test_claim_file_and_parent_directory_are_fsynced_before_test_access(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    module = sys.modules["scripts.run_ego_motion_one_shot_test"]
    original_open = Path.open
    original_fsync = module.os.fsync
    events: list[str] = []
    claim_file_closed = False

    class TrackedClaimFile:
        def __init__(self, raw_file: BinaryIO) -> None:
            self.raw_file = raw_file

        def __enter__(self) -> TrackedClaimFile:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            nonlocal claim_file_closed
            self.raw_file.close()
            claim_file_closed = True
            events.append("claim file close")

        def write(self, payload: bytes) -> int:
            events.append("claim write")
            return self.raw_file.write(payload)

        def flush(self) -> None:
            events.append("claim flush")
            self.raw_file.flush()

        def fileno(self) -> int:
            return self.raw_file.fileno()

    def tracked_open(
        path: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> object:
        raw_file = original_open(path, mode, *args, **kwargs)
        if path == paths.claim_path and mode == "xb":
            events.append("claim exclusive create")
            return TrackedClaimFile(raw_file)
        return raw_file

    def tracked_file_fsync(descriptor: int) -> None:
        events.append("claim file fsync")
        original_fsync(descriptor)

    def tracked_directory_fsync(path: Path) -> None:
        assert claim_file_closed
        assert path == paths.claim_path.parent
        events.append("claim parent directory fsync")

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(module.os, "fsync", tracked_file_fsync)
    monkeypatch.setattr(module, "_fsync_directory", tracked_directory_fsync)
    claim = create_execution_claim(paths, provenance, preconditions)

    class AccessTrackingRow(GuardedRow):
        def get(self, key: str, default: object = None) -> object:
            if key in {"meta_action", "current_ego_motion"}:
                if dict.get(self, "split") == "test":
                    events.append("test sealed field access")
            return super().get(key, default)

    rows = [
        AccessTrackingRow(row, claim_path=paths.claim_path)
        for row in synthetic_rows()
    ]
    load_test_samples_after_claim(rows, provenance.protocol, claim)

    required_order = (
        "claim exclusive create",
        "claim write",
        "claim flush",
        "claim file fsync",
        "claim file close",
        "claim parent directory fsync",
        "test sealed field access",
    )
    positions = tuple(events.index(event) for event in required_order)
    assert positions == tuple(sorted(positions))


def test_fsync_directory_fsyncs_and_closes_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = sys.modules["scripts.run_ego_motion_one_shot_test"]
    events: list[tuple[str, object]] = []

    def tracked_open(path: Path, flags: int) -> int:
        events.append(("open", (path, flags)))
        return 42

    def tracked_fsync(descriptor: int) -> None:
        events.append(("fsync", descriptor))

    def tracked_close(descriptor: int) -> None:
        events.append(("close", descriptor))

    monkeypatch.setattr(module.os, "open", tracked_open)
    monkeypatch.setattr(module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(module.os, "close", tracked_close)

    module._fsync_directory(tmp_path)

    assert events == [
        ("open", (tmp_path, module.os.O_RDONLY)),
        ("fsync", 42),
        ("close", 42),
    ]


def test_directory_fsync_failure_preserves_claim_and_blocks_test_access(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    test_loader_called = False

    def fail_directory_fsync(path: Path) -> None:
        assert path == paths.claim_path.parent
        raise OSError("directory fsync failed")

    def forbidden_test_loader(*args: object, **kwargs: object) -> None:
        nonlocal test_loader_called
        test_loader_called = True
        raise AssertionError("test loader must not run after directory fsync failure")

    monkeypatch.setattr(
        "scripts.run_ego_motion_one_shot_test._fsync_directory",
        fail_directory_fsync,
    )
    monkeypatch.setattr(
        "scripts.run_ego_motion_one_shot_test.load_test_samples_after_claim",
        forbidden_test_loader,
    )

    with pytest.raises(OSError, match="directory fsync failed"):
        run_one_shot_execution(paths, provenance)

    assert not test_loader_called
    assert paths.claim_path.exists()
    assert all(
        not (paths.output_dir / filename).exists()
        for filename in DECLARED_TEST_OUTPUTS
    )
    with pytest.raises(FileExistsError):
        create_execution_claim(paths, provenance, preconditions)


class GuardedRow(dict[str, object]):
    def __init__(self, *args: object, claim_path: Path | None = None, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.claim_path = claim_path

    def get(self, key: str, default: object = None) -> object:
        if key in {"meta_action", "current_ego_motion"}:
            is_test = dict.get(self, "split") == "test"
            if is_test and (
                self.claim_path is None or not self.claim_path.exists()
            ):
                raise AssertionError(f"sealed field accessed before claim: {key}")
        return super().get(key, default)


def guarded_rows(claim_path: Path | None = None) -> list[GuardedRow]:
    return [GuardedRow(row, claim_path=claim_path) for row in synthetic_rows()]


def minimal_provenance(
    source: ExecutionProvenance,
) -> ExecutionProvenance:
    return replace(
        source,
        expected_split_sample_counts={
            "train": 2,
            "validation": 1,
            "test": 3799,
        },
        expected_split_scene_counts={
            "train": 2,
            "validation": 1,
            "test": 150,
        },
    )


def test_metadata_gate_never_accesses_test_label_or_motion(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    _, provenance = execution_tree

    summary = validate_execution_manifest_metadata(
        guarded_rows(), minimal_provenance(provenance)
    )

    assert summary.split_sample_counts["test"] == 3799


def test_train_loader_skips_test_before_accessing_label() -> None:
    rows = guarded_rows()

    samples = load_train_samples_without_test_access(rows)

    assert {sample.sample_token for sample in samples} == {"train-0", "train-1"}
    assert all(sample.split == "train" for sample in samples)


def test_test_loader_allows_sealed_access_only_after_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    rows = guarded_rows(paths.claim_path)
    missing_claim = ExecutionClaim(paths.claim_path, {})
    with pytest.raises(FileNotFoundError, match="claim must exist"):
        load_test_samples_after_claim(rows, provenance.protocol, missing_claim)
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)

    samples = load_test_samples_after_claim(rows, provenance.protocol, claim)

    assert len(samples) == 3799
    assert all(sample.split == "test" for sample in samples)


def test_preconditions_do_not_call_predictor_or_classification_evaluator(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("test evaluator called before claim")

    monkeypatch.setattr(
        "src.baselines.ego_motion_test.evaluate_frozen_rule_test_samples",
        forbidden,
    )
    monkeypatch.setattr("src.phase0.protocol.evaluate_classification", forbidden)

    validate_execution_preconditions(paths, provenance)


def claimed_test_rows(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> tuple[
    list[dict[str, object]],
    ExecutionClaim,
    ExecutionProvenance,
]:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    return synthetic_rows(), claim, provenance


def test_test_loader_rejects_wrong_count(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)
    rows.pop()

    with pytest.raises(ValueError, match="test sample count"):
        load_test_samples_after_claim(rows, provenance.protocol, claim)


def test_test_loader_rejects_wrong_scene_count(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)
    for row in rows:
        if row["split"] == "test":
            row["scene_token"] = "one-test-scene"

    with pytest.raises(ValueError, match="test scene count"):
        load_test_samples_after_claim(rows, provenance.protocol, claim)


def test_test_loader_rejects_duplicate_token(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_rows[1]["sample_token"] = test_rows[0]["sample_token"]

    with pytest.raises(ValueError, match="duplicate test sample_token"):
        load_test_samples_after_claim(rows, provenance.protocol, claim)


def test_metadata_rejects_scene_overlap(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    _, provenance = execution_tree
    rows = synthetic_rows()
    rows[0]["scene_token"] = "test-scene-000"

    with pytest.raises(ValueError, match="scene_token spans splits"):
        validate_execution_manifest_metadata(rows, provenance)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("official_split", "train", "official val"),
        ("manifest_schema_version", "wrong", "manifest_schema_version"),
        ("label_rule_version", "wrong", "label_rule_version"),
        ("split_mapping_sha256", "e" * 64, "split_mapping_sha256"),
    ),
)
def test_test_loader_rejects_contract_mismatch(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    field: str,
    value: object,
    message: str,
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)
    next(row for row in rows if row["split"] == "test")[field] = value

    with pytest.raises(ValueError, match=message):
        load_test_samples_after_claim(rows, provenance.protocol, claim)


def test_forbidden_manifest_fields_never_enter_evaluator_samples(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)

    samples = load_test_samples_after_claim(rows, provenance.protocol, claim)

    assert FORBIDDEN_TEST_FIELDS.isdisjoint(asdict(samples[0]))


def test_frozen_evaluator_rejects_forbidden_sample_field(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    _, provenance = execution_tree
    payload = {
        "sample_token": "sample",
        "scene_token": "scene",
        "split": "test",
        "features": EgoMotionFeatures(3.0, 0.0, 0.0, "full", 0.5, 0.5),
        "ground_truth_action": "keep",
        "label_rule_version": provenance.protocol.label_rule_version,
        "manifest_schema_version": provenance.protocol.manifest_schema_version,
        "split_mapping_sha256": provenance.protocol.split_mapping_sha256,
        "future_ego_trajectory": [],
    }

    with pytest.raises(ValueError, match="forbidden fields"):
        evaluate_frozen_rule_test_samples([payload], provenance.protocol)


def test_unavailable_and_partial_samples_are_not_filtered(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    rows, claim, provenance = claimed_test_rows(execution_tree)

    samples = load_test_samples_after_claim(rows, provenance.protocol, claim)

    assert len(samples) == 3799
    assert samples[0].features.availability == "unavailable"
    assert samples[1].features.availability == "partial"


def test_formal_outputs_reuse_frozen_rule_and_train_only_majority(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    samples = load_test_samples_after_claim(
        synthetic_rows(), provenance.protocol, claim
    )

    outputs = build_formal_outputs(samples, provenance, preconditions)
    predictions = [
        json.loads(line)
        for line in outputs.serialized_files["test_predictions.jsonl"].splitlines()
    ]
    majority = json.loads(
        outputs.serialized_files["majority_test_metrics.json"]
    )

    assert len(predictions) == 3799
    assert predictions == sorted(predictions, key=lambda row: row["sample_token"])
    assert all(tuple(row) == TEST_PREDICTION_FIELDS for row in predictions)
    assert all(row["candidate_id"] == "candidate-0293" for row in predictions)
    assert all(
        row["thresholds_sha256"] == provenance.protocol.thresholds_sha256
        for row in predictions
    )
    assert majority["majority_action"] == "keep"
    assert majority["metrics_schema_version"] == (
        "phase0.2_majority_test_metrics_v0.1"
    )


def test_formal_builders_are_called(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    samples = load_test_samples_after_claim(
        synthetic_rows(), provenance.protocol, claim
    )
    calls = {"rule": 0, "majority": 0, "comparison": 0, "diagnostics": 0}
    module = sys.modules["scripts.run_ego_motion_one_shot_test"]
    for name, key in (
        ("evaluate_frozen_rule_test_samples", "rule"),
        ("evaluate_majority_on_test_samples", "majority"),
        ("build_validation_to_test_comparison", "comparison"),
        ("build_test_diagnostics", "diagnostics"),
    ):
        original = getattr(module, name)

        def wrapper(
            *args: object,
            _original: object = original,
            _key: str = key,
            **kwargs: object,
        ) -> object:
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    build_formal_outputs(samples, provenance, preconditions)

    assert calls == {"rule": 1, "majority": 1, "comparison": 1, "diagnostics": 1}


def test_written_hashes_and_receipt_exclusions(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree

    outputs = run_one_shot_execution(paths, provenance)
    receipt = json.loads(
        (paths.output_dir / "one_shot_test_receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(receipt["output_sha256"]) == set(FORMAL_RESULT_SHA_FILENAMES)
    assert "one_shot_test_receipt.json" not in receipt["output_sha256"]
    assert EXECUTION_CLAIM_FILENAME not in receipt["output_sha256"]
    assert "one_shot_preflight_receipt.json" not in receipt["output_sha256"]
    for filename, digest in receipt["output_sha256"].items():
        assert sha256(paths.output_dir / filename) == digest
    for filename in DECLARED_TEST_OUTPUTS:
        assert sha256(paths.output_dir / filename) == hashlib.sha256(
            outputs.serialized_files[filename]
        ).hexdigest()
    assert receipt["execution_source_sha256"] == sha256(
        paths.execution_source_path
    )
    assert receipt["execution_count"] == 1
    assert receipt["rerun_permitted"] is False
    assert receipt["test_sample_count"] == 3799
    assert receipt["test_scene_count"] == 150


def test_receipt_is_written_last(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    payloads = {name: name.encode("utf-8") for name in DECLARED_TEST_OUTPUTS}
    writes: list[str] = []

    def record_write(path: Path, payload: bytes) -> None:
        assert payload == path.name.encode("utf-8")
        writes.append(path.name)

    monkeypatch.setattr(
        "scripts.run_ego_motion_one_shot_test._write_atomic_once", record_write
    )

    write_formal_outputs_once(paths, claim, FormalOutputs(payloads))

    assert writes == list(DECLARED_TEST_OUTPUTS)
    assert writes[-1] == "one_shot_test_receipt.json"


def test_writer_and_precondition_share_temporary_path_helper(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = execution_tree
    module = sys.modules["scripts.run_ego_motion_one_shot_test"]
    original_helper = module.formal_temporary_path
    helper_calls: list[tuple[Path, str]] = []

    def tracked_helper(output_dir: Path, filename: str) -> Path:
        helper_calls.append((output_dir, filename))
        return original_helper(output_dir, filename)

    monkeypatch.setattr(module, "formal_temporary_path", tracked_helper)

    _validate_no_outputs_or_claim(paths)
    formal_path = paths.output_dir / DECLARED_TEST_OUTPUTS[0]
    _write_atomic_once(formal_path, b"synthetic-output")

    assert helper_calls[: len(DECLARED_TEST_OUTPUTS)] == [
        (paths.output_dir, filename) for filename in DECLARED_TEST_OUTPUTS
    ]
    assert helper_calls[-1] == (paths.output_dir, formal_path.name)
    assert formal_path.read_bytes() == b"synthetic-output"


def test_second_execution_is_permanently_blocked_by_claim(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    run_one_shot_execution(paths, provenance)

    with pytest.raises(FileExistsError, match="claim already exists"):
        run_one_shot_execution(paths, provenance)


def test_identical_inputs_produce_identical_formal_outputs(tmp_path: Path) -> None:
    first_paths, first_provenance = build_tree(tmp_path / "first")
    second_paths, second_provenance = build_tree(tmp_path / "second")

    run_one_shot_execution(first_paths, first_provenance)
    run_one_shot_execution(second_paths, second_provenance)

    for filename in DECLARED_TEST_OUTPUTS:
        assert (first_paths.output_dir / filename).read_bytes() == (
            second_paths.output_dir / filename
        ).read_bytes()


def test_comparison_uses_fixed_validation_metrics_source(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
) -> None:
    paths, provenance = execution_tree
    preconditions = validate_execution_preconditions(paths, provenance)
    producer_artifact = json.loads(
        paths.validation_metrics_path.read_text(encoding="utf-8")
    )
    test_metrics = {
        "sample_count": 3799,
        "accuracy": 0.6,
        "macro_f1": 0.5,
        "per_class_f1": {action: 0.5 for action in ACTION_SCHEMA},
        "prediction_class_distribution": {
            action: 3799 if action == "keep" else 0 for action in ACTION_SCHEMA
        },
    }

    comparison = build_validation_to_test_comparison(
        preconditions.validation_metrics, test_metrics
    )

    assert "metrics" in producer_artifact
    assert "predicted_class_distribution" in producer_artifact
    assert "per_class_f1" not in producer_artifact
    assert "prediction_class_distribution" not in producer_artifact
    assert tuple(preconditions.validation_metrics) == (
        "sample_count",
        "accuracy",
        "macro_f1",
        "per_class_f1",
        "prediction_class_distribution",
    )
    assert comparison["validation_sample_count"] == 3594
    assert comparison["test_sample_count"] == 3799
    assert comparison["rule_modified_from_test_results"] is False


def test_cli_has_no_rerun_or_overwrite_escape_hatches() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_ego_motion_one_shot_test.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = (
        "--force",
        "--retry",
        "--resume",
        "--reset",
        "--overwrite",
        "--cleanup",
        "--delete-claim",
    )

    assert all(option not in result.stdout for option in forbidden)


@pytest.mark.parametrize(
    "path",
    tuple(EVALUATOR_SOURCE_PATHS) + ("docs/progress.md",),
)
def test_frozen_sources_and_progress_have_no_worktree_change(path: str) -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", path],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_execution_tests_do_not_reference_real_derived_manifest() -> None:
    source = Path(__file__).read_text(encoding="utf-8")

    assert "/Volumes" + "/T7" not in source


def test_synthetic_execution_writes_only_inside_temporary_tree(
    execution_tree: tuple[ExecutionPaths, ExecutionProvenance],
    tmp_path: Path,
) -> None:
    paths, provenance = execution_tree

    run_one_shot_execution(paths, provenance)

    assert paths.output_dir.is_relative_to(tmp_path)
    assert set(path.name for path in paths.output_dir.iterdir()) == {
        "one_shot_preflight_receipt.json",
        EXECUTION_CLAIM_FILENAME,
        *DECLARED_TEST_OUTPUTS,
    }


def test_project_fixed_split_constants_remain_declared() -> None:
    assert EXPECTED_SPLIT_SAMPLE_COUNTS == {
        "train": 14253,
        "validation": 3594,
        "test": 3799,
    }
    assert EXPECTED_SPLIT_SCENE_COUNTS == {
        "train": 560,
        "validation": 140,
        "test": 150,
    }

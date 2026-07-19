from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_ego_motion_one_shot_test import (
    DECLARED_OUTPUTS,
    ManifestPreflightSummary,
    _verified_sha256,
    build_preflight_receipt,
    parse_args,
    run_preflight,
    validate_freeze_artifact,
    validate_manifest_preflight_rows,
    validate_preflight_config,
)
from src.actions.schema import ACTION_SCHEMA
from src.baselines import ego_motion_test
from src.baselines.ego_motion import EgoMotionFeatures, EgoMotionRuleThresholds
from src.baselines.ego_motion_analysis import DiagnosticMargins
from src.baselines.ego_motion_test import (
    FORBIDDEN_TEST_FIELDS,
    TEST_PREDICTION_FIELDS,
    FrozenRuleTestProtocol,
    FrozenRuleTestSample,
    build_test_diagnostics,
    build_validation_to_test_comparison,
    evaluate_frozen_rule_test_samples,
    evaluate_majority_on_test_samples,
    validate_frozen_rule_contract,
)
from src.baselines.majority import ManifestSample
from src.phase0.protocol import evaluate_classification


THRESHOLD_SHA = "43feb5e2baad95bed63e98557eb63c7c2a8fdfbca07a503825f60d41b08d82c9"
MAPPING_SHA = "a96e04aaf068e75b0aa3ecb8412dc5b35fea2412d7090bbee0a6661132923b12"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> FrozenRuleTestProtocol:
    return FrozenRuleTestProtocol(
        frozen_rule_version="phase0.2-ego-motion-rule-v0.1",
        source_rule_version="phase0.2b-ego-motion-rule-v0.1",
        candidate_id="candidate-0293",
        thresholds=EgoMotionRuleThresholds(0.2, 0.05, 0.5, 0.3),
        thresholds_sha256=THRESHOLD_SHA,
        label_rule_version="phase-1.6-meta-action-v0.2",
        manifest_schema_version="phase0_trainval_dataset_manifest_v1",
        split_mapping_sha256=MAPPING_SHA,
    )


def features(
    *,
    speed: float | None = 4.0,
    acceleration: float | None = 0.0,
    yaw_rate: float | None = 0.0,
    availability: str = "full",
) -> EgoMotionFeatures:
    return EgoMotionFeatures(
        speed_mps=speed,
        longitudinal_acceleration_mps2=acceleration,
        yaw_rate_radps=yaw_rate,
        availability=availability,
        history_interval_sec=None if availability == "unavailable" else 0.5,
        acceleration_interval_sec=0.5 if availability == "full" else None,
    )


def sample(
    token: str = "sample",
    *,
    split: str = "test",
    action: str = "keep",
    motion: EgoMotionFeatures | None = None,
) -> FrozenRuleTestSample:
    source = protocol()
    return FrozenRuleTestSample(
        sample_token=token,
        scene_token=f"scene-{token}",
        split=split,
        features=motion or features(),
        ground_truth_action=action,
        label_rule_version=source.label_rule_version,
        manifest_schema_version=source.manifest_schema_version,
        split_mapping_sha256=source.split_mapping_sha256,
    )


def manifest_row(
    token: str,
    *,
    scene: str = "scene",
    split: str = "test",
) -> dict[str, object]:
    return {
        "sample_token": token,
        "scene_token": scene,
        "split": split,
        "manifest_schema_version": "phase0_trainval_dataset_manifest_v1",
        "label_rule_version": "phase-1.6-meta-action-v0.2",
        "split_seed": 20260710,
        "split_strategy_version": "official_train_scene_label_stratified_v1",
        "split_mapping_sha256": MAPPING_SHA,
        "official_split": "val" if split == "test" else "train",
        "meta_action": "sealed-test-label",
        "current_ego_motion": {"speed_mps": "sealed-test-motion"},
    }


def validate_rows(
    rows: list[dict[str, object]],
    *,
    sample_count: int,
    scene_count: int,
) -> ManifestPreflightSummary:
    return validate_manifest_preflight_rows(
        rows,
        expected_schema_version="phase0_trainval_dataset_manifest_v1",
        expected_label_rule_version="phase-1.6-meta-action-v0.2",
        expected_split_seed=20260710,
        expected_split_strategy_version=(
            "official_train_scene_label_stratified_v1"
        ),
        expected_split_mapping_sha256=MAPPING_SHA,
        expected_test_sample_count=sample_count,
        expected_test_scene_count=scene_count,
    )


def freeze_payload() -> dict[str, object]:
    source = protocol()
    return {
        "freeze_status": "frozen",
        "frozen_rule_version": source.frozen_rule_version,
        "source_rule_version": source.source_rule_version,
        "selected_candidate_id": source.candidate_id,
        "thresholds": source.thresholds.as_dict(),
        "thresholds_sha256": source.thresholds_sha256,
        "next_gate": "phase0.2d_one_shot_test",
        "test_evaluation_performed": False,
        "manifest_sha256": "a" * 64,
        "manifest_schema_version": source.manifest_schema_version,
        "label_rule_version": source.label_rule_version,
        "split_mapping_sha256": source.split_mapping_sha256,
    }


def base_config() -> dict[str, object]:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/phase0_2_one_shot_test_v0_1.yaml").read_text(
            encoding="utf-8"
        )
    )


def build_preflight_tree(tmp_path: Path) -> tuple[Path, Path]:
    derived_root = tmp_path / "derived"
    manifest_path = (
        derived_root / "phase_0_1b/trainval_manifest_v1/manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True)
    rows = [
        manifest_row(
            f"test-{index:04d}", scene=f"scene-{index % 150:03d}"
        )
        for index in range(3799)
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    freeze_dir = derived_root / "phase_0_2/ego_motion_rule_freeze_v0_1"
    freeze_dir.mkdir(parents=True)
    freeze = freeze_payload()
    freeze["manifest_sha256"] = sha256(manifest_path)
    freeze_path = freeze_dir / "rule_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    failure_analysis = freeze_dir / "failure_analysis.json"
    validation_failures = freeze_dir / "validation_failures.jsonl"
    failure_analysis.write_text("{}", encoding="utf-8")
    validation_failures.write_text("", encoding="utf-8")
    config = base_config()
    config["expected_manifest_sha256"] = sha256(manifest_path)
    config["expected_freeze_sha256"] = sha256(freeze_path)
    config["expected_freeze_source_sha256"] = {
        "failure_analysis.json": sha256(failure_analysis),
        "validation_failures.jsonl": sha256(validation_failures),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, derived_root


def test_freeze_artifact_sha_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "rule_freeze.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze artifact SHA-256 mismatch"):
        _verified_sha256(path, "0" * 64, "freeze artifact")


def test_manifest_sha_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        _verified_sha256(path, "0" * 64, "manifest")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("freeze_status", "draft"),
        ("frozen_rule_version", "wrong"),
        ("selected_candidate_id", "candidate-0001"),
        ("thresholds", {"stop_speed_threshold_mps": 0.1}),
        ("thresholds_sha256", "0" * 64),
    ),
)
def test_freeze_contract_mismatch_fails(field: str, value: object) -> None:
    freeze = freeze_payload()
    freeze[field] = value

    with pytest.raises(ValueError, match=field):
        validate_freeze_artifact(
            freeze, protocol(), expected_manifest_sha256="a" * 64
        )


def test_threshold_values_must_match_threshold_sha() -> None:
    source = replace(
        protocol(), thresholds=EgoMotionRuleThresholds(0.1, 0.05, 0.5, 0.3)
    )

    with pytest.raises(ValueError, match="threshold values"):
        validate_frozen_rule_contract(source)


def test_expected_test_count_is_frozen_to_3799() -> None:
    config = base_config()
    config["expected_test_sample_count"] = 3798

    with pytest.raises(ValueError, match="must be 3799"):
        validate_preflight_config(config)


def test_expected_test_scene_count_is_frozen_to_150() -> None:
    config = base_config()
    config["expected_test_scene_count"] = 149

    with pytest.raises(ValueError, match="must be 150"):
        validate_preflight_config(config)


def test_duplicate_test_sample_token_is_rejected() -> None:
    rows = [manifest_row("duplicate"), manifest_row("duplicate")]

    with pytest.raises(ValueError, match="duplicate sample_token"):
        validate_rows(rows, sample_count=2, scene_count=1)


def test_scene_split_overlap_is_rejected() -> None:
    rows = [
        manifest_row("train", scene="shared", split="train"),
        manifest_row("test", scene="shared", split="test"),
    ]

    with pytest.raises(ValueError, match="scene_token spans splits"):
        validate_rows(rows, sample_count=1, scene_count=1)


class GuardedTestRow(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key in {"meta_action", "current_ego_motion"}:
            raise AssertionError(f"preflight accessed sealed field: {key}")
        return super().get(key, default)


def test_preflight_does_not_access_test_meta_action() -> None:
    row = GuardedTestRow(manifest_row("test"))

    assert validate_rows([row], sample_count=1, scene_count=1).test_sample_count == 1


def test_preflight_does_not_access_test_current_ego_motion() -> None:
    row = GuardedTestRow(manifest_row("test"))

    assert validate_rows([row], sample_count=1, scene_count=1).test_scene_count == 1


def test_preflight_does_not_call_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, derived_root = build_preflight_tree(tmp_path)
    monkeypatch.setattr(
        ego_motion_test,
        "predict_ego_motion_action",
        lambda *_: pytest.fail("preflight called predictor"),
    )

    run_preflight(config_path, derived_root)


def test_preflight_does_not_call_evaluate_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, derived_root = build_preflight_tree(tmp_path)
    monkeypatch.setattr(
        ego_motion_test,
        "evaluate_classification",
        lambda *_: pytest.fail("preflight called evaluate_classification"),
    )

    run_preflight(config_path, derived_root)


def test_preflight_does_not_generate_predictions_or_metrics(tmp_path: Path) -> None:
    config_path, derived_root = build_preflight_tree(tmp_path)
    run_preflight(config_path, derived_root)
    output_dir = derived_root / "phase_0_2/ego_motion_one_shot_test_v0_1"

    assert {path.name for path in output_dir.iterdir()} == {
        "one_shot_preflight_receipt.json"
    }


def test_unknown_execute_argument_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--config", "config.yaml", "--execute"])


def test_existing_formal_test_output_blocks_preflight(tmp_path: Path) -> None:
    config_path, derived_root = build_preflight_tree(tmp_path)
    output_dir = derived_root / "phase_0_2/ego_motion_one_shot_test_v0_1"
    output_dir.mkdir(parents=True)
    (output_dir / "test_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="formal one-shot test output"):
        run_preflight(config_path, derived_root)


@pytest.mark.parametrize("split", ("train", "validation"))
def test_synthetic_evaluation_rejects_non_test_split(split: str) -> None:
    with pytest.raises(ValueError, match="only accepts test"):
        evaluate_frozen_rule_test_samples((sample(split=split),), protocol())


def test_synthetic_evaluation_rejects_duplicate_token() -> None:
    with pytest.raises(ValueError, match="duplicate sample_token"):
        evaluate_frozen_rule_test_samples(
            (sample("duplicate"), sample("duplicate")), protocol()
        )


def test_synthetic_evaluation_rejects_illegal_action() -> None:
    with pytest.raises(ValueError, match="illegal ground-truth action"):
        evaluate_frozen_rule_test_samples(
            (sample(action="illegal"),), protocol()
        )


def test_synthetic_evaluation_rejects_forbidden_field() -> None:
    payload = asdict(sample())
    payload["features"] = features()
    payload["future_ego_trajectory"] = []

    with pytest.raises(ValueError, match="forbidden fields"):
        evaluate_frozen_rule_test_samples((payload,), protocol())


@pytest.mark.parametrize(
    ("token", "motion", "expected"),
    (
        ("stop", features(speed=0.2, acceleration=9.0, yaw_rate=9.0), "stop"),
        ("left", features(acceleration=9.0, yaw_rate=0.05), "left_lateral"),
        ("right", features(acceleration=9.0, yaw_rate=-0.05), "right_lateral"),
        ("accelerate", features(acceleration=0.5), "accelerate"),
        ("decelerate", features(acceleration=-0.3), "decelerate"),
        ("keep", features(), "keep"),
    ),
)
def test_synthetic_predictions_preserve_frozen_priority(
    token: str, motion: EgoMotionFeatures, expected: str
) -> None:
    records, _ = evaluate_frozen_rule_test_samples(
        (sample(token, action=expected, motion=motion),), protocol()
    )

    assert records[0].predicted_action == expected


def test_synthetic_unavailable_motion_falls_back_to_keep() -> None:
    motion = features(
        speed=None,
        acceleration=None,
        yaw_rate=None,
        availability="unavailable",
    )
    records, _ = evaluate_frozen_rule_test_samples(
        (sample(action="keep", motion=motion),), protocol()
    )

    assert records[0].decision_reason == "unavailable_motion_fallback_keep"


@pytest.mark.parametrize(
    ("speed", "yaw_rate", "expected"),
    ((4.0, 0.0, "keep"), (0.1, 0.0, "stop"), (4.0, 0.1, "left_lateral")),
)
def test_synthetic_partial_never_predicts_longitudinal(
    speed: float, yaw_rate: float, expected: str
) -> None:
    motion = features(
        speed=speed,
        acceleration=None,
        yaw_rate=yaw_rate,
        availability="partial",
    )
    records, _ = evaluate_frozen_rule_test_samples(
        (sample(action=expected, motion=motion),), protocol()
    )

    assert records[0].predicted_action == expected


def test_synthetic_metrics_reuse_evaluate_classification() -> None:
    samples = (sample("a", action="keep"), sample("b", action="stop"))
    records, metrics = evaluate_frozen_rule_test_samples(samples, protocol())
    expected = evaluate_classification(
        tuple(item.ground_truth_action for item in records),
        tuple(item.predicted_action for item in records),
    )

    assert metrics["accuracy"] == expected.accuracy
    assert metrics["macro_f1"] == expected.macro_f1
    assert metrics["confusion_matrix"] == [
        list(row) for row in expected.confusion_matrix
    ]


def test_synthetic_majority_action_is_fit_from_train_only() -> None:
    train = (
        ManifestSample("1", "train", "keep", "train"),
        ManifestSample("2", "train", "keep", "train"),
        ManifestSample("3", "train", "stop", "train"),
    )
    test_samples = (sample("test", action="keep"),)
    _, frozen_metrics = evaluate_frozen_rule_test_samples(test_samples, protocol())

    result = evaluate_majority_on_test_samples(
        train, test_samples, frozen_metrics, protocol()
    )

    assert result["majority_action"] == "keep"


def test_synthetic_majority_rejects_validation_fit_rows() -> None:
    train = (ManifestSample("1", "scene", "keep", "validation"),)
    test_samples = (sample(),)
    _, frozen_metrics = evaluate_frozen_rule_test_samples(test_samples, protocol())

    with pytest.raises(ValueError, match="train samples only"):
        evaluate_majority_on_test_samples(
            train, test_samples, frozen_metrics, protocol()
        )


def test_validation_to_test_comparison_schema_is_fixed() -> None:
    distribution = {action: 1 for action in ACTION_SCHEMA}
    f1 = {action: 0.5 for action in ACTION_SCHEMA}
    validation = {
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "per_class_f1": f1,
        "prediction_class_distribution": distribution,
    }
    test = {
        "accuracy": 0.6,
        "macro_f1": 0.55,
        "per_class_f1": {action: 0.6 for action in ACTION_SCHEMA},
        "prediction_class_distribution": {action: 2 for action in ACTION_SCHEMA},
    }

    comparison = build_validation_to_test_comparison(validation, test)

    assert tuple(comparison) == (
        "comparison_schema_version",
        "test_minus_validation_accuracy",
        "test_minus_validation_macro_f1",
        "test_minus_validation_per_class_f1",
        "prediction_distribution_count_difference",
        "rule_modified_from_test_results",
    )
    assert comparison["rule_modified_from_test_results"] is False


def test_declared_output_list_is_complete_and_fixed() -> None:
    assert DECLARED_OUTPUTS == (
        "test_predictions.jsonl",
        "test_metrics.json",
        "majority_test_metrics.json",
        "validation_to_test_comparison.json",
        "test_diagnostics.json",
        "one_shot_test_receipt.json",
    )


def test_sample_level_record_has_exact_fields_and_no_forbidden_fields() -> None:
    records, _ = evaluate_frozen_rule_test_samples((sample(),), protocol())
    fields = tuple(asdict(records[0]))

    assert fields == TEST_PREDICTION_FIELDS
    assert not FORBIDDEN_TEST_FIELDS.intersection(fields)


def test_evaluator_has_no_threshold_search_or_candidate_selection() -> None:
    source = inspect.getsource(ego_motion_test)

    assert "build_rule_candidates" not in source
    assert "select_best_rule_candidate" not in source
    assert "evaluate_rule_candidate" not in source


def test_preflight_receipt_records_sealed_information_boundary() -> None:
    receipt = build_preflight_receipt(
        protocol=protocol(),
        manifest_sha256="a" * 64,
        freeze_sha256="b" * 64,
        freeze_source_sha256={"failure_analysis.json": "c" * 64},
        manifest_summary=ManifestPreflightSummary(3799, 150),
        evaluator_source_sha256={"source.py": "d" * 64},
    )

    assert receipt["test_label_value_accessed_by_application_logic"] is False
    assert receipt["test_motion_value_accessed_by_application_logic"] is False
    assert receipt["test_predictions_generated"] is False
    assert receipt["test_metrics_generated"] is False
    assert receipt["one_shot_execution_performed"] is False
    assert receipt["ready_for_execution"] is True


def test_repeated_preflight_receipt_is_byte_identical(tmp_path: Path) -> None:
    config_path, derived_root = build_preflight_tree(tmp_path)
    receipt_path, first = run_preflight(config_path, derived_root)
    first_bytes = receipt_path.read_bytes()
    second_path, second = run_preflight(config_path, derived_root)

    assert second_path == receipt_path
    assert first == second
    assert second_path.read_bytes() == first_bytes


def test_test_diagnostics_schema_is_predeclared() -> None:
    records, _ = evaluate_frozen_rule_test_samples((sample(),), protocol())
    diagnostics = build_test_diagnostics(
        records, protocol(), DiagnosticMargins(0.05, 0.01, 0.05)
    )

    assert tuple(diagnostics) == (
        "diagnostics_schema_version",
        "availability",
        "decision_reason",
        "confusion_pairs",
        "threshold_boundary",
        "trigger_overlap",
        "post_hoc_subgroups_added",
    )
    assert diagnostics["post_hoc_subgroups_added"] is False


def test_progress_document_has_no_worktree_change() -> None:
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "docs/progress.md"],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert result.returncode == 0

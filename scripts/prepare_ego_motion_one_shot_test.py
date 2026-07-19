from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePath
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_ego_motion_inputs import load_config, sha256_file
from src.baselines.ego_motion import EgoMotionRuleThresholds
from src.baselines.ego_motion_test import (
    DECLARED_TEST_OUTPUTS,
    EVALUATOR_SOURCE_PATHS,
    EXPECTED_TEST_SAMPLE_COUNT,
    EXPECTED_TEST_SCENE_COUNT,
    FORMAL_RESULT_SCHEMA,
    FrozenRuleTestProtocol,
    validate_frozen_rule_contract,
)
from src.phase0.manifest import write_canonical_json
from src.phase0.protocol import iter_manifest_rows, validate_sha256


DECLARED_OUTPUTS = DECLARED_TEST_OUTPUTS
PREFLIGHT_OUTPUTS = ("one_shot_preflight_receipt.json",)
FREEZE_SOURCE_FILENAMES = (
    "failure_analysis.json",
    "validation_failures.jsonl",
)


@dataclass(frozen=True)
class ManifestPreflightSummary:
    test_sample_count: int
    test_scene_count: int


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _required_mapping(
    mapping: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration missing {key}")
    return value


def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"configuration {key} must be a number")
    return float(value)


def _relative_path(mapping: Mapping[str, object], key: str) -> Path:
    value = _required_string(mapping, key)
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be relative to VLA_DERIVED_ROOT")
    return Path(value)


def _string_sequence(
    mapping: Mapping[str, object], key: str
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"configuration {key} must be a list of strings")
    return tuple(value)


def _thresholds(config: Mapping[str, object]) -> EgoMotionRuleThresholds:
    values = _required_mapping(config, "thresholds")
    return EgoMotionRuleThresholds(
        stop_speed_threshold_mps=_required_number(
            values, "stop_speed_threshold_mps"
        ),
        lateral_yaw_rate_threshold_radps=_required_number(
            values, "lateral_yaw_rate_threshold_radps"
        ),
        accelerate_threshold_mps2=_required_number(
            values, "accelerate_threshold_mps2"
        ),
        decelerate_threshold_mps2=_required_number(
            values, "decelerate_threshold_mps2"
        ),
    )


def _protocol(config: Mapping[str, object]) -> FrozenRuleTestProtocol:
    return FrozenRuleTestProtocol(
        frozen_rule_version=_required_string(config, "frozen_rule_version"),
        source_rule_version=_required_string(config, "source_rule_version"),
        candidate_id=_required_string(config, "selected_candidate_id"),
        thresholds=_thresholds(config),
        thresholds_sha256=validate_sha256(
            config.get("expected_thresholds_sha256"),
            "expected_thresholds_sha256",
        ),
        label_rule_version=_required_string(
            config, "expected_label_rule_version"
        ),
        manifest_schema_version=_required_string(
            config, "expected_manifest_schema_version"
        ),
        split_mapping_sha256=validate_sha256(
            config.get("expected_split_mapping_sha256"),
            "expected_split_mapping_sha256",
        ),
    )


def validate_preflight_config(
    config: Mapping[str, object],
) -> FrozenRuleTestProtocol:
    if _required_string(config, "one_shot_schema_version") != (
        "phase0.2_one_shot_test_v0.1"
    ):
        raise ValueError("unsupported one_shot_schema_version")
    if _required_string(config, "protocol_status") != "preflight_only":
        raise ValueError("protocol_status must remain preflight_only")
    if config.get("expected_test_sample_count") != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError("expected_test_sample_count must be 3799")
    if config.get("expected_test_scene_count") != EXPECTED_TEST_SCENE_COUNT:
        raise ValueError("expected_test_scene_count must be 150")
    if _string_sequence(config, "declared_outputs") != DECLARED_OUTPUTS:
        raise ValueError("declared_outputs do not match the frozen protocol")
    if _string_sequence(config, "preflight_output") != PREFLIGHT_OUTPUTS:
        raise ValueError("preflight_output must contain only the receipt")
    protocol = _protocol(config)
    validate_frozen_rule_contract(protocol)
    return protocol


def _manifest_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest row missing {key}")
    return value


def validate_manifest_preflight_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_schema_version: str,
    expected_label_rule_version: str,
    expected_split_seed: int,
    expected_split_strategy_version: str,
    expected_split_mapping_sha256: str,
    expected_test_sample_count: int,
    expected_test_scene_count: int,
) -> ManifestPreflightSummary:
    sample_tokens: set[str] = set()
    scene_splits: dict[str, str] = {}
    test_scene_tokens: set[str] = set()
    test_sample_count = 0
    row_count = 0
    for row in rows:
        row_count += 1
        split = _manifest_string(row, "split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split!r}")
        sample_token = _manifest_string(row, "sample_token")
        if sample_token in sample_tokens:
            raise ValueError(f"duplicate sample_token: {sample_token}")
        sample_tokens.add(sample_token)
        scene_token = _manifest_string(row, "scene_token")
        existing_split = scene_splits.setdefault(scene_token, split)
        if existing_split != split:
            raise ValueError(
                f"scene_token spans splits: {scene_token} "
                f"({existing_split}, {split})"
            )
        if _manifest_string(row, "manifest_schema_version") != (
            expected_schema_version
        ):
            raise ValueError("manifest_schema_version does not match config")
        if _manifest_string(row, "label_rule_version") != (
            expected_label_rule_version
        ):
            raise ValueError("label_rule_version does not match config")
        split_seed = row.get("split_seed")
        if split_seed != expected_split_seed or isinstance(split_seed, bool):
            raise ValueError("split_seed does not match config")
        if _manifest_string(row, "split_strategy_version") != (
            expected_split_strategy_version
        ):
            raise ValueError("split_strategy_version does not match config")
        mapping_sha = validate_sha256(
            row.get("split_mapping_sha256"), "split_mapping_sha256"
        )
        if mapping_sha != expected_split_mapping_sha256:
            raise ValueError("split_mapping_sha256 does not match config")
        official_split = _manifest_string(row, "official_split")
        if split == "test" and official_split != "val":
            raise ValueError("project test rows must come from official val")
        if split != "test" and official_split != "train":
            raise ValueError(
                "project train/validation rows must come from official train"
            )
        if split == "test":
            test_sample_count += 1
            test_scene_tokens.add(scene_token)
    if row_count == 0:
        raise ValueError("manifest must contain at least one row")
    if test_sample_count != expected_test_sample_count:
        raise ValueError(
            f"test sample count must be {expected_test_sample_count}, "
            f"got {test_sample_count}"
        )
    if len(test_scene_tokens) != expected_test_scene_count:
        raise ValueError(
            f"test scene count must be {expected_test_scene_count}, "
            f"got {len(test_scene_tokens)}"
        )
    return ManifestPreflightSummary(
        test_sample_count=test_sample_count,
        test_scene_count=len(test_scene_tokens),
    )


def validate_freeze_artifact(
    freeze: Mapping[str, object],
    protocol: FrozenRuleTestProtocol,
    *,
    expected_manifest_sha256: str,
) -> None:
    expected = {
        "freeze_status": "frozen",
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "selected_candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "next_gate": "phase0.2d_one_shot_test",
        "test_evaluation_performed": False,
        "manifest_sha256": expected_manifest_sha256,
        "manifest_schema_version": protocol.manifest_schema_version,
        "label_rule_version": protocol.label_rule_version,
        "split_mapping_sha256": protocol.split_mapping_sha256,
    }
    for field_name, expected_value in expected.items():
        if freeze.get(field_name) != expected_value:
            raise ValueError(f"freeze artifact {field_name} does not match config")


def _verified_sha256(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    validate_sha256(expected, f"expected {description} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{description} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _load_json_object(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _validate_no_formal_outputs(output_dir: Path) -> None:
    existing = [name for name in DECLARED_OUTPUTS if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"formal one-shot test output already exists: {existing}"
        )


def _source_hashes(config_path: Path) -> dict[str, str]:
    result = {}
    for relative_path in EVALUATOR_SOURCE_PATHS:
        path = config_path if relative_path.startswith("configs/") else (
            PROJECT_ROOT / relative_path
        )
        if not path.is_file():
            raise FileNotFoundError(f"evaluator source not found: {path}")
        result[relative_path] = sha256_file(path)
    return result


def build_preflight_receipt(
    *,
    protocol: FrozenRuleTestProtocol,
    manifest_sha256: str,
    freeze_sha256: str,
    freeze_source_sha256: Mapping[str, str],
    manifest_summary: ManifestPreflightSummary,
    evaluator_source_sha256: Mapping[str, str],
) -> dict[str, object]:
    return {
        "preflight_schema_version": "phase0.2_one_shot_test_preflight_v0.1",
        "protocol_status": "preflight_passed",
        "manifest_sha256": manifest_sha256,
        "freeze_sha256": freeze_sha256,
        "freeze_source_sha256": dict(freeze_source_sha256),
        "frozen_rule_version": protocol.frozen_rule_version,
        "source_rule_version": protocol.source_rule_version,
        "candidate_id": protocol.candidate_id,
        "thresholds": protocol.thresholds.as_dict(),
        "thresholds_sha256": protocol.thresholds_sha256,
        "test_sample_count": manifest_summary.test_sample_count,
        "test_scene_count": manifest_summary.test_scene_count,
        "evaluator_source_sha256": dict(evaluator_source_sha256),
        "declared_outputs": list(DECLARED_OUTPUTS),
        "formal_result_schema": dict(FORMAL_RESULT_SCHEMA),
        "test_manifest_rows_parsed": True,
        "test_label_value_accessed_by_application_logic": False,
        "test_motion_value_accessed_by_application_logic": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "one_shot_execution_performed": False,
        "ready_for_execution": True,
    }


def run_preflight(
    config_path: Path,
    derived_root: Path,
) -> tuple[Path, dict[str, object]]:
    config = load_config(config_path)
    protocol = validate_preflight_config(config)
    manifest_path = derived_root / _relative_path(
        config, "manifest_relative_path"
    )
    freeze_path = derived_root / _relative_path(config, "freeze_relative_path")
    output_dir = derived_root / _relative_path(config, "output_relative_dir")
    _validate_no_formal_outputs(output_dir)

    expected_manifest_sha = validate_sha256(
        config.get("expected_manifest_sha256"), "expected_manifest_sha256"
    )
    manifest_sha = _verified_sha256(
        manifest_path, expected_manifest_sha, "manifest"
    )
    expected_freeze_sha = validate_sha256(
        config.get("expected_freeze_sha256"), "expected_freeze_sha256"
    )
    freeze_sha = _verified_sha256(
        freeze_path, expected_freeze_sha, "freeze artifact"
    )
    freeze = _load_json_object(freeze_path)
    validate_freeze_artifact(
        freeze, protocol, expected_manifest_sha256=manifest_sha
    )

    expected_freeze_sources = _required_mapping(
        config, "expected_freeze_source_sha256"
    )
    if set(expected_freeze_sources) != set(FREEZE_SOURCE_FILENAMES):
        raise ValueError(
            "expected_freeze_source_sha256 must list the two source artifacts"
        )
    freeze_source_hashes = {
        filename: _verified_sha256(
            freeze_path.parent / filename,
            _required_string(expected_freeze_sources, filename),
            filename,
        )
        for filename in FREEZE_SOURCE_FILENAMES
    }
    split_seed = config.get("expected_split_seed")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("expected_split_seed must be an integer")
    manifest_summary = validate_manifest_preflight_rows(
        iter_manifest_rows(manifest_path),
        expected_schema_version=protocol.manifest_schema_version,
        expected_label_rule_version=protocol.label_rule_version,
        expected_split_seed=split_seed,
        expected_split_strategy_version=_required_string(
            config, "expected_split_strategy_version"
        ),
        expected_split_mapping_sha256=protocol.split_mapping_sha256,
        expected_test_sample_count=EXPECTED_TEST_SAMPLE_COUNT,
        expected_test_scene_count=EXPECTED_TEST_SCENE_COUNT,
    )
    receipt = build_preflight_receipt(
        protocol=protocol,
        manifest_sha256=manifest_sha,
        freeze_sha256=freeze_sha,
        freeze_source_sha256=freeze_source_hashes,
        manifest_summary=manifest_summary,
        evaluator_source_sha256=_source_hashes(config_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / PREFLIGHT_OUTPUTS[0]
    write_canonical_json(receipt, receipt_path)
    return receipt_path, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the sealed Phase 0.2d one-shot test protocol."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    derived_root_value = os.environ.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        raise ValueError("VLA_DERIVED_ROOT is not set")
    receipt_path, receipt = run_preflight(
        arguments.config, Path(derived_root_value)
    )
    print(
        json.dumps(
            {
                "receipt_path": receipt_path.as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
                "protocol_status": receipt["protocol_status"],
                "ready_for_execution": receipt["ready_for_execution"],
                "one_shot_execution_performed": receipt[
                    "one_shot_execution_performed"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_ego_motion_inputs import load_config, sha256_file
from scripts.prepare_ego_motion_one_shot_test import (
    _relative_path,
    _required_mapping,
    _required_number,
    _required_string,
    _source_hashes,
    validate_freeze_artifact,
    validate_manifest_preflight_rows,
    validate_preflight_config,
)
from src.baselines.ego_motion import parse_ego_motion_features
from src.baselines.ego_motion_analysis import DiagnosticMargins
from src.baselines.ego_motion_test import (
    DECLARED_TEST_OUTPUTS,
    EXPECTED_TEST_SAMPLE_COUNT,
    EXPECTED_TEST_SCENE_COUNT,
    FORMAL_RESULT_SHA_FILENAMES,
    TEST_PREDICTION_FIELDS,
    FrozenRuleTestProtocol,
    FrozenRuleTestSample,
    build_one_shot_receipt,
    build_test_diagnostics,
    build_validation_to_test_comparison,
    evaluate_frozen_rule_test_samples,
    evaluate_majority_on_test_samples,
    validate_preflight_receipt_for_execution,
)
from src.phase0.protocol import ManifestSample, iter_manifest_rows, validate_sha256


CONFIRMATION: Final = "phase0.2d-execute-once"
FROZEN_CONFIG_PATH: Final = PROJECT_ROOT / (
    "configs/phase0_2_one_shot_test_v0_1.yaml"
)
FROZEN_CONFIG_SHA256: Final = (
    "943f81dea9dc6970e21997686dae302a644861f15c0e0225540017e5f6717749"
)
PREFLIGHT_RECEIPT_SHA256: Final = (
    "2f8644b267048043412d05a602e34070efa374e794a161541d1f0422a9cb683a"
)
VALIDATION_METRICS_RELATIVE_PATH: Final = Path(
    "phase_0_2/ego_motion_rule_v0_1/validation_metrics.json"
)
VALIDATION_METRICS_SHA256: Final = (
    "a56e8c9a49c05d6a4f0c09489398fcf214e6733e7ee642029533a73f66edff51"
)
PREFLIGHT_RECEIPT_FILENAME: Final = "one_shot_preflight_receipt.json"
EXECUTION_CLAIM_FILENAME: Final = "one_shot_execution_claim.json"
EXECUTION_CLAIM_SCHEMA_VERSION: Final = (
    "phase0.2_one_shot_execution_claim_v0.1"
)
EXPECTED_SPLIT_SAMPLE_COUNTS: Final = {
    "train": 14253,
    "validation": 3594,
    "test": EXPECTED_TEST_SAMPLE_COUNT,
}
EXPECTED_SPLIT_SCENE_COUNTS: Final = {
    "train": 560,
    "validation": 140,
    "test": EXPECTED_TEST_SCENE_COUNT,
}


@dataclass(frozen=True)
class ExecutionPaths:
    config_path: Path
    manifest_path: Path
    freeze_path: Path
    validation_metrics_path: Path
    preflight_receipt_path: Path
    output_dir: Path
    claim_path: Path
    execution_source_path: Path


@dataclass(frozen=True)
class ExecutionProvenance:
    protocol: FrozenRuleTestProtocol
    expected_config_sha256: str
    expected_manifest_sha256: str
    expected_freeze_sha256: str
    expected_validation_metrics_sha256: str
    expected_preflight_receipt_sha256: str
    diagnostic_margins: DiagnosticMargins
    expected_split_seed: int
    expected_split_strategy_version: str
    expected_split_sample_counts: Mapping[str, int]
    expected_split_scene_counts: Mapping[str, int]


@dataclass(frozen=True)
class ExecutionManifestMetadata:
    split_sample_counts: Mapping[str, int]
    split_scene_counts: Mapping[str, int]


@dataclass(frozen=True)
class ExecutionPreconditions:
    preflight_receipt_bytes: bytes
    preflight_receipt: Mapping[str, object]
    evaluator_source_sha256: Mapping[str, str]
    execution_source_sha256: str
    manifest_sha256: str
    freeze_sha256: str
    validation_metrics_sha256: str
    validation_metrics: Mapping[str, object]
    train_samples: tuple[ManifestSample, ...]
    manifest_metadata: ExecutionManifestMetadata


@dataclass(frozen=True)
class ExecutionClaim:
    path: Path
    payload: Mapping[str, object]


@dataclass(frozen=True)
class FormalOutputs:
    serialized_files: Mapping[str, bytes]


def _load_json_object_bytes(payload: bytes, description: str) -> Mapping[str, object]:
    try:
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must contain valid JSON") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return loaded


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _prediction_jsonl_bytes(records: Sequence[object]) -> bytes:
    lines = []
    for record in records:
        payload = asdict(record)
        if tuple(payload) != TEST_PREDICTION_FIELDS:
            raise ValueError("test prediction field contract changed")
        lines.append(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


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


def _validate_no_outputs_or_claim(paths: ExecutionPaths) -> None:
    if paths.claim_path.exists():
        raise FileExistsError(
            "one-shot execution claim already exists; retry is permanently forbidden"
        )
    existing_outputs = [
        name for name in DECLARED_TEST_OUTPUTS if (paths.output_dir / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"formal one-shot test output already exists: {existing_outputs}"
        )


def validate_execution_manifest_metadata(
    rows: Iterable[Mapping[str, object]],
    provenance: ExecutionProvenance,
) -> ExecutionManifestMetadata:
    rows = tuple(rows)
    validate_manifest_preflight_rows(
        rows,
        expected_schema_version=provenance.protocol.manifest_schema_version,
        expected_label_rule_version=provenance.protocol.label_rule_version,
        expected_split_seed=provenance.expected_split_seed,
        expected_split_strategy_version=(
            provenance.expected_split_strategy_version
        ),
        expected_split_mapping_sha256=(
            provenance.protocol.split_mapping_sha256
        ),
        expected_test_sample_count=EXPECTED_TEST_SAMPLE_COUNT,
        expected_test_scene_count=EXPECTED_TEST_SCENE_COUNT,
    )
    sample_counts = {split: 0 for split in EXPECTED_SPLIT_SAMPLE_COUNTS}
    scene_tokens = {split: set() for split in EXPECTED_SPLIT_SCENE_COUNTS}
    for row in rows:
        split = _required_string(row, "split")
        sample_counts[split] += 1
        scene_tokens[split].add(_required_string(row, "scene_token"))
    scene_counts = {
        split: len(tokens) for split, tokens in scene_tokens.items()
    }
    if sample_counts != dict(provenance.expected_split_sample_counts):
        raise ValueError("manifest split sample counts do not match execution contract")
    if scene_counts != dict(provenance.expected_split_scene_counts):
        raise ValueError("manifest split scene counts do not match execution contract")
    return ExecutionManifestMetadata(sample_counts, scene_counts)


def load_train_samples_without_test_access(
    rows: Iterable[Mapping[str, object]],
) -> tuple[ManifestSample, ...]:
    samples = []
    for row in rows:
        split = _required_string(row, "split")
        if split != "train":
            continue
        samples.append(
            ManifestSample(
                sample_token=_required_string(row, "sample_token"),
                scene_token=_required_string(row, "scene_token"),
                meta_action=_required_string(row, "meta_action"),
                split=split,
                label_rule_version=_required_string(row, "label_rule_version"),
            )
        )
    if not samples:
        raise ValueError("train Majority fitting requires train samples")
    return tuple(samples)


def validate_execution_preconditions(
    paths: ExecutionPaths,
    provenance: ExecutionProvenance,
    *,
    preflight_receipt: Mapping[str, object] | None = None,
) -> ExecutionPreconditions:
    _verified_sha256(
        paths.config_path,
        provenance.expected_config_sha256,
        "frozen config",
    )
    manifest_sha = _verified_sha256(
        paths.manifest_path,
        provenance.expected_manifest_sha256,
        "manifest",
    )
    freeze_sha = _verified_sha256(
        paths.freeze_path,
        provenance.expected_freeze_sha256,
        "freeze artifact",
    )
    validation_metrics_sha = _verified_sha256(
        paths.validation_metrics_path,
        provenance.expected_validation_metrics_sha256,
        "validation metrics",
    )
    preflight_sha = _verified_sha256(
        paths.preflight_receipt_path,
        provenance.expected_preflight_receipt_sha256,
        "preflight receipt",
    )
    preflight_bytes = paths.preflight_receipt_path.read_bytes()
    parsed_preflight = _load_json_object_bytes(
        preflight_bytes, "preflight receipt"
    )
    supplied_preflight = preflight_receipt or parsed_preflight
    evaluator_hashes = _source_hashes(paths.config_path)
    validate_preflight_receipt_for_execution(
        preflight_bytes,
        supplied_preflight,
        actual_preflight_receipt_sha256=preflight_sha,
        actual_evaluator_source_sha256=evaluator_hashes,
        protocol=provenance.protocol,
        manifest_sha256=manifest_sha,
        freeze_sha256=freeze_sha,
    )
    freeze = _load_json_object_bytes(
        paths.freeze_path.read_bytes(), "freeze artifact"
    )
    validate_freeze_artifact(
        freeze,
        provenance.protocol,
        expected_manifest_sha256=manifest_sha,
    )
    validation_metrics = _load_json_object_bytes(
        paths.validation_metrics_path.read_bytes(), "validation metrics"
    )
    execution_source_sha = sha256_file(paths.execution_source_path)
    validate_sha256(execution_source_sha, "execution_source_sha256")
    _validate_no_outputs_or_claim(paths)
    metadata = validate_execution_manifest_metadata(
        iter_manifest_rows(paths.manifest_path), provenance
    )
    train_samples = load_train_samples_without_test_access(
        iter_manifest_rows(paths.manifest_path)
    )
    return ExecutionPreconditions(
        preflight_receipt_bytes=preflight_bytes,
        preflight_receipt=supplied_preflight,
        evaluator_source_sha256=evaluator_hashes,
        execution_source_sha256=execution_source_sha,
        manifest_sha256=manifest_sha,
        freeze_sha256=freeze_sha,
        validation_metrics_sha256=validation_metrics_sha,
        validation_metrics=validation_metrics,
        train_samples=train_samples,
        manifest_metadata=metadata,
    )


def _claim_payload(
    provenance: ExecutionProvenance,
    preconditions: ExecutionPreconditions,
) -> dict[str, object]:
    return {
        "claim_schema_version": EXECUTION_CLAIM_SCHEMA_VERSION,
        "protocol_status": "execution_claimed",
        "preflight_receipt_sha256": (
            provenance.expected_preflight_receipt_sha256
        ),
        "manifest_sha256": preconditions.manifest_sha256,
        "freeze_sha256": preconditions.freeze_sha256,
        "validation_metrics_sha256": preconditions.validation_metrics_sha256,
        "execution_source_sha256": preconditions.execution_source_sha256,
        "candidate_id": provenance.protocol.candidate_id,
        "thresholds_sha256": provenance.protocol.thresholds_sha256,
        "execution_count": 1,
        "rerun_permitted": False,
        "test_label_access_allowed_after_claim": True,
        "test_motion_access_allowed_after_claim": True,
    }


def create_execution_claim(
    paths: ExecutionPaths,
    provenance: ExecutionProvenance,
    preconditions: ExecutionPreconditions,
) -> ExecutionClaim:
    payload = _claim_payload(provenance, preconditions)
    serialized = _canonical_json_bytes(payload)
    paths.claim_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.claim_path.open("xb") as claim_file:
        claim_file.write(serialized)
        claim_file.flush()
        os.fsync(claim_file.fileno())
    return ExecutionClaim(paths.claim_path, payload)


def _validate_claim(claim: ExecutionClaim) -> None:
    if not claim.path.is_file():
        raise FileNotFoundError("execution claim must exist before test access")
    if claim.path.read_bytes() != _canonical_json_bytes(claim.payload):
        raise ValueError("execution claim bytes do not match the claim payload")


def load_test_samples_after_claim(
    rows: Iterable[Mapping[str, object]],
    protocol: FrozenRuleTestProtocol,
    claim: ExecutionClaim,
) -> tuple[FrozenRuleTestSample, ...]:
    _validate_claim(claim)
    samples = []
    sample_tokens: set[str] = set()
    scene_tokens: set[str] = set()
    for row in rows:
        split = _required_string(row, "split")
        if split != "test":
            continue
        sample_token = _required_string(row, "sample_token")
        if sample_token in sample_tokens:
            raise ValueError(f"duplicate test sample_token: {sample_token}")
        sample_tokens.add(sample_token)
        scene_token = _required_string(row, "scene_token")
        scene_tokens.add(scene_token)
        if _required_string(row, "official_split") != "val":
            raise ValueError("project test rows must come from official val")
        trace = {
            "label_rule_version": protocol.label_rule_version,
            "manifest_schema_version": protocol.manifest_schema_version,
            "split_mapping_sha256": protocol.split_mapping_sha256,
        }
        for field_name, expected in trace.items():
            if _required_string(row, field_name) != expected:
                raise ValueError(f"test sample {field_name} is inconsistent")
        samples.append(
            FrozenRuleTestSample(
                sample_token=sample_token,
                scene_token=scene_token,
                split=split,
                features=parse_ego_motion_features(row),
                ground_truth_action=_required_string(row, "meta_action"),
                label_rule_version=protocol.label_rule_version,
                manifest_schema_version=protocol.manifest_schema_version,
                split_mapping_sha256=protocol.split_mapping_sha256,
            )
        )
    if len(samples) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            f"test sample count must be {EXPECTED_TEST_SAMPLE_COUNT}, "
            f"got {len(samples)}"
        )
    if len(scene_tokens) != EXPECTED_TEST_SCENE_COUNT:
        raise ValueError(
            f"test scene count must be {EXPECTED_TEST_SCENE_COUNT}, "
            f"got {len(scene_tokens)}"
        )
    return tuple(samples)


def build_formal_outputs(
    test_samples: Sequence[FrozenRuleTestSample],
    provenance: ExecutionProvenance,
    preconditions: ExecutionPreconditions,
) -> FormalOutputs:
    records, test_metrics = evaluate_frozen_rule_test_samples(
        test_samples, provenance.protocol
    )
    majority_metrics = evaluate_majority_on_test_samples(
        preconditions.train_samples,
        test_samples,
        test_metrics,
        provenance.protocol,
    )
    comparison = build_validation_to_test_comparison(
        preconditions.validation_metrics, test_metrics
    )
    diagnostics = build_test_diagnostics(
        records, provenance.protocol, provenance.diagnostic_margins
    )
    serialized_results = {
        "test_predictions.jsonl": _prediction_jsonl_bytes(records),
        "test_metrics.json": _canonical_json_bytes(
            {
                "metrics_schema_version": "phase0.2_test_metrics_v0.1",
                **test_metrics,
            }
        ),
        "majority_test_metrics.json": _canonical_json_bytes(
            {
                "metrics_schema_version": (
                    "phase0.2_majority_test_metrics_v0.1"
                ),
                **majority_metrics,
            }
        ),
        "validation_to_test_comparison.json": _canonical_json_bytes(comparison),
        "test_diagnostics.json": _canonical_json_bytes(diagnostics),
    }
    output_hashes = {
        filename: hashlib.sha256(serialized_results[filename]).hexdigest()
        for filename in FORMAL_RESULT_SHA_FILENAMES
    }
    receipt = build_one_shot_receipt(
        provenance.protocol,
        preflight_receipt_bytes=preconditions.preflight_receipt_bytes,
        preflight_receipt=preconditions.preflight_receipt,
        preflight_receipt_sha256=(
            provenance.expected_preflight_receipt_sha256
        ),
        actual_evaluator_source_sha256=(
            preconditions.evaluator_source_sha256
        ),
        execution_source_sha256=preconditions.execution_source_sha256,
        manifest_sha256=preconditions.manifest_sha256,
        freeze_sha256=preconditions.freeze_sha256,
        output_sha256=output_hashes,
        test_sample_count=len(test_samples),
    )
    serialized_results["one_shot_test_receipt.json"] = _canonical_json_bytes(
        receipt
    )
    if tuple(serialized_results) != DECLARED_TEST_OUTPUTS:
        raise ValueError("formal output order differs from the frozen contract")
    return FormalOutputs(serialized_results)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic_once(path: Path, payload: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.phase0.2d.tmp")
    with temporary_path.open("xb") as output_file:
        output_file.write(payload)
        output_file.flush()
        os.fsync(output_file.fileno())
    with path.open("xb") as reservation:
        reservation.flush()
        os.fsync(reservation.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def write_formal_outputs_once(
    paths: ExecutionPaths,
    claim: ExecutionClaim,
    outputs: FormalOutputs,
) -> None:
    _validate_claim(claim)
    if tuple(outputs.serialized_files) != DECLARED_TEST_OUTPUTS:
        raise ValueError("formal outputs must match the declared output contract")
    for filename in DECLARED_TEST_OUTPUTS:
        _write_atomic_once(
            paths.output_dir / filename,
            outputs.serialized_files[filename],
        )


def run_one_shot_execution(
    paths: ExecutionPaths,
    provenance: ExecutionProvenance,
) -> FormalOutputs:
    preconditions = validate_execution_preconditions(paths, provenance)
    claim = create_execution_claim(paths, provenance, preconditions)
    try:
        test_samples = load_test_samples_after_claim(
            iter_manifest_rows(paths.manifest_path),
            provenance.protocol,
            claim,
        )
        outputs = build_formal_outputs(test_samples, provenance, preconditions)
        write_formal_outputs_once(paths, claim, outputs)
    except Exception as error:
        raise RuntimeError(
            "one-shot execution was consumed after claim creation and did not "
            "complete; retry, resume, reset, and claim deletion are forbidden"
        ) from error
    return outputs


def _diagnostic_margins(config: Mapping[str, object]) -> DiagnosticMargins:
    values = _required_mapping(config, "diagnostic_margin")
    return DiagnosticMargins(
        stop_speed_mps=_required_number(values, "stop_speed_mps"),
        lateral_yaw_rate_radps=_required_number(
            values, "lateral_yaw_rate_radps"
        ),
        longitudinal_acceleration_mps2=_required_number(
            values, "longitudinal_acceleration_mps2"
        ),
    )


def build_execution_context(
    config_path: Path,
    derived_root: Path,
) -> tuple[ExecutionPaths, ExecutionProvenance]:
    config = load_config(config_path)
    protocol = validate_preflight_config(config)
    output_dir = derived_root / _relative_path(config, "output_relative_dir")
    split_seed = config.get("expected_split_seed")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("expected_split_seed must be an integer")
    paths = ExecutionPaths(
        config_path=config_path,
        manifest_path=derived_root / _relative_path(
            config, "manifest_relative_path"
        ),
        freeze_path=derived_root / _relative_path(config, "freeze_relative_path"),
        validation_metrics_path=derived_root / VALIDATION_METRICS_RELATIVE_PATH,
        preflight_receipt_path=output_dir / PREFLIGHT_RECEIPT_FILENAME,
        output_dir=output_dir,
        claim_path=output_dir / EXECUTION_CLAIM_FILENAME,
        execution_source_path=Path(__file__).resolve(),
    )
    provenance = ExecutionProvenance(
        protocol=protocol,
        expected_config_sha256=FROZEN_CONFIG_SHA256,
        expected_manifest_sha256=validate_sha256(
            config.get("expected_manifest_sha256"), "expected_manifest_sha256"
        ),
        expected_freeze_sha256=validate_sha256(
            config.get("expected_freeze_sha256"), "expected_freeze_sha256"
        ),
        expected_validation_metrics_sha256=VALIDATION_METRICS_SHA256,
        expected_preflight_receipt_sha256=PREFLIGHT_RECEIPT_SHA256,
        diagnostic_margins=_diagnostic_margins(config),
        expected_split_seed=split_seed,
        expected_split_strategy_version=_required_string(
            config, "expected_split_strategy_version"
        ),
        expected_split_sample_counts=EXPECTED_SPLIT_SAMPLE_COUNTS,
        expected_split_scene_counts=EXPECTED_SPLIT_SCENE_COUNTS,
    )
    return paths, provenance


def validate_cli_config_path(config_path: Path) -> Path:
    if config_path.resolve() != FROZEN_CONFIG_PATH.resolve():
        raise ValueError(
            "--config must be configs/phase0_2_one_shot_test_v0_1.yaml"
        )
    return config_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the sealed Phase 0.2d one-shot test exactly once."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--confirm-one-shot",
        choices=(CONFIRMATION,),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    config_path = validate_cli_config_path(arguments.config)
    derived_root_value = os.environ.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        raise ValueError("VLA_DERIVED_ROOT is not set")
    paths, provenance = build_execution_context(
        config_path, Path(derived_root_value)
    )
    outputs = run_one_shot_execution(paths, provenance)
    print(
        json.dumps(
            {
                "protocol_status": "executed_once",
                "execution_count": 1,
                "rerun_permitted": False,
                "output_dir": paths.output_dir.as_posix(),
                "formal_outputs": list(outputs.serialized_files),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

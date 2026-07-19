from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath
import sys
from typing import Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.ego_motion import (
    audit_manifest_rows,
    build_test_label_access_evidence,
)
from src.phase0.manifest import write_canonical_json
from src.phase0.protocol import (
    PHASE0_SPLIT_SEED,
    iter_manifest_rows,
    validate_manifest,
)
from src.phase0.stratified_split import SPLIT_STRATEGY_VERSION


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration missing {key}")
    return value


def _relative_path(mapping: Mapping[str, object], key: str) -> Path:
    value = _required_string(mapping, key)
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{key} must be relative to VLA_DERIVED_ROOT")
    return Path(value)


def load_config(config_path: Path) -> Mapping[str, object]:
    loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration root must be a mapping")
    return loaded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_manifest_sha256(path: Path, expected_sha256: str) -> str:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def run_audit(config_path: Path, derived_root: Path) -> tuple[Path, dict[str, object]]:
    config = load_config(config_path)
    manifest_relative_path = _relative_path(config, "manifest_relative_path")
    output_relative_dir = _relative_path(config, "output_relative_dir")
    manifest_path = derived_root / manifest_relative_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    expected_sha256 = _required_string(config, "expected_manifest_sha256")
    manifest_sha256 = require_manifest_sha256(manifest_path, expected_sha256)
    validation = validate_manifest(manifest_path)

    expected_schema = _required_string(
        config,
        "expected_manifest_schema_version",
    )
    if validation.manifest_schema_version != expected_schema:
        raise ValueError("manifest schema version does not match audit config")
    expected_label_rule = _required_string(config, "expected_label_rule_version")
    if validation.label_rule_version != expected_label_rule:
        raise ValueError("label rule version does not match audit config")
    expected_split_seed = config.get("expected_split_seed")
    if expected_split_seed != PHASE0_SPLIT_SEED:
        raise ValueError("split seed does not match the frozen manifest protocol")
    expected_split_strategy = _required_string(
        config,
        "expected_split_strategy_version",
    )
    if expected_split_strategy != SPLIT_STRATEGY_VERSION:
        raise ValueError(
            "split strategy does not match the frozen manifest protocol"
        )
    if validation.split_mapping_sha256 is None:
        raise ValueError("trainval manifest must provide split_mapping_sha256")

    audit = audit_manifest_rows(iter_manifest_rows(manifest_path))
    sample_count_by_split = audit.get("sample_count_by_split")
    if not isinstance(sample_count_by_split, Mapping):
        raise ValueError("audit result missing sample_count_by_split")
    if sum(sample_count_by_split.values()) != validation.sample_count:
        raise ValueError("audit sample count does not match manifest validation")
    audit.update(
        {
            "audit_version": _required_string(config, "version"),
            "manifest_path_relative_to_VLA_DERIVED_ROOT": (
                manifest_relative_path.as_posix()
            ),
            "manifest_sha256": manifest_sha256,
            "split_seed": expected_split_seed,
            "split_strategy_version": expected_split_strategy,
        }
    )
    audit.update(build_test_label_access_evidence())
    output_path = derived_root / output_relative_dir / "ego_motion_input_audit.json"
    write_canonical_json(audit, output_path)
    return output_path, audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Phase 0.2 current/past ego-motion inputs."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    derived_root_value = os.environ.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        raise ValueError("VLA_DERIVED_ROOT is not set")
    output_path, audit = run_audit(arguments.config, Path(derived_root_value))
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"audit_output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.phase0.qwen3vl_dataset_adapter import (  # noqa: E402
    build_qwen3vl_dataset_adapter,
    collect_git_provenance,
    load_config,
)


DEFAULT_CONFIG_PATH = Path("configs/phase0_3_dataset_adapter.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the consumed-test-safe Qwen3-VL dataset adapter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def _derived_root() -> Path:
    value = os.environ.get("VLA_DERIVED_ROOT")
    if not value:
        raise ValueError("VLA_DERIVED_ROOT must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("VLA_DERIVED_ROOT must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError("VLA_DERIVED_ROOT must be an existing directory")
    return resolved


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_relative_path = config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("adapter config must be inside the repository") from error
    result = build_qwen3vl_dataset_adapter(
        config=load_config(config_path),
        config_relative_path=config_relative_path,
        repository_root=REPOSITORY_ROOT,
        derived_root=_derived_root(),
        git_provenance=collect_git_provenance(REPOSITORY_ROOT),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

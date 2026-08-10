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

from src.phase0.qwen3vl_dataset_adapter import collect_git_provenance  # noqa: E402
from src.phase0.qwen3vl_zero_shot import (  # noqa: E402
    load_config,
    run_zero_shot,
)


DEFAULT_CONFIG_PATH = Path("configs/phase0_3_zero_shot.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 0.3c Qwen3-VL zero-shot validation inference."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-variant", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-samples", type=int)
    return parser


def _required_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    return resolved


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.split != "validation":
        print(
            "Phase 0.3c zero-shot runner only permits validation",
            file=sys.stderr,
        )
        return 2
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_relative_path = config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("zero-shot config must be inside repository") from error
    try:
        result = run_zero_shot(
            config=load_config(config_path),
            config_relative_path=config_relative_path,
            repository_root=REPOSITORY_ROOT,
            nuscenes_root=_required_root("NUSCENES_ROOT"),
            derived_root=_required_root("VLA_DERIVED_ROOT"),
            input_variant=arguments.input_variant,
            split=arguments.split,
            max_samples=arguments.max_samples,
            git_provenance=collect_git_provenance(REPOSITORY_ROOT),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(result["output_dir"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

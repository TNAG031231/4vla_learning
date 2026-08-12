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
from src.phase0.qwen3vl_lora_smoke import (  # noqa: E402
    EVALUATION_SPLIT,
    OPTIMIZATION_SPLIT,
    load_config,
    run_lora_smoke,
)


DEFAULT_CONFIG_PATH = Path("configs/phase0_3_lora_smoke.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 0.3d Qwen3-VL LoRA tiny-overfit smoke."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--optimization-split", default=OPTIMIZATION_SPLIT
    )
    parser.add_argument("--evaluation-split", default=EVALUATION_SPLIT)
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
    if arguments.optimization_split != OPTIMIZATION_SPLIT:
        print("LoRA optimization only permits train", file=sys.stderr)
        return 2
    if arguments.evaluation_split != EVALUATION_SPLIT:
        print("LoRA smoke evaluation only permits validation", file=sys.stderr)
        return 2
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_relative_path = config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("LoRA smoke config must be inside repository") from error
    try:
        result = run_lora_smoke(
            config=load_config(config_path),
            config_relative_path=config_relative_path,
            repository_root=REPOSITORY_ROOT,
            nuscenes_root=_required_root("NUSCENES_ROOT"),
            derived_root=_required_root("VLA_DERIVED_ROOT"),
            optimization_split=arguments.optimization_split,
            evaluation_split=arguments.evaluation_split,
            git_provenance=collect_git_provenance(REPOSITORY_ROOT),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": result["checkpoint"]["adapter_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

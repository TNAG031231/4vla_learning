#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from nuscenes.nuscenes import NuScenes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / "data"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(DATA_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DATA_DIRECTORY))

from data.build_trainval_manifest import (  # noqa: E402
    build_records,
    load_config as load_trainval_config,
)
from derive_meta_action import load_meta_action_rules  # noqa: E402
from inspect_nuscenes_sample import load_trajectory_config  # noqa: E402
from src.actions.schema import LABEL_RULE_VERSION  # noqa: E402
from src.phase0.development_projection import (  # noqa: E402
    ProducerInputs,
    build_development_projection,
    collect_projection_git_provenance,
    load_config,
)


DEFAULT_CONFIG_PATH = Path("configs/phase0_3_development_projection.yaml")
TRAINVAL_CONFIG_PATH = Path("configs/trainval_manifest.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the consumed-test-safe Phase 0.3 development projection."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser


def _environment_root(name: str) -> Path:
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
    git_provenance = collect_projection_git_provenance(REPOSITORY_ROOT)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path
    config_path = config_path.resolve()
    try:
        config_relative_path = config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("projection config must be inside the repository") from error

    projection_config = load_config(config_path)
    trainval_config = load_trainval_config(
        REPOSITORY_ROOT / TRAINVAL_CONFIG_PATH
    )
    trajectory_config = load_trajectory_config(
        REPOSITORY_ROOT / trainval_config.data_config_path
    )
    rules = load_meta_action_rules(
        REPOSITORY_ROOT / trainval_config.action_config_path
    )
    if trainval_config.version != projection_config.nuscenes_version:
        raise ValueError("trainval and projection nuScenes versions differ")
    if trainval_config.split_seed != projection_config.expected_split_seed:
        raise ValueError("trainval and projection split seeds differ")
    if (
        trainval_config.split_strategy_version
        != projection_config.expected_split_strategy_version
    ):
        raise ValueError("trainval and projection split strategies differ")
    if rules.label_rule_version != LABEL_RULE_VERSION:
        raise ValueError("action rules do not use the frozen label rule version")
    if (
        rules.horizon_sec != trajectory_config.horizon_sec
        or rules.sample_interval_sec != trajectory_config.sample_interval_sec
    ):
        raise ValueError("trajectory and action rule timing must match")

    nuscenes_root = _environment_root("NUSCENES_ROOT")
    derived_root = _environment_root("VLA_DERIVED_ROOT")
    nuscenes = NuScenes(
        version=projection_config.nuscenes_version,
        dataroot=str(nuscenes_root),
        verbose=False,
    )
    result = build_development_projection(
        config=projection_config,
        config_relative_path=config_relative_path,
        repository_root=REPOSITORY_ROOT,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        nuscenes=nuscenes,
        producer=build_records,
        producer_inputs=ProducerInputs(
            rules=rules,
            horizon_sec=trajectory_config.horizon_sec,
            sample_interval_sec=trajectory_config.sample_interval_sec,
            time_tolerance_sec=(
                trajectory_config.trajectory_time_tolerance_sec
            ),
            agent_radius_m=trajectory_config.nearby_radius_m,
        ),
        git_provenance=git_provenance,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

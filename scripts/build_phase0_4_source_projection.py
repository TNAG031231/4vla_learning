#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "data"))

from nuscenes.nuscenes import NuScenes  # noqa: E402
from data.build_trainval_manifest import (  # noqa: E402
    build_records,
    load_config as load_trainval_config,
)
from derive_meta_action import load_meta_action_rules  # noqa: E402
from inspect_nuscenes_sample import load_trajectory_config  # noqa: E402
from scripts.build_phase0_3_development_projection import (  # noqa: E402
    _environment_root,
)
from src.phase0.development_projection import (  # noqa: E402
    ProducerInputs,
    collect_projection_git_provenance,
)
from src.phase0.phase0_4_source_projection import (  # noqa: E402
    build_source_projection,
    load_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the train/validation-only Phase 0.4 raw source projection."
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/phase0_4_source_projection.yaml"),
    )
    args = parser.parse_args(argv)
    git = collect_projection_git_provenance(REPOSITORY_ROOT)
    config = load_config(REPOSITORY_ROOT / args.config, REPOSITORY_ROOT)
    source = config.source_contract
    trainval = load_trainval_config(config.trainval_config)
    trajectory = load_trajectory_config(REPOSITORY_ROOT / trainval.data_config_path)
    rules = load_meta_action_rules(REPOSITORY_ROOT / trainval.action_config_path)
    if (
        trainval.version != source.nuscenes_version
        or trainval.split_seed != source.expected_split_seed
        or trainval.split_strategy_version != source.expected_split_strategy_version
        or rules.label_rule_version != source.expected_label_rule_version
        or rules.horizon_sec != config.horizon_sec
        or rules.sample_interval_sec != config.sample_interval_sec
    ):
        raise ValueError("trainval producer configuration differs from frozen source")
    nuscenes_root = _environment_root("NUSCENES_ROOT")
    receipt = build_source_projection(
        config=config,
        repository_root=REPOSITORY_ROOT,
        derived_root=_environment_root("VLA_DERIVED_ROOT"),
        nuscenes_root=nuscenes_root,
        reader_factory=lambda: NuScenes(
            version=source.nuscenes_version,
            dataroot=str(nuscenes_root),
            verbose=False,
        ),
        producer=build_records,
        producer_inputs=ProducerInputs(
            rules=rules,
            horizon_sec=trajectory.horizon_sec,
            sample_interval_sec=trajectory.sample_interval_sec,
            time_tolerance_sec=trajectory.trajectory_time_tolerance_sec,
            agent_radius_m=trajectory.nearby_radius_m,
        ),
        git_provenance=git,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

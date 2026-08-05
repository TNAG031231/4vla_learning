from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.phase0.qwen3vl_smoke import (  # noqa: E402
    load_smoke_config,
    run_smoke,
    smoke_exit_code,
    smoke_output_path,
    write_smoke_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 0.3a-2 Qwen3-VL single-sample smoke."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned model and processor to already be cached.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_smoke_config(args.config)
    nuscenes_value = os.environ.get("NUSCENES_ROOT")
    derived_value = os.environ.get("VLA_DERIVED_ROOT")
    if not nuscenes_value or not derived_value:
        missing = [
            name
            for name, value in (
                ("NUSCENES_ROOT", nuscenes_value),
                ("VLA_DERIVED_ROOT", derived_value),
            )
            if not value
        ]
        print(
            f"missing required environment variables: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2

    nuscenes_root = Path(nuscenes_value).expanduser().resolve()
    derived_root = Path(derived_value).expanduser().resolve()
    if not nuscenes_root.is_dir() or not derived_root.is_dir():
        print(
            "NUSCENES_ROOT and VLA_DERIVED_ROOT must be directories",
            file=sys.stderr,
        )
        return 2

    output_path = smoke_output_path(config, derived_root, REPOSITORY_ROOT)
    artifact = run_smoke(
        config=config,
        repository_root=REPOSITORY_ROOT,
        nuscenes_root=nuscenes_root,
        derived_root=derived_root,
        local_files_only=(True if args.local_files_only else None),
    )
    try:
        write_smoke_artifact(artifact, output_path)
    except OSError as error:
        print(f"artifact write failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": artifact["status"], "artifact": str(output_path)}))
    return smoke_exit_code(str(artifact["status"]))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase0.qwen_preflight import (  # noqa: E402
    load_preflight_config,
    resolve_output_path,
    run_preflight,
    write_preflight_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Phase 0.3a Qwen3-VL environment metadata.",
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _print_summary(artifact: Mapping[str, object], output_path: Path) -> None:
    git = artifact.get("git")
    cuda = artifact.get("cuda")
    manifest = artifact.get("manifest")
    gpus = artifact.get("gpus")
    git_commit = git.get("git_commit") if isinstance(git, Mapping) else None
    gpu_count = cuda.get("gpu_count") if isinstance(cuda, Mapping) else None
    manifest_match = (
        manifest.get("sha256_matches")
        if isinstance(manifest, Mapping)
        else None
    )
    gpu_names = []
    if isinstance(gpus, list):
        for gpu in gpus:
            if isinstance(gpu, Mapping) and isinstance(gpu.get("gpu_name"), str):
                gpu_names.append(gpu["gpu_name"])
    print(f"artifact_path={output_path}")
    print(f"status={artifact.get('status')}")
    print(f"git_commit={git_commit}")
    print(f"gpu_count={gpu_count}")
    print(f"gpu_names={','.join(gpu_names) if gpu_names else 'none'}")
    print(f"manifest_sha256_matches={manifest_match}")


def main(
    argv: Sequence[str] | None = None,
    *,
    preflight_runner: Callable[..., dict[str, object]] = run_preflight,
    artifact_writer: Callable[[Mapping[str, object], Path], None] = (
        write_preflight_artifact
    ),
) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_preflight_config(arguments.config)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"preflight configuration failed: {error}", file=sys.stderr)
        return 1

    environment = dict(os.environ)
    artifact = preflight_runner(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
    )
    status = artifact.get("status")
    derived_root_value = environment.get("VLA_DERIVED_ROOT")
    if not derived_root_value:
        print(
            "preflight artifact was not written: VLA_DERIVED_ROOT is unset",
            file=sys.stderr,
        )
        return 2 if status == "blocked" else 1
    derived_root = Path(derived_root_value).expanduser()
    if not derived_root.is_dir():
        print(
            "preflight artifact was not written: VLA_DERIVED_ROOT is not an "
            "existing directory",
            file=sys.stderr,
        )
        return 2 if status == "blocked" else 1
    try:
        output_path = resolve_output_path(
            derived_root,
            config.output_relative_path,
            PROJECT_ROOT,
        )
        artifact_writer(artifact, output_path)
    except (OSError, ValueError, TypeError) as error:
        print(f"preflight artifact write failed: {error}", file=sys.stderr)
        return 1

    _print_summary(artifact, output_path)
    if status == "passed":
        return 0
    if status == "blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys


CLEANABLE_DIRECTORY_REASONS = {
    "__pycache__": "Python bytecode cache directory",
    ".pytest_cache": "pytest cache directory",
    ".mypy_cache": "mypy cache directory",
    ".ruff_cache": "Ruff cache directory",
    ".ipynb_checkpoints": "Jupyter checkpoint directory",
    "tmp": "temporary directory",
    "temp": "temporary directory",
}
SCRATCH_DIRECTORY = "scratch"
ROOT_SCRIPT_PATTERNS = (
    "debug_*.py",
    "try_*.py",
    "tmp_*.py",
    "temp_*.py",
    "check_*_tmp.py",
    "test_*_tmp.py",
)
EXCLUDED_ROOT_DIRECTORIES = {
    ".git",
    ".venv",
    "can_bus",
    "conda-meta",
    "data",
    "datasets",
    "nuscenes",
    "venv",
}
PROTECTED_FORMAL_ROOT_DIRECTORIES = {
    "configs",
    "demo",
    "reports",
    "src",
    "tests",
}
PROTECTED_TEMPORARY_DIRECTORY_NAMES = {
    "scratch",
    "temp",
    "tmp",
}


@dataclass(frozen=True)
class CleanupCandidate:
    path: Path
    reason: str
    is_directory: bool


def root_script_reason(path: Path) -> str | None:
    for pattern in ROOT_SCRIPT_PATTERNS:
        if path.match(pattern):
            return f"project-root one-off script matching {pattern}"
    return None


def collect_candidates(
    root: Path,
    include_scratch: bool,
) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []

    def visit(directory: Path, in_protected_formal_root: bool) -> None:
        for path in directory.iterdir():
            if path.is_symlink():
                continue

            if path.is_dir():
                if directory == root and path.name in EXCLUDED_ROOT_DIRECTORIES:
                    continue
                path_is_protected = in_protected_formal_root or (
                    directory == root
                    and path.name in PROTECTED_FORMAL_ROOT_DIRECTORIES
                )
                if path.name == SCRATCH_DIRECTORY:
                    if include_scratch and not path_is_protected:
                        candidates.append(
                            CleanupCandidate(
                                path=path,
                                reason="scratch directory explicitly included",
                                is_directory=True,
                            )
                        )
                    elif path_is_protected:
                        visit(path, in_protected_formal_root=True)
                    continue
                reason = CLEANABLE_DIRECTORY_REASONS.get(path.name)
                if reason is not None:
                    if (
                        path.name in PROTECTED_TEMPORARY_DIRECTORY_NAMES
                        and path_is_protected
                    ):
                        visit(path, in_protected_formal_root=True)
                        continue
                    candidates.append(
                        CleanupCandidate(
                            path=path,
                            reason=reason,
                            is_directory=True,
                        )
                    )
                    continue
                visit(path, path_is_protected)
                continue

            if path.suffix == ".pyc":
                reason = "compiled Python bytecode"
            elif path.suffix == ".log":
                reason = "log file"
            elif directory == root:
                reason = root_script_reason(path)
            else:
                reason = None

            if reason is not None:
                candidates.append(
                    CleanupCandidate(
                        path=path,
                        reason=reason,
                        is_directory=False,
                    )
                )

    visit(root, in_protected_formal_root=False)
    return sorted(
        candidates,
        key=lambda candidate: candidate.path.relative_to(root).as_posix(),
    )


def print_candidates(
    candidates: list[CleanupCandidate],
    root: Path,
    action: str,
    verbose: bool,
) -> None:
    for candidate in candidates:
        relative_path = candidate.path.relative_to(root).as_posix()
        print(f"path: {relative_path}")
        print(f"reason: {candidate.reason}")
        print(f"action: {action}")
        if verbose:
            item_type = "directory" if candidate.is_directory else "file"
            print(f"type: {item_type}")
            print(f"absolute_path: {candidate.path}")
        print()


def delete_candidate(
    candidate: CleanupCandidate,
    include_scratch: bool,
) -> None:
    if candidate.is_directory:
        allowed_directories = set(CLEANABLE_DIRECTORY_REASONS)
        if include_scratch:
            allowed_directories.add(SCRATCH_DIRECTORY)
        if candidate.path.name not in allowed_directories:
            raise ValueError(
                f"Refusing to delete non-allowlisted directory: {candidate.path}"
            )
        shutil.rmtree(candidate.path)
        return

    candidate.path.unlink()


def run(
    root: Path,
    apply: bool,
    include_scratch: bool,
    verbose: bool,
) -> int:
    candidates = collect_candidates(root, include_scratch)
    if not candidates:
        print("Workspace is clean. No temporary files found.")
        return 0

    action = "delete" if apply else "would_delete"
    print_candidates(candidates, root, action, verbose)
    if not apply:
        return 0

    for candidate in candidates:
        delete_candidate(candidate, include_scratch)
    print(f"Deleted {len(candidates)} temporary item(s).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely review or remove temporary workspace files."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete listed candidates. The default is dry-run.",
    )
    parser.add_argument(
        "--include-scratch",
        action="store_true",
        help="Include scratch/ directories in the cleanup.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print candidate type and absolute path.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    try:
        return run(
            root=project_root,
            apply=arguments.apply,
            include_scratch=arguments.include_scratch,
            verbose=arguments.verbose,
        )
    except (OSError, ValueError) as error:
        print(f"Workspace cleanup failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Workspace cleanup failed with an unknown error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

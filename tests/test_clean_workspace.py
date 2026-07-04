import importlib.util
from pathlib import Path
import sys
from typing import Protocol, cast

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANUP_SCRIPT = PROJECT_ROOT / "scripts" / "clean_workspace.py"


class CleanupRunner(Protocol):
    def __call__(
        self,
        root: Path,
        apply: bool,
        include_scratch: bool,
        verbose: bool,
    ) -> int:
        ...


def load_cleanup_runner() -> CleanupRunner:
    assert CLEANUP_SCRIPT.is_file(), "scripts/clean_workspace.py is missing"
    specification = importlib.util.spec_from_file_location(
        "clean_workspace",
        CLEANUP_SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    runner = getattr(module, "run", None)
    assert callable(runner)
    return cast(CleanupRunner, runner)


def create_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_dry_run_lists_candidates_without_deleting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = load_cleanup_runner()
    cache_file = tmp_path / "src" / "__pycache__" / "module.pyc"
    debug_script = tmp_path / "debug_probe.py"
    create_file(cache_file)
    create_file(debug_script)

    exit_code = run(
        root=tmp_path,
        apply=False,
        include_scratch=False,
        verbose=False,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "path: debug_probe.py" in output
    assert "path: src/__pycache__" in output
    assert "action: would_delete" in output
    assert cache_file.exists()
    assert debug_script.exists()


def test_apply_deletes_only_allowed_temporary_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = load_cleanup_runner()
    cache_directory = tmp_path / "src" / "__pycache__"
    log_file = tmp_path / "run.log"
    debug_script = tmp_path / "debug_probe.py"
    formal_source = tmp_path / "src" / "module.py"
    formal_test = tmp_path / "tests" / "test_module.py"
    nested_debug_script = tmp_path / "src" / "debug_keep.py"
    source_in_temp_directory = tmp_path / "src" / "temp" / "formal_module.py"
    report_in_tmp_directory = tmp_path / "reports" / "tmp" / "review.md"
    create_file(cache_directory / "module.pyc")
    create_file(log_file)
    create_file(debug_script)
    create_file(formal_source)
    create_file(formal_test)
    create_file(nested_debug_script)
    create_file(source_in_temp_directory)
    create_file(report_in_tmp_directory)

    exit_code = run(
        root=tmp_path,
        apply=True,
        include_scratch=False,
        verbose=False,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "action: delete" in output
    assert not cache_directory.exists()
    assert not log_file.exists()
    assert not debug_script.exists()
    assert formal_source.exists()
    assert formal_test.exists()
    assert nested_debug_script.exists()
    assert source_in_temp_directory.exists()
    assert report_in_tmp_directory.exists()


def test_scratch_cleanup_requires_explicit_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = load_cleanup_runner()
    scratch_directory = tmp_path / "scratch"
    create_file(scratch_directory / "notes.txt")

    first_exit_code = run(
        root=tmp_path,
        apply=True,
        include_scratch=False,
        verbose=False,
    )
    first_output = capsys.readouterr().out
    second_exit_code = run(
        root=tmp_path,
        apply=True,
        include_scratch=True,
        verbose=False,
    )

    assert first_exit_code == 0
    assert "Workspace is clean. No temporary files found." in first_output
    assert second_exit_code == 0
    assert not scratch_directory.exists()


def test_clean_workspace_reports_when_no_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = load_cleanup_runner()
    create_file(tmp_path / "README.md")

    exit_code = run(
        root=tmp_path,
        apply=False,
        include_scratch=False,
        verbose=False,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.strip() == "Workspace is clean. No temporary files found."

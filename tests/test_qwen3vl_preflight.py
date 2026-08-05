from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_qwen3vl_preflight import main as cli_main
from src.phase0.qwen_preflight import (
    PreflightConfig,
    collect_package_versions,
    load_preflight_config,
    resolve_output_path,
    run_preflight,
    sha256_file,
    write_preflight_artifact,
)


GIB = 1024**3


class FakeGpuProperties:
    def __init__(
        self,
        name: str,
        total_memory: int,
        major: int,
        minor: int,
    ) -> None:
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor


class FakeCuda:
    def __init__(
        self,
        properties: tuple[FakeGpuProperties, ...] = (),
        *,
        available: bool = True,
        bf16_supported: bool = True,
    ) -> None:
        self._properties = properties
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._properties)

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, index: int) -> FakeGpuProperties:
        return self._properties[index]

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class FakeTorch:
    __version__ = "2.7.1"
    version = SimpleNamespace(cuda="12.8")

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def config_payload(manifest_sha256: str = "a" * 64) -> dict[str, object]:
    return {
        "preflight_version": "phase0.3a-qwen-preflight-v0.1",
        "required_conda_environment": "codex4vla_env",
        "required_environment_variables": [
            "NUSCENES_ROOT",
            "VLA_DERIVED_ROOT",
        ],
        "optional_cache_environment_variables": ["HF_HOME"],
        "manifest_relative_path": (
            "phase_0_1b/trainval_manifest_v1/manifest.jsonl"
        ),
        "expected_manifest_sha256": manifest_sha256,
        "output_relative_path": (
            "phase_0_3/qwen3vl_preflight_v0_1/environment_preflight.json"
        ),
        "required_packages": ["torch", "transformers", "PyYAML"],
        "optional_packages": ["qwen-vl-utils"],
        "minimum_free_disk_gib": 1.0,
    }


def write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_config(
    tmp_path: Path,
    payload: dict[str, object] | None = None,
) -> PreflightConfig:
    return load_preflight_config(
        write_config(tmp_path / "preflight.yaml", payload or config_payload())
    )


def installed_versions(*missing: str) -> Callable[[str], str]:
    missing_names = set(missing)

    def lookup(package_name: str) -> str:
        if package_name in missing_names:
            raise metadata.PackageNotFoundError(package_name)
        return "1.0.0"

    return lookup


def clean_git_runner(repository: Path, arguments: tuple[str, ...]) -> str:
    del repository
    outputs = {
        ("rev-parse", "HEAD"): "d" * 40,
        ("branch", "--show-current"): "phase-0.3a1-autodl-preflight",
        ("status", "--porcelain"): "",
    }
    return outputs[arguments]


def disk_usage_with_free(free_bytes: int) -> Callable[[Path], SimpleNamespace]:
    def disk_usage(path: Path):
        del path
        return SimpleNamespace(
            total=100 * GIB,
            used=100 * GIB - free_bytes,
            free=free_bytes,
        )

    return disk_usage


def prepared_run(tmp_path: Path, **overrides: object):
    nuscenes_root = tmp_path / "nuscenes"
    derived_root = tmp_path / "derived"
    nuscenes_root.mkdir()
    manifest_path = (
        derived_root / "phase_0_1b/trainval_manifest_v1/manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"frozen-manifest-bytes\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = load_config(tmp_path, config_payload(digest))
    environment = {
        "CONDA_DEFAULT_ENV": "codex4vla_env",
        "NUSCENES_ROOT": str(nuscenes_root),
        "VLA_DERIVED_ROOT": str(derived_root),
    }
    arguments = {
        "config": config,
        "repository_root": PROJECT_ROOT,
        "environment": environment,
        "version_lookup": installed_versions("qwen-vl-utils"),
        "torch_loader": lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        "git_runner": clean_git_runner,
        "disk_usage": disk_usage_with_free(50 * GIB),
        "now_utc": lambda: "2026-08-05T12:00:00Z",
    }
    arguments.update(overrides)
    return run_preflight(**arguments), config, environment


def failure_codes(artifact: dict[str, object]) -> set[str]:
    failures = artifact["failures"]
    assert isinstance(failures, list)
    return {failure["code"] for failure in failures}


def warning_codes(artifact: dict[str, object]) -> set[str]:
    warnings = artifact["warnings"]
    assert isinstance(warnings, list)
    return {warning["code"] for warning in warnings}


def test_loads_valid_config(tmp_path: Path) -> None:
    config = load_config(tmp_path)

    assert config.preflight_version == "phase0.3a-qwen-preflight-v0.1"
    assert config.minimum_free_disk_gib == 1.0


def test_config_rejects_missing_field(tmp_path: Path) -> None:
    payload = config_payload()
    del payload["required_packages"]

    with pytest.raises(ValueError, match="missing required fields"):
        load_config(tmp_path, payload)


def test_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "preflight.yaml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_preflight_config(config_path)


def test_config_rejects_invalid_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        load_config(tmp_path, config_payload("ABC"))


@pytest.mark.parametrize(
    "manifest_path",
    ["/tmp/manifest.jsonl", "C:\\data\\manifest.jsonl"],
)
def test_config_rejects_absolute_manifest_path(
    tmp_path: Path,
    manifest_path: str,
) -> None:
    payload = config_payload()
    payload["manifest_relative_path"] = manifest_path

    with pytest.raises(ValueError, match="manifest_relative_path"):
        load_config(tmp_path, payload)


def test_config_rejects_parent_traversal(tmp_path: Path) -> None:
    payload = config_payload()
    payload["manifest_relative_path"] = "phase_0_1b/../test.jsonl"

    with pytest.raises(ValueError, match="manifest_relative_path"):
        load_config(tmp_path, payload)


def test_output_path_cannot_resolve_into_repository() -> None:
    with pytest.raises(ValueError, match="must not be written"):
        resolve_output_path(
            PROJECT_ROOT,
            "phase_0_3/preflight.json",
            PROJECT_ROOT,
        )


def test_required_environment_variable_unset_blocks(tmp_path: Path) -> None:
    artifact, config, environment = prepared_run(tmp_path)
    del environment["NUSCENES_ROOT"]

    artifact = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    assert artifact["status"] == "blocked"
    assert "required_environment_variable_unset" in failure_codes(artifact)


def test_required_root_missing_blocks(tmp_path: Path) -> None:
    _, config, environment = prepared_run(tmp_path)
    environment["NUSCENES_ROOT"] = str(tmp_path / "missing")

    artifact = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    assert "required_root_invalid" in failure_codes(artifact)


def test_required_package_installed_and_missing() -> None:
    records, warnings, failures = collect_package_versions(
        required_packages=("torch", "transformers"),
        optional_packages=(),
        version_lookup=installed_versions("transformers"),
    )

    assert records[0]["installed"] is True
    assert records[1]["installed"] is False
    assert warnings == []
    assert failures[0]["code"] == "required_package_missing"


def test_optional_package_missing_is_warning_only() -> None:
    records, warnings, failures = collect_package_versions(
        required_packages=("torch",),
        optional_packages=("qwen-vl-utils",),
        version_lookup=installed_versions("qwen-vl-utils"),
    )

    assert records[1]["required_or_optional"] == "optional"
    assert failures == []
    assert warnings[0]["code"] == "optional_package_missing"


def test_cuda_available_records_gpu(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(tmp_path)

    assert artifact["cuda"]["cuda_available"] is True
    assert artifact["cuda"]["gpu_count"] == 1
    assert artifact["gpus"][0]["gpu_name"] == "Fake GPU"


def test_cuda_unavailable_is_structured_block(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(
        tmp_path,
        torch_loader=lambda: FakeTorch(FakeCuda((), available=False)),
    )

    assert artifact["status"] == "blocked"
    assert artifact["cuda"]["cuda_available"] is False
    assert "cuda_unavailable" in failure_codes(artifact)


def test_multiple_gpu_metadata(tmp_path: Path) -> None:
    properties = (
        FakeGpuProperties("GPU 0", 24 * GIB, 8, 6),
        FakeGpuProperties("GPU 1", 48 * GIB, 9, 0),
    )
    artifact, _, _ = prepared_run(
        tmp_path,
        torch_loader=lambda: FakeTorch(FakeCuda(properties)),
    )

    assert artifact["cuda"]["gpu_count"] == 2
    assert [gpu["compute_capability"] for gpu in artifact["gpus"]] == [
        "8.6",
        "9.0",
    ]


def test_bf16_capability_is_recorded(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(
        tmp_path,
        torch_loader=lambda: FakeTorch(
            FakeCuda(
                (FakeGpuProperties("GPU", 24 * GIB, 8, 0),),
                bf16_supported=False,
            )
        ),
    )

    assert artifact["cuda"]["bf16_supported"] is False
    assert artifact["gpus"][0]["bf16_hardware_capable"] is True


def test_manifest_missing_blocks_without_parsing(tmp_path: Path) -> None:
    artifact, config, environment = prepared_run(tmp_path)
    manifest_path = (
        Path(environment["VLA_DERIVED_ROOT"])
        / config.manifest_relative_path
    )
    manifest_path.unlink()

    artifact = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    assert artifact["manifest"] == {
        "exists": False,
        "is_file": False,
        "file_size_bytes": None,
        "sha256": None,
        "sha256_matches": False,
    }
    assert artifact["manifest_records_parsed"] == 0


def test_manifest_sha_matches(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(tmp_path)

    assert artifact["manifest"]["sha256_matches"] is True
    assert artifact["status"] == "passed"


def test_manifest_sha_mismatch_hard_fails(tmp_path: Path) -> None:
    artifact, config, environment = prepared_run(tmp_path)
    config = replace(config, expected_manifest_sha256="0" * 64)

    artifact = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    assert artifact["status"] == "failed"
    assert "manifest_sha256_mismatch" in failure_codes(artifact)


def test_manifest_sha_uses_streaming_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b"0123456789")
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: pytest.fail("read_bytes must not be used"),
    )

    digest = sha256_file(path, chunk_size=3)

    assert digest == hashlib.sha256(b"0123456789").hexdigest()


def test_low_disk_space_blocks(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(
        tmp_path,
        disk_usage=disk_usage_with_free(512 * 1024**2),
    )

    assert artifact["status"] == "blocked"
    assert "disk_space_below_minimum" in failure_codes(artifact)


def test_secret_like_environment_value_is_not_recorded(tmp_path: Path) -> None:
    _, original_config, environment = prepared_run(tmp_path)
    config = replace(
        original_config,
        optional_cache_environment_variables=("HF_TOKEN",),
    )
    environment["HF_TOKEN"] = "do-not-record-this-secret"

    artifact = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    serialized = json.dumps(artifact, sort_keys=True)
    assert "do-not-record-this-secret" not in serialized
    assert artifact["paths"]["HF_TOKEN"]["value_recorded"] is False


def test_dirty_worktree_blocks(tmp_path: Path) -> None:
    def dirty_git_runner(repository: Path, arguments: tuple[str, ...]) -> str:
        if arguments == ("status", "--porcelain"):
            return " M src/phase0/qwen_preflight.py"
        return clean_git_runner(repository, arguments)

    artifact, _, _ = prepared_run(tmp_path, git_runner=dirty_git_runner)

    assert artifact["git"]["git_worktree_clean"] is False
    assert "git_worktree_dirty" in failure_codes(artifact)


def test_atomic_json_write_replaces_complete_file(tmp_path: Path) -> None:
    output_path = tmp_path / "nested/artifact.json"
    output_path.parent.mkdir()
    output_path.write_text("old", encoding="utf-8")

    write_preflight_artifact({"status": "passed", "value": 1}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "status": "passed",
        "value": 1,
    }
    assert list(output_path.parent.glob(".*.tmp")) == []


def test_atomic_json_write_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "artifact.json"
    output_path.write_text("old", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_preflight_artifact({"status": "passed"}, output_path)

    assert output_path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_status_is_derived_from_real_gates(tmp_path: Path) -> None:
    passed, config, environment = prepared_run(tmp_path)
    blocked = run_preflight(
        config=config,
        repository_root=PROJECT_ROOT,
        environment=environment,
        version_lookup=installed_versions("transformers", "qwen-vl-utils"),
        torch_loader=lambda: FakeTorch(
            FakeCuda((FakeGpuProperties("Fake GPU", 24 * GIB, 8, 9),))
        ),
        git_runner=clean_git_runner,
        disk_usage=disk_usage_with_free(50 * GIB),
        now_utc=lambda: "2026-08-05T12:00:00Z",
    )

    assert passed["status"] == "passed"
    assert blocked["status"] == "blocked"


def test_protection_fields_are_fixed(tmp_path: Path) -> None:
    artifact, _, _ = prepared_run(tmp_path)

    assert artifact["test_evaluation_performed"] is False
    assert artifact["model_download_performed"] is False
    assert artifact["model_load_performed"] is False
    assert artifact["manifest_records_parsed"] == 0


def cli_artifact(status: str) -> dict[str, object]:
    return {
        "status": status,
        "git": {"git_commit": "d" * 40},
        "cuda": {"gpu_count": 1},
        "gpus": [{"gpu_name": "Fake GPU"}],
        "manifest": {"sha256_matches": True},
    }


def test_cli_success_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    derived_root = tmp_path / "derived"
    derived_root.mkdir()
    config_path = write_config(tmp_path / "preflight.yaml", config_payload())
    monkeypatch.setenv("VLA_DERIVED_ROOT", str(derived_root))

    exit_code = cli_main(
        ["--config", str(config_path)],
        preflight_runner=lambda **kwargs: cli_artifact("passed"),
    )

    assert exit_code == 0
    assert (derived_root / config_payload()["output_relative_path"]).is_file()


def test_cli_blocked_is_nonzero_and_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived_root = tmp_path / "derived"
    derived_root.mkdir()
    config_path = write_config(tmp_path / "preflight.yaml", config_payload())
    monkeypatch.setenv("VLA_DERIVED_ROOT", str(derived_root))

    exit_code = cli_main(
        ["--config", str(config_path)],
        preflight_runner=lambda **kwargs: cli_artifact("blocked"),
    )

    assert exit_code != 0
    output_path = derived_root / config_payload()["output_relative_path"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == (
        "blocked"
    )


def test_cli_rejects_unknown_arguments(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "preflight.yaml", config_payload())

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--download"])

    assert error.value.code == 2

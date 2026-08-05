from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import subprocess
import tempfile
from typing import Protocol

import yaml

from src.phase0.protocol import validate_sha256


ARTIFACT_SCHEMA_VERSION = "phase0.3a_environment_preflight_artifact_v0.1"
GIB = 1024**3
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "preflight_version",
        "required_conda_environment",
        "required_environment_variables",
        "optional_cache_environment_variables",
        "manifest_relative_path",
        "expected_manifest_sha256",
        "output_relative_path",
        "required_packages",
        "optional_packages",
        "minimum_free_disk_gib",
    }
)


class GpuProperties(Protocol):
    name: str
    total_memory: int
    major: int
    minor: int


class CudaRuntime(Protocol):
    def is_available(self) -> bool:
        ...

    def device_count(self) -> int:
        ...

    def current_device(self) -> int:
        ...

    def get_device_properties(self, index: int) -> GpuProperties:
        ...

    def is_bf16_supported(self) -> bool:
        ...


class TorchVersion(Protocol):
    cuda: str | None


class TorchRuntime(Protocol):
    __version__: str
    cuda: CudaRuntime
    version: TorchVersion


class DiskUsage(Protocol):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class PreflightConfig:
    preflight_version: str
    required_conda_environment: str
    required_environment_variables: tuple[str, ...]
    optional_cache_environment_variables: tuple[str, ...]
    manifest_relative_path: str
    expected_manifest_sha256: str
    output_relative_path: str
    required_packages: tuple[str, ...]
    optional_packages: tuple[str, ...]
    minimum_free_disk_gib: float
    config_sha256: str


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_tuple(
    mapping: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{key} must be {qualifier}")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} entries must be non-empty strings")
    entries = tuple(value)
    if len(set(entries)) != len(entries):
        raise ValueError(f"{key} entries must be unique")
    return entries


def _relative_posix_path(value: str, field_name: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or "." in posix_path.parts
    ):
        raise ValueError(
            f"{field_name} must be a traversal-free relative POSIX path"
        )
    return posix_path.as_posix()


def load_preflight_config(path: Path) -> PreflightConfig:
    config_bytes = path.read_bytes()
    raw: object = yaml.safe_load(config_bytes.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("preflight config root must be a mapping")
    missing_fields = REQUIRED_CONFIG_FIELDS - set(raw)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"preflight config missing required fields: {missing}")

    required_environment_variables = _string_tuple(
        raw,
        "required_environment_variables",
    )
    required_roots = {"NUSCENES_ROOT", "VLA_DERIVED_ROOT"}
    if not required_roots.issubset(required_environment_variables):
        raise ValueError(
            "required_environment_variables must include NUSCENES_ROOT and "
            "VLA_DERIVED_ROOT"
        )
    optional_cache_variables = _string_tuple(
        raw,
        "optional_cache_environment_variables",
        allow_empty=True,
    )
    if set(required_environment_variables) & set(optional_cache_variables):
        raise ValueError("required and optional environment variables must differ")

    manifest_relative_path = _relative_posix_path(
        _required_string(raw, "manifest_relative_path"),
        "manifest_relative_path",
    )
    output_relative_path = _relative_posix_path(
        _required_string(raw, "output_relative_path"),
        "output_relative_path",
    )
    if PurePosixPath(output_relative_path).parts[0] != "phase_0_3":
        raise ValueError("output_relative_path must be under phase_0_3")
    if manifest_relative_path == output_relative_path:
        raise ValueError("output_relative_path must not overwrite the manifest")

    minimum_free_disk_gib = raw.get("minimum_free_disk_gib")
    if (
        isinstance(minimum_free_disk_gib, bool)
        or not isinstance(minimum_free_disk_gib, (int, float))
        or minimum_free_disk_gib < 0
    ):
        raise ValueError("minimum_free_disk_gib must be a non-negative number")

    return PreflightConfig(
        preflight_version=_required_string(raw, "preflight_version"),
        required_conda_environment=_required_string(
            raw,
            "required_conda_environment",
        ),
        required_environment_variables=required_environment_variables,
        optional_cache_environment_variables=optional_cache_variables,
        manifest_relative_path=manifest_relative_path,
        expected_manifest_sha256=validate_sha256(
            raw.get("expected_manifest_sha256"),
            "expected_manifest_sha256",
        ),
        output_relative_path=output_relative_path,
        required_packages=_string_tuple(raw, "required_packages"),
        optional_packages=_string_tuple(
            raw,
            "optional_packages",
            allow_empty=True,
        ),
        minimum_free_disk_gib=float(minimum_free_disk_gib),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _failure(code: str, message: str, kind: str) -> dict[str, str]:
    if kind not in {"blocked", "failed"}:
        raise ValueError("failure kind must be blocked or failed")
    return {"code": code, "message": message, "kind": kind}


def collect_package_versions(
    *,
    required_packages: Sequence[str],
    optional_packages: Sequence[str],
    version_lookup: Callable[[str], str] = metadata.version,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    records: list[dict[str, object]] = []
    warnings: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for requirement, package_names in (
        ("required", required_packages),
        ("optional", optional_packages),
    ):
        for package_name in package_names:
            try:
                version = version_lookup(package_name)
            except metadata.PackageNotFoundError:
                records.append(
                    {
                        "package_name": package_name,
                        "installed": False,
                        "version": None,
                        "required_or_optional": requirement,
                    }
                )
                if requirement == "required":
                    failures.append(
                        _failure(
                            "required_package_missing",
                            f"required package is not installed: {package_name}",
                            "blocked",
                        )
                    )
                else:
                    warnings.append(
                        _warning(
                            "optional_package_missing",
                            f"optional package is not installed: {package_name}",
                        )
                    )
            else:
                records.append(
                    {
                        "package_name": package_name,
                        "installed": True,
                        "version": version,
                        "required_or_optional": requirement,
                    }
                )
    return records, warnings, failures


def _load_torch() -> TorchRuntime:
    return import_module("torch")


def collect_cuda_metadata(
    torch_loader: Callable[[], TorchRuntime] = _load_torch,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    cuda: dict[str, object] = {
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu_count": 0,
        "current_device": None,
        "bf16_supported": False,
    }
    gpus: list[dict[str, object]] = []
    warnings: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    try:
        torch = torch_loader()
    except (ImportError, OSError) as error:
        failures.append(
            _failure(
                "torch_runtime_unavailable",
                f"PyTorch runtime metadata is unavailable: {error}",
                "blocked",
            )
        )
        return cuda, gpus, warnings, failures

    cuda.update(
        {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if not cuda["cuda_available"]:
        failures.append(
            _failure(
                "cuda_unavailable",
                "CUDA is not available in the current PyTorch runtime",
                "blocked",
            )
        )
        return cuda, gpus, warnings, failures

    try:
        gpu_count = torch.cuda.device_count()
    except RuntimeError as error:
        failures.append(
            _failure(
                "cuda_runtime_query_failed",
                f"CUDA device count could not be queried: {error}",
                "failed",
            )
        )
        return cuda, gpus, warnings, failures
    cuda["gpu_count"] = gpu_count
    if gpu_count <= 0:
        failures.append(
            _failure(
                "gpu_unavailable",
                "CUDA reports no visible GPU",
                "blocked",
            )
        )
        return cuda, gpus, warnings, failures

    try:
        cuda["current_device"] = torch.cuda.current_device()
    except RuntimeError as error:
        failures.append(
            _failure(
                "cuda_runtime_query_failed",
                f"current CUDA device could not be queried: {error}",
                "failed",
            )
        )
        return cuda, gpus, warnings, failures
    try:
        cuda["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError) as error:
        warnings.append(
            _warning(
                "bf16_runtime_check_unavailable",
                f"BF16 runtime capability could not be queried: {error}",
            )
        )

    for index in range(gpu_count):
        try:
            properties = torch.cuda.get_device_properties(index)
        except RuntimeError as error:
            failures.append(
                _failure(
                    "gpu_metadata_unavailable",
                    f"GPU {index} metadata could not be queried: {error}",
                    "failed",
                )
            )
            continue
        gpus.append(
            {
                "device_index": index,
                "gpu_name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_hardware_capable": properties.major >= 8,
            }
        )
    return cuda, gpus, warnings, failures


def _run_git(repository_root: Path, arguments: tuple[str, ...]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def collect_git_provenance(
    repository_root: Path,
    git_runner: Callable[[Path, tuple[str, ...]], str] = _run_git,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    try:
        commit = git_runner(repository_root, ("rev-parse", "HEAD"))
        branch = git_runner(repository_root, ("branch", "--show-current"))
        status = git_runner(repository_root, ("status", "--porcelain"))
        if not commit or not branch:
            raise ValueError("Git commit or branch is empty")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return (
            {
                "git_commit": None,
                "git_branch": None,
                "git_worktree_clean": False,
            },
            [
                _failure(
                    "git_provenance_unavailable",
                    f"Git provenance could not be read: {error}",
                    "failed",
                )
            ],
        )

    worktree_clean = not status
    failures = []
    if not worktree_clean:
        failures.append(
            _failure(
                "git_worktree_dirty",
                "formal preflight requires a clean Git worktree",
                "blocked",
            )
        )
    return (
        {
            "git_commit": commit,
            "git_branch": branch,
            "git_worktree_clean": worktree_clean,
        },
        failures,
    )


def _is_secret_like(variable_name: str) -> bool:
    upper_name = variable_name.upper()
    return any(marker in upper_name for marker in SECRET_MARKERS)


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def resolve_output_path(
    derived_root: Path,
    output_relative_path: str,
    repository_root: Path,
) -> Path:
    resolved_root = derived_root.expanduser().resolve()
    output_path = (resolved_root / output_relative_path).resolve()
    try:
        output_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("output path must remain within VLA_DERIVED_ROOT") from error
    try:
        output_path.relative_to(repository_root.resolve())
    except ValueError:
        return output_path
    raise ValueError("preflight output must not be written into the repository")


def collect_paths(
    *,
    config: PreflightConfig,
    repository_root: Path,
    environment: Mapping[str, str],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, Path],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    paths: dict[str, dict[str, object]] = {
        "repository": {
            "set": True,
            "path": str(repository_root.resolve()),
            "exists": repository_root.exists(),
            "is_dir": repository_root.is_dir(),
            "value_recorded": True,
        }
    }
    usable_paths: dict[str, Path] = {"repository": repository_root.resolve()}
    warnings: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    required_names = set(config.required_environment_variables)
    variable_names = (
        config.required_environment_variables
        + config.optional_cache_environment_variables
    )
    for variable_name in variable_names:
        raw_value = environment.get(variable_name)
        required = variable_name in required_names
        secret_like = _is_secret_like(variable_name)
        record: dict[str, object] = {
            "set": bool(raw_value),
            "value_recorded": not secret_like,
        }
        if not raw_value:
            if not secret_like:
                record.update({"path": None, "exists": False, "is_dir": False})
            if required:
                failures.append(
                    _failure(
                        "required_environment_variable_unset",
                        f"required environment variable is unset: {variable_name}",
                        "blocked",
                    )
                )
            else:
                warnings.append(
                    _warning(
                        "optional_cache_environment_variable_unset",
                        f"optional cache variable is unset: {variable_name}",
                    )
                )
            paths[variable_name] = record
            continue

        resolved = _resolved_path(raw_value)
        exists = resolved.exists()
        is_dir = resolved.is_dir()
        if not secret_like:
            record.update(
                {
                    "path": str(resolved),
                    "exists": exists,
                    "is_dir": is_dir,
                }
            )
        if exists and is_dir and not secret_like:
            usable_paths[variable_name] = resolved
        elif required:
            failures.append(
                _failure(
                    "required_root_invalid",
                    f"required root is not an existing directory: {variable_name}",
                    "blocked",
                )
            )
        else:
            warnings.append(
                _warning(
                    "optional_cache_root_invalid",
                    f"optional cache path is not an existing directory: {variable_name}",
                )
            )
        paths[variable_name] = record

    derived_root = usable_paths.get("VLA_DERIVED_ROOT")
    if derived_root is not None:
        try:
            output_path = resolve_output_path(
                derived_root,
                config.output_relative_path,
                repository_root,
            )
        except ValueError as error:
            failures.append(
                _failure("output_path_invalid", str(error), "failed")
            )
        else:
            paths["preflight_output"] = {
                "set": True,
                "path": str(output_path),
                "exists": output_path.exists(),
                "is_dir": output_path.is_dir(),
                "value_recorded": True,
            }
    return paths, usable_paths, warnings, failures


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest_integrity(
    path: Path | None,
    expected_sha256: str,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    result: dict[str, object] = {
        "exists": False,
        "is_file": False,
        "file_size_bytes": None,
        "sha256": None,
        "sha256_matches": False,
    }
    if path is None:
        return result, []
    result["exists"] = path.exists()
    result["is_file"] = path.is_file()
    if not result["exists"]:
        return result, [
            _failure(
                "manifest_missing",
                "frozen manifest does not exist",
                "blocked",
            )
        ]
    if not result["is_file"]:
        return result, [
            _failure(
                "manifest_not_file",
                "frozen manifest path is not a file",
                "blocked",
            )
        ]
    try:
        result["file_size_bytes"] = path.stat().st_size
        actual_sha256 = sha256_file(path)
    except OSError as error:
        return result, [
            _failure(
                "manifest_read_failed",
                f"frozen manifest metadata could not be read: {error}",
                "failed",
            )
        ]
    result["sha256"] = actual_sha256
    result["sha256_matches"] = actual_sha256 == expected_sha256
    if not result["sha256_matches"]:
        return result, [
            _failure(
                "manifest_sha256_mismatch",
                "frozen manifest SHA-256 does not match the configured digest",
                "failed",
            )
        ]
    return result, []


def collect_disk_usage(
    *,
    paths: Mapping[str, Path],
    minimum_free_disk_gib: float,
    disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    records: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    minimum_free_bytes = int(minimum_free_disk_gib * GIB)
    for logical_name, path in paths.items():
        try:
            usage = disk_usage(path)
        except OSError as error:
            failures.append(
                _failure(
                    "disk_usage_unavailable",
                    f"disk usage could not be read for {logical_name}: {error}",
                    "failed",
                )
            )
            continue
        sufficient = usage.free >= minimum_free_bytes
        records[logical_name] = {
            "path": str(path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": usage.free / GIB,
            "minimum_free_gib": minimum_free_disk_gib,
            "minimum_satisfied": sufficient,
        }
        if not sufficient:
            failures.append(
                _failure(
                    "disk_space_below_minimum",
                    f"free disk space is below the configured minimum for {logical_name}",
                    "blocked",
                )
            )
    return records, failures


def _derive_status(failures: Sequence[Mapping[str, str]]) -> str:
    if any(failure.get("kind") == "failed" for failure in failures):
        return "failed"
    if failures:
        return "blocked"
    return "passed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_preflight(
    *,
    config: PreflightConfig,
    repository_root: Path,
    environment: Mapping[str, str],
    version_lookup: Callable[[str], str] = metadata.version,
    torch_loader: Callable[[], TorchRuntime] = _load_torch,
    git_runner: Callable[[Path, tuple[str, ...]], str] = _run_git,
    disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
    now_utc: Callable[[], str] = _utc_now,
) -> dict[str, object]:
    warnings: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    git, git_failures = collect_git_provenance(repository_root, git_runner)
    failures.extend(git_failures)

    paths, usable_paths, path_warnings, path_failures = collect_paths(
        config=config,
        repository_root=repository_root,
        environment=environment,
    )
    warnings.extend(path_warnings)
    failures.extend(path_failures)

    active_conda_environment = environment.get("CONDA_DEFAULT_ENV")
    if active_conda_environment != config.required_conda_environment:
        failures.append(
            _failure(
                "conda_environment_mismatch",
                "active conda environment does not match the configured environment",
                "blocked",
            )
        )

    packages, package_warnings, package_failures = collect_package_versions(
        required_packages=config.required_packages,
        optional_packages=config.optional_packages,
        version_lookup=version_lookup,
    )
    warnings.extend(package_warnings)
    failures.extend(package_failures)

    cuda, gpus, cuda_warnings, cuda_failures = collect_cuda_metadata(torch_loader)
    warnings.extend(cuda_warnings)
    failures.extend(cuda_failures)

    derived_root = usable_paths.get("VLA_DERIVED_ROOT")
    manifest_path = (
        derived_root / config.manifest_relative_path
        if derived_root is not None
        else None
    )
    manifest, manifest_failures = check_manifest_integrity(
        manifest_path,
        config.expected_manifest_sha256,
    )
    failures.extend(manifest_failures)

    disk_paths = {
        name: path
        for name, path in usable_paths.items()
        if name == "repository"
        or name in config.required_environment_variables
        or name in config.optional_cache_environment_variables
    }
    disk, disk_failures = collect_disk_usage(
        paths=disk_paths,
        minimum_free_disk_gib=config.minimum_free_disk_gib,
        disk_usage=disk_usage,
    )
    failures.extend(disk_failures)

    artifact: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "preflight_version": config.preflight_version,
        "config_sha256": config.config_sha256,
        "generated_at_utc": now_utc(),
        "deterministic_comparison_excludes": ["generated_at_utc"],
        "status": _derive_status(failures),
        "git": git,
        "environment": {
            "python_version": platform.python_version(),
            "required_conda_environment": config.required_conda_environment,
            "active_conda_environment": active_conda_environment,
        },
        "packages": packages,
        "cuda": cuda,
        "gpus": gpus,
        "paths": paths,
        "manifest": manifest,
        "disk": disk,
        "warnings": warnings,
        "failures": failures,
        "test_evaluation_performed": False,
        "model_download_performed": False,
        "model_load_performed": False,
        "manifest_records_parsed": 0,
    }
    return artifact


def write_preflight_artifact(
    artifact: Mapping[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(
                artifact,
                output_file,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

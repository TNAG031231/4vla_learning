#!/usr/bin/env python3

import importlib
import os
import platform
from pathlib import Path


REQUIRED_MODULES = (
    ("nuscenes", "nuscenes-devkit"),
    ("nuscenes.can_bus.can_bus_api", "nuScenes CAN bus API"),
    ("cv2", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("pyquaternion", "pyquaternion"),
    ("shapely", "shapely"),
)


def check_torch() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        print(f"[FAIL] torch import: {error}")
        return False

    print(f"[PASS] torch version: {torch.__version__}")
    print(f"[PASS] CUDA available: {torch.cuda.is_available()}")
    print(f"[PASS] CUDA version: {torch.version.cuda}")
    return True


def check_imports() -> bool:
    imports_succeeded = True
    for module_name, display_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            imports_succeeded = False
            print(f"[FAIL] import {display_name}: {error}")
        else:
            print(f"[PASS] import {display_name}")
    return imports_succeeded


def check_nuscenes_root() -> bool:
    root_value = os.environ.get("NUSCENES_ROOT")
    if not root_value:
        print("[SKIP] NUSCENES_ROOT is not set.")
        print("Please set NUSCENES_ROOT, for example:")
        print("export NUSCENES_ROOT=/path/to/nuscenes")
        return True

    root = Path(root_value).expanduser()
    if not root.is_dir():
        print(f"[FAIL] NUSCENES_ROOT does not exist: {root}")
        return False

    print(f"[PASS] NUSCENES_ROOT exists: {root}")
    version_directory = root / "v1.0-mini"
    if version_directory.is_dir():
        try:
            from nuscenes.nuscenes import NuScenes

            nuscenes = NuScenes(
                version="v1.0-mini",
                dataroot=str(root),
                verbose=True,
            )
        except Exception as error:
            print(f"[FAIL] NuScenes initialization: {error}")
            return False

        print(f"[PASS] sample count: {len(nuscenes.sample)}")
        print(f"[PASS] scene count: {len(nuscenes.scene)}")
    else:
        print(f"[SKIP] v1.0-mini not found under {root}")

    can_bus_directory = root / "can_bus"
    if can_bus_directory.is_dir():
        file_count = sum(path.is_file() for path in can_bus_directory.rglob("*"))
        print(f"[PASS] CAN bus directory: {can_bus_directory}")
        print(f"[PASS] CAN bus file count: {file_count}")
    else:
        print(f"[SKIP] CAN bus directory not found: {can_bus_directory}")

    return True


def main() -> int:
    print(f"Python version: {platform.python_version()}")

    checks_passed = all(
        (
            check_torch(),
            check_imports(),
            check_nuscenes_root(),
        )
    )
    if checks_passed:
        print("Environment check: PASS")
        return 0

    print("Environment check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

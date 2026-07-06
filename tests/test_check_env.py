import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_check_env_without_nuscenes_root_is_non_fatal() -> None:
    environment = os.environ.copy()
    environment.pop("NUSCENES_ROOT", None)

    result = subprocess.run(
        [sys.executable, "scripts/check_env.py"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Environment check: PASS" in result.stdout
    assert "Please set NUSCENES_ROOT, for example:" in result.stdout
    assert "export NUSCENES_ROOT=/path/to/nuscenes" in result.stdout

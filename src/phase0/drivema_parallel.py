from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run_concurrent(commands: dict[str, list[str]], worktrees: dict[str, Path],
                   environment: dict[str, str], directory: Path, sample_resources=None) -> dict:
    processes = {}
    with ExitStack() as files:
        try:
            for worker, command in commands.items():
                log = files.enter_context((directory / f'{worker}.log').open('x'))
                processes[worker] = subprocess.Popen(command, cwd=worktrees[worker], env=environment,
                                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with (directory / 'resources.jsonl').open('x') as telemetry:
                while True:
                    codes = {worker: process.poll() for worker, process in processes.items()}
                    if sample_resources:
                        telemetry.write(json.dumps({'time': time.time(), 'exit_codes': codes,
                                                    'resources': sample_resources()}) + '\n')
                        telemetry.flush()
                    if any(code not in (None, 0) for code in codes.values()):
                        raise RuntimeError(f'worker failed: {codes}; inspect worker logs; outputs preserved')
                    if all(code == 0 for code in codes.values()):
                        return {'exit_codes': codes, 'pids': {w: p.pid for w, p in processes.items()}}
                    time.sleep(.1 if sample_resources is None else 5)
        finally:
            for process in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes.values():
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (directory / 'process_status.json').write_text(json.dumps(
                {w: {'pid': p.pid, 'exit_code': p.returncode} for w, p in processes.items()}, indent=2) + '\n')

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase0.drivema_grpo_diagnostics import analyze_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Read saved Stage 3D ZIP artifacts; CPU-only offline diagnostics.')
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new JSON file outside the repository')
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261116)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        parser.error('output must be a new file outside the repository')
    result = analyze_archive(args.archive, repetitions=args.bootstrap_repetitions, seed=args.seed)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'status': 'completed'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

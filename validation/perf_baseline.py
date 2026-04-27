"""Wall-clock baseline measurement for all demos in mnpbem_demo_comparison.

Iterates each demo directory under ``DEMO_ROOT`` and runs whichever Python
entry-point script is available (``run_python.py`` preferred, ``run.py``
fallback). Records wall-clock seconds via ``time.perf_counter`` (subprocess).

Skipped demos (no Python entry-point script) are recorded with status=SKIP.

Output CSV columns
------------------
demo_name, wall_seconds, status

Status values
-------------
- ``OK``    — script returned exit code 0
- ``FAIL``  — script returned non-zero exit code
- ``SKIP``  — no run_python.py / run.py present
- ``TIMEOUT`` — exceeded ``--timeout`` seconds
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


DEMO_ROOT = Path('/home/yoojk20/scratch/mnpbem_demo_comparison')


def find_entry(demo_dir: Path) -> Path | None:
    for name in ('run_python.py', 'run.py'):
        cand = demo_dir / name
        if cand.is_file():
            return cand
    return None


def run_demo(demo_dir: Path, timeout: int, python_exe: str) -> tuple[float, str]:
    entry = find_entry(demo_dir)
    if entry is None:
        return float('nan'), 'SKIP'

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [python_exe, entry.name],
            cwd=str(demo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        status = 'OK' if proc.returncode == 0 else 'FAIL'
        return elapsed, status
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return elapsed, 'TIMEOUT'
    except Exception:
        elapsed = time.perf_counter() - t0
        return elapsed, 'FAIL'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default='/home/yoojk20/workspace/MNPBEM/validation/perf_baseline_2026-04-27.csv',
    )
    parser.add_argument('--timeout', type=int, default=1800,
                        help='per-demo timeout (sec)')
    parser.add_argument('--python', default=sys.executable,
                        help='Python interpreter to use')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='demo names to skip (already measured)')
    parser.add_argument('--start-from', default=None,
                        help='start from this demo name (sorted order)')
    args = parser.parse_args()

    demo_dirs = sorted(d for d in DEMO_ROOT.iterdir()
                       if d.is_dir() and d.name.startswith('demo'))
    if args.start_from is not None:
        demo_dirs = [d for d in demo_dirs if d.name >= args.start_from]
    if args.skip:
        skip_set = set(args.skip)
        demo_dirs = [d for d in demo_dirs if d.name not in skip_set]
    print(f'found {len(demo_dirs)} demo dirs', flush=True)

    rows = []
    for demo in demo_dirs:
        wall, status = run_demo(demo, args.timeout, args.python)
        rows.append({'demo_name': demo.name, 'wall_seconds': wall, 'status': status})
        if isinstance(wall, float) and (wall != wall):  # NaN
            print(f'  {demo.name:30s} SKIP', flush=True)
        else:
            print(f'  {demo.name:30s} {wall:8.2f}s [{status}]', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['demo_name', 'wall_seconds', 'status'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()

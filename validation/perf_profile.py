"""cProfile harness for top-N slowest demos.

Runs a demo's Python entry-point under ``cProfile``, then prints the top 10
functions by cumulative time. Output is written to ``/tmp/perf_profile_<demo>.txt``.

Usage
-----
python validation/perf_profile.py <demo_name> [<demo_name> ...]

Each <demo_name> must be a subdirectory under
``/home/yoojk20/scratch/mnpbem_demo_comparison`` containing either
``run_python.py`` or ``run.py``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import runpy
import sys
import traceback
from pathlib import Path


DEMO_ROOT = Path('/home/yoojk20/scratch/mnpbem_demo_comparison')


def find_entry(demo_dir: Path) -> Path | None:
    for name in ('run_python.py', 'run.py'):
        cand = demo_dir / name
        if cand.is_file():
            return cand
    return None


def profile_demo(demo_name: str, top: int, out_dir: Path) -> str:
    demo_dir = DEMO_ROOT / demo_name
    if not demo_dir.is_dir():
        return f'NOT_FOUND: {demo_dir}'
    entry = find_entry(demo_dir)
    if entry is None:
        return f'NO_ENTRY: {demo_dir}'

    out_path = out_dir / f'perf_profile_{demo_name}.txt'

    # cd into demo_dir so relative paths inside the script work
    saved_cwd = os.getcwd()
    saved_argv = list(sys.argv)
    saved_path = list(sys.path)
    try:
        os.chdir(demo_dir)
        sys.argv = [str(entry)]
        # Ensure demo dir is on sys.path so its imports work
        sys.path.insert(0, str(demo_dir))

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit:
            pass  # Some demos call sys.exit(0/1)
        except Exception:
            traceback.print_exc()
        finally:
            profiler.disable()

        # Top-N by cumulative time
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats('cumulative')
        stats.print_stats(top)
        report = buf.getvalue()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(f'=== cProfile top {top} (cumulative) — {demo_name} ===\n\n')
            f.write(report)

        return f'wrote {out_path}'
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        sys.path[:] = saved_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('demos', nargs='+',
                        help='demo directory names (e.g. demospecstat2)')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--out-dir', default='/tmp')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for demo in args.demos:
        print(f'==> profiling {demo}', flush=True)
        msg = profile_demo(demo, args.top, out_dir)
        print(f'    {msg}', flush=True)


if __name__ == '__main__':
    main()

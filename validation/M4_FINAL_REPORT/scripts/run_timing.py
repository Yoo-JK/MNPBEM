#!/usr/bin/env python
"""Run all 58 demo run_python.py with timing collection.

Backends:
  - python_cpu: MNPBEM_GPU=0 MNPBEM_NUMBA=1
  - python_gpu: MNPBEM_GPU=1 MNPBEM_NUMBA=1

Output:
  validation/M4_FINAL_REPORT/data/timing_<backend>.csv
"""
import os, sys, glob, time, subprocess, csv

DEMO_ROOT = '/home/yoojk20/scratch/mnpbem_demo_comparison'
OUT_DIR = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT/data'
PYTHON = '/home/yoojk20/miniconda3/envs/mnpbem/bin/python'
TIMEOUT_SEC = 600


def find_demos():
    demos = []
    for d in sorted(glob.glob(os.path.join(DEMO_ROOT, 'demo*'))):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, 'run_python.py')):
            demos.append(d)
    return demos


def run_one(demo, env_extra):
    run_py = os.path.join(demo, 'run_python.py')
    env = {**os.environ, **env_extra}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, run_py],
            env=env,
            capture_output=True,
            timeout=TIMEOUT_SEC,
            text=True,
        )
        wall = time.time() - t0
        status = 'ok' if proc.returncode == 0 else f'error_rc{proc.returncode}'
        tail = (proc.stdout or '').splitlines()[-3:]
        snippet = ' | '.join(tail)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        status = 'timeout'
        snippet = ''
    except Exception as e:
        wall = time.time() - t0
        status = f'exception_{type(e).__name__}'
        snippet = str(e)[:200]
    return wall, status, snippet


def main():
    backends = []
    if len(sys.argv) > 1:
        which = sys.argv[1]
    else:
        which = 'cpu'
    if which in ('cpu', 'all'):
        backends.append(('python_cpu', {'MNPBEM_GPU': '0', 'MNPBEM_NUMBA': '1'}))
    if which in ('gpu', 'all'):
        backends.append(('python_gpu', {'MNPBEM_GPU': '1', 'MNPBEM_NUMBA': '1'}))

    demos = find_demos()
    print(f'Found {len(demos)} demos with run_python.py')

    os.makedirs(OUT_DIR, exist_ok=True)

    for backend, env_extra in backends:
        out_csv = os.path.join(OUT_DIR, f'timing_{backend}.csv')
        print(f'\n=== Backend: {backend} -> {out_csv} ===', flush=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['demo', 'backend', 'wall_sec', 'status', 'tail'])
            for i, d in enumerate(demos):
                name = os.path.basename(d)
                print(f'[{i+1}/{len(demos)}] {name}...', end=' ', flush=True)
                wall, status, snippet = run_one(d, env_extra)
                print(f'{wall:.2f}s [{status}]', flush=True)
                w.writerow([name, backend, f'{wall:.4f}', status, snippet[:300]])
                f.flush()
        print(f'  wrote {out_csv}')


if __name__ == '__main__':
    main()

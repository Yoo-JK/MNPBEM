#!/usr/bin/env python
"""Parallel timing collection using multiprocessing.

Backends:
  - python_cpu: MNPBEM_GPU=0 MNPBEM_NUMBA=1, parallel via process pool
  - python_gpu: MNPBEM_GPU=1, sequential (single GPU)

Usage:
  run_timing_parallel.py cpu [N_PARALLEL]
  run_timing_parallel.py gpu
"""
import os, sys, glob, time, subprocess, csv
from concurrent.futures import ProcessPoolExecutor, as_completed

DEMO_ROOT = '/home/yoojk20/scratch/mnpbem_demo_comparison'
OUT_DIR = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT/data'
PYTHON = '/home/yoojk20/miniconda3/envs/mnpbem/bin/python'
TIMEOUT_SEC = 900


def find_demos():
    demos = []
    for d in sorted(glob.glob(os.path.join(DEMO_ROOT, 'demo*'))):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, 'run_python.py')):
            demos.append(d)
    return demos


def run_one(args):
    demo, env_extra, n_threads = args
    run_py = os.path.join(demo, 'run_python.py')
    name = os.path.basename(demo)
    env = {**os.environ, **env_extra,
           'OMP_NUM_THREADS': str(n_threads),
           'MKL_NUM_THREADS': str(n_threads),
           'OPENBLAS_NUM_THREADS': str(n_threads),
           'NUMBA_NUM_THREADS': str(n_threads)}
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
        out = proc.stdout or ''
        err = proc.stderr or ''
        tail = (out + err).splitlines()[-3:]
        snippet = ' | '.join(s[:100] for s in tail)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        status = 'timeout'
        snippet = ''
    except Exception as e:
        wall = time.time() - t0
        status = f'exc_{type(e).__name__}'
        snippet = str(e)[:150]
    return name, wall, status, snippet


def main():
    if len(sys.argv) < 2:
        print('usage: run_timing_parallel.py {cpu|gpu} [n_parallel]')
        sys.exit(1)

    which = sys.argv[1]
    demos = find_demos()
    print(f'Found {len(demos)} demos')

    os.makedirs(OUT_DIR, exist_ok=True)

    if which == 'cpu':
        n_par = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        threads = max(1, 64 // n_par)
        env_extra = {'MNPBEM_GPU': '0', 'MNPBEM_NUMBA': '1'}
        backend = 'python_cpu'
        out_csv = os.path.join(OUT_DIR, f'timing_{backend}.csv')
        print(f'Parallel: {n_par} workers, {threads} threads each')
        results = {}
        tasks = [(d, env_extra, threads) for d in demos]
        with ProcessPoolExecutor(max_workers=n_par) as ex:
            futs = {ex.submit(run_one, t): t[0] for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                name, wall, status, snippet = fut.result()
                results[name] = (wall, status, snippet)
                print(f'[{done}/{len(demos)}] {name}: {wall:.1f}s [{status}]', flush=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['demo', 'backend', 'wall_sec', 'status', 'tail'])
            for name in sorted(results):
                wall, status, snippet = results[name]
                w.writerow([name, backend, f'{wall:.4f}', status, snippet[:300]])
        print(f'wrote {out_csv}')
    elif which == 'gpu':
        env_extra = {'MNPBEM_GPU': '1', 'MNPBEM_NUMBA': '1'}
        backend = 'python_gpu'
        out_csv = os.path.join(OUT_DIR, f'timing_{backend}.csv')
        print('Sequential GPU run')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['demo', 'backend', 'wall_sec', 'status', 'tail'])
            for i, d in enumerate(demos):
                name, wall, status, snippet = run_one((d, env_extra, 8))
                print(f'[{i+1}/{len(demos)}] {name}: {wall:.1f}s [{status}]', flush=True)
                w.writerow([name, backend, f'{wall:.4f}', status, snippet[:300]])
                f.flush()
        print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()

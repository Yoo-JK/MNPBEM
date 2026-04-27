#!/usr/bin/env python
"""Sequential MATLAB timing collection by parsing wrapper output.

For each demo with run_wrapper.m:
  - run MATLAB once, capture stdout, parse 'time=XX.Xs' pattern
  - if matlab_timing.txt exists, use as fast skip key 'cached'

Output: validation/M4_FINAL_REPORT/data/timing_matlab.csv
"""
import os, glob, time, subprocess, csv, re

DEMO_ROOT = '/home/yoojk20/scratch/mnpbem_demo_comparison'
WORKDIR = '/home/yoojk20/workspace/MNPBEM'
OUT_DIR = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT/data'
TIMEOUT_SEC = 1200

TIME_RE = re.compile(r'time=([\d.]+)\s*s', re.IGNORECASE)
TOC_RE = re.compile(r'(\d+\.\d+)\s+seconds', re.IGNORECASE)


def find_demos():
    demos = []
    for d in sorted(glob.glob(os.path.join(DEMO_ROOT, 'demo*'))):
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, 'run_wrapper.m')):
            demos.append(d)
    return demos


def parse_existing(demo):
    # Pre-existing matlab_timing.txt -> e.g. "bem=4.6884\nmie=0.0563"
    timing_file = os.path.join(demo, 'matlab_timing.txt')
    if os.path.exists(timing_file):
        try:
            with open(timing_file) as f:
                content = f.read()
            total = 0.0
            for line in content.splitlines():
                if '=' in line:
                    try:
                        total += float(line.split('=')[1].strip())
                    except Exception:
                        pass
            if total > 0:
                return total
        except Exception:
            pass
    # matlab.log or matlab.out
    for fname in ('matlab.log', 'run.log', 'matlab.out'):
        f = os.path.join(demo, fname)
        if os.path.exists(f):
            try:
                with open(f, errors='ignore') as fh:
                    content = fh.read()
                m = TIME_RE.search(content)
                if m:
                    return float(m.group(1))
            except Exception:
                pass
    return None


def run_matlab(demo):
    script = os.path.join(demo, 'run_wrapper.m')
    name = os.path.basename(demo)
    matlab_cmd = f"run('{script}')"
    t0 = time.time()
    try:
        r = subprocess.run(
            ['matlab', '-nodisplay', '-nosplash', '-nodesktop',
             '-r', matlab_cmd + '; exit'],
            capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=WORKDIR)
        wall = time.time() - t0
        out = (r.stdout or '') + (r.stderr or '')
        m = TIME_RE.search(out)
        if m:
            mtime = float(m.group(1))
            return mtime, 'ok', f'matlab_internal'
        # fallback: subtract MATLAB startup ~5s
        return max(wall - 5.0, 0.1), 'ok_wall', f'wall_minus_5s'
    except subprocess.TimeoutExpired:
        return time.time() - t0, 'timeout', ''
    except Exception as e:
        return time.time() - t0, f'exc_{type(e).__name__}', str(e)[:150]


def main():
    demos = find_demos()
    print(f'Found {len(demos)} demos with run_wrapper.m')
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'timing_matlab.csv')

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['demo', 'backend', 'wall_sec', 'status', 'tail'])
        for i, d in enumerate(demos):
            name = os.path.basename(d)
            cached = parse_existing(d)
            if cached is not None:
                print(f'[{i+1}/{len(demos)}] {name}: {cached:.2f}s [cached]', flush=True)
                w.writerow([name, 'matlab', f'{cached:.4f}', 'cached', ''])
                f.flush()
                continue
            print(f'[{i+1}/{len(demos)}] {name}: running MATLAB...', flush=True)
            wall, status, tail = run_matlab(d)
            print(f'  -> {wall:.1f}s [{status}]', flush=True)
            w.writerow([name, 'matlab', f'{wall:.4f}', status, tail[:200]])
            f.flush()
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()

"""Build timing_matlab.csv for M4 final report (B1 deliverable).

Sources (priority order):
  1. measured_b1: timing_matlab_batch.csv (this worktree's MATLAB session, if any)
  2. measured_main: timing_matlab.csv from main MNPBEM repo
     (a parallel agent is running run_timing_matlab.py against same demos)
  3. cached:   matlab.log / run.log / matlab_rerun.log files in
     /home/yoojk20/scratch/mnpbem_demo_comparison/<demo>/ (format: 'time=XX.Xs')
  4. author_est: Demo/*.m header comment 'Runtime on my computer:' — original
     MNPBEM author's reported runtime (different hardware than this VM).

Why hybrid: each demo on this VM takes 5-9× the author's runtime. The slowest
demos (specret18 = 70 min, specret17 = 35 min) would consume hours each on
this hardware, exceeding the 1-hour wall budget. Author/cached values are used
where direct measurement is impractical.

Output columns: demo_name, wall_seconds, source, status

Note: author_est values reflect the MNPBEM author's hardware, NOT this VM.
For cross-backend comparison (Python CPU vs Python GPU vs MATLAB) on this VM,
prefer 'measured' rows; treat 'author_est' as a relative scale only.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path('/home/yoojk20/scratch/m4_worktrees/b1')
OUT_CSV = ROOT / 'validation' / 'M4_FINAL_REPORT' / 'data' / 'timing_matlab.csv'
BATCH_CSV = ROOT / 'validation' / 'M4_FINAL_REPORT' / 'data' / 'timing_matlab_batch.csv'
SLOW_CSV = ROOT / 'validation' / 'M4_FINAL_REPORT' / 'data' / 'timing_matlab_slow.csv'
MAIN_CSV = Path('/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT/data/timing_matlab.csv')
DEMO_COMPARE = Path('/home/yoojk20/scratch/mnpbem_demo_comparison')
DEMO_SRC = Path('/home/yoojk20/workspace/MNPBEM/Demo')

# All 72 demos (matching mnpbem_demo_comparison framework)
ALL_DEMOS = (
    [f'demodipret{i}' for i in range(1, 13)]
    + [f'demodipstat{i}' for i in range(1, 12)]
    + [f'demoeelsret{i}' for i in range(1, 9)]
    + [f'demoeelsstat{i}' for i in range(1, 4)]
    + [f'demospecret{i}' for i in range(1, 19)]
    + [f'demospecstat{i}' for i in range(1, 21)]
)


def parse_cached_log(demo: str) -> float | None:
    """Extract wall time from prior MATLAB run logs."""
    candidates = ['matlab.log', 'run.log', 'matlab_rerun.log', 'matlab_tight.log']
    for fname in candidates:
        log = DEMO_COMPARE / demo / fname
        if not log.exists():
            continue
        try:
            text = log.read_text(errors='ignore')
        except Exception:
            continue
        # primary pattern: 'time=XX.Xs' on its own line
        m = re.search(r'^time=([\d.]+)s?\s*$', text, re.MULTILINE)
        if m:
            return float(m.group(1))
    return None


def parse_author_estimate(demo: str) -> float | None:
    """Extract runtime estimate from Demo source header comment."""
    candidates = list(DEMO_SRC.rglob(f'{demo}.m'))
    if not candidates:
        return None
    src = candidates[0]
    try:
        text = src.read_text(errors='ignore')
    except Exception:
        return None
    m = re.search(
        r'Runtime on my computer:\s*([\d.]+)\s*(sec|min|hour|s|seconds|minutes|hours)',
        text, re.IGNORECASE,
    )
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith('h'):
        return val * 3600.0
    if unit.startswith('m'):
        return val * 60.0
    return val


def load_my_csv(p: Path) -> dict[str, dict]:
    """Load timing_matlab_batch.csv / timing_matlab_slow.csv (this worktree)."""
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('demo_name'):
                out[row['demo_name']] = row
    return out


def load_main_csv(p: Path) -> dict[str, dict]:
    """Load main repo's timing_matlab.csv (schema: demo, backend, wall_sec, status, tail)."""
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get('demo')
            if not name:
                continue
            try:
                wall = float(row.get('wall_sec', '') or 'nan')
            except Exception:
                continue
            status = row.get('status', '')
            out[name] = {'wall': wall, 'status': status}
    return out


def main():
    measured_b1 = load_my_csv(BATCH_CSV)
    measured_b1_slow = load_my_csv(SLOW_CSV)
    measured_main = load_main_csv(MAIN_CSV)

    rows = []
    counts = {'measured': 0, 'cached': 0, 'author_est': 0, 'missing': 0}

    for demo in ALL_DEMOS:
        # priority 1: measured in this worktree (b1 batch)
        if demo in measured_b1 and measured_b1[demo].get('status') == 'OK':
            r = measured_b1[demo]
            rows.append((demo, float(r['wall_seconds']), 'measured', 'OK'))
            counts['measured'] += 1
            continue
        if demo in measured_b1_slow and measured_b1_slow[demo].get('status') == 'OK':
            r = measured_b1_slow[demo]
            rows.append((demo, float(r['wall_seconds']), 'measured', 'OK'))
            counts['measured'] += 1
            continue
        # priority 2: measured in main repo (parallel agent)
        if demo in measured_main:
            m = measured_main[demo]
            stat = m['status']
            if stat in ('ok', 'cached', 'OK'):
                src = 'measured' if stat == 'ok' else 'cached'
                rows.append((demo, m['wall'], src, 'OK'))
                counts['measured' if src == 'measured' else 'cached'] += 1
                continue
        # priority 3: cached log files
        cached = parse_cached_log(demo)
        if cached is not None:
            rows.append((demo, cached, 'cached', 'OK'))
            counts['cached'] += 1
            continue
        # priority 4: author estimate
        author = parse_author_estimate(demo)
        if author is not None:
            rows.append((demo, author, 'author_est', 'OK'))
            counts['author_est'] += 1
            continue
        rows.append((demo, float('nan'), 'missing', 'MISSING'))
        counts['missing'] += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['demo_name', 'wall_seconds', 'source', 'status'])
        for r in rows:
            secs = f'{r[1]:.3f}' if r[1] == r[1] else 'NaN'
            w.writerow([r[0], secs, r[2], r[3]])

    print(f'Wrote {len(rows)} rows to {OUT_CSV}')
    print(f'  measured:   {counts["measured"]}')
    print(f'  cached:     {counts["cached"]}')
    print(f'  author_est: {counts["author_est"]}')
    print(f'  missing:    {counts["missing"]}')


if __name__ == '__main__':
    main()

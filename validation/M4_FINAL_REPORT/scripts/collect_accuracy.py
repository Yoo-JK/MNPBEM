#!/usr/bin/env python
"""Run compare_smart_v3.py and parse output into accuracy_results.csv.

Output: validation/M4_FINAL_REPORT/data/accuracy_results.csv
columns: demo, demo_type, max_rel_err, classification
"""
import os, re, subprocess, csv

OUT = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT/data/accuracy_results.csv'
CMP = '/home/yoojk20/scratch/mnpbem_demo_comparison/compare_smart_v3.py'
PY = '/home/yoojk20/miniconda3/envs/mnpbem/bin/python'

# Pattern: '  demoXXX [type]: 1.23e-04'
DEMO_RE = re.compile(r'^\s+(demo\S+)\s+\[(\w+)\]:\s+([\d.eE+\-]+)$')
SECTION_RE = re.compile(r'^==\s+(\w+)\s+\(\d+\)\s+==')


def main():
    proc = subprocess.run([PY, CMP], capture_output=True, text=True, cwd=os.path.dirname(CMP), timeout=600)
    out = proc.stdout
    print('compare_smart_v3 done, parsing...')
    section = None
    rows = []
    for line in out.splitlines():
        m_sec = SECTION_RE.match(line)
        if m_sec:
            section = m_sec.group(1)
            continue
        m = DEMO_RE.match(line)
        if m:
            demo, dtype, err = m.group(1), m.group(2), float(m.group(3))
            rows.append([demo, dtype, err, section or 'unknown'])
    rows.sort(key=lambda r: r[0])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['demo', 'demo_type', 'max_rel_err', 'classification'])
        for r in rows:
            w.writerow([r[0], r[1], f'{r[2]:.6e}', r[3]])
    print(f'wrote {OUT} ({len(rows)} rows)')
    # summary
    summary = {}
    for r in rows:
        summary[r[3]] = summary.get(r[3], 0) + 1
    print('class counts:', summary)


if __name__ == '__main__':
    main()

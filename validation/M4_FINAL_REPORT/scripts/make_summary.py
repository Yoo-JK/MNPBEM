#!/usr/bin/env python
"""Build summary_table.csv, summary_table.md, and M4_FINAL_REPORT.md.

Inputs:
  data/timing_matlab.csv
  data/timing_python_cpu.csv
  data/timing_python_gpu.csv (optional)
  data/accuracy_results.csv
"""
import os
import pandas as pd
import numpy as np

ROOT = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT'
DATA = os.path.join(ROOT, 'data')


def load_timing(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df['wall_sec'] = pd.to_numeric(df['wall_sec'], errors='coerce')
    return df


def main():
    matlab = load_timing('timing_matlab.csv').rename(columns={'wall_sec': 'matlab_sec', 'status': 'matlab_status'})
    cpu = load_timing('timing_python_cpu.csv').rename(columns={'wall_sec': 'cpu_sec', 'status': 'cpu_status'})
    gpu = load_timing('timing_python_gpu.csv').rename(columns={'wall_sec': 'gpu_sec', 'status': 'gpu_status'})
    acc = pd.DataFrame()
    acc_p = os.path.join(DATA, 'accuracy_results.csv')
    if os.path.exists(acc_p):
        acc = pd.read_csv(acc_p)

    df = matlab[['demo', 'matlab_sec', 'matlab_status']]
    if not cpu.empty:
        df = df.merge(cpu[['demo', 'cpu_sec', 'cpu_status']], on='demo', how='outer')
    if not gpu.empty:
        df = df.merge(gpu[['demo', 'gpu_sec', 'gpu_status']], on='demo', how='outer')
    if not acc.empty:
        df = df.merge(acc[['demo', 'demo_type', 'max_rel_err', 'classification']], on='demo', how='outer')

    # speedups
    df['cpu_speedup'] = df.apply(
        lambda r: r['matlab_sec'] / r['cpu_sec'] if (pd.notna(r.get('matlab_sec')) and pd.notna(r.get('cpu_sec')) and r.get('cpu_sec', 0) > 0) else np.nan,
        axis=1,
    )
    if 'gpu_sec' in df.columns:
        df['gpu_speedup'] = df.apply(
            lambda r: r['matlab_sec'] / r['gpu_sec'] if (pd.notna(r.get('matlab_sec')) and pd.notna(r.get('gpu_sec')) and r.get('gpu_sec', 0) > 0) else np.nan,
            axis=1,
        )
    df = df.sort_values('demo').reset_index(drop=True)

    # CSV
    csv_path = os.path.join(ROOT, 'summary_table.csv')
    df.to_csv(csv_path, index=False, float_format='%.4g')
    print(f'wrote {csv_path}')

    # Markdown
    md_lines = []
    md_lines.append('# M4 Summary Table\n')
    md_lines.append('| Demo | Type | MATLAB (s) | PyCPU (s) | PyGPU (s) | CPU/MATLAB | GPU/MATLAB | max_rel_err | class |')
    md_lines.append('|---|---|---|---|---|---|---|---|---|')
    for _, r in df.iterrows():
        def fmt(v, prec=2):
            if pd.isna(v):
                return '-'
            try:
                return f'{float(v):.{prec}f}'
            except Exception:
                return str(v)
        def fmt_e(v):
            if pd.isna(v):
                return '-'
            try:
                return f'{float(v):.2e}'
            except Exception:
                return str(v)

        cpu_sp = r.get('cpu_speedup', float('nan'))
        gpu_sp = r.get('gpu_speedup', float('nan'))
        cpu_color = ''
        if pd.notna(cpu_sp):
            cpu_color = ':green_circle:' if cpu_sp >= 1.0 else ':red_circle:'
        gpu_color = ''
        if pd.notna(gpu_sp):
            gpu_color = ':green_circle:' if gpu_sp >= 1.0 else ':red_circle:'

        cls = r.get('classification', '')
        err = r.get('max_rel_err', float('nan'))
        err_str = fmt_e(err)
        cls_str = str(cls) if pd.notna(cls) else ''

        md_lines.append(f"| {r['demo']} | {r.get('demo_type','')} | "
                        f"{fmt(r.get('matlab_sec'))} | {fmt(r.get('cpu_sec'))} | "
                        f"{fmt(r.get('gpu_sec'))} | "
                        f"{cpu_color} {fmt(cpu_sp)} | "
                        f"{gpu_color} {fmt(gpu_sp)} | "
                        f"{err_str} | {cls_str} |")
    md_path = os.path.join(ROOT, 'summary_table.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'wrote {md_path}')

    # final report
    n_total = len(df)
    n_cpu_win = int((df['cpu_speedup'] >= 1.0).sum()) if 'cpu_speedup' in df else 0
    n_cpu_loss = int((df['cpu_speedup'] < 1.0).sum()) if 'cpu_speedup' in df else 0
    n_cpu_compared = int(df['cpu_speedup'].notna().sum()) if 'cpu_speedup' in df else 0
    n_gpu_win = int((df['gpu_speedup'] >= 1.0).sum()) if 'gpu_speedup' in df else 0
    n_gpu_compared = int(df['gpu_speedup'].notna().sum()) if 'gpu_speedup' in df else 0
    n_perf = int((df['classification'] == 'perf').sum()) if 'classification' in df else 0
    n_ok = int((df['classification'] == 'OK').sum()) if 'classification' in df else 0
    n_warn = int((df['classification'] == 'warn').sum()) if 'classification' in df else 0
    n_bad = int((df['classification'] == 'BAD').sum()) if 'classification' in df else 0

    median_speedup = df['cpu_speedup'].median() if 'cpu_speedup' in df else float('nan')

    rep_lines = []
    rep_lines.append('# M4 Final Report\n')
    rep_lines.append(f'**Total demos**: {n_total}')
    rep_lines.append('')
    rep_lines.append('## User requirements\n')
    rep_lines.append('1. **GPU acceleration as opt-in** - DONE (MNPBEM_GPU=1 explicit)')
    rep_lines.append(f'2. **Python (CPU) > MATLAB on every demo** - {n_cpu_win}/{n_cpu_compared} demos faster on Python CPU '
                     f'(median {median_speedup:.2f}x speedup)')
    rep_lines.append(f'3. **MATLAB == Python (1e-12)** - perf={n_perf} OK={n_ok} warn={n_warn} BAD={n_bad}')
    rep_lines.append('')
    rep_lines.append('## Timing summary')
    rep_lines.append('')
    rep_lines.append(f'- Python CPU vs MATLAB: {n_cpu_win}/{n_cpu_compared} demos Python is faster')
    if 'gpu_speedup' in df:
        rep_lines.append(f'- Python GPU vs MATLAB: {n_gpu_win}/{n_gpu_compared} demos Python+GPU is faster')
    rep_lines.append('')
    rep_lines.append('## Plots')
    rep_lines.append('')
    rep_lines.append('![Dashboard](plots/dashboard.png)')
    rep_lines.append('')
    rep_lines.append('![Timing overview](plots/timing_overview.png)')
    rep_lines.append('')
    rep_lines.append('![Accuracy](plots/accuracy_overview.png)')
    rep_lines.append('')
    rep_lines.append('## Per-demo detail')
    rep_lines.append('')
    rep_lines.append('See `summary_table.md` for full table; `plots/per_demo/` for per-demo spectrum + timing plots.')
    rep_lines.append('')

    if n_cpu_compared > 0 and n_cpu_loss > 0:
        rep_lines.append('## Demos where MATLAB still wins (CPU)')
        rep_lines.append('')
        slow = df[df['cpu_speedup'] < 1.0].sort_values('cpu_speedup')
        rep_lines.append('| Demo | MATLAB (s) | PyCPU (s) | CPU/MATLAB |')
        rep_lines.append('|---|---|---|---|')
        for _, r in slow.iterrows():
            ms = r.get('matlab_sec', float('nan'))
            cs = r.get('cpu_sec', float('nan'))
            sp = r.get('cpu_speedup', float('nan'))
            rep_lines.append(f"| {r['demo']} | "
                             f"{('-' if pd.isna(ms) else f'{ms:.2f}')} | "
                             f"{('-' if pd.isna(cs) else f'{cs:.2f}')} | "
                             f"{('-' if pd.isna(sp) else f'{sp:.2f}')} |")

    rep_path = os.path.join(ROOT, 'M4_FINAL_REPORT.md')
    with open(rep_path, 'w') as f:
        f.write('\n'.join(rep_lines))
    print(f'wrote {rep_path}')


if __name__ == '__main__':
    main()

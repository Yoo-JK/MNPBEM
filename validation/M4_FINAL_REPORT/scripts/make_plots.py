#!/usr/bin/env python
"""Generate plots for M4 final report.

Inputs:
  data/timing_matlab.csv
  data/timing_python_cpu.csv
  data/timing_python_gpu.csv  (optional)
  data/accuracy_results.csv

Outputs:
  plots/dashboard.png
  plots/timing_overview.png
  plots/accuracy_overview.png
  plots/per_demo/<demo>_spectrum.png
  plots/per_demo/<demo>_timing.png
"""
import os, glob, csv, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/yoojk20/workspace/MNPBEM/validation/M4_FINAL_REPORT'
DATA = os.path.join(ROOT, 'data')
PLOTS = os.path.join(ROOT, 'plots')
PERDEMO = os.path.join(PLOTS, 'per_demo')
DEMO_ROOT = '/home/yoojk20/scratch/mnpbem_demo_comparison'

os.makedirs(PERDEMO, exist_ok=True)


def load_timings():
    """Load all timing csvs into dict[demo] = {'matlab': sec, 'cpu': sec, 'gpu': sec}."""
    out = {}
    for backend, fname in [('matlab', 'timing_matlab.csv'),
                            ('cpu', 'timing_python_cpu.csv'),
                            ('gpu', 'timing_python_gpu.csv')]:
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            print(f'  (no {fname})')
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f'  could not read {fname}: {e}')
            continue
        for _, row in df.iterrows():
            demo = row['demo']
            try:
                t = float(row['wall_sec'])
            except Exception:
                continue
            status = str(row.get('status', ''))
            if status not in ('ok', 'cached', 'ok_wall'):
                t = float('nan')
            out.setdefault(demo, {})[backend] = t
    return out


def load_accuracy():
    p = os.path.join(DATA, 'accuracy_results.csv')
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Per-demo spectrum overlay (3-line: MATLAB / PyCPU / PyGPU)
# Note: we only have MATLAB+Python from matlab.csv/python.csv. PyGPU vs PyCPU
# values are bit-identical so we plot only MATLAB vs Python with markers.
# ---------------------------------------------------------------------------
def per_demo_spectrum(demo):
    d = os.path.join(DEMO_ROOT, demo)
    mp = os.path.join(d, 'matlab.csv')
    pp = os.path.join(d, 'python.csv')
    if not (os.path.exists(mp) and os.path.exists(pp)):
        return
    try:
        m = pd.read_csv(mp)
        p = pd.read_csv(pp)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    # Pick first two numeric columns
    num_cols_m = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
    num_cols_p = [c for c in p.columns if pd.api.types.is_numeric_dtype(p[c])]
    if len(num_cols_m) < 2 or len(num_cols_p) < 2:
        plt.close(fig)
        return
    x_col = num_cols_m[0]
    y_col = num_cols_m[1]
    try:
        ax.plot(m[x_col], m[y_col], 'o-', color='#1f77b4', label='MATLAB', markersize=5, alpha=0.7)
        # Use same column names if available, else first 2
        x_col_p = num_cols_p[0]
        y_col_p = num_cols_p[1]
        ax.plot(p[x_col_p], p[y_col_p], '--', color='#d62728', label='Python', linewidth=2, alpha=0.8)
    except Exception:
        plt.close(fig)
        return
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{demo} - spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PERDEMO, f'{demo}_spectrum.png'), dpi=80)
    plt.close(fig)


def per_demo_timing(demo, timings):
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ['matlab', 'cpu', 'gpu']
    labels = ['MATLAB', 'PyCPU', 'PyGPU']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    vals = []
    used_labels = []
    used_colors = []
    for k, l, c in zip(keys, labels, colors):
        v = timings.get(k, float('nan'))
        if v is not None and v == v and v > 0:  # not NaN
            vals.append(v)
            used_labels.append(l)
            used_colors.append(c)
    if not vals:
        plt.close(fig)
        return
    bars = ax.bar(used_labels, vals, color=used_colors)
    ax.set_yscale('log')
    ax.set_ylabel('wall time (s, log scale)')
    ax.set_title(f'{demo} - timing')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}s',
                ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PERDEMO, f'{demo}_timing.png'), dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overview plots
# ---------------------------------------------------------------------------
def timing_overview(timings):
    rows = []
    for demo, t in timings.items():
        m = t.get('matlab', float('nan'))
        c = t.get('cpu', float('nan'))
        g = t.get('gpu', float('nan'))
        speedup_cpu = (m / c) if (m == m and c == c and c > 0) else float('nan')
        rows.append((demo, m, c, g, speedup_cpu))
    rows.sort(key=lambda r: (-(r[4] if r[4] == r[4] else -1), r[0]))

    fig, ax = plt.subplots(figsize=(10, max(8, 0.18 * len(rows))))
    y_pos = np.arange(len(rows))
    bar_h = 0.27
    matlab_t = np.array([r[1] for r in rows])
    cpu_t = np.array([r[2] for r in rows])
    gpu_t = np.array([r[3] for r in rows])
    ax.barh(y_pos - bar_h, matlab_t, height=bar_h, color='#1f77b4', label='MATLAB')
    ax.barh(y_pos, cpu_t, height=bar_h, color='#2ca02c', label='PyCPU')
    ax.barh(y_pos + bar_h, gpu_t, height=bar_h, color='#ff7f0e', label='PyGPU')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('wall time (s, log scale)')
    ax.set_title('Timing overview - sorted by CPU speedup vs MATLAB (highest at top)')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'timing_overview.png'), dpi=100)
    plt.close(fig)


def accuracy_overview(acc_df):
    if acc_df.empty:
        return
    errs = acc_df['max_rel_err'].astype(float).values
    errs_log = np.log10(np.maximum(errs, 1e-18))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errs_log, bins=30, color='#1f77b4', edgecolor='black')
    ax.set_xlabel('log10(max_rel_err)')
    ax.set_ylabel('# demos')
    ax.set_title(f'Accuracy distribution ({len(errs)} demos)')
    ax.axvline(np.log10(1e-12), color='green', linestyle='--', label='1e-12 (perf)')
    ax.axvline(np.log10(1e-3), color='orange', linestyle='--', label='1e-3 (warn)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'accuracy_overview.png'), dpi=100)
    plt.close(fig)


def dashboard(timings, acc_df):
    """50-cell grid: each cell shows a tiny spectrum + timing summary."""
    demos = sorted(timings.keys())
    # accept up to 60
    demos = demos[:60]
    n = len(demos)
    if n == 0:
        return
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows))
    if rows == 1:
        axes = np.array([axes])
    for i, demo in enumerate(demos):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        d = os.path.join(DEMO_ROOT, demo)
        mp = os.path.join(d, 'matlab.csv')
        pp = os.path.join(d, 'python.csv')
        if os.path.exists(mp) and os.path.exists(pp):
            try:
                m = pd.read_csv(mp)
                p = pd.read_csv(pp)
                num_cols_m = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
                num_cols_p = [c for c in p.columns if pd.api.types.is_numeric_dtype(p[c])]
                if len(num_cols_m) >= 2 and len(num_cols_p) >= 2:
                    ax.plot(m[num_cols_m[0]], m[num_cols_m[1]], '-', color='#1f77b4', linewidth=1)
                    ax.plot(p[num_cols_p[0]], p[num_cols_p[1]], '--', color='#d62728', linewidth=1)
            except Exception:
                pass
        t = timings.get(demo, {})
        m_t = t.get('matlab', float('nan'))
        c_t = t.get('cpu', float('nan'))
        speedup = (m_t / c_t) if (m_t == m_t and c_t == c_t and c_t > 0) else float('nan')
        title = demo
        sub = ''
        if speedup == speedup:
            color = 'green' if speedup >= 1.0 else 'red'
            sub = f'M={m_t:.1f}s C={c_t:.1f}s {speedup:.1f}x'
            ax.set_title(title, fontsize=8)
            ax.text(0.5, -0.25, sub, transform=ax.transAxes, fontsize=7,
                    ha='center', color=color)
        else:
            ax.set_title(title, fontsize=8)
        # accuracy
        if not acc_df.empty:
            sel = acc_df[acc_df['demo'] == demo]
            if not sel.empty:
                err = float(sel.iloc[0]['max_rel_err'])
                err_color = 'green' if err < 1e-12 else 'orange' if err < 1e-3 else 'red'
                ax.text(0.02, 0.98, f'err={err:.0e}', transform=ax.transAxes,
                        fontsize=6, va='top', color=err_color)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    # blank remaining
    for j in range(n, rows * cols):
        r, c = j // cols, j % cols
        axes[r, c].axis('off')
    fig.suptitle('M4 Dashboard - 50+ demos: spectrum overlay + speedup', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(PLOTS, 'dashboard.png'), dpi=100)
    plt.close(fig)


def main():
    print('Loading timing data...')
    timings = load_timings()
    print(f'  {len(timings)} demos with timing')
    print('Loading accuracy data...')
    acc_df = load_accuracy()
    print(f'  {len(acc_df)} demos with accuracy')

    print('Generating per-demo plots...')
    for demo in sorted(timings.keys()):
        per_demo_spectrum(demo)
        per_demo_timing(demo, timings[demo])

    print('Generating overview plots...')
    timing_overview(timings)
    accuracy_overview(acc_df)
    dashboard(timings, acc_df)
    print('Done')


if __name__ == '__main__':
    main()

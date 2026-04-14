"""
Summary: collect timing and RMS error data from all validation directories (01-13).

Generates:
  summary/data/summary_data.csv
  summary/figures/summary_table.png
  summary/figures/summary_bar.png
  summary/figures/summary_timing.png
  summary/summary_report.md
"""

import os
import sys
import csv

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VAL_DIR = '/home/yoojk20/workspace/MNPBEM/validation'
DATA_DIR = os.path.join(VAL_DIR, 'summary', 'data')
FIG_DIR = os.path.join(VAL_DIR, 'summary', 'figures')


# =========================================================================
# Configuration: tests and their MATLAB/Python file pairs
# =========================================================================
TESTS = [
    {
        'id': '01_mie',
        'subtests': [
            {
                'name': '01 MieStat',
                'matlab_file': '01_mie/data/miestat_matlab.csv',
                'python_file': '01_mie/data/miestat_python.csv',
                'columns': ['extinction', 'scattering', 'absorption'],
            },
            {
                'name': '01 MieRet',
                'matlab_file': '01_mie/data/mieret_matlab.csv',
                'python_file': '01_mie/data/mieret_python.csv',
                'columns': ['extinction', 'scattering', 'absorption'],
            },
            {
                'name': '01 MieGans',
                'matlab_file': '01_mie/data/miegans_matlab.csv',
                'python_file': '01_mie/data/miegans_python.csv',
                'columns': ['extinction_xpol', 'extinction_zpol'],
            },
        ],
        'matlab_timing': '01_mie/data/matlab_timing.csv',
        'python_timing': '01_mie/data/python_timing.csv',
        'timing_key': 'time_seconds',
        'timing_tests': ['miestat', 'mieret', 'miegans'],
    },
    {
        'id': '02_bemstat_sphere',
        'subtests': [
            {
                'name': '02 BEMStat sphere',
                'matlab_file': '02_bemstat_sphere/data/matlab_bemstat.csv',
                'python_file': '02_bemstat_sphere/data/python_bemstat.csv',
                'columns': ['extinction', 'scattering', 'absorption'],
            },
        ],
        'matlab_timing': '02_bemstat_sphere/data/matlab_timing.csv',
        'python_timing': '02_bemstat_sphere/data/python_timing.csv',
        'timing_key': 'time_sec',
        'timing_tests': ['BEM'],
    },
    {
        'id': '03_bemret_sphere',
        'subtests': [
            {
                'name': '03 BEMRet sphere',
                'matlab_file': '03_bemret_sphere/data/matlab_bemret.csv',
                'python_file': '03_bemret_sphere/data/python_bemret.csv',
                'columns': ['extinction', 'scattering', 'absorption'],
            },
        ],
        'matlab_timing': '03_bemret_sphere/data/matlab_timing.csv',
        'python_timing': '03_bemret_sphere/data/python_timing.csv',
        'timing_key': 'time_seconds',
        'timing_tests': ['bemret'],
    },
    {
        'id': '04_bemstat_layer',
        'subtests': [
            {
                'name': '04 BEMStat layer (normal)',
                'matlab_file': '04_bemstat_layer/data/matlab_normal.csv',
                'python_file': '04_bemstat_layer/data/python_normal.csv',
                'columns': ['extinction', 'scattering'],
            },
            {
                'name': '04 BEMStat layer (oblique)',
                'matlab_file': '04_bemstat_layer/data/matlab_oblique.csv',
                'python_file': '04_bemstat_layer/data/python_oblique.csv',
                'columns': ['extinction', 'scattering'],
            },
        ],
        'matlab_timing': '04_bemstat_layer/data/matlab_timing.csv',
        'python_timing': '04_bemstat_layer/data/python_timing.csv',
        'timing_key': 'time_sec',
        'timing_tests': ['normal', 'oblique'],
    },
    {
        'id': '05_bemret_layer',
        'subtests': [
            {
                'name': '05 BEMRet layer',
                'matlab_file': '05_bemret_layer/data/matlab_retlayer.csv',
                'python_file': '05_bemret_layer/data/python_retlayer.csv',
                'columns': ['extinction', 'scattering', 'absorption'],
            },
        ],
        'matlab_timing': '05_bemret_layer/data/matlab_retlayer_timing.csv',
        'python_timing': '05_bemret_layer/data/python_retlayer_timing.csv',
        'timing_key': 'total_sec',
        'timing_tests': [None],  # Single row, no test column
    },
    {
        'id': '13_shapes',
        'subtests': [
            {
                'name': '13 trisphere',
                'matlab_file': '13_shapes/data/trisphere_matlab.csv',
                'python_file': '13_shapes/data/trisphere_python.csv',
                'columns': ['extinction'],
            },
            {
                'name': '13 trirod',
                'matlab_file': '13_shapes/data/trirod_matlab.csv',
                'python_file': '13_shapes/data/trirod_python.csv',
                'columns': ['extinction_xpol', 'extinction_zpol'],
            },
            {
                'name': '13 tricube',
                'matlab_file': '13_shapes/data/tricube_matlab.csv',
                'python_file': '13_shapes/data/tricube_python.csv',
                'columns': ['extinction'],
            },
            {
                'name': '13 tritorus',
                'matlab_file': '13_shapes/data/tritorus_matlab.csv',
                'python_file': '13_shapes/data/tritorus_python.csv',
                'columns': ['extinction'],
            },
            {
                'name': '13 trispheresegment',
                'matlab_file': '13_shapes/data/trispheresegment_matlab.csv',
                'python_file': '13_shapes/data/trispheresegment_python.csv',
                'columns': ['extinction'],
            },
            {
                'name': '13 trispherescale',
                'matlab_file': '13_shapes/data/trispherescale_matlab.csv',
                'python_file': '13_shapes/data/trispherescale_python.csv',
                'columns': ['extinction'],
            },
            {
                'name': '13 tripolygon',
                'matlab_file': '13_shapes/data/tripolygon_matlab.csv',
                'python_file': '13_shapes/data/tripolygon_python.csv',
                'columns': ['extinction'],
            },
        ],
        'matlab_timing': '13_shapes/data/matlab_timing.csv',
        'python_timing': '13_shapes/data/python_timing.csv',
        'timing_key': 'time_seconds',
        'timing_tests': [
            'trisphere', 'trirod', 'tricube', 'tritorus',
            'trispheresegment', 'trispherescale', 'tripolygon',
        ],
    },
]


def load_csv_data(path):
    """Load CSV and return header + data."""
    with open(path) as f:
        header = f.readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def load_timing_csv(path):
    """Load timing CSV as list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def rms_relative_error(a, b):
    """RMS relative error between arrays a and b."""
    denom = np.maximum(np.abs(b), 1e-30)
    return float(np.sqrt(np.mean(((a - b) / denom) ** 2)))


def get_timing(timing_rows, test_name, key):
    """Extract timing value from timing rows."""
    if test_name is None:
        # Single row format
        return float(timing_rows[0][key])
    for row in timing_rows:
        # Find matching test row
        for k, v in row.items():
            if k != key and v.strip().lower() == test_name.lower():
                return float(row[key])
    return None


def compute_rms_error(matlab_path, python_path, columns):
    """Compute RMS relative error between MATLAB and Python data files."""
    try:
        header_m, data_m = load_csv_data(os.path.join(VAL_DIR, matlab_path))
        header_p, data_p = load_csv_data(os.path.join(VAL_DIR, python_path))
    except Exception as e:
        print('  [warn] Could not load files: {}'.format(e))
        return None

    errors = []
    for col_name in columns:
        # Find column index in both headers
        idx_m = None
        idx_p = None
        for i, h in enumerate(header_m):
            if h.strip().lower() == col_name.lower():
                idx_m = i
                break
        for i, h in enumerate(header_p):
            if h.strip().lower() == col_name.lower():
                idx_p = i
                break

        if idx_m is None or idx_p is None:
            print('  [warn] Column {} not found in headers'.format(col_name))
            continue

        m_col = data_m[:, idx_m]
        p_col = data_p[:, idx_p]

        # Interpolate if wavelength grids differ
        if len(m_col) != len(p_col):
            wl_m = data_m[:, 0]
            wl_p = data_p[:, 0]
            p_col = np.interp(wl_m, wl_p, p_col)

        err = rms_relative_error(p_col, m_col)
        errors.append(err)

    if errors:
        return max(errors)
    return None


def main():
    print('=== Generating Summary ===')
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Collect all results
    results = []  # list of dicts: name, matlab_time, python_time, speedup, rms_error

    for test_group in TESTS:
        test_id = test_group['id']
        print('\nProcessing: {}'.format(test_id))

        # Load timing data
        matlab_timing_path = os.path.join(VAL_DIR, test_group['matlab_timing'])
        python_timing_path = os.path.join(VAL_DIR, test_group['python_timing'])

        matlab_timing_rows = None
        python_timing_rows = None
        if os.path.exists(matlab_timing_path):
            matlab_timing_rows = load_timing_csv(matlab_timing_path)
        if os.path.exists(python_timing_path):
            python_timing_rows = load_timing_csv(python_timing_path)

        for si, subtest in enumerate(test_group['subtests']):
            name = subtest['name']

            # RMS error
            rms_err = compute_rms_error(
                subtest['matlab_file'],
                subtest['python_file'],
                subtest['columns'],
            )
            if rms_err is not None:
                print('  {} : RMS err = {:.4e}'.format(name, rms_err))
            else:
                print('  {} : RMS err = N/A'.format(name))

            # Timing
            matlab_time = None
            python_time = None
            timing_key = test_group['timing_key']
            timing_tests = test_group['timing_tests']

            # For group tests with many subtests, match timing to subtest
            if len(timing_tests) > 1 and si < len(timing_tests):
                tname = timing_tests[si]
            elif len(timing_tests) == 1:
                tname = timing_tests[0]
            else:
                tname = None

            if matlab_timing_rows is not None:
                matlab_time = get_timing(matlab_timing_rows, tname, timing_key)
            if python_timing_rows is not None:
                python_time = get_timing(python_timing_rows, tname, timing_key)

            speedup = None
            if matlab_time is not None and python_time is not None and python_time > 0:
                speedup = matlab_time / python_time

            results.append({
                'name': name,
                'matlab_time': matlab_time,
                'python_time': python_time,
                'speedup': speedup,
                'rms_error': rms_err,
            })

    # Save summary data CSV
    summary_csv = os.path.join(DATA_DIR, 'summary_data.csv')
    with open(summary_csv, 'w') as f:
        f.write('test,matlab_time_s,python_time_s,speedup,rms_error\n')
        for r in results:
            f.write('{},{},{},{},{}\n'.format(
                r['name'],
                '{:.4f}'.format(r['matlab_time']) if r['matlab_time'] is not None else '',
                '{:.4f}'.format(r['python_time']) if r['python_time'] is not None else '',
                '{:.2f}'.format(r['speedup']) if r['speedup'] is not None else '',
                '{:.4e}'.format(r['rms_error']) if r['rms_error'] is not None else '',
            ))
    print('\n[saved] {}'.format(summary_csv))

    # =====================================================================
    # Figure 1: summary_table.png
    # =====================================================================
    fig, ax = plt.subplots(figsize=(14, max(4, 0.5 * len(results) + 2)))
    ax.axis('off')

    col_labels = ['Test', 'MATLAB (s)', 'Python (s)', 'Speedup', 'RMS Error']
    cell_text = []
    for r in results:
        row = [
            r['name'],
            '{:.4f}'.format(r['matlab_time']) if r['matlab_time'] is not None else '--',
            '{:.4f}'.format(r['python_time']) if r['python_time'] is not None else '--',
            '{:.2f}x'.format(r['speedup']) if r['speedup'] is not None else '--',
            '{:.2e}'.format(r['rms_error']) if r['rms_error'] is not None else '--',
        ]
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j, label in enumerate(col_labels):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color code RMS errors
    for i, r in enumerate(results):
        if r['rms_error'] is not None:
            if r['rms_error'] < 1e-6:
                table[i + 1, 4].set_facecolor('#C6EFCE')  # green
            elif r['rms_error'] < 1e-2:
                table[i + 1, 4].set_facecolor('#FFEB9C')  # yellow
            else:
                table[i + 1, 4].set_facecolor('#FFC7CE')  # red

    ax.set_title('MNPBEM Validation Summary: MATLAB vs Python', fontsize=14, pad=20)
    fig.tight_layout()
    table_path = os.path.join(FIG_DIR, 'summary_table.png')
    fig.savefig(table_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[saved] {}'.format(table_path))

    # =====================================================================
    # Figure 2: summary_bar.png (RMS relative error per test)
    # =====================================================================
    valid_results = [r for r in results if r['rms_error'] is not None]
    if valid_results:
        names = [r['name'] for r in valid_results]
        errors = [r['rms_error'] for r in valid_results]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(names))
        colors = []
        for e in errors:
            if e < 1e-6:
                colors.append('#2E7D32')  # dark green
            elif e < 1e-2:
                colors.append('#F9A825')  # amber
            else:
                colors.append('#C62828')  # red

        bars = ax.bar(x, errors, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('RMS Relative Error')
        ax.set_title('MATLAB vs Python: RMS Relative Error per Test')
        ax.grid(True, alpha=0.3, axis='y')

        # Add threshold lines
        ax.axhline(y=1e-2, color='orange', linestyle='--', linewidth=0.8, label='1% threshold')
        ax.axhline(y=1e-6, color='green', linestyle='--', linewidth=0.8, label='1e-6 threshold')
        ax.legend(loc='upper right', fontsize=8)

        fig.tight_layout()
        bar_path = os.path.join(FIG_DIR, 'summary_bar.png')
        fig.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('[saved] {}'.format(bar_path))

    # =====================================================================
    # Figure 3: summary_timing.png (grouped bar chart MATLAB vs Python)
    # =====================================================================
    timed_results = [r for r in results if r['matlab_time'] is not None and r['python_time'] is not None]
    if timed_results:
        names = [r['name'] for r in timed_results]
        matlab_times = [r['matlab_time'] for r in timed_results]
        python_times = [r['python_time'] for r in timed_results]

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, matlab_times, width, label='MATLAB',
                        color='#4472C4', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, python_times, width, label='Python',
                        color='#ED7D31', edgecolor='black', linewidth=0.5)

        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('MATLAB vs Python: Computation Time per Test')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add speedup annotations
        for i, r in enumerate(timed_results):
            if r['speedup'] is not None:
                max_h = max(r['matlab_time'], r['python_time'])
                ax.annotate(
                    '{:.1f}x'.format(r['speedup']),
                    xy=(i, max_h),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=7,
                    color='gray',
                )

        fig.tight_layout()
        timing_path = os.path.join(FIG_DIR, 'summary_timing.png')
        fig.savefig(timing_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('[saved] {}'.format(timing_path))

    # =====================================================================
    # Generate summary_report.md
    # =====================================================================
    report_path = os.path.join(VAL_DIR, 'summary', 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write('# MNPBEM Validation Summary Report\n\n')
        f.write('Generated: 2026-04-13\n\n')
        f.write('## Overview\n\n')
        f.write('This report summarizes the validation of the Python MNPBEM implementation\n')
        f.write('against the original MATLAB MNPBEM toolbox.\n\n')

        n_total = len(results)
        n_with_err = len([r for r in results if r['rms_error'] is not None])
        n_pass = len([r for r in results if r['rms_error'] is not None and r['rms_error'] < 1e-2])
        n_exact = len([r for r in results if r['rms_error'] is not None and r['rms_error'] < 1e-6])

        f.write('- **Total tests**: {}\n'.format(n_total))
        f.write('- **Tests with MATLAB-Python comparison**: {}\n'.format(n_with_err))
        f.write('- **Tests passing (RMS < 1%)**: {}\n'.format(n_pass))
        f.write('- **Near-exact match (RMS < 1e-6)**: {}\n\n'.format(n_exact))

        f.write('## Results Table\n\n')
        f.write('| Test | MATLAB (s) | Python (s) | Speedup | RMS Error |\n')
        f.write('|------|-----------|-----------|---------|----------|\n')
        for r in results:
            mt = '{:.4f}'.format(r['matlab_time']) if r['matlab_time'] is not None else '--'
            pt = '{:.4f}'.format(r['python_time']) if r['python_time'] is not None else '--'
            sp = '{:.2f}x'.format(r['speedup']) if r['speedup'] is not None else '--'
            re = '{:.2e}'.format(r['rms_error']) if r['rms_error'] is not None else '--'
            f.write('| {} | {} | {} | {} | {} |\n'.format(r['name'], mt, pt, sp, re))

        f.write('\n## Timing Comparison\n\n')
        if timed_results:
            total_m = sum(r['matlab_time'] for r in timed_results)
            total_p = sum(r['python_time'] for r in timed_results)
            f.write('- **Total MATLAB time**: {:.2f} s\n'.format(total_m))
            f.write('- **Total Python time**: {:.2f} s\n'.format(total_p))
            f.write('- **Overall speedup**: {:.2f}x\n\n'.format(total_m / total_p if total_p > 0 else 0))

        f.write('## Figures\n\n')
        f.write('- `summary_table.png`: Summary table with all tests\n')
        f.write('- `summary_bar.png`: RMS relative error bar chart\n')
        f.write('- `summary_timing.png`: MATLAB vs Python timing comparison\n\n')

        f.write('## Notes\n\n')
        f.write('- RMS error is computed as the root mean square of the relative error\n')
        f.write('  |Python - MATLAB| / max(|MATLAB|, 1e-30) across all wavelength points.\n')
        f.write('- For tests with multiple output columns (e.g., extinction, scattering),\n')
        f.write('  the maximum RMS error across all columns is reported.\n')
        f.write('- Speedup = MATLAB time / Python time. Values > 1 mean Python is faster.\n')
        f.write('- Tests 06-12 have Python-only data (no MATLAB reference for comparison).\n')

    print('[saved] {}'.format(report_path))
    print('\n[info] Summary generation complete.')


if __name__ == '__main__':
    main()

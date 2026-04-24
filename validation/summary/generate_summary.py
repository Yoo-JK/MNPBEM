import os
import sys

from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MNPBEM_ROOT = '/home/yoojk20/workspace/MNPBEM'
sys.path.insert(0, os.path.join(MNPBEM_ROOT, 'validation'))

from _common import compute_rms


VALIDATION_ROOT = '/home/yoojk20/workspace/MNPBEM/validation'
SUMMARY_DATA = os.path.join(VALIDATION_ROOT, 'summary', 'data')
SUMMARY_FIG = os.path.join(VALIDATION_ROOT, 'summary', 'figures')


def read_timing_csv(path: str) -> Dict[str, float]:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def pair_rms(folder: str) -> Dict[str, float]:
    rms_by_case = {}
    if not os.path.isdir(folder):
        return rms_by_case
    files = os.listdir(folder)
    py_files = [f for f in files if f.endswith('_python.csv')]
    for pyf in py_files:
        base = pyf[:-len('_python.csv')]
        if base == 'python':
            continue
        mlf = '{}_matlab.csv'.format(base)
        if mlf not in files:
            continue
        try:
            py = np.loadtxt(os.path.join(folder, pyf), delimiter = ',', skiprows = 1)
            ml = np.loadtxt(os.path.join(folder, mlf), delimiter = ',', skiprows = 1)
        except Exception:
            continue
        if py.shape != ml.shape or py.ndim != 2 or py.shape[1] < 2:
            continue
        rms_cols = []
        for j in range(1, py.shape[1]):
            rms_cols.append(compute_rms(py[:, j], ml[:, j]))
        rms_by_case[base] = float(max(rms_cols)) if rms_cols else 0.0
    return rms_by_case


def walk_validation() -> List[Dict]:
    rows = []
    for top in sorted(os.listdir(VALIDATION_ROOT)):
        if top.startswith('_') or top in ('summary',):
            continue
        top_path = os.path.join(VALIDATION_ROOT, top)
        if not os.path.isdir(top_path):
            continue

        direct_data = os.path.join(top_path, 'data')
        if os.path.isdir(direct_data):
            row = collect_row(top, None, direct_data)
            if row:
                rows.append(row)
            continue

        for sub in sorted(os.listdir(top_path)):
            sub_data = os.path.join(top_path, sub, 'data')
            if not os.path.isdir(sub_data):
                continue
            row = collect_row(top, sub, sub_data)
            if row:
                rows.append(row)
    return rows


def collect_row(top: str, sub, data_dir: str) -> Dict:
    py_timing = read_timing_csv(os.path.join(data_dir, 'python_timing.csv'))
    ml_timing = read_timing_csv(os.path.join(data_dir, 'matlab_timing.csv'))
    rms_by_case = pair_rms(data_dir)

    t_py = sum(py_timing.values())
    t_ml = sum(ml_timing.values())
    if not rms_by_case:
        max_rms = float('nan')
    else:
        max_rms = max(rms_by_case.values())

    name = top if sub is None else '{}/{}'.format(top, sub)
    return {
        'name': name,
        't_py': t_py,
        't_ml': t_ml,
        'speedup': t_ml / t_py if t_py > 0 else 0.0,
        'max_rms': max_rms,
        'rms_cases': rms_by_case,
    }


def plot_timing(rows: List[Dict], savepath: str) -> None:
    names = [r['name'] for r in rows]
    t_py = [r['t_py'] for r in rows]
    t_ml = [r['t_ml'] for r in rows]

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize = (14, 6))
    ax.bar(x - w / 2, t_ml, width = w, label = 'MATLAB', color = '#4477AA')
    ax.bar(x + w / 2, t_py, width = w, label = 'Python', color = '#CC6677')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation = 40, ha = 'right')
    ax.set_ylabel('Time (s)')
    ax.set_title('MATLAB vs Python timing (total per validation case)')
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'y')
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def plot_rms(rows: List[Dict], savepath: str) -> None:
    names = [r['name'] for r in rows]
    rms_vals = [r['max_rms'] if not np.isnan(r['max_rms']) else 0 for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize = (14, 6))
    ax.bar(x, rms_vals, color = '#CC6677')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation = 40, ha = 'right')
    ax.set_yscale('log')
    ax.set_ylabel('max RMS relative error (log)')
    ax.set_title('MATLAB vs Python — RMS relative error per validation case')
    ax.axhline(1e-3, color = 'g', ls = '--', lw = 1.0, label = '1e-3 (0.1%)')
    ax.axhline(1e-1, color = 'orange', ls = '--', lw = 1.0, label = '1e-1 (10%)')
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'y')
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def write_summary_md(rows: List[Dict], savepath: str) -> None:
    total_py = sum(r['t_py'] for r in rows)
    total_ml = sum(r['t_ml'] for r in rows)

    lines = []
    lines.append('# MNPBEM Validation Summary\n')
    lines.append('Total cases: {}\n'.format(len(rows)))
    lines.append('Total MATLAB time: {:.2f} s\n'.format(total_ml))
    lines.append('Total Python time: {:.2f} s\n'.format(total_py))
    if total_py > 0:
        lines.append('Overall Python speedup: x{:.2f}\n'.format(total_ml / total_py))
    lines.append('\n## Results\n')
    lines.append('| Test | MATLAB (s) | Python (s) | Speedup | Max RMS |\n')
    lines.append('|------|-----------|------------|---------|---------|\n')
    for r in rows:
        rms_str = '{:.2e}'.format(r['max_rms']) if not np.isnan(r['max_rms']) else 'N/A'
        lines.append('| {} | {:.3f} | {:.3f} | x{:.2f} | {} |\n'.format(
            r['name'], r['t_ml'], r['t_py'], r['speedup'], rms_str))
    lines.append('\n## Notes\n')
    lines.append('- RMS is computed per matching `*_python.csv` / `*_matlab.csv` pair '
        'as `sqrt(mean((|py-ml|/max(|ml|,1e-30))^2))`, '
        '최대값이 "Max RMS"로 보고됨.\n')
    lines.append('- Timing은 각 폴더의 `python_timing.csv` / `matlab_timing.csv` 합계.\n')
    lines.append('- `Speedup = MATLAB time / Python time`. >1 이면 Python이 빠름.\n')
    lines.append('- 06_mirror/rod: trirod quarter-mesh 생성 한계로 skip (README 참고).\n')
    lines.append('- 10_dipole_layer: Python BEMRetLayer 성능 이슈로 ret 파트 스크립트 내 skip 플래그.\n')

    with open(savepath, 'w') as f:
        f.writelines(lines)


def save_csv(rows: List[Dict], savepath: str) -> None:
    with open(savepath, 'w') as f:
        f.write('name,matlab_s,python_s,speedup,max_rms\n')
        for r in rows:
            rms_v = r['max_rms'] if not np.isnan(r['max_rms']) else ''
            f.write('{},{:.6f},{:.6f},{:.6f},{}\n'.format(
                r['name'], r['t_ml'], r['t_py'], r['speedup'], rms_v))


def main() -> None:
    os.makedirs(SUMMARY_DATA, exist_ok = True)
    os.makedirs(SUMMARY_FIG, exist_ok = True)

    print('[info] walking validation tree ...')
    rows = walk_validation()
    print('[info] collected {} cases'.format(len(rows)))
    for r in rows:
        print('  {:<35} MATLAB={:.3f}s  Python={:.3f}s  speedup={:.2f}x  RMS={}'.format(
            r['name'], r['t_ml'], r['t_py'], r['speedup'],
            '{:.2e}'.format(r['max_rms']) if not np.isnan(r['max_rms']) else 'N/A'))

    save_csv(rows, os.path.join(SUMMARY_DATA, 'summary_data.csv'))
    plot_timing(rows, os.path.join(SUMMARY_FIG, 'summary_timing.png'))
    plot_rms(rows, os.path.join(SUMMARY_FIG, 'summary_rms.png'))
    write_summary_md(rows, os.path.join(VALIDATION_ROOT, 'summary', 'summary_report.md'))

    print('[info] summary written to {}'.format(SUMMARY_FIG))


if __name__ == '__main__':
    main()

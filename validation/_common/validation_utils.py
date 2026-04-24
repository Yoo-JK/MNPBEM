import os
import subprocess

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_csv(filepath: str,
        x: np.ndarray,
        ys: Sequence[np.ndarray],
        col_names: Sequence[str],
        x_name: str = 'wavelength_nm') -> None:
    header = x_name + ',' + ','.join(col_names)
    n = x.shape[0]
    ncol = 1 + len(ys)
    data = np.empty((n, ncol), dtype = float)
    data[:, 0] = x
    for i, y in enumerate(ys):
        data[:, i + 1] = np.asarray(y).ravel()
    np.savetxt(filepath, data, delimiter = ',', header = header, comments = '', fmt = '%.10e')


def load_csv(filepath: str) -> Optional[np.ndarray]:
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath, delimiter = ',', skiprows = 1)


def save_timing(filepath: str, timings: Dict[str, float]) -> None:
    with open(filepath, 'w') as f:
        f.write('case,time_sec\n')
        for name, t in timings.items():
            f.write('{},{:.6f}\n'.format(name, t))


def compute_rms(py: np.ndarray, ml: np.ndarray) -> float:
    py = np.asarray(py).ravel()
    ml = np.asarray(ml).ravel()
    denom = np.maximum(np.abs(ml), 1e-30)
    rel = np.abs(py - ml) / denom
    return float(np.sqrt(np.mean(rel ** 2)))


def compute_rms_max(py_arr: np.ndarray, ml_arr: np.ndarray) -> float:
    py_arr = np.atleast_2d(py_arr)
    ml_arr = np.atleast_2d(ml_arr)
    if py_arr.ndim == 1:
        return compute_rms(py_arr, ml_arr)
    rms_vals = []
    for j in range(py_arr.shape[1]):
        rms_vals.append(compute_rms(py_arr[:, j], ml_arr[:, j]))
    return float(max(rms_vals))


def plot_spectrum(x: np.ndarray,
        ys: Sequence[np.ndarray],
        labels: Sequence[str],
        title: str,
        savepath: str,
        xlabel: str = 'Wavelength (nm)',
        ylabel: str = 'Cross section (nm$^2$)',
        styles: Optional[Sequence[str]] = None) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    if styles is None:
        styles = ['b-', 'r--', 'g:', 'k-.', 'm-', 'c--']
    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot(x, y, styles[i % len(styles)], linewidth = 1.5, label = lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc = 'best')
    ax.grid(True, alpha = 0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def plot_comparison(x: np.ndarray,
        py_cols: Sequence[np.ndarray],
        ml_cols: Sequence[np.ndarray],
        col_names: Sequence[str],
        title: str,
        savepath: str,
        t_py: float = 0.0,
        t_ml: float = 0.0,
        xlabel: str = 'Wavelength (nm)',
        ylabel: str = 'Cross section (nm$^2$)') -> Tuple[float, Dict[str, float]]:
    rms_dict = {}
    for name, py, ml in zip(col_names, py_cols, ml_cols):
        rms_dict[name] = compute_rms(py, ml)
    max_rms = max(rms_dict.values())

    fig, axes = plt.subplots(2, 1, figsize = (8, 8))
    colors = ['b', 'r', 'g', 'k', 'm', 'c']

    ax = axes[0]
    for i, (name, py, ml) in enumerate(zip(col_names, py_cols, ml_cols)):
        c = colors[i % len(colors)]
        ax.plot(x, ml, c + '-', linewidth = 1.5, label = 'MATLAB {}'.format(name))
        ax.plot(x, py, c + '--', linewidth = 1.5, label = 'Python {}'.format(name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    speed = t_ml / t_py if t_py > 0 else 0.0
    ax.set_title('{} | t_ML={:.3f}s, t_PY={:.3f}s (x{:.2f})'.format(title, t_ml, t_py, speed))
    ax.legend(loc = 'best', fontsize = 8)
    ax.grid(True, alpha = 0.3)

    ax = axes[1]
    for i, (name, py, ml) in enumerate(zip(col_names, py_cols, ml_cols)):
        c = colors[i % len(colors)]
        denom = np.maximum(np.abs(ml), 1e-30)
        rel = np.abs(py - ml) / denom
        ax.semilogy(x, rel, c + '-', linewidth = 1.5,
            label = '{} (RMS={:.2e})'.format(name, rms_dict[name]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('|Python - MATLAB| / |MATLAB|')
    ax.set_title('Relative error, max RMS = {:.2e}'.format(max_rms))
    ax.legend(loc = 'best', fontsize = 8)
    ax.grid(True, alpha = 0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)

    return max_rms, rms_dict


def run_matlab_script(script_path: str,
        cwd: Optional[str] = None,
        timeout: int = 3600) -> Tuple[int, str, str]:
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    script_dir = os.path.dirname(os.path.abspath(script_path))
    cmd = ['matlab', '-batch', 'cd(\'{}\'); {}'.format(script_dir, script_name)]
    result = subprocess.run(cmd,
        capture_output = True, text = True, timeout = timeout,
        cwd = cwd or script_dir)
    return result.returncode, result.stdout, result.stderr

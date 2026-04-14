import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok = True)


def load_csv(path: str) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter = ',', skip_header = 1)
    return {
        'wavelength': data[:, 0],
        'scattering': data[:, 1],
        'extinction': data[:, 2],
        'absorption': data[:, 3],
    }


def load_timing(path: str) -> Dict[str, float]:
    data = np.genfromtxt(path, delimiter = ',', skip_header = 1)
    return {
        'total_sec': float(data[0]),
        'n_wavelengths': int(data[1]),
        'per_wavelength_sec': float(data[2]),
    }


def relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(b), 1e-30)
    return np.abs(a - b) / denom


# ---- Load data ----

matlab_path = os.path.join(DATA_DIR, 'matlab_retlayer.csv')
python_path = os.path.join(DATA_DIR, 'python_retlayer.csv')
matlab_timing_path = os.path.join(DATA_DIR, 'matlab_retlayer_timing.csv')
python_timing_path = os.path.join(DATA_DIR, 'python_retlayer_timing.csv')

have_matlab = os.path.exists(matlab_path)
have_python = os.path.exists(python_path)
have_matlab_timing = os.path.exists(matlab_timing_path)
have_python_timing = os.path.exists(python_timing_path)

if not have_matlab and not have_python:
    print('[error] No data files found in {}'.format(DATA_DIR))
    sys.exit(1)


# ---- Figure 1: MATLAB results ----

if have_matlab:
    matlab = load_csv(matlab_path)
    wl_m = matlab['wavelength']

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    axes[0].plot(wl_m, matlab['extinction'], 'b-o', markersize = 4, label = 'Extinction')
    axes[0].plot(wl_m, matlab['scattering'], 'r-s', markersize = 4, label = 'Scattering')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Cross section (nm$^2$)')
    axes[0].set_title('MATLAB BEMRetLayer')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    axes[1].plot(wl_m, matlab['extinction'], 'b-o', markersize = 4, label = 'Extinction')
    axes[1].plot(wl_m, matlab['absorption'], 'g-^', markersize = 4, label = 'Absorption')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Cross section (nm$^2$)')
    axes[1].set_title('MATLAB BEMRetLayer (ext + abs)')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    fig.suptitle('20nm Au sphere on glass (n=1.5) - MATLAB retarded + layer + greentab',
        fontsize = 13, fontweight = 'bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'retlayer_matlab.png')
    plt.savefig(path, dpi = 150)
    plt.close()
    print('[info] Saved {}'.format(path))


# ---- Figure 2: Python results ----

if have_python:
    python = load_csv(python_path)
    wl_p = python['wavelength']

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    axes[0].plot(wl_p, python['extinction'], 'b-o', markersize = 4, label = 'Extinction')
    axes[0].plot(wl_p, python['scattering'], 'r-s', markersize = 4, label = 'Scattering')
    axes[0].set_xlabel('Wavelength (nm)')
    axes[0].set_ylabel('Cross section (nm$^2$)')
    axes[0].set_title('Python BEMRetLayer')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    axes[1].plot(wl_p, python['extinction'], 'b-o', markersize = 4, label = 'Extinction')
    axes[1].plot(wl_p, python['absorption'], 'g-^', markersize = 4, label = 'Absorption')
    axes[1].set_xlabel('Wavelength (nm)')
    axes[1].set_ylabel('Cross section (nm$^2$)')
    axes[1].set_title('Python BEMRetLayer (ext + abs)')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    fig.suptitle('20nm Au sphere on glass (n=1.5) - Python retarded + layer + greentab',
        fontsize = 13, fontweight = 'bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'retlayer_python.png')
    plt.savefig(path, dpi = 150)
    plt.close()
    print('[info] Saved {}'.format(path))


# ---- Figure 3: Comparison ----

if have_matlab and have_python:
    matlab = load_csv(matlab_path)
    python = load_csv(python_path)

    # Interpolate if grids differ
    if not np.allclose(matlab['wavelength'], python['wavelength'], atol = 0.01):
        print('[info] Wavelength grids differ; interpolating Python to MATLAB grid')
        for key in ['scattering', 'extinction', 'absorption']:
            python[key] = np.interp(matlab['wavelength'], python['wavelength'], python[key])
        python['wavelength'] = matlab['wavelength'].copy()

    wl = matlab['wavelength']

    fig, axes = plt.subplots(2, 2, figsize = (13, 10))

    # Ext+Sca overlay
    ax = axes[0, 0]
    ax.plot(wl, matlab['extinction'], 'bo-', markersize = 5, label = 'MATLAB ext')
    ax.plot(wl, python['extinction'], 'b^--', markersize = 5, label = 'Python ext')
    ax.plot(wl, matlab['scattering'], 'rs-', markersize = 5, label = 'MATLAB sca')
    ax.plot(wl, python['scattering'], 'rv--', markersize = 5, label = 'Python sca')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm$^2$)')
    ax.set_title('Extinction + Scattering')
    ax.legend(fontsize = 9)
    ax.grid(True, alpha = 0.3)

    # Absorption overlay
    ax = axes[0, 1]
    ax.plot(wl, matlab['absorption'], 'go-', markersize = 5, label = 'MATLAB')
    ax.plot(wl, python['absorption'], 'g^--', markersize = 5, label = 'Python')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm$^2$)')
    ax.set_title('Absorption')
    ax.legend()
    ax.grid(True, alpha = 0.3)

    # Relative error
    ax = axes[1, 0]
    for key, color in zip(['extinction', 'scattering', 'absorption'], ['b', 'r', 'g']):
        rel = relative_error(python[key], matlab[key]) * 100
        ax.semilogy(wl, rel, '-o', markersize = 4, color = color, label = key)
    ax.axhline(y = 5, color = 'k', linestyle = '--', alpha = 0.5, label = '5% limit')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Relative Error')
    ax.legend(fontsize = 9)
    ax.grid(True, alpha = 0.3)

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    lines = []
    all_pass = True

    for key in ['extinction', 'scattering', 'absorption']:
        rel = relative_error(python[key], matlab[key])
        rms = np.sqrt(np.mean(rel ** 2))
        max_err = np.max(rel)

        m_peak_idx = np.argmax(np.abs(matlab[key]))
        p_peak_idx = np.argmax(np.abs(python[key]))
        peak_shift = abs(wl[m_peak_idx] - wl[p_peak_idx])

        passed = (rms < 0.03) and (max_err < 0.05) and (peak_shift < 5.0)
        if not passed:
            all_pass = False
        status = 'PASS' if passed else 'FAIL'

        lines.append('[{}] {}'.format(status, key))
        lines.append('  RMS rel err: {:.4f}'.format(rms))
        lines.append('  Max rel err: {:.4f}'.format(max_err))
        lines.append('  Peak shift:  {:.1f} nm'.format(peak_shift))
        lines.append('')

    overall = 'ALL PASS' if all_pass else 'SOME FAIL'
    lines.insert(0, 'Overall: {}\n'.format(overall))
    ax.text(0.05, 0.95, '\n'.join(lines), transform = ax.transAxes,
        fontsize = 10, verticalalignment = 'top', fontfamily = 'monospace')

    fig.suptitle('BEMRetLayer validation: 20nm Au on glass - MATLAB vs Python',
        fontsize = 14, fontweight = 'bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'retlayer_comparison.png')
    plt.savefig(path, dpi = 150)
    plt.close()
    print('[info] Saved {}'.format(path))

    # Print summary to console
    print('\n' + '=' * 60)
    for line in lines:
        print('  {}'.format(line))
    print('=' * 60)


# ---- Figure 4: Timing bar chart ----

if have_matlab_timing or have_python_timing:
    labels = []
    totals = []
    per_wl = []
    colors = []

    if have_matlab_timing:
        mt = load_timing(matlab_timing_path)
        labels.append('MATLAB')
        totals.append(mt['total_sec'])
        per_wl.append(mt['per_wavelength_sec'])
        colors.append('#2196F3')

    if have_python_timing:
        pt = load_timing(python_timing_path)
        labels.append('Python')
        totals.append(pt['total_sec'])
        per_wl.append(pt['per_wavelength_sec'])
        colors.append('#FF9800')

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))

    x = np.arange(len(labels))
    width = 0.5

    # Total time
    ax = axes[0]
    bars = ax.bar(x, totals, width, color = colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (sec)')
    ax.set_title('Total simulation time')
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            '{:.1f}s'.format(val), ha = 'center', va = 'bottom', fontweight = 'bold')

    # Per-wavelength time
    ax = axes[1]
    bars = ax.bar(x, per_wl, width, color = colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (sec)')
    ax.set_title('Time per wavelength')
    for bar, val in zip(bars, per_wl):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            '{:.2f}s'.format(val), ha = 'center', va = 'bottom', fontweight = 'bold')

    fig.suptitle('BEMRetLayer timing: MATLAB vs Python (with greentab)',
        fontsize = 13, fontweight = 'bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'retlayer_timing.png')
    plt.savefig(path, dpi = 150)
    plt.close()
    print('[info] Saved {}'.format(path))

print('\n[info] All plots generated in {}'.format(FIG_DIR))

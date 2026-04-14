import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def load_csv(path: str) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter = ',', skip_header = 1)
    return {
        'wavelength': data[:, 0],
        'extinction': data[:, 1],
        'scattering': data[:, 2],
    }


def relative_error(
        a: np.ndarray,
        b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(b), 1e-30)
    return np.abs(a - b) / denom


def compare_case(
        case_name: str,
        matlab_path: str,
        python_path: str) -> Dict[str, Any]:

    matlab = load_csv(matlab_path)
    python = load_csv(python_path)

    # Interpolate if wavelength grids differ
    if not np.allclose(matlab['wavelength'], python['wavelength'], atol = 0.01):
        print('[info] Wavelength grids differ for {}. Interpolating.'.format(case_name))
        for key in ['extinction', 'scattering']:
            python[key] = np.interp(matlab['wavelength'], python['wavelength'], python[key])
        python['wavelength'] = matlab['wavelength'].copy()

    wl = matlab['wavelength']
    results = {}
    all_pass = True

    print('\n--- {} ---'.format(case_name))

    for key in ['extinction', 'scattering']:
        m = matlab[key]
        p = python[key]
        rel_err = relative_error(p, m)
        rms_err = np.sqrt(np.mean(rel_err ** 2))
        max_err = np.max(rel_err)

        m_peak_idx = np.argmax(np.abs(m))
        p_peak_idx = np.argmax(np.abs(p))
        peak_shift = abs(wl[m_peak_idx] - wl[p_peak_idx])

        # Pass/fail criteria
        pass_rms = rms_err < 0.03
        pass_max = np.all(rel_err < 0.05)
        pass_peak = peak_shift < 2.0

        passed = pass_rms and pass_max and pass_peak
        if not passed:
            all_pass = False

        results[key] = {
            'rms_err': rms_err,
            'max_err': max_err,
            'peak_shift': peak_shift,
            'passed': passed,
            'rel_err': rel_err,
        }

        status = 'PASS' if passed else 'FAIL'
        print('  [{}] {}:'.format(status, key))
        print('    RMS relative error: {:.6f} (limit: 0.03)'.format(rms_err))
        print('    Max relative error: {:.6f} (limit: 0.05)'.format(max_err))
        print('    Peak shift: {:.1f} nm (limit: 2.0 nm)'.format(peak_shift))
        print('    MATLAB peak at {:.1f} nm = {:.4e}'.format(wl[m_peak_idx], m[m_peak_idx]))
        print('    Python peak at {:.1f} nm = {:.4e}'.format(wl[p_peak_idx], p[p_peak_idx]))

    return {
        'matlab': matlab,
        'python': python,
        'results': results,
        'all_pass': all_pass,
        'wl': wl,
    }


def plot_comparison(
        case_data: Dict[str, Any],
        case_name: str,
        save_path: str) -> None:

    matlab = case_data['matlab']
    python = case_data['python']
    results = case_data['results']
    wl = case_data['wl']

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))

    # Extinction overlay
    ax = axes[0]
    ax.plot(wl, matlab['extinction'], 'b-o', markersize = 3, label = 'MATLAB')
    ax.plot(wl, python['extinction'], 'r-s', markersize = 3, label = 'Python')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm^2)')
    ax.set_title('Extinction')
    ax.legend()
    ax.grid(True, alpha = 0.3)

    # Scattering overlay
    ax = axes[1]
    ax.plot(wl, matlab['scattering'], 'b-o', markersize = 3, label = 'MATLAB')
    ax.plot(wl, python['scattering'], 'r-s', markersize = 3, label = 'Python')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm^2)')
    ax.set_title('Scattering')
    ax.legend()
    ax.grid(True, alpha = 0.3)

    # Relative error
    ax = axes[2]
    for key, color in zip(['extinction', 'scattering'], ['b', 'r']):
        ax.semilogy(wl, results[key]['rel_err'] * 100, '-', label = key, color = color)
    ax.axhline(y = 5, color = 'k', linestyle = '--', alpha = 0.5, label = '5% limit')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Relative Error')
    ax.legend()
    ax.grid(True, alpha = 0.3)

    status = 'PASS' if case_data['all_pass'] else 'FAIL'
    fig.suptitle('BEMStatLayer {} - [{}]'.format(case_name, status),
        fontsize = 14, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150)
    plt.close(fig)
    print('[info] Comparison figure saved: {}'.format(save_path))


def main() -> int:

    # Check all required files exist
    required_files = [
        ('matlab_normal.csv', DATA_DIR),
        ('matlab_oblique.csv', DATA_DIR),
        ('python_normal.csv', DATA_DIR),
        ('python_oblique.csv', DATA_DIR),
    ]

    for fname, ddir in required_files:
        fpath = os.path.join(ddir, fname)
        if not os.path.exists(fpath):
            print('[error] Required file not found: {}'.format(fpath))
            return 1

    # Load timing data
    print('=' * 60)
    print('BEMStatLayer Comparison: MATLAB vs Python')
    print('=' * 60)

    for lang in ['matlab', 'python']:
        timing_path = os.path.join(DATA_DIR, '{}_timing.csv'.format(lang))
        if os.path.exists(timing_path):
            timing = np.genfromtxt(timing_path, delimiter = ',', skip_header = 1,
                dtype = None, encoding = 'utf-8')
            print('\n{} timing:'.format(lang.upper()))
            for row in timing:
                print('  {}: {:.2f} sec'.format(row[0], float(row[1])))

    # Compare normal incidence
    normal_data = compare_case(
        'Normal incidence (theta=0)',
        os.path.join(DATA_DIR, 'matlab_normal.csv'),
        os.path.join(DATA_DIR, 'python_normal.csv'))

    plot_comparison(normal_data, 'Normal incidence',
        os.path.join(FIG_DIR, 'comparison_normal.png'))

    # Compare oblique incidence
    oblique_data = compare_case(
        'Oblique incidence (theta=45, TM)',
        os.path.join(DATA_DIR, 'matlab_oblique.csv'),
        os.path.join(DATA_DIR, 'python_oblique.csv'))

    plot_comparison(oblique_data, 'Oblique incidence (theta=45, TM)',
        os.path.join(FIG_DIR, 'comparison_oblique.png'))

    # Overall result
    overall_pass = normal_data['all_pass'] and oblique_data['all_pass']
    status = 'ALL PASS' if overall_pass else 'SOME FAIL'

    print('\n' + '=' * 60)
    print('Overall result: {}'.format(status))
    print('=' * 60)

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())

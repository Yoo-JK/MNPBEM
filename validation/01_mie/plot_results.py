"""
Plot Mie Theory Validation Results

Generates 9 figures (3 per sub-test):
  - *_matlab.png: MATLAB data only
  - *_python.png: Python data only
  - *_comparison.png: overlay + relative error

Sub-tests:
  1. MieStat: 20nm Au sphere
  2. MieRet: 100nm Au sphere
  3. MieGans: [20,10,10]nm ellipsoid (x-pol and z-pol extinction)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
fig_dir = os.path.join(base_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# Load timing
matlab_timing = {}
with open(os.path.join(data_dir, 'matlab_timing.csv'), 'r') as f:
    for line in f:
        if line.startswith('test'):
            continue
        parts = line.strip().split(',')
        if len(parts) == 2:
            matlab_timing[parts[0]] = float(parts[1])

python_timing = {}
with open(os.path.join(data_dir, 'python_timing.csv'), 'r') as f:
    for line in f:
        if line.startswith('test'):
            continue
        parts = line.strip().split(',')
        if len(parts) == 2:
            python_timing[parts[0]] = float(parts[1])


def load_csv_3col(filepath):
    """Load CSV with columns: wavelength_nm, extinction, scattering, absorption."""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def load_csv_gans(filepath):
    """Load CSV with columns: wavelength_nm, extinction_xpol, extinction_zpol."""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data[:, 0], data[:, 1], data[:, 2]


def plot_single(wavelength, ext, sca, abso, title, filename, color, marker, linestyle):
    """Plot a single dataset: ext, sca, abs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelength, ext, color=color, marker=marker, linestyle=linestyle,
            markersize=4, label='Extinction')
    ax.plot(wavelength, sca, color=color, marker=marker, linestyle=linestyle,
            markersize=4, alpha=0.7, label='Scattering')
    ax.plot(wavelength, abso, color=color, marker=marker, linestyle=linestyle,
            markersize=4, alpha=0.5, label='Absorption')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross Section (nm$^2$)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close(fig)
    print(f'  Saved: {filename}')


def plot_comparison(wl_m, ext_m, sca_m, abs_m,
                    wl_p, ext_p, sca_p, abs_p,
                    title, filename, t_matlab, t_python):
    """Plot comparison: overlay + relative error."""
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3, figure=fig)

    # Top: overlay
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(wl_m, ext_m, 'bo-', markersize=4, label='Extinction (MATLAB)')
    ax_top.plot(wl_p, ext_p, 'rs--', markersize=4, label='Extinction (Python)')
    ax_top.plot(wl_m, sca_m, 'bo-', markersize=3, alpha=0.5, label='Scattering (MATLAB)')
    ax_top.plot(wl_p, sca_p, 'rs--', markersize=3, alpha=0.5, label='Scattering (Python)')
    ax_top.plot(wl_m, abs_m, 'bo-', markersize=3, alpha=0.3, label='Absorption (MATLAB)')
    ax_top.plot(wl_p, abs_p, 'rs--', markersize=3, alpha=0.3, label='Absorption (Python)')
    ax_top.set_ylabel('Cross Section (nm$^2$)', fontsize=12)
    ax_top.set_title(f'{title}\nMATLAB: {t_matlab:.3f}s, Python: {t_python:.4f}s', fontsize=13)
    ax_top.legend(fontsize=8, ncol=2)
    ax_top.grid(True, alpha=0.3)

    # Bottom: relative error (%)
    ax_bot = fig.add_subplot(gs[1])
    eps = 1e-30  # avoid division by zero
    rel_err_ext = np.abs((ext_p - ext_m) / (np.abs(ext_m) + eps)) * 100
    rel_err_sca = np.abs((sca_p - sca_m) / (np.abs(sca_m) + eps)) * 100
    rel_err_abs = np.abs((abs_p - abs_m) / (np.abs(abs_m) + eps)) * 100

    ax_bot.semilogy(wl_m, rel_err_ext, 'k-o', markersize=3, label='Extinction')
    ax_bot.semilogy(wl_m, rel_err_sca, 'g-s', markersize=3, label='Scattering')
    ax_bot.semilogy(wl_m, rel_err_abs, 'm-^', markersize=3, label='Absorption')
    ax_bot.axhline(y=5.0, color='gray', linestyle='--', linewidth=1.5, label='5% threshold')
    ax_bot.set_xlabel('Wavelength (nm)', fontsize=12)
    ax_bot.set_ylabel('Relative Error (%)', fontsize=12)
    ax_bot.legend(fontsize=8, ncol=2)
    ax_bot.grid(True, alpha=0.3)

    fig.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close(fig)
    print(f'  Saved: {filename}')


def plot_gans_single(wavelength, ext_x, ext_z, title, filename, color, marker, linestyle):
    """Plot MieGans single dataset: x-pol and z-pol extinction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wavelength, ext_x, color=color, marker=marker, linestyle=linestyle,
            markersize=4, label='Extinction (x-pol)')
    ax.plot(wavelength, ext_z, color=color, marker=marker, linestyle=linestyle,
            markersize=4, alpha=0.6, label='Extinction (z-pol)')
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross Section (nm$^2$)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close(fig)
    print(f'  Saved: {filename}')


def plot_gans_comparison(wl_m, extx_m, extz_m,
                         wl_p, extx_p, extz_p,
                         title, filename, t_matlab, t_python):
    """Plot MieGans comparison: overlay + relative error."""
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3, figure=fig)

    # Top: overlay
    ax_top = fig.add_subplot(gs[0])
    ax_top.plot(wl_m, extx_m, 'bo-', markersize=4, label='x-pol Ext (MATLAB)')
    ax_top.plot(wl_p, extx_p, 'rs--', markersize=4, label='x-pol Ext (Python)')
    ax_top.plot(wl_m, extz_m, 'bo-', markersize=3, alpha=0.5, label='z-pol Ext (MATLAB)')
    ax_top.plot(wl_p, extz_p, 'rs--', markersize=3, alpha=0.5, label='z-pol Ext (Python)')
    ax_top.set_ylabel('Cross Section (nm$^2$)', fontsize=12)
    ax_top.set_title(f'{title}\nMATLAB: {t_matlab:.3f}s, Python: {t_python:.4f}s', fontsize=13)
    ax_top.legend(fontsize=9, ncol=2)
    ax_top.grid(True, alpha=0.3)

    # Bottom: relative error (%)
    ax_bot = fig.add_subplot(gs[1])
    eps = 1e-30
    rel_err_x = np.abs((extx_p - extx_m) / (np.abs(extx_m) + eps)) * 100
    rel_err_z = np.abs((extz_p - extz_m) / (np.abs(extz_m) + eps)) * 100

    ax_bot.semilogy(wl_m, rel_err_x, 'k-o', markersize=3, label='x-pol Extinction')
    ax_bot.semilogy(wl_m, rel_err_z, 'g-s', markersize=3, label='z-pol Extinction')
    ax_bot.axhline(y=5.0, color='gray', linestyle='--', linewidth=1.5, label='5% threshold')
    ax_bot.set_xlabel('Wavelength (nm)', fontsize=12)
    ax_bot.set_ylabel('Relative Error (%)', fontsize=12)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    fig.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close(fig)
    print(f'  Saved: {filename}')


# =============================================================================
# 1. MieStat
# =============================================================================
print('=== Plotting MieStat ===')
wl_m, ext_m, sca_m, abs_m = load_csv_3col(os.path.join(data_dir, 'miestat_matlab.csv'))
wl_p, ext_p, sca_p, abs_p = load_csv_3col(os.path.join(data_dir, 'miestat_python.csv'))

plot_single(wl_m, ext_m, sca_m, abs_m,
            'MieStat: 20nm Au Sphere (MATLAB)', 'miestat_matlab.png',
            'blue', 'o', '-')
plot_single(wl_p, ext_p, sca_p, abs_p,
            'MieStat: 20nm Au Sphere (Python)', 'miestat_python.png',
            'red', 's', '--')
plot_comparison(wl_m, ext_m, sca_m, abs_m,
                wl_p, ext_p, sca_p, abs_p,
                'MieStat: 20nm Au Sphere', 'miestat_comparison.png',
                matlab_timing['miestat'], python_timing['miestat'])

# =============================================================================
# 2. MieRet
# =============================================================================
print('=== Plotting MieRet ===')
wl_m, ext_m, sca_m, abs_m = load_csv_3col(os.path.join(data_dir, 'mieret_matlab.csv'))
wl_p, ext_p, sca_p, abs_p = load_csv_3col(os.path.join(data_dir, 'mieret_python.csv'))

plot_single(wl_m, ext_m, sca_m, abs_m,
            'MieRet: 100nm Au Sphere (MATLAB)', 'mieret_matlab.png',
            'blue', 'o', '-')
plot_single(wl_p, ext_p, sca_p, abs_p,
            'MieRet: 100nm Au Sphere (Python)', 'mieret_python.png',
            'red', 's', '--')
plot_comparison(wl_m, ext_m, sca_m, abs_m,
                wl_p, ext_p, sca_p, abs_p,
                'MieRet: 100nm Au Sphere', 'mieret_comparison.png',
                matlab_timing['mieret'], python_timing['mieret'])

# =============================================================================
# 3. MieGans
# =============================================================================
print('=== Plotting MieGans ===')
wl_m, extx_m, extz_m = load_csv_gans(os.path.join(data_dir, 'miegans_matlab.csv'))
wl_p, extx_p, extz_p = load_csv_gans(os.path.join(data_dir, 'miegans_python.csv'))

plot_gans_single(wl_m, extx_m, extz_m,
                 'MieGans: [20,10,10]nm Au Ellipsoid (MATLAB)', 'miegans_matlab.png',
                 'blue', 'o', '-')
plot_gans_single(wl_p, extx_p, extz_p,
                 'MieGans: [20,10,10]nm Au Ellipsoid (Python)', 'miegans_python.png',
                 'red', 's', '--')
plot_gans_comparison(wl_m, extx_m, extz_m,
                     wl_p, extx_p, extz_p,
                     'MieGans: [20,10,10]nm Au Ellipsoid', 'miegans_comparison.png',
                     matlab_timing['miegans'], python_timing['miegans'])

print(f'\nAll 9 figures saved to: {fig_dir}')
print('Done.')

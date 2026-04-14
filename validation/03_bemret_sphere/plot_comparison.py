"""
BEMRet Sphere Validation - Comparison Plots

Generates 3 images:
  1. bemret_matlab.png   - MATLAB BEMRet + MieRet overlay
  2. bemret_python.png   - Python BEMRet + MieRet overlay
  3. bemret_comparison.png - MATLAB vs Python vs MieRet side-by-side
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = '/home/yoojk20/workspace/MNPBEM/validation/03_bemret_sphere/data'
fig_dir = '/home/yoojk20/workspace/MNPBEM/validation/03_bemret_sphere/figures'


def load_csv(filename):
    """Load CSV with columns: wavelength, extinction, scattering, absorption."""
    path = os.path.join(data_dir, filename)
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    return {
        'wavelength': data[:, 0],
        'extinction': data[:, 1],
        'scattering': data[:, 2],
        'absorption': data[:, 3],
    }


def load_timing(filename):
    """Load timing CSV with columns: method, time_seconds."""
    path = os.path.join(data_dir, filename)
    timing = {}
    with open(path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            timing[parts[0]] = float(parts[1])
    return timing


# Load data
matlab_bem = load_csv('matlab_bemret.csv')
matlab_mie = load_csv('matlab_mie.csv')
python_bem = load_csv('python_bemret.csv')
python_mie = load_csv('python_mie.csv')
matlab_time = load_timing('matlab_timing.csv')
python_time = load_timing('python_timing.csv')


# ============================================================
# Plot 1: MATLAB BEMRet + MieRet overlay
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MATLAB BEMRet Sphere (20nm Au, trisphere 144)', fontsize=14, fontweight='bold')

titles = ['Extinction', 'Scattering', 'Absorption']
keys = ['extinction', 'scattering', 'absorption']

for ax, title, key in zip(axes, titles, keys):
    ax.plot(matlab_bem['wavelength'], matlab_bem[key], 'o-', ms=3, lw=1.2,
            label='BEMRet', color='#1f77b4')
    ax.plot(matlab_mie['wavelength'], matlab_mie[key], '--', lw=2.0,
            label='MieRet', color='#ff7f0e')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(f'{title} cross section (nm$^2$)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

timing_text = f'BEMRet: {matlab_time["bemret"]:.2f}s / MieRet: {matlab_time["mieret"]:.2f}s'
fig.text(0.5, 0.01, timing_text, ha='center', fontsize=10, style='italic')
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(fig_dir, 'bemret_matlab.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {os.path.join(fig_dir, "bemret_matlab.png")}')


# ============================================================
# Plot 2: Python BEMRet + MieRet overlay
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Python BEMRet Sphere (20nm Au, trisphere 144)', fontsize=14, fontweight='bold')

for ax, title, key in zip(axes, titles, keys):
    ax.plot(python_bem['wavelength'], python_bem[key], 'o-', ms=3, lw=1.2,
            label='BEMRet', color='#2ca02c')
    ax.plot(python_mie['wavelength'], python_mie[key], '--', lw=2.0,
            label='MieRet', color='#d62728')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(f'{title} cross section (nm$^2$)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

timing_text = f'BEMRet: {python_time["bemret"]:.2f}s / MieRet: {python_time["mieret"]:.2f}s'
fig.text(0.5, 0.01, timing_text, ha='center', fontsize=10, style='italic')
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(fig_dir, 'bemret_python.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {os.path.join(fig_dir, "bemret_python.png")}')


# ============================================================
# Plot 3: MATLAB vs Python vs MieRet comparison
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('BEMRet Sphere Comparison: MATLAB vs Python (20nm Au, trisphere 144)',
             fontsize=14, fontweight='bold')

colors_matlab = '#1f77b4'
colors_python = '#2ca02c'
colors_mie = '#ff7f0e'

# Top row: cross sections overlay
for ax, title, key in zip(axes[0], titles, keys):
    ax.plot(matlab_bem['wavelength'], matlab_bem[key], 'o-', ms=3, lw=1.2,
            label='MATLAB BEMRet', color=colors_matlab)
    ax.plot(python_bem['wavelength'], python_bem[key], 's-', ms=3, lw=1.2,
            label='Python BEMRet', color=colors_python)
    ax.plot(matlab_mie['wavelength'], matlab_mie[key], '--', lw=2.0,
            label='MATLAB MieRet', color=colors_mie)
    ax.plot(python_mie['wavelength'], python_mie[key], ':', lw=2.0,
            label='Python MieRet', color='#d62728')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(f'{title} cross section (nm$^2$)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Bottom row: relative errors
for ax, title, key in zip(axes[1], titles, keys):
    # MATLAB BEMRet vs MATLAB MieRet
    ref_matlab = matlab_mie[key]
    mask_matlab = np.abs(ref_matlab) > 1e-30
    err_matlab = np.full_like(ref_matlab, np.nan)
    err_matlab[mask_matlab] = np.abs(
        (matlab_bem[key][mask_matlab] - ref_matlab[mask_matlab]) / ref_matlab[mask_matlab]
    ) * 100

    # Python BEMRet vs Python MieRet
    ref_python = python_mie[key]
    mask_python = np.abs(ref_python) > 1e-30
    err_python = np.full_like(ref_python, np.nan)
    err_python[mask_python] = np.abs(
        (python_bem[key][mask_python] - ref_python[mask_python]) / ref_python[mask_python]
    ) * 100

    # MATLAB BEMRet vs Python BEMRet
    ref_cross = matlab_bem[key]
    mask_cross = np.abs(ref_cross) > 1e-30
    err_cross = np.full_like(ref_cross, np.nan)
    err_cross[mask_cross] = np.abs(
        (python_bem[key][mask_cross] - ref_cross[mask_cross]) / ref_cross[mask_cross]
    ) * 100

    ax.plot(matlab_bem['wavelength'], err_matlab, 'o-', ms=3, lw=1.2,
            label='MATLAB BEM vs Mie', color=colors_matlab)
    ax.plot(python_bem['wavelength'], err_python, 's-', ms=3, lw=1.2,
            label='Python BEM vs Mie', color=colors_python)
    ax.plot(matlab_bem['wavelength'], err_cross, '^-', ms=3, lw=1.2,
            label='Python vs MATLAB BEM', color='#9467bd')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'{title} Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

# Timing annotation
timing_text = (
    f'MATLAB: BEMRet={matlab_time["bemret"]:.2f}s, MieRet={matlab_time["mieret"]:.2f}s  |  '
    f'Python: BEMRet={python_time["bemret"]:.2f}s, MieRet={python_time["mieret"]:.2f}s'
)
fig.text(0.5, 0.01, timing_text, ha='center', fontsize=10, style='italic')
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(fig_dir, 'bemret_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {os.path.join(fig_dir, "bemret_comparison.png")}')

# Print summary statistics
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)

for key, title in zip(keys, titles):
    # MATLAB BEM vs Mie
    ref = matlab_mie[key]
    mask = np.abs(ref) > 1e-30
    if np.any(mask):
        err = np.abs((matlab_bem[key][mask] - ref[mask]) / ref[mask]) * 100
        print(f'  MATLAB BEM vs Mie {title}: max={np.max(err):.4f}%, mean={np.mean(err):.4f}%')

    # Python BEM vs Mie
    ref = python_mie[key]
    mask = np.abs(ref) > 1e-30
    if np.any(mask):
        err = np.abs((python_bem[key][mask] - ref[mask]) / ref[mask]) * 100
        print(f'  Python BEM vs Mie {title}: max={np.max(err):.4f}%, mean={np.mean(err):.4f}%')

    # Python vs MATLAB BEM
    ref = matlab_bem[key]
    mask = np.abs(ref) > 1e-30
    if np.any(mask):
        err = np.abs((python_bem[key][mask] - ref[mask]) / ref[mask]) * 100
        print(f'  Python vs MATLAB BEM {title}: max={np.max(err):.4f}%, mean={np.mean(err):.4f}%')
    print()

print(f'Timing:')
print(f'  MATLAB BEMRet: {matlab_time["bemret"]:.3f}s')
print(f'  Python BEMRet: {python_time["bemret"]:.3f}s')
print(f'  MATLAB MieRet: {matlab_time["mieret"]:.3f}s')
print(f'  Python MieRet: {python_time["mieret"]:.3f}s')

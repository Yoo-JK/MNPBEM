import os
import sys
import time

from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import EELSStat, EELSRet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok = True)
os.makedirs(FIG_DIR, exist_ok = True)


# ==============================================================================
# common setup
# ==============================================================================

epstab = [EpsConst(1.0), EpsTable('gold.dat')]
sphere = trisphere(144, 20.0)
p = ComParticle(epstab, [sphere], [[2, 1]], [1])

enei_arr = np.linspace(450, 650, 21)
WIDTH = 0.5
VEL = 0.5

IMP_SCAN = np.array([5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0])
LAMBDA_MAP = 520.0

timings = {}


# ==============================================================================
# Test 1: EELSStat spectrum
# ==============================================================================

print('=' * 60)
print('Test 1: EELSStat spectrum (impact=[15,0], 450-650nm, 21pt)')
print('=' * 60)

t0 = time.time()

bem_s = BEMStat(p)
eels_s = EELSStat(p, impact = np.array([[15.0, 0.0]]), width = WIDTH, vel = VEL)

psurf_stat = np.empty(len(enei_arr))
pbulk_stat = np.empty(len(enei_arr))

for i, enei in enumerate(enei_arr):
    exc = eels_s(p, enei)
    sig, _ = bem_s.solve(exc)
    ps, pb = eels_s.loss(sig)
    psurf_stat[i] = np.real(np.ravel(ps)[0])
    pbulk_stat[i] = np.real(np.ravel(pb)[0])
    print('  [{:2d}/{:2d}] lambda={:.1f}nm  psurf={:.6e}  pbulk={:.6e}'.format(
        i + 1, len(enei_arr), enei, psurf_stat[i], pbulk_stat[i]))

t_stat = time.time() - t0
timings['stat_spectrum'] = t_stat
print('[info] EELSStat spectrum: {:.3f} s'.format(t_stat))


# ==============================================================================
# Test 2: EELSRet spectrum
# ==============================================================================

print()
print('=' * 60)
print('Test 2: EELSRet spectrum (impact=[15,0], 450-650nm, 21pt)')
print('=' * 60)

t0 = time.time()

bem_r = BEMRet(p)
eels_r = EELSRet(p, impact = np.array([[15.0, 0.0]]), width = WIDTH, vel = VEL)

psurf_ret = np.empty(len(enei_arr))
pbulk_ret = np.empty(len(enei_arr))

for i, enei in enumerate(enei_arr):
    exc = eels_r(p, enei)
    sig, _ = bem_r.solve(exc)
    ps, pb = eels_r.loss(sig)
    psurf_ret[i] = np.real(np.ravel(ps)[0])
    pbulk_ret[i] = np.real(np.ravel(pb)[0])
    print('  [{:2d}/{:2d}] lambda={:.1f}nm  psurf={:.6e}  pbulk={:.6e}'.format(
        i + 1, len(enei_arr), enei, psurf_ret[i], pbulk_ret[i]))

t_ret = time.time() - t0
timings['ret_spectrum'] = t_ret
print('[info] EELSRet spectrum: {:.3f} s'.format(t_ret))


# ==============================================================================
# Test 3: Loss map (impact scan at lambda=520nm)
# ==============================================================================

print()
print('=' * 60)
print('Test 3: Loss map (impact scan at lambda={}nm)'.format(LAMBDA_MAP))
print('=' * 60)

t0 = time.time()

# stat map
psurf_map_stat = np.empty(len(IMP_SCAN))
pbulk_map_stat = np.empty(len(IMP_SCAN))

for j, imp in enumerate(IMP_SCAN):
    eels_map_s = EELSStat(p, impact = np.array([[imp, 0.0]]), width = WIDTH, vel = VEL)
    exc = eels_map_s(p, LAMBDA_MAP)
    sig, _ = bem_s.solve(exc)
    ps, pb = eels_map_s.loss(sig)
    psurf_map_stat[j] = np.real(np.ravel(ps)[0])
    pbulk_map_stat[j] = np.real(np.ravel(pb)[0])
    print('  [stat] impact={:5.1f}nm  psurf={:.6e}  pbulk={:.6e}'.format(
        imp, psurf_map_stat[j], pbulk_map_stat[j]))

# ret map
psurf_map_ret = np.empty(len(IMP_SCAN))
pbulk_map_ret = np.empty(len(IMP_SCAN))

for j, imp in enumerate(IMP_SCAN):
    eels_map_r = EELSRet(p, impact = np.array([[imp, 0.0]]), width = WIDTH, vel = VEL)
    exc = eels_map_r(p, LAMBDA_MAP)
    sig, _ = bem_r.solve(exc)
    ps, pb = eels_map_r.loss(sig)
    psurf_map_ret[j] = np.real(np.ravel(ps)[0])
    pbulk_map_ret[j] = np.real(np.ravel(pb)[0])
    print('  [ret]  impact={:5.1f}nm  psurf={:.6e}  pbulk={:.6e}'.format(
        imp, psurf_map_ret[j], pbulk_map_ret[j]))

t_map = time.time() - t0
timings['map'] = t_map
print('[info] Loss map: {:.3f} s'.format(t_map))


# ==============================================================================
# Save Python CSVs
# ==============================================================================

print()
print('=' * 60)
print('Saving Python CSVs')
print('=' * 60)

df_spectrum = pd.DataFrame({
    'wavelength_nm': enei_arr,
    'psurf_stat': psurf_stat,
    'pbulk_stat': pbulk_stat,
    'psurf_ret': psurf_ret,
    'pbulk_ret': pbulk_ret,
})
df_spectrum.to_csv(os.path.join(DATA_DIR, 'python_eels_spectrum.csv'), index = False)
print('[info] Saved python_eels_spectrum.csv')

df_map = pd.DataFrame({
    'impact_nm': IMP_SCAN,
    'psurf_stat': psurf_map_stat,
    'pbulk_stat': pbulk_map_stat,
    'psurf_ret': psurf_map_ret,
    'pbulk_ret': pbulk_map_ret,
})
df_map.to_csv(os.path.join(DATA_DIR, 'python_eels_map.csv'), index = False)
print('[info] Saved python_eels_map.csv')

df_timing = pd.DataFrame({
    'test': list(timings.keys()),
    'time_s': list(timings.values()),
})
df_timing.to_csv(os.path.join(DATA_DIR, 'python_eels_timing.csv'), index = False)
print('[info] Saved python_eels_timing.csv')


# ==============================================================================
# comparison with MATLAB (if data exists)
# ==============================================================================

matlab_spectrum_path = os.path.join(DATA_DIR, 'matlab_eels_spectrum.csv')
matlab_map_path = os.path.join(DATA_DIR, 'matlab_eels_map.csv')

has_matlab = os.path.exists(matlab_spectrum_path) and os.path.exists(matlab_map_path)

if has_matlab:
    print()
    print('=' * 60)
    print('MATLAB comparison')
    print('=' * 60)

    df_m_spec = pd.read_csv(matlab_spectrum_path)
    df_m_map = pd.read_csv(matlab_map_path)

    # compute relative errors
    m_psurf_stat = df_m_spec['psurf_stat'].values
    m_pbulk_stat = df_m_spec['pbulk_stat'].values
    m_psurf_ret = df_m_spec['psurf_ret'].values
    m_pbulk_ret = df_m_spec['pbulk_ret'].values

    def rel_err(p_val: np.ndarray, m_val: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(m_val), 1e-30)
        return np.abs(p_val - m_val) / denom

    err_psurf_stat = rel_err(psurf_stat, m_psurf_stat)
    err_psurf_ret = rel_err(psurf_ret, m_psurf_ret)

    print('  psurf_stat  max_rel_err = {:.2e}  mean_rel_err = {:.2e}'.format(
        np.max(err_psurf_stat), np.mean(err_psurf_stat)))
    print('  psurf_ret   max_rel_err = {:.2e}  mean_rel_err = {:.2e}'.format(
        np.max(err_psurf_ret), np.mean(err_psurf_ret)))

    m_map_psurf_stat = df_m_map['psurf_stat'].values
    m_map_psurf_ret = df_m_map['psurf_ret'].values
    err_map_stat = rel_err(psurf_map_stat, m_map_psurf_stat)
    err_map_ret = rel_err(psurf_map_ret, m_map_psurf_ret)

    print('  map_stat    max_rel_err = {:.2e}  mean_rel_err = {:.2e}'.format(
        np.max(err_map_stat), np.mean(err_map_stat)))
    print('  map_ret     max_rel_err = {:.2e}  mean_rel_err = {:.2e}'.format(
        np.max(err_map_ret), np.mean(err_map_ret)))
else:
    print()
    print('[info] MATLAB reference data not found -- skipping comparison.')
    print('[info] Run generate_matlab_data.m in MATLAB first.')

    # placeholder for plotting: set MATLAB values to None
    m_psurf_stat = None
    m_pbulk_stat = None
    m_psurf_ret = None
    m_pbulk_ret = None
    m_map_psurf_stat = None
    m_map_psurf_ret = None


# ==============================================================================
# Figure 1: EELSStat spectrum  (MATLAB / Python / comparison)
# ==============================================================================

print()
print('=' * 60)
print('Generating figures')
print('=' * 60)

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

# -- panel (a): MATLAB --
ax = axes[0]
ax.set_title('EELSStat (MATLAB)')
if m_psurf_stat is not None:
    ax.plot(enei_arr, m_psurf_stat, 'o-', label = 'psurf', markersize = 4)
    ax.plot(enei_arr, m_pbulk_stat, 's--', label = 'pbulk', markersize = 4)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

# -- panel (b): Python --
ax = axes[1]
ax.set_title('EELSStat (Python)')
ax.plot(enei_arr, psurf_stat, 'o-', label = 'psurf', markersize = 4)
ax.plot(enei_arr, pbulk_stat, 's--', label = 'pbulk', markersize = 4)
ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

# -- panel (c): comparison --
ax = axes[2]
ax.set_title('EELSStat comparison')
if m_psurf_stat is not None:
    ax.plot(enei_arr, psurf_stat, 'b-', label = 'Python psurf', linewidth = 2)
    ax.plot(enei_arr, m_psurf_stat, 'r--', label = 'MATLAB psurf', linewidth = 2)
    ax.legend()
else:
    ax.plot(enei_arr, psurf_stat, 'b-', label = 'Python psurf', linewidth = 2)
    ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_stat_spectrum.png'), dpi = 150)
plt.close()
print('[info] Saved eels_stat_spectrum.png')


# ==============================================================================
# Figure 2: EELSRet spectrum  (MATLAB / Python / comparison)
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

ax = axes[0]
ax.set_title('EELSRet (MATLAB)')
if m_psurf_ret is not None:
    ax.plot(enei_arr, m_psurf_ret, 'o-', label = 'psurf', markersize = 4)
    ax.plot(enei_arr, m_pbulk_ret, 's--', label = 'pbulk', markersize = 4)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[1]
ax.set_title('EELSRet (Python)')
ax.plot(enei_arr, psurf_ret, 'o-', label = 'psurf', markersize = 4)
ax.plot(enei_arr, pbulk_ret, 's--', label = 'pbulk', markersize = 4)
ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[2]
ax.set_title('EELSRet comparison')
if m_psurf_ret is not None:
    ax.plot(enei_arr, psurf_ret, 'b-', label = 'Python psurf', linewidth = 2)
    ax.plot(enei_arr, m_psurf_ret, 'r--', label = 'MATLAB psurf', linewidth = 2)
    ax.legend()
else:
    ax.plot(enei_arr, psurf_ret, 'b-', label = 'Python psurf', linewidth = 2)
    ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_ret_spectrum.png'), dpi = 150)
plt.close()
print('[info] Saved eels_ret_spectrum.png')


# ==============================================================================
# Figure 3: Loss map  (MATLAB / Python / difference)
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

# -- panel (a): MATLAB --
ax = axes[0]
ax.set_title('Loss map (MATLAB)')
if m_map_psurf_stat is not None:
    ax.plot(IMP_SCAN, m_map_psurf_stat, 'o-', label = 'stat psurf', markersize = 5)
    ax.plot(IMP_SCAN, m_map_psurf_ret, 's-', label = 'ret psurf', markersize = 5)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Impact parameter (nm)')
ax.set_ylabel('Loss probability (1/eV)')

# -- panel (b): Python --
ax = axes[1]
ax.set_title('Loss map (Python)')
ax.plot(IMP_SCAN, psurf_map_stat, 'o-', label = 'stat psurf', markersize = 5)
ax.plot(IMP_SCAN, psurf_map_ret, 's-', label = 'ret psurf', markersize = 5)
ax.legend()
ax.set_xlabel('Impact parameter (nm)')
ax.set_ylabel('Loss probability (1/eV)')

# -- panel (c): comparison / difference --
ax = axes[2]
ax.set_title('Loss map comparison')
if m_map_psurf_stat is not None:
    diff_stat = psurf_map_stat - m_map_psurf_stat
    diff_ret = psurf_map_ret - m_map_psurf_ret
    ax.plot(IMP_SCAN, diff_stat, 'o-', label = 'stat diff (P-M)', markersize = 5)
    ax.plot(IMP_SCAN, diff_ret, 's-', label = 'ret diff (P-M)', markersize = 5)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    ax.legend()
else:
    ax.plot(IMP_SCAN, psurf_map_stat, 'b-o', label = 'stat (Python)', markersize = 5)
    ax.plot(IMP_SCAN, psurf_map_ret, 'r-s', label = 'ret (Python)', markersize = 5)
    ax.legend()
ax.set_xlabel('Impact parameter (nm)')
ax.set_ylabel('Loss probability diff (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_loss_map.png'), dpi = 150)
plt.close()
print('[info] Saved eels_loss_map.png')


# ==============================================================================
# Figure 4-6: 2D heatmap for loss map (stat, ret, both)
# ==============================================================================

# build symmetric 2D map: mirror impact along x-axis
imp_2d = np.empty(len(IMP_SCAN) * 2 - 1)
psurf_2d_stat = np.empty(len(IMP_SCAN) * 2 - 1)
psurf_2d_ret = np.empty(len(IMP_SCAN) * 2 - 1)

n_imp = len(IMP_SCAN)
for k in range(n_imp):
    imp_2d[n_imp - 1 + k] = IMP_SCAN[k]
    imp_2d[n_imp - 1 - k] = -IMP_SCAN[k]
    psurf_2d_stat[n_imp - 1 + k] = psurf_map_stat[k]
    psurf_2d_stat[n_imp - 1 - k] = psurf_map_stat[k]
    psurf_2d_ret[n_imp - 1 + k] = psurf_map_ret[k]
    psurf_2d_ret[n_imp - 1 - k] = psurf_map_ret[k]

# create 2D grid (x=impact, y=impact, value=product for radial-ish view)
XX, YY = np.meshgrid(imp_2d, imp_2d)
RR = np.sqrt(XX ** 2 + YY ** 2)

# interpolate loss vs radial distance
from scipy.interpolate import interp1d
interp_stat = interp1d(IMP_SCAN, psurf_map_stat, kind = 'cubic',
                       bounds_error = False, fill_value = 0.0)
interp_ret = interp1d(IMP_SCAN, psurf_map_ret, kind = 'cubic',
                      bounds_error = False, fill_value = 0.0)
ZZ_stat = interp_stat(RR)
ZZ_ret = interp_ret(RR)

# mask inside sphere (r < 10nm)
mask_inside = RR < 10.0
ZZ_stat[mask_inside] = np.nan
ZZ_ret[mask_inside] = np.nan


def plot_heatmap(ax: plt.Axes, XX: np.ndarray, YY: np.ndarray, ZZ: np.ndarray,
        title: str, cmap: str = 'hot') -> None:

    im = ax.pcolormesh(XX, YY, ZZ, cmap = cmap, shading = 'auto')
    plt.colorbar(im, ax = ax, label = 'Loss probability (1/eV)')

    # draw sphere outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(10 * np.cos(theta), 10 * np.sin(theta), 'w-', linewidth = 1.5)

    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title(title)
    ax.set_aspect('equal')


# Figure 4: stat heatmap
fig, axes = plt.subplots(1, 3, figsize = (20, 6))

if has_matlab:
    m_interp_stat = interp1d(IMP_SCAN, df_m_map['psurf_stat'].values, kind = 'cubic',
                             bounds_error = False, fill_value = 0.0)
    ZZ_m_stat = m_interp_stat(RR)
    ZZ_m_stat[mask_inside] = np.nan
    plot_heatmap(axes[0], XX, YY, ZZ_m_stat, 'EELSStat map (MATLAB)')
else:
    axes[0].text(0.5, 0.5, 'No MATLAB data', transform = axes[0].transAxes,
                 ha = 'center', va = 'center', fontsize = 14, color = 'gray')
    axes[0].set_title('EELSStat map (MATLAB)')

plot_heatmap(axes[1], XX, YY, ZZ_stat, 'EELSStat map (Python)')

if has_matlab:
    ZZ_diff_stat = ZZ_stat - ZZ_m_stat
    plot_heatmap(axes[2], XX, YY, ZZ_diff_stat, 'EELSStat diff (P-M)', cmap = 'RdBu_r')
else:
    plot_heatmap(axes[2], XX, YY, ZZ_stat, 'EELSStat map (Python only)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_stat_map_2d.png'), dpi = 150)
plt.close()
print('[info] Saved eels_stat_map_2d.png')

# Figure 5: ret heatmap
fig, axes = plt.subplots(1, 3, figsize = (20, 6))

if has_matlab:
    m_interp_ret = interp1d(IMP_SCAN, df_m_map['psurf_ret'].values, kind = 'cubic',
                            bounds_error = False, fill_value = 0.0)
    ZZ_m_ret = m_interp_ret(RR)
    ZZ_m_ret[mask_inside] = np.nan
    plot_heatmap(axes[0], XX, YY, ZZ_m_ret, 'EELSRet map (MATLAB)')
else:
    axes[0].text(0.5, 0.5, 'No MATLAB data', transform = axes[0].transAxes,
                 ha = 'center', va = 'center', fontsize = 14, color = 'gray')
    axes[0].set_title('EELSRet map (MATLAB)')

plot_heatmap(axes[1], XX, YY, ZZ_ret, 'EELSRet map (Python)')

if has_matlab:
    ZZ_diff_ret = ZZ_ret - ZZ_m_ret
    plot_heatmap(axes[2], XX, YY, ZZ_diff_ret, 'EELSRet diff (P-M)', cmap = 'RdBu_r')
else:
    plot_heatmap(axes[2], XX, YY, ZZ_ret, 'EELSRet map (Python only)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_ret_map_2d.png'), dpi = 150)
plt.close()
print('[info] Saved eels_ret_map_2d.png')

# Figure 6: combined stat+ret comparison
fig, axes = plt.subplots(1, 3, figsize = (20, 6))

if has_matlab:
    plot_heatmap(axes[0], XX, YY, ZZ_m_stat, 'MATLAB stat map')
    plot_heatmap(axes[1], XX, YY, ZZ_m_ret, 'MATLAB ret map')
    ZZ_stat_ret_diff = ZZ_stat - ZZ_ret
    plot_heatmap(axes[2], XX, YY, ZZ_stat_ret_diff, 'Python stat-ret diff', cmap = 'RdBu_r')
else:
    plot_heatmap(axes[0], XX, YY, ZZ_stat, 'Python stat map')
    plot_heatmap(axes[1], XX, YY, ZZ_ret, 'Python ret map')
    ZZ_stat_ret_diff = ZZ_stat - ZZ_ret
    plot_heatmap(axes[2], XX, YY, ZZ_stat_ret_diff, 'stat-ret diff', cmap = 'RdBu_r')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_combined_map_2d.png'), dpi = 150)
plt.close()
print('[info] Saved eels_combined_map_2d.png')


# ==============================================================================
# Figure 7-9: additional comparison panels
# ==============================================================================

# Figure 7: stat overlay with bulk+surface
fig, axes = plt.subplots(1, 3, figsize = (18, 5))

ax = axes[0]
ax.set_title('EELSStat total loss (MATLAB)')
if has_matlab:
    ax.plot(enei_arr, m_psurf_stat + m_pbulk_stat, 'k-o', label = 'total', markersize = 4)
    ax.plot(enei_arr, m_psurf_stat, 'b--', label = 'surface', markersize = 3)
    ax.plot(enei_arr, m_pbulk_stat, 'r--', label = 'bulk', markersize = 3)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[1]
ax.set_title('EELSStat total loss (Python)')
ax.plot(enei_arr, psurf_stat + pbulk_stat, 'k-o', label = 'total', markersize = 4)
ax.plot(enei_arr, psurf_stat, 'b--', label = 'surface', markersize = 3)
ax.plot(enei_arr, pbulk_stat, 'r--', label = 'bulk', markersize = 3)
ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[2]
ax.set_title('EELSStat total comparison')
if has_matlab:
    total_m = m_psurf_stat + m_pbulk_stat
    total_p = psurf_stat + pbulk_stat
    ax.plot(enei_arr, total_p, 'b-', label = 'Python', linewidth = 2)
    ax.plot(enei_arr, total_m, 'r--', label = 'MATLAB', linewidth = 2)
    ax.legend()
else:
    ax.plot(enei_arr, psurf_stat + pbulk_stat, 'b-', label = 'Python', linewidth = 2)
    ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_stat_total.png'), dpi = 150)
plt.close()
print('[info] Saved eels_stat_total.png')

# Figure 8: ret overlay with bulk+surface
fig, axes = plt.subplots(1, 3, figsize = (18, 5))

ax = axes[0]
ax.set_title('EELSRet total loss (MATLAB)')
if has_matlab:
    ax.plot(enei_arr, m_psurf_ret + m_pbulk_ret, 'k-o', label = 'total', markersize = 4)
    ax.plot(enei_arr, m_psurf_ret, 'b--', label = 'surface', markersize = 3)
    ax.plot(enei_arr, m_pbulk_ret, 'r--', label = 'bulk', markersize = 3)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[1]
ax.set_title('EELSRet total loss (Python)')
ax.plot(enei_arr, psurf_ret + pbulk_ret, 'k-o', label = 'total', markersize = 4)
ax.plot(enei_arr, psurf_ret, 'b--', label = 'surface', markersize = 3)
ax.plot(enei_arr, pbulk_ret, 'r--', label = 'bulk', markersize = 3)
ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[2]
ax.set_title('EELSRet total comparison')
if has_matlab:
    total_m = m_psurf_ret + m_pbulk_ret
    total_p = psurf_ret + pbulk_ret
    ax.plot(enei_arr, total_p, 'b-', label = 'Python', linewidth = 2)
    ax.plot(enei_arr, total_m, 'r--', label = 'MATLAB', linewidth = 2)
    ax.legend()
else:
    ax.plot(enei_arr, psurf_ret + pbulk_ret, 'b-', label = 'Python', linewidth = 2)
    ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_ret_total.png'), dpi = 150)
plt.close()
print('[info] Saved eels_ret_total.png')

# Figure 9: stat vs ret overlay
fig, axes = plt.subplots(1, 3, figsize = (18, 5))

ax = axes[0]
ax.set_title('Stat vs Ret surface loss (MATLAB)')
if has_matlab:
    ax.plot(enei_arr, m_psurf_stat, 'b-o', label = 'stat', markersize = 4)
    ax.plot(enei_arr, m_psurf_ret, 'r-s', label = 'ret', markersize = 4)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No MATLAB data', transform = ax.transAxes,
            ha = 'center', va = 'center', fontsize = 14, color = 'gray')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[1]
ax.set_title('Stat vs Ret surface loss (Python)')
ax.plot(enei_arr, psurf_stat, 'b-o', label = 'stat', markersize = 4)
ax.plot(enei_arr, psurf_ret, 'r-s', label = 'ret', markersize = 4)
ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Loss probability (1/eV)')

ax = axes[2]
ax.set_title('Stat vs Ret comparison')
if has_matlab:
    ax.plot(enei_arr, psurf_stat - m_psurf_stat, 'b-', label = 'stat err', linewidth = 1.5)
    ax.plot(enei_arr, psurf_ret - m_psurf_ret, 'r-', label = 'ret err', linewidth = 1.5)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    ax.legend()
else:
    ax.plot(enei_arr, psurf_stat - psurf_ret, 'g-', label = 'stat-ret', linewidth = 1.5)
    ax.axhline(0, color = 'k', linestyle = '--', linewidth = 0.5)
    ax.legend()
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Difference (1/eV)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eels_stat_vs_ret.png'), dpi = 150)
plt.close()
print('[info] Saved eels_stat_vs_ret.png')


# ==============================================================================
# summary
# ==============================================================================

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
total_time = sum(timings.values())
for name, t in timings.items():
    print('  {:20s} {:.3f} s'.format(name, t))
print('  {:20s} {:.3f} s'.format('TOTAL', total_time))
print()

if has_matlab:
    print('  MATLAB comparison: AVAILABLE')
    print('  psurf_stat  max_rel_err = {:.2e}'.format(np.max(err_psurf_stat)))
    print('  psurf_ret   max_rel_err = {:.2e}'.format(np.max(err_psurf_ret)))
    print('  map_stat    max_rel_err = {:.2e}'.format(np.max(err_map_stat)))
    print('  map_ret     max_rel_err = {:.2e}'.format(np.max(err_map_ret)))
else:
    print('  MATLAB comparison: NOT AVAILABLE')

print()
print('Figures saved to: {}'.format(FIG_DIR))
print('Data saved to:    {}'.format(DATA_DIR))
print()
print('[info] 9 images: stat_spectrum, ret_spectrum, loss_map,')
print('       stat_map_2d, ret_map_2d, combined_map_2d,')
print('       stat_total, ret_total, stat_vs_ret')

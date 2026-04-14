import os
import sys
import time
from typing import Dict, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, trispheresegment, ComParticle, ComParticleMirror
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.bem.bem_stat_mirror import BEMStatMirror
from mnpbem.bem.bem_ret_mirror import BEMRetMirror
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet
from mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror
from mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
WAVELENGTHS = np.linspace(400, 800, 41)


# ============================================================================
# Geometry helpers
# ============================================================================


def make_mirror_sphere() -> ComParticleMirror:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    n = 13
    phi = np.linspace(0, np.pi / 2, n)
    theta = np.linspace(0, np.pi, 2 * n - 1)
    seg = trispheresegment(phi, theta, diameter = 20.0)
    p_mir = ComParticleMirror(epstab, [seg], [[2, 1]], sym = 'xy', closed_args = (1,))
    return p_mir


# ============================================================================
# Quasistatic
# ============================================================================


def run_stat_full(p_full: ComParticle) -> Tuple[np.ndarray, float]:
    bem = BEMStat(p_full)
    exc = PlaneWaveStat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float))

    n_wl = len(WAVELENGTHS)
    ext = np.empty((n_wl, 3), dtype = float)

    t0 = time.perf_counter()
    for i, enei in enumerate(WAVELENGTHS):
        pot = exc.potential(p_full, enei)
        sig, bem = bem.solve(pot)
        ext[i, :] = exc.extinction(sig)
    elapsed = time.perf_counter() - t0

    print('[info] stat full: {:.2f} s ({} faces)'.format(elapsed, p_full.nfaces))
    return ext, elapsed


def run_stat_mirror(p_mir: ComParticleMirror) -> Tuple[np.ndarray, float]:
    bem = BEMStatMirror(p_mir)
    exc = PlaneWaveStatMirror(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float))

    n_wl = len(WAVELENGTHS)
    ext = np.empty((n_wl, 3), dtype = float)

    t0 = time.perf_counter()
    for i, enei in enumerate(WAVELENGTHS):
        pot = exc.potential(p_mir, enei)
        sig, bem = bem.solve(pot)
        ext[i, :] = exc.extinction(sig)
    elapsed = time.perf_counter() - t0

    print('[info] stat mirror: {:.2f} s ({} faces, 1/4)'.format(elapsed, p_mir.nfaces))
    return ext, elapsed


# ============================================================================
# Retarded
# ============================================================================


def run_ret_full(p_full: ComParticle) -> Tuple[np.ndarray, float]:
    bem = BEMRet(p_full)
    pol = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
    dir_vec = np.array([[0, 0, 1], [0, 0, 1]], dtype = float)
    exc = PlaneWaveRet(pol, dir_vec)

    n_wl = len(WAVELENGTHS)
    ext = np.empty((n_wl, 2), dtype = float)

    t0 = time.perf_counter()
    for i, enei in enumerate(WAVELENGTHS):
        pot = exc.potential(p_full, enei)
        sig, bem = bem.solve(pot)
        ext_val = exc.extinction(sig)
        if np.isscalar(ext_val):
            ext[i, :] = ext_val
        else:
            ext[i, :] = ext_val
    elapsed = time.perf_counter() - t0

    print('[info] ret full: {:.2f} s ({} faces)'.format(elapsed, p_full.nfaces))
    return ext, elapsed


def run_ret_mirror(p_mir: ComParticleMirror) -> Tuple[np.ndarray, float]:
    bem = BEMRetMirror(p_mir)
    pol = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
    dir_vec = np.array([[0, 0, 1], [0, 0, 1]], dtype = float)
    exc = PlaneWaveRetMirror(pol, dir_vec)

    n_wl = len(WAVELENGTHS)
    ext = np.empty((n_wl, 2), dtype = float)

    t0 = time.perf_counter()
    for i, enei in enumerate(WAVELENGTHS):
        pot = exc.potential(p_mir, enei)
        sig, bem = bem.solve(pot)
        ext_val = exc.extinction(sig)
        if np.isscalar(ext_val):
            ext[i, :] = ext_val
        else:
            ext[i, :] = ext_val
    elapsed = time.perf_counter() - t0

    print('[info] ret mirror: {:.2f} s ({} faces, 1/4)'.format(elapsed, p_mir.nfaces))
    return ext, elapsed


# ============================================================================
# CSV I/O
# ============================================================================


def save_csv(fname: str, wl: np.ndarray, ext: np.ndarray, header: str) -> None:
    path = os.path.join(DATA_DIR, fname)
    np.savetxt(path, np.column_stack([wl, ext]), delimiter = ',', header = header, comments = '')
    print('[info] saved {}'.format(path))


def load_csv(fname: str) -> Optional[np.ndarray]:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, delimiter = ',', skiprows = 1)
    return data


# ============================================================================
# Plotting
# ============================================================================


def plot_stat_mirror(
        ext_mirror: np.ndarray,
        ext_full: np.ndarray) -> None:

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    labels = ['x-pol', 'y-pol', 'z-pol']

    for j in range(3):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_full[:, j], 'b-', linewidth = 2, label = 'Full BEMStat')
        ax.plot(WAVELENGTHS, ext_mirror[:, j], 'r--', linewidth = 2, label = 'BEMStatMirror')
        relerr = np.max(np.abs(ext_mirror[:, j] - ext_full[:, j])) / (np.max(np.abs(ext_full[:, j])) + 1e-30)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('{} (max rel err = {:.2e})'.format(labels[j], relerr))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    fig.suptitle('Quasistatic: BEMStatMirror vs BEMStat (same mesh)', fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stat_mirror.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved stat_mirror.png')


def plot_stat_python(
        ext_mirror: np.ndarray,
        ext_full: np.ndarray) -> None:

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    labels = ['x-pol', 'y-pol', 'z-pol']

    for j in range(3):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_full[:, j], 'b-', linewidth = 2, label = 'Full BEMStat')
        ax.plot(WAVELENGTHS, ext_mirror[:, j], 'r--', linewidth = 2, label = 'BEMStatMirror')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('Stat {}'.format(labels[j]))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    fig.suptitle('Quasistatic: Mirror vs Full (Python, same mesh)', fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stat_python.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved stat_python.png')


def plot_ret_mirror(
        ext_mirror: np.ndarray,
        ext_full: np.ndarray) -> None:

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    labels = ['x-pol', 'y-pol']

    for j in range(2):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_full[:, j], 'b-', linewidth = 2, label = 'Full BEMRet')
        ax.plot(WAVELENGTHS, ext_mirror[:, j], 'r--', linewidth = 2, label = 'BEMRetMirror')
        relerr = np.max(np.abs(ext_mirror[:, j] - ext_full[:, j])) / (np.max(np.abs(ext_full[:, j])) + 1e-30)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('{} (max rel err = {:.2e})'.format(labels[j], relerr))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    fig.suptitle('Retarded: BEMRetMirror vs BEMRet (same mesh)', fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ret_mirror.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved ret_mirror.png')


def plot_ret_python(
        ext_mirror: np.ndarray,
        ext_full: np.ndarray) -> None:

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    labels = ['x-pol', 'y-pol']

    for j in range(2):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_full[:, j], 'b-', linewidth = 2, label = 'Full BEMRet')
        ax.plot(WAVELENGTHS, ext_mirror[:, j], 'r--', linewidth = 2, label = 'BEMRetMirror')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('Ret {}'.format(labels[j]))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    fig.suptitle('Retarded: Mirror vs Full (Python, same mesh)', fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ret_python.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved ret_python.png')


def plot_stat_comp(
        ext_python: np.ndarray,
        matlab_data: Optional[np.ndarray]) -> None:

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))
    labels = ['x-pol', 'y-pol', 'z-pol']

    for j in range(3):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_python[:, j], 'r-', linewidth = 2, label = 'Python mirror')
        if matlab_data is not None:
            ax.plot(matlab_data[:, 0], matlab_data[:, j + 1], 'b+', markersize = 7, label = 'MATLAB mirror')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('Stat mirror {}'.format(labels[j]))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    title = 'Quasistatic Mirror: Python vs MATLAB'
    if matlab_data is None:
        title += ' (MATLAB data not available)'
    fig.suptitle(title, fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stat_comp.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved stat_comp.png')


def plot_ret_comp(
        ext_python: np.ndarray,
        matlab_data: Optional[np.ndarray]) -> None:

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    labels = ['x-pol', 'y-pol']

    for j in range(2):
        ax = axes[j]
        ax.plot(WAVELENGTHS, ext_python[:, j], 'r-', linewidth = 2, label = 'Python mirror')
        if matlab_data is not None:
            ax.plot(matlab_data[:, 0], matlab_data[:, j + 1], 'b+', markersize = 7, label = 'MATLAB mirror')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm^2)')
        ax.set_title('Ret mirror {}'.format(labels[j]))
        ax.legend(fontsize = 9)
        ax.grid(True, alpha = 0.3)

    title = 'Retarded Mirror: Python vs MATLAB'
    if matlab_data is None:
        title += ' (MATLAB data not available)'
    fig.suptitle(title, fontsize = 14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ret_comp.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved ret_comp.png')


def plot_speedup(timings: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))

    methods = ['stat', 'ret']
    full_times = [timings['stat_full'], timings['ret_full']]
    mirror_times = [timings['stat_mirror'], timings['ret_mirror']]
    speedups = [f / m if m > 0 else 0 for f, m in zip(full_times, mirror_times)]

    x = np.arange(len(methods))
    width = 0.3
    ax.bar(x - width / 2, full_times, width, label = 'Full sphere', color = '#4477AA')
    ax.bar(x + width / 2, mirror_times, width, label = 'Mirror (1/4)', color = '#CC6677')

    for i, sp in enumerate(speedups):
        y_pos = max(full_times[i], mirror_times[i]) * 1.05
        ax.text(i, y_pos, '{:.1f}x'.format(sp), ha = 'center', fontsize = 12, fontweight = 'bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Quasistatic', 'Retarded'])
    ax.set_ylabel('Time (s)')
    ax.set_title('Mirror Symmetry Speedup (same expanded mesh)')
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'speedup.png'), dpi = 150)
    plt.close(fig)
    print('[info] saved speedup.png')


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    print('=' * 60)
    print('Mirror Symmetry Validation')
    print('Gold nanosphere, d=20nm, 400-800nm, 41pt')
    print('1/4 sphere (trispheresegment) + sym=xy')
    print('BEMStatMirror / BEMRetMirror vs BEMStat / BEMRet')
    print('Same expanded mesh for fair comparison')
    print('=' * 60)

    # --- Build mirror particle ---
    p_mir = make_mirror_sphere()
    p_full = p_mir.full()  # expanded full ComParticle (same mesh)
    print('[info] mirror nfaces (1/4): {}'.format(p_mir.nfaces))
    print('[info] full nfaces: {}'.format(p_full.nfaces))

    # --- Quasistatic ---
    print('\n--- Quasistatic ---')
    ext_stat_full, t_stat_full = run_stat_full(p_full)
    ext_stat_mirror, t_stat_mirror = run_stat_mirror(p_mir)

    save_csv('python_full_stat.csv', WAVELENGTHS, ext_stat_full,
             'wavelength_nm,ext_x,ext_y,ext_z')
    save_csv('python_mirror_stat.csv', WAVELENGTHS, ext_stat_mirror,
             'wavelength_nm,ext_x,ext_y,ext_z')

    # --- Retarded ---
    print('\n--- Retarded ---')
    ext_ret_full, t_ret_full = run_ret_full(p_full)
    ext_ret_mirror, t_ret_mirror = run_ret_mirror(p_mir)

    save_csv('python_full_ret.csv', WAVELENGTHS, ext_ret_full,
             'wavelength_nm,ext_x,ext_y')
    save_csv('python_mirror_ret.csv', WAVELENGTHS, ext_ret_mirror,
             'wavelength_nm,ext_x,ext_y')

    # --- Timings ---
    timings = {
        'stat_full': t_stat_full,
        'stat_mirror': t_stat_mirror,
        'ret_full': t_ret_full,
        'ret_mirror': t_ret_mirror,
    }

    timing_arr = np.array([
        [t_stat_full, t_stat_mirror, t_stat_full / t_stat_mirror if t_stat_mirror > 0 else 0],
        [t_ret_full, t_ret_mirror, t_ret_full / t_ret_mirror if t_ret_mirror > 0 else 0],
    ])
    np.savetxt(os.path.join(DATA_DIR, 'python_timing.csv'), timing_arr,
               delimiter = ',', header = 'full_s,mirror_s,speedup', comments = '')

    # --- Plots ---
    plot_stat_mirror(ext_stat_mirror, ext_stat_full)
    plot_stat_python(ext_stat_mirror, ext_stat_full)
    plot_ret_mirror(ext_ret_mirror, ext_ret_full)
    plot_ret_python(ext_ret_mirror, ext_ret_full)
    plot_speedup(timings)

    # --- MATLAB comparison plots (always generated) ---
    m_mir_stat = load_csv('matlab_mirror_stat.csv')
    m_mir_ret = load_csv('matlab_mirror_ret.csv')
    plot_stat_comp(ext_stat_mirror, m_mir_stat)
    plot_ret_comp(ext_ret_mirror, m_mir_ret)

    # --- Self-consistency check ---
    print('\n--- Self-consistency (Mirror vs Full, same mesh) ---')

    for j, label in enumerate(['x', 'y', 'z']):
        relerr = np.max(np.abs(ext_stat_mirror[:, j] - ext_stat_full[:, j])) / (np.max(np.abs(ext_stat_full[:, j])) + 1e-30)
        print('[info] stat {}: max rel err = {:.4e}'.format(label, relerr))

    for j, label in enumerate(['x', 'y']):
        relerr = np.max(np.abs(ext_ret_mirror[:, j] - ext_ret_full[:, j])) / (np.max(np.abs(ext_ret_full[:, j])) + 1e-30)
        print('[info] ret  {}: max rel err = {:.4e}'.format(label, relerr))

    # --- MATLAB comparison metrics (if available) ---
    if m_mir_stat is not None:
        print('\n--- MATLAB comparison (stat mirror) ---')
        for j, label in enumerate(['x', 'y', 'z']):
            matlab_ext = np.interp(WAVELENGTHS, m_mir_stat[:, 0], m_mir_stat[:, j + 1])
            relerr = np.max(np.abs(ext_stat_mirror[:, j] - matlab_ext)) / (np.max(np.abs(matlab_ext)) + 1e-30)
            print('[info] stat mirror {}: max rel err vs MATLAB = {:.4e}'.format(label, relerr))

    if m_mir_ret is not None:
        print('\n--- MATLAB comparison (ret mirror) ---')
        for j, label in enumerate(['x', 'y']):
            matlab_ext = np.interp(WAVELENGTHS, m_mir_ret[:, 0], m_mir_ret[:, j + 1])
            relerr = np.max(np.abs(ext_ret_mirror[:, j] - matlab_ext)) / (np.max(np.abs(matlab_ext)) + 1e-30)
            print('[info] ret mirror {}: max rel err vs MATLAB = {:.4e}'.format(label, relerr))

    # --- Speedup report ---
    print('\n--- Speedup ---')
    stat_speedup = t_stat_full / t_stat_mirror if t_stat_mirror > 0 else 0
    ret_speedup = t_ret_full / t_ret_mirror if t_ret_mirror > 0 else 0
    print('[info] stat: full={:.2f}s, mirror={:.2f}s, speedup={:.2f}x'.format(
        t_stat_full, t_stat_mirror, stat_speedup))
    print('[info] ret:  full={:.2f}s, mirror={:.2f}s, speedup={:.2f}x'.format(
        t_ret_full, t_ret_mirror, ret_speedup))

    print('\n' + '=' * 60)
    print('Validation complete.')
    print('Data:    {}'.format(DATA_DIR))
    print('Figures: {}'.format(FIG_DIR))
    print('=' * 60)


if __name__ == '__main__':
    main()

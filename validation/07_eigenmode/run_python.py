import os
import sys
import time
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem import (
    EpsConst, EpsTable, trisphere, ComParticle,
    PlaneWaveStat, plasmonmode,
)
from mnpbem.bem import BEMStat, BEMStatEig


DATA_DIR = '/home/yoojk20/workspace/MNPBEM/validation/07_eigenmode/data'
FIG_DIR = '/home/yoojk20/workspace/MNPBEM/validation/07_eigenmode/figures'


def setup_particle() -> ComParticle:
    eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 20.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])
    return p


def compute_eig_spectrum(
        p: ComParticle,
        wavelengths: np.ndarray,
        nev: int = 20) -> Tuple[np.ndarray, float]:

    bem_eig = BEMStatEig(p, nev = nev)
    exc = PlaneWaveStat([1, 0, 0])

    n = len(wavelengths)
    ext = np.empty(n, dtype = np.float64)

    t0 = time.perf_counter()
    for i, enei in enumerate(wavelengths):
        pot = exc.potential(p, enei)
        sig, bem_eig = bem_eig.solve(pot)
        ext[i] = exc.extinction(sig)
    elapsed = time.perf_counter() - t0

    return ext, elapsed


def compute_dir_spectrum(
        p: ComParticle,
        wavelengths: np.ndarray) -> Tuple[np.ndarray, float]:

    bem_dir = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    n = len(wavelengths)
    ext = np.empty(n, dtype = np.float64)

    t0 = time.perf_counter()
    for i, enei in enumerate(wavelengths):
        pot = exc.potential(p, enei)
        sig, bem_dir = bem_dir.solve(pot)
        ext[i] = exc.extinction(sig)
    elapsed = time.perf_counter() - t0

    return ext, elapsed


def compute_plasmonmode(
        p: ComParticle,
        nev: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ene, ur, ul = plasmonmode(p, nev = nev)
    return ene, ur, ul


def save_csv_spectrum(
        path: str,
        wavelengths: np.ndarray,
        ext: np.ndarray) -> None:

    with open(path, 'w') as f:
        f.write('wavelength_nm,extinction\n')
        for i in range(len(wavelengths)):
            f.write('{:.6f},{:.15e}\n'.format(wavelengths[i], ext[i]))


def save_csv_eigenvalues(
        path: str,
        ene: np.ndarray) -> None:

    with open(path, 'w') as f:
        f.write('mode_index,eigenvalue\n')
        for i in range(len(ene)):
            f.write('{},{:.15e}\n'.format(i + 1, ene[i]))


def save_csv_mode_charge(
        path: str,
        pos: np.ndarray,
        charge: np.ndarray) -> None:

    with open(path, 'w') as f:
        f.write('x,y,z,charge_real,charge_imag\n')
        for j in range(pos.shape[0]):
            f.write('{:.15e},{:.15e},{:.15e},{:.15e},{:.15e}\n'.format(
                pos[j, 0], pos[j, 1], pos[j, 2],
                np.real(charge[j]), np.imag(charge[j])))


def load_matlab_csv(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    data = np.genfromtxt(path, delimiter = ',', names = True)
    result = {}
    for name in data.dtype.names:
        result[name] = data[name]
    return result


def plot_eig_spectrum_matlab(
        wavelengths: np.ndarray,
        ext_eig: np.ndarray,
        ext_dir: np.ndarray,
        t_eig: float,
        t_dir: float) -> None:

    matlab_eig = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_eig_spectrum.csv'))
    matlab_dir = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_dir_spectrum.csv'))

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.plot(wavelengths, ext_eig, 'b-', linewidth = 1.5, label = 'Python Eigenmode (nev=20)')
    ax.plot(wavelengths, ext_dir, 'r--', linewidth = 1.5, label = 'Python Direct')

    if matlab_eig:
        ax.plot(matlab_eig['wavelength_nm'], matlab_eig['extinction'],
                'bs', markersize = 4, markerfacecolor = 'none', label = 'MATLAB Eigenmode')
    if matlab_dir:
        ax.plot(matlab_dir['wavelength_nm'], matlab_dir['extinction'],
                'ro', markersize = 4, markerfacecolor = 'none', label = 'MATLAB Direct')

    ax.set_xlabel('Wavelength (nm)', fontsize = 12)
    ax.set_ylabel('Extinction (nm^2)', fontsize = 12)
    ax.set_title('BEMStatEig vs BEMStat (t_eig={:.3f}s, t_dir={:.3f}s)'.format(t_eig, t_dir), fontsize = 13)
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eig_spectrum_matlab.png'), dpi = 150)
    plt.close()


def plot_eig_spectrum_python(
        wavelengths: np.ndarray,
        ext_eig: np.ndarray,
        ext_dir: np.ndarray,
        t_eig: float,
        t_dir: float) -> None:

    fig, axes = plt.subplots(2, 1, figsize = (10, 8), gridspec_kw = {'height_ratios': [3, 1]})

    ax = axes[0]
    ax.plot(wavelengths, ext_eig, 'b-', linewidth = 1.5, label = 'Eigenmode (nev=20)')
    ax.plot(wavelengths, ext_dir, 'r--', linewidth = 1.5, label = 'Direct')
    ax.set_xlabel('Wavelength (nm)', fontsize = 12)
    ax.set_ylabel('Extinction (nm^2)', fontsize = 12)
    ax.set_title('Python: BEMStatEig vs BEMStat (t_eig={:.3f}s, t_dir={:.3f}s)'.format(t_eig, t_dir), fontsize = 13)
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    ax2 = axes[1]
    rel_diff = np.abs(ext_eig - ext_dir) / (np.abs(ext_dir) + 1e-30)
    ax2.semilogy(wavelengths, rel_diff, 'k-', linewidth = 1.0)
    ax2.set_xlabel('Wavelength (nm)', fontsize = 12)
    ax2.set_ylabel('Relative difference', fontsize = 12)
    ax2.set_title('Relative difference (Eigenmode vs Direct)', fontsize = 11)
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eig_spectrum_python.png'), dpi = 150)
    plt.close()


def plot_eig_spectrum_comparison(
        wavelengths: np.ndarray,
        ext_eig: np.ndarray,
        ext_dir: np.ndarray,
        t_eig: float,
        t_dir: float) -> None:

    matlab_eig = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_eig_spectrum.csv'))
    matlab_dir = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_dir_spectrum.csv'))

    has_matlab = bool(matlab_eig) and bool(matlab_dir)

    fig, axes = plt.subplots(2, 1, figsize = (10, 8), gridspec_kw = {'height_ratios': [3, 1]})

    ax = axes[0]
    ax.plot(wavelengths, ext_eig, 'b-', linewidth = 2.0, label = 'Python Eigenmode')
    ax.plot(wavelengths, ext_dir, 'r--', linewidth = 1.5, label = 'Python Direct')

    if has_matlab:
        ax.plot(matlab_eig['wavelength_nm'], matlab_eig['extinction'],
                'bs', markersize = 5, markerfacecolor = 'none', label = 'MATLAB Eigenmode')
        ax.plot(matlab_dir['wavelength_nm'], matlab_dir['extinction'],
                'ro', markersize = 5, markerfacecolor = 'none', label = 'MATLAB Direct')

    ax.set_xlabel('Wavelength (nm)', fontsize = 12)
    ax.set_ylabel('Extinction (nm^2)', fontsize = 12)
    ax.set_title('Eigenmode vs Direct: Python & MATLAB comparison', fontsize = 13)
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    ax2 = axes[1]
    rel_py = np.abs(ext_eig - ext_dir) / (np.abs(ext_dir) + 1e-30)
    ax2.semilogy(wavelengths, rel_py, 'k-', linewidth = 1.0, label = 'Py Eig vs Py Dir')

    if has_matlab:
        m_wl = matlab_eig['wavelength_nm']
        m_ext_eig = matlab_eig['extinction']
        m_ext_dir = matlab_dir['extinction']

        # MATLAB eig vs Python eig (interpolate if needed)
        py_eig_interp = np.interp(m_wl, wavelengths, ext_eig)
        rel_mp = np.abs(py_eig_interp - m_ext_eig) / (np.abs(m_ext_eig) + 1e-30)
        ax2.semilogy(m_wl, rel_mp, 'b--', linewidth = 1.0, label = 'Py Eig vs MATLAB Eig')

    ax2.set_xlabel('Wavelength (nm)', fontsize = 12)
    ax2.set_ylabel('Relative difference', fontsize = 12)
    ax2.legend(fontsize = 9)
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eig_spectrum_comp.png'), dpi = 150)
    plt.close()


def plot_eigenvalues_comparison(
        ene_py: np.ndarray) -> None:

    matlab_ev = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_eigenvalues.csv'))

    nev = len(ene_py)
    x = np.arange(1, nev + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize = (10, 6))

    if matlab_ev:
        nev_m = len(matlab_ev['eigenvalue'])
        nev_common = min(nev, nev_m)
        x_common = np.arange(1, nev_common + 1)

        ax.bar(x_common - width / 2, matlab_ev['eigenvalue'][:nev_common], width,
               label = 'MATLAB', color = 'steelblue', alpha = 0.8)
        ax.bar(x_common + width / 2, ene_py[:nev_common], width,
               label = 'Python', color = 'coral', alpha = 0.8)
    else:
        ax.bar(x, ene_py, width, label = 'Python', color = 'coral', alpha = 0.8)

    ax.set_xlabel('Mode index', fontsize = 12)
    ax.set_ylabel('Eigenvalue', fontsize = 12)
    ax.set_title('Plasmon eigenvalues (nev=10): MATLAB vs Python', fontsize = 13)
    ax.legend(fontsize = 11)
    ax.grid(True, alpha = 0.3, axis = 'y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'eigenvalues_comp.png'), dpi = 150)
    plt.close()


def plot_plasmon_modes(
        pos: np.ndarray,
        ur: np.ndarray) -> None:

    fig = plt.figure(figsize = (18, 5))

    for k in range(3):
        ax = fig.add_subplot(1, 3, k + 1, projection = '3d')
        charge = np.real(ur[:, k])

        vmax = np.max(np.abs(charge))
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                        c = charge, cmap = 'RdBu_r', vmin = -vmax, vmax = vmax,
                        s = 20, alpha = 0.9, edgecolors = 'none')
        fig.colorbar(sc, ax = ax, shrink = 0.6, label = 'Charge')

        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.set_title('Mode {} (eig={:.4f})'.format(k + 1, 0.0))

        ax.set_box_aspect([1, 1, 1])

    plt.suptitle('Top 3 plasmon mode surface charge patterns', fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'plasmon_modes.png'), dpi = 150)
    plt.close()


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    print('=' * 60)
    print('[info] Eigenmode validation (07_eigenmode)')
    print('=' * 60)

    # setup
    p = setup_particle()
    wavelengths = np.linspace(400, 800, 41)

    # --- Test 1: BEMStatEig vs BEMStat extinction spectrum ---
    print('\n--- Test 1: BEMStatEig vs BEMStat extinction spectrum ---')

    print('[info] Computing BEMStatEig spectrum (nev=20) ...')
    ext_eig, t_eig = compute_eig_spectrum(p, wavelengths, nev = 20)
    print('[info] BEMStatEig time: {:.4f} s'.format(t_eig))

    print('[info] Computing BEMStat direct spectrum ...')
    ext_dir, t_dir = compute_dir_spectrum(p, wavelengths)
    print('[info] BEMStat time: {:.4f} s'.format(t_dir))

    # relative difference
    rel_diff = np.abs(ext_eig - ext_dir) / (np.abs(ext_dir) + 1e-30)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    print('[info] Max relative difference (Eig vs Dir): {:.6e}'.format(max_rel_diff))
    print('[info] Mean relative difference (Eig vs Dir): {:.6e}'.format(mean_rel_diff))

    if max_rel_diff < 0.01:
        print('[info] PASS: Eigenmode matches direct solver')
    else:
        print('[info] WARN: Large difference detected')

    # save CSVs
    save_csv_spectrum(os.path.join(DATA_DIR, 'python_eig_spectrum.csv'), wavelengths, ext_eig)
    save_csv_spectrum(os.path.join(DATA_DIR, 'python_dir_spectrum.csv'), wavelengths, ext_dir)

    # save timing
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('solver,time_sec\n')
        f.write('bemstateig,{:.6f}\n'.format(t_eig))
        f.write('bemstat_direct,{:.6f}\n'.format(t_dir))

    # --- Test 2: PlasmonMode eigenvalues ---
    print('\n--- Test 2: PlasmonMode eigenvalues (nev=10) ---')
    ene, ur, ul = compute_plasmonmode(p, nev = 10)
    nev_actual = len(ene)
    print('[info] Computed {} eigenvalues'.format(nev_actual))
    for i in range(min(nev_actual, 10)):
        print('[info]   mode {}: eigenvalue = {:.10f}'.format(i + 1, ene[i]))

    save_csv_eigenvalues(os.path.join(DATA_DIR, 'python_eigenvalues.csv'), ene)

    # compare with MATLAB
    matlab_ev = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_eigenvalues.csv'))
    if matlab_ev:
        nev_common = min(nev_actual, len(matlab_ev['eigenvalue']))
        ev_diff = np.abs(ene[:nev_common] - matlab_ev['eigenvalue'][:nev_common])
        ev_rel = ev_diff / (np.abs(matlab_ev['eigenvalue'][:nev_common]) + 1e-30)
        print('[info] Eigenvalue comparison (MATLAB vs Python):')
        for i in range(nev_common):
            print('[info]   mode {}: MATLAB={:.10f}  Python={:.10f}  rel_diff={:.6e}'.format(
                i + 1, matlab_ev['eigenvalue'][i], ene[i], ev_rel[i]))
        print('[info] Max eigenvalue rel diff: {:.6e}'.format(np.max(ev_rel)))
    else:
        print('[info] No MATLAB eigenvalue data found, skipping comparison')

    # --- Test 3: Top 3 plasmon mode surface charges ---
    print('\n--- Test 3: Top 3 plasmon mode surface charges ---')
    pos = p.pos
    for k in range(3):
        charge = ur[:, k]
        save_csv_mode_charge(
            os.path.join(DATA_DIR, 'python_mode{}_charge.csv'.format(k + 1)),
            pos, charge)
        print('[info] Mode {} surface charge saved ({} faces, max|charge|={:.6e})'.format(
            k + 1, pos.shape[0], np.max(np.abs(charge))))

    # --- Plots ---
    print('\n--- Generating plots ---')

    # 1) eig_spectrum_matlab.png
    plot_eig_spectrum_matlab(wavelengths, ext_eig, ext_dir, t_eig, t_dir)
    print('[info] Saved: eig_spectrum_matlab.png')

    # 2) eig_spectrum_python.png
    plot_eig_spectrum_python(wavelengths, ext_eig, ext_dir, t_eig, t_dir)
    print('[info] Saved: eig_spectrum_python.png')

    # 3) eig_spectrum_comp.png
    plot_eig_spectrum_comparison(wavelengths, ext_eig, ext_dir, t_eig, t_dir)
    print('[info] Saved: eig_spectrum_comp.png')

    # 4) eigenvalues_comp.png
    plot_eigenvalues_comparison(ene)
    print('[info] Saved: eigenvalues_comp.png')

    # 5) plasmon_modes.png (update with eigenvalues in title)
    fig = plt.figure(figsize = (18, 5))
    for k in range(3):
        ax = fig.add_subplot(1, 3, k + 1, projection = '3d')
        charge = np.real(ur[:, k])
        vmax = np.max(np.abs(charge))
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                        c = charge, cmap = 'RdBu_r', vmin = -vmax, vmax = vmax,
                        s = 20, alpha = 0.9, edgecolors = 'none')
        fig.colorbar(sc, ax = ax, shrink = 0.6, label = 'Charge')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.set_title('Mode {} (eig={:.4f})'.format(k + 1, ene[k]))
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle('Top 3 plasmon mode surface charge patterns', fontsize = 14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'plasmon_modes.png'), dpi = 150)
    plt.close()
    print('[info] Saved: plasmon_modes.png')

    # --- MATLAB comparison summary ---
    print('\n--- MATLAB comparison ---')
    matlab_eig = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_eig_spectrum.csv'))
    if matlab_eig:
        py_interp = np.interp(matlab_eig['wavelength_nm'], wavelengths, ext_eig)
        ext_rel = np.abs(py_interp - matlab_eig['extinction']) / (np.abs(matlab_eig['extinction']) + 1e-30)
        print('[info] Extinction (Eig): max_rel_diff = {:.6e}'.format(np.max(ext_rel)))
        print('[info] Extinction (Eig): mean_rel_diff = {:.6e}'.format(np.mean(ext_rel)))
    else:
        print('[info] No MATLAB extinction data found')

    # --- Summary ---
    print('\n' + '=' * 60)
    print('[info] Validation Summary')
    print('=' * 60)
    print('[info] Test 1 (Eig vs Dir): max_rel_diff = {:.6e}'.format(max_rel_diff))
    print('[info] Test 2 (Eigenvalues): {} modes computed'.format(nev_actual))
    print('[info] Test 3 (Mode charges): top 3 modes saved')
    print('[info] Timing: BEMStatEig = {:.4f}s, BEMStat = {:.4f}s'.format(t_eig, t_dir))
    print('[info] Figures: 5 plots saved to {}'.format(FIG_DIR))
    print('[info] CSVs: eigenvalues, spectra, charges saved to {}'.format(DATA_DIR))


if __name__ == '__main__':
    main()

import os
import sys
import time
from typing import Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, ComPoint
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import DipoleStat, DipoleRet


DATA_DIR = '/home/yoojk20/workspace/MNPBEM/validation/09_dipole/data'
FIG_DIR = '/home/yoojk20/workspace/MNPBEM/validation/09_dipole/figures'


def make_particle_stat() -> ComParticle:
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    return p


def make_particle_ret() -> ComParticle:
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    return p


def run_dipole_stat(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    p = make_particle_stat()
    pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
    dip = DipoleStat(pt, dip = np.array([[0.0, 0.0, 1.0]]))
    bem = BEMStat(p)

    n = len(enei)
    tot = np.zeros(n)
    rad = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        exc = dip(p, enei[i])
        sig, bem = bem.solve(exc)
        t_arr, r_arr, _ = dip.decayrate(sig)
        tot[i] = t_arr[0, 0]
        rad[i] = r_arr[0, 0]
    elapsed = time.time() - t0

    return tot, rad, elapsed


def run_dipole_ret(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    p = make_particle_ret()
    pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
    dip = DipoleRet(pt, dip = np.array([[0.0, 0.0, 1.0]]))
    bem = BEMRet(p)

    n = len(enei)
    tot = np.zeros(n)
    rad = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        exc = dip(p, enei[i])
        sig, bem = bem.solve(exc)
        t_arr, r_arr, _ = dip.decayrate(sig)
        tot[i] = t_arr[0, 0]
        rad[i] = r_arr[0, 0]
    elapsed = time.time() - t0

    return tot, rad, elapsed


def run_distance(z_vals: np.ndarray, lambda_fix: float) -> Tuple[np.ndarray, np.ndarray, float]:
    p = make_particle_stat()
    bem = BEMStat(p)

    nz = len(z_vals)
    tot = np.zeros(nz)
    rad = np.zeros(nz)

    t0 = time.time()
    for j in range(nz):
        pt = ComPoint(p, np.array([[0.0, 0.0, z_vals[j]]]))
        dip = DipoleStat(pt, dip = np.array([[0.0, 0.0, 1.0]]))
        exc = dip(p, lambda_fix)
        sig, bem = bem.solve(exc)
        t_arr, r_arr, _ = dip.decayrate(sig)
        tot[j] = t_arr[0, 0]
        rad[j] = r_arr[0, 0]
    elapsed = time.time() - t0

    return tot, rad, elapsed


def save_csv_wavelength(filepath: str,
        enei: np.ndarray,
        tot: np.ndarray,
        rad: np.ndarray) -> None:
    header = 'wavelength_nm,tot,rad'
    n = len(enei)
    data = np.empty((n, 3))
    data[:, 0] = enei
    data[:, 1] = tot
    data[:, 2] = rad
    np.savetxt(filepath, data, delimiter = ',', header = header, comments = '')


def save_csv_distance(filepath: str,
        z_vals: np.ndarray,
        tot: np.ndarray,
        rad: np.ndarray) -> None:
    header = 'z_nm,tot,rad'
    nz = len(z_vals)
    data = np.empty((nz, 3))
    data[:, 0] = z_vals
    data[:, 1] = tot
    data[:, 2] = rad
    np.savetxt(filepath, data, delimiter = ',', header = header, comments = '')


def plot_python_stat(enei: np.ndarray,
        tot: np.ndarray,
        rad: np.ndarray,
        elapsed: float) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(enei, tot, 'b-', linewidth = 1.5, label = 'tot')
    ax.plot(enei, rad, 'r--', linewidth = 1.5, label = 'rad')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Decay rate')
    ax.set_title('Python DipoleStat - z-dipole at [0,0,15] (t={:.3f}s)'.format(elapsed))
    ax.legend(loc = 'best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dipole_stat_python.png'), dpi = 150)
    plt.close(fig)


def plot_python_ret(enei: np.ndarray,
        tot: np.ndarray,
        rad: np.ndarray,
        elapsed: float) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(enei, tot, 'b-', linewidth = 1.5, label = 'tot')
    ax.plot(enei, rad, 'r--', linewidth = 1.5, label = 'rad')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Decay rate')
    ax.set_title('Python DipoleRet - z-dipole at [0,0,15] (t={:.3f}s)'.format(elapsed))
    ax.legend(loc = 'best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dipole_ret_python.png'), dpi = 150)
    plt.close(fig)


def plot_python_distance(z_vals: np.ndarray,
        tot: np.ndarray,
        rad: np.ndarray,
        elapsed: float) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(z_vals, tot, 'bo-', linewidth = 1.5, label = 'tot')
    ax.plot(z_vals, rad, 'rs--', linewidth = 1.5, label = 'rad')
    ax.set_xlabel('Dipole distance z (nm)')
    ax.set_ylabel('Decay rate')
    ax.set_title('Python DipoleStat distance - lambda=520nm (t={:.3f}s)'.format(elapsed))
    ax.legend(loc = 'best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dipole_distance_python.png'), dpi = 150)
    plt.close(fig)


def load_matlab_csv(filepath: str) -> Optional[np.ndarray]:
    if os.path.exists(filepath):
        return np.genfromtxt(filepath, delimiter = ',', skip_header = 1)
    return None


def plot_comparison_wavelength(enei: np.ndarray,
        py_tot: np.ndarray, py_rad: np.ndarray,
        ml_tot: np.ndarray, ml_rad: np.ndarray,
        t_py: float, t_ml: float,
        label: str, filename: str) -> None:
    rel_err_tot = np.abs(py_tot - ml_tot) / (np.abs(ml_tot) + 1e-30)
    rel_err_rad = np.abs(py_rad - ml_rad) / (np.abs(ml_rad) + 1e-30)
    max_err_tot = np.max(rel_err_tot)
    max_err_rad = np.max(rel_err_rad)

    fig, axes = plt.subplots(2, 1, figsize = (8, 8))

    ax = axes[0]
    ax.plot(enei, ml_tot, 'b-', linewidth = 1.5, label = 'MATLAB tot')
    ax.plot(enei, ml_rad, 'r-', linewidth = 1.5, label = 'MATLAB rad')
    ax.plot(enei, py_tot, 'b--', linewidth = 1.5, label = 'Python tot')
    ax.plot(enei, py_rad, 'r--', linewidth = 1.5, label = 'Python rad')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Decay rate')
    ax.set_title('{} | t_ML={:.3f}s, t_PY={:.3f}s'.format(label, t_ml, t_py))
    ax.legend(loc = 'best')
    ax.grid(True)

    ax = axes[1]
    ax.semilogy(enei, rel_err_tot, 'b-', linewidth = 1.5, label = 'tot rel err')
    ax.semilogy(enei, rel_err_rad, 'r-', linewidth = 1.5, label = 'rad rel err')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative error')
    ax.set_title('max rel err: tot={:.2e}, rad={:.2e}'.format(max_err_tot, max_err_rad))
    ax.legend(loc = 'best')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi = 150)
    plt.close(fig)

    print('[info] {} max rel err: tot={:.2e}, rad={:.2e}'.format(label, max_err_tot, max_err_rad))


def plot_comparison_distance(z_vals: np.ndarray,
        py_tot: np.ndarray, py_rad: np.ndarray,
        ml_tot: np.ndarray, ml_rad: np.ndarray,
        t_py: float, t_ml: float) -> None:
    rel_err_tot = np.abs(py_tot - ml_tot) / (np.abs(ml_tot) + 1e-30)
    rel_err_rad = np.abs(py_rad - ml_rad) / (np.abs(ml_rad) + 1e-30)
    max_err_tot = np.max(rel_err_tot)
    max_err_rad = np.max(rel_err_rad)

    fig, axes = plt.subplots(2, 1, figsize = (8, 8))

    ax = axes[0]
    ax.plot(z_vals, ml_tot, 'bo-', linewidth = 1.5, label = 'MATLAB tot')
    ax.plot(z_vals, ml_rad, 'rs-', linewidth = 1.5, label = 'MATLAB rad')
    ax.plot(z_vals, py_tot, 'b^--', linewidth = 1.5, label = 'Python tot')
    ax.plot(z_vals, py_rad, 'rv--', linewidth = 1.5, label = 'Python rad')
    ax.set_xlabel('Dipole distance z (nm)')
    ax.set_ylabel('Decay rate')
    ax.set_title('Distance dependence | t_ML={:.3f}s, t_PY={:.3f}s'.format(t_ml, t_py))
    ax.legend(loc = 'best')
    ax.grid(True)

    ax = axes[1]
    ax.semilogy(z_vals, rel_err_tot, 'b-', linewidth = 1.5, label = 'tot rel err')
    ax.semilogy(z_vals, rel_err_rad, 'r-', linewidth = 1.5, label = 'rad rel err')
    ax.set_xlabel('Dipole distance z (nm)')
    ax.set_ylabel('Relative error')
    ax.set_title('max rel err: tot={:.2e}, rad={:.2e}'.format(max_err_tot, max_err_rad))
    ax.legend(loc = 'best')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dipole_distance_comparison.png'), dpi = 150)
    plt.close(fig)

    print('[info] Distance max rel err: tot={:.2e}, rad={:.2e}'.format(max_err_tot, max_err_rad))


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(500, 700, 21)
    z_vals = np.array([12.0, 15.0, 20.0, 30.0])
    lambda_fix = 520.0

    # ========== Test 1: DipoleStat — wavelength scan ==========
    print('[info] Running Python DipoleStat ...')
    tot_s, rad_s, t_stat = run_dipole_stat(enei)
    print('[info] Python DipoleStat time: {:.4f} sec'.format(t_stat))
    save_csv_wavelength(os.path.join(DATA_DIR, 'python_dipole_stat.csv'), enei, tot_s, rad_s)
    plot_python_stat(enei, tot_s, rad_s, t_stat)
    print('[info] Saved dipole_stat_python.png')

    # ========== Test 2: DipoleRet — wavelength scan ==========
    print('[info] Running Python DipoleRet ...')
    tot_r, rad_r, t_ret = run_dipole_ret(enei)
    print('[info] Python DipoleRet time: {:.4f} sec'.format(t_ret))
    save_csv_wavelength(os.path.join(DATA_DIR, 'python_dipole_ret.csv'), enei, tot_r, rad_r)
    plot_python_ret(enei, tot_r, rad_r, t_ret)
    print('[info] Saved dipole_ret_python.png')

    # ========== Test 3: DipoleStat — distance dependence ==========
    print('[info] Running Python Distance dependence ...')
    tot_d, rad_d, t_dist = run_distance(z_vals, lambda_fix)
    print('[info] Python Distance time: {:.4f} sec'.format(t_dist))
    save_csv_distance(os.path.join(DATA_DIR, 'python_dipole_distance.csv'), z_vals, tot_d, rad_d)
    plot_python_distance(z_vals, tot_d, rad_d, t_dist)
    print('[info] Saved dipole_distance_python.png')

    # Timing CSV
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('test,time_sec\n')
        f.write('DipoleStat,{:.6f}\n'.format(t_stat))
        f.write('DipoleRet,{:.6f}\n'.format(t_ret))
        f.write('Distance,{:.6f}\n'.format(t_dist))

    # ========== Comparison with MATLAB ==========
    # DipoleStat comparison
    ml_stat = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_dipole_stat.csv'))
    ml_timing = {}
    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    ml_timing[parts[0]] = float(parts[1])

    if ml_stat is not None:
        print('[info] Loading MATLAB DipoleStat data for comparison ...')
        ml_tot_s = ml_stat[:, 1]
        ml_rad_s = ml_stat[:, 2]
        t_ml_stat = ml_timing.get('DipoleStat', 0.0)
        plot_comparison_wavelength(
            enei, tot_s, rad_s, ml_tot_s, ml_rad_s,
            t_stat, t_ml_stat,
            'DipoleStat MATLAB vs Python', 'dipole_stat_comparison.png')
        print('[info] Saved dipole_stat_comparison.png')
    else:
        print('[info] MATLAB DipoleStat data not found, skipping comparison.')

    # DipoleRet comparison
    ml_ret = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_dipole_ret.csv'))
    if ml_ret is not None:
        print('[info] Loading MATLAB DipoleRet data for comparison ...')
        ml_tot_r = ml_ret[:, 1]
        ml_rad_r = ml_ret[:, 2]
        t_ml_ret = ml_timing.get('DipoleRet', 0.0)
        plot_comparison_wavelength(
            enei, tot_r, rad_r, ml_tot_r, ml_rad_r,
            t_ret, t_ml_ret,
            'DipoleRet MATLAB vs Python', 'dipole_ret_comparison.png')
        print('[info] Saved dipole_ret_comparison.png')
    else:
        print('[info] MATLAB DipoleRet data not found, skipping comparison.')

    # Distance comparison
    ml_dist = load_matlab_csv(os.path.join(DATA_DIR, 'matlab_dipole_distance.csv'))
    if ml_dist is not None:
        print('[info] Loading MATLAB Distance data for comparison ...')
        ml_tot_d = ml_dist[:, 1]
        ml_rad_d = ml_dist[:, 2]
        t_ml_dist = ml_timing.get('Distance', 0.0)
        plot_comparison_distance(
            z_vals, tot_d, rad_d, ml_tot_d, ml_rad_d,
            t_dist, t_ml_dist)
        print('[info] Saved dipole_distance_comparison.png')
    else:
        print('[info] MATLAB Distance data not found, skipping comparison.')

    print('[info] Python validation complete.')


if __name__ == '__main__':
    main()

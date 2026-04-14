import os
import sys
import time
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat
from mnpbem.simulation import PlaneWaveStat
from mnpbem.mie import MieStat


DATA_DIR = '/home/yoojk20/workspace/MNPBEM/validation/02_bemstat_sphere/data'
FIG_DIR = '/home/yoojk20/workspace/MNPBEM/validation/02_bemstat_sphere/figures'


def run_bem(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    n = len(enei)
    ext = np.zeros(n)
    sca = np.zeros(n)
    absc = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        sig, bem = bem.solve(exc(p, enei[i]))
        ext[i] = exc.extinction(sig)
        sca[i] = exc.scattering(sig)
        absc[i] = ext[i] - sca[i]
    t_bem = time.time() - t0

    return ext, sca, absc, t_bem


def run_mie(enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    epsin = EpsTable('gold.dat')
    epsout = EpsConst(1)
    mie = MieStat(epsin, epsout, 20)

    n = len(enei)
    mie_ext = np.zeros(n)
    mie_sca = np.zeros(n)
    mie_abs = np.zeros(n)

    t0 = time.time()
    for i in range(n):
        mie_ext[i] = mie.extinction(enei[i])
        mie_sca[i] = mie.scattering(enei[i])
        mie_abs[i] = mie_ext[i] - mie_sca[i]
    t_mie = time.time() - t0

    return mie_ext, mie_sca, mie_abs, t_mie


def save_csv(filepath: str, enei: np.ndarray,
        ext: np.ndarray, sca: np.ndarray, absc: np.ndarray) -> None:
    header = 'wavelength_nm,extinction,scattering,absorption'
    data = np.column_stack([enei, ext, sca, absc])
    np.savetxt(filepath, data, delimiter = ',', header = header, comments = '')


def plot_python(enei: np.ndarray,
        ext: np.ndarray, sca: np.ndarray,
        mie_ext: np.ndarray, t_bem: float) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(enei, ext, 'b-', linewidth = 1.5, label = 'BEM ext')
    ax.plot(enei, sca, 'r--', linewidth = 1.5, label = 'BEM sca')
    ax.plot(enei, mie_ext, 'ko', markersize = 4, label = 'Mie ext')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm$^2$)')
    ax.set_title('Python BEMStat - 20nm Au sphere (t_BEM={:.3f}s)'.format(t_bem))
    ax.legend(loc = 'best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'bemstat_python.png'), dpi = 150)
    plt.close(fig)


def plot_comparison(enei: np.ndarray,
        py_ext: np.ndarray, py_sca: np.ndarray,
        ml_ext: np.ndarray, ml_sca: np.ndarray,
        t_py: float, t_ml: float) -> None:
    rel_err_ext = np.abs(py_ext - ml_ext) / (np.abs(ml_ext) + 1e-30)
    rel_err_sca = np.abs(py_sca - ml_sca) / (np.abs(ml_sca) + 1e-30)
    max_err_ext = np.max(rel_err_ext)
    max_err_sca = np.max(rel_err_sca)

    fig, axes = plt.subplots(2, 1, figsize = (8, 8))

    # Cross section overlay
    ax = axes[0]
    ax.plot(enei, ml_ext, 'b-', linewidth = 1.5, label = 'MATLAB ext')
    ax.plot(enei, ml_sca, 'r-', linewidth = 1.5, label = 'MATLAB sca')
    ax.plot(enei, py_ext, 'b--', linewidth = 1.5, label = 'Python ext')
    ax.plot(enei, py_sca, 'r--', linewidth = 1.5, label = 'Python sca')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm$^2$)')
    ax.set_title('MATLAB vs Python BEMStat | t_ML={:.3f}s, t_PY={:.3f}s'.format(t_ml, t_py))
    ax.legend(loc = 'best')
    ax.grid(True)

    # Relative error
    ax = axes[1]
    ax.semilogy(enei, rel_err_ext, 'b-', linewidth = 1.5, label = 'ext rel err')
    ax.semilogy(enei, rel_err_sca, 'r-', linewidth = 1.5, label = 'sca rel err')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative error')
    ax.set_title('max rel err: ext={:.2e}, sca={:.2e}'.format(max_err_ext, max_err_sca))
    ax.legend(loc = 'best')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'bemstat_comparison.png'), dpi = 150)
    plt.close(fig)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei = np.linspace(400, 800, 41)

    # BEM
    print('[info] Running Python BEMStat ...')
    ext, sca, absc, t_bem = run_bem(enei)
    print('[info] Python BEM solve time: {:.4f} sec'.format(t_bem))
    save_csv(os.path.join(DATA_DIR, 'python_bemstat.csv'), enei, ext, sca, absc)

    # Mie
    print('[info] Running Python MieStat ...')
    mie_ext, mie_sca, mie_abs, t_mie = run_mie(enei)
    print('[info] Python Mie solve time: {:.4f} sec'.format(t_mie))
    save_csv(os.path.join(DATA_DIR, 'python_mie.csv'), enei, mie_ext, mie_sca, mie_abs)

    # Timing CSV
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('solver,time_sec\n')
        f.write('BEM,{:.6f}\n'.format(t_bem))
        f.write('Mie,{:.6f}\n'.format(t_mie))

    # Python plot
    plot_python(enei, ext, sca, mie_ext, t_bem)
    print('[info] Saved bemstat_python.png')

    # Comparison with MATLAB (if MATLAB data exists)
    matlab_bem_path = os.path.join(DATA_DIR, 'matlab_bemstat.csv')
    matlab_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    if os.path.exists(matlab_bem_path):
        print('[info] Loading MATLAB BEM data for comparison ...')
        ml_data = np.genfromtxt(matlab_bem_path, delimiter = ',', skip_header = 1)
        ml_ext = ml_data[:, 1]
        ml_sca = ml_data[:, 2]

        t_ml = 0.0
        if os.path.exists(matlab_timing_path):
            with open(matlab_timing_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if parts[0] == 'BEM':
                        t_ml = float(parts[1])

        plot_comparison(enei, ext, sca, ml_ext, ml_sca, t_bem, t_ml)
        print('[info] Saved bemstat_comparison.png')

        rel_err = np.max(np.abs(ext - ml_ext) / (np.abs(ml_ext) + 1e-30))
        print('[info] Max relative error (extinction): {:.2e}'.format(rel_err))
    else:
        print('[info] MATLAB data not found, skipping comparison plot.')
        print('[info] Run run_matlab.m first to generate MATLAB reference data.')

    print('[info] Python validation complete.')


if __name__ == '__main__':
    main()

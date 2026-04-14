import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials.eps_table import EpsTable
from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet, MeshField
from mnpbem.bem import BEMStat, BEMRet


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
WAVELENGTH = 520.0
GRID_N = 31
GRID_RANGE = 30.0


def setup_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_1d = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N)
    z_1d = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_N)
    x, z = np.meshgrid(x_1d, z_1d)
    y = np.zeros_like(x)
    return x, y, z


def run_stat(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    t0 = time.time()
    pot = exc.potential(p, WAVELENGTH)
    sig, _ = bem.solve(pot)
    mf = MeshField(p, x, y, z)
    e, h = mf.field(sig)
    t_elapsed = time.time() - t0

    # |E| and |E|^2
    enorm = np.sqrt(np.sum(np.abs(e) ** 2, axis = -1))  # (31, 31)
    e2 = enorm ** 2

    return enorm, e2, t_elapsed


def run_ret(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    t0 = time.time()
    pot = exc.potential(p, WAVELENGTH)
    sig, _ = bem.solve(pot)
    mf = MeshField(p, x, y, z, sim = 'ret')
    e, h = mf.field(sig)
    t_elapsed = time.time() - t0

    # |E| and |E|^2
    enorm = np.sqrt(np.sum(np.abs(e) ** 2, axis = -1))  # (31, 31)
    e2 = enorm ** 2

    return enorm, e2, t_elapsed


def save_csv(x: np.ndarray, z: np.ndarray, enorm: np.ndarray, e2: np.ndarray,
        mode: str) -> None:
    df = pd.DataFrame({
        'x_nm': x.ravel(),
        'z_nm': z.ravel(),
        'enorm': enorm.ravel(),
        'e2': e2.ravel()
    })
    df.to_csv(os.path.join(DATA_DIR, 'python_{}.csv'.format(mode)), index = False)

    # x=0 linecut (column 15, 0-indexed middle of 31-point grid)
    mid = GRID_N // 2  # 15
    z_cut = z[:, mid]
    enorm_cut = enorm[:, mid]
    df_lc = pd.DataFrame({
        'z_nm': z_cut,
        'enorm': enorm_cut
    })
    df_lc.to_csv(os.path.join(DATA_DIR, 'python_{}_linecut.csv'.format(mode)), index = False)


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    x, y, z = setup_grid()

    # --- stat ---
    print('[info] Running stat meshfield ...')
    enorm_s, e2_s, t_stat = run_stat(x, y, z)
    save_csv(x, z, enorm_s, e2_s, 'stat')
    print('[info] Python stat meshfield time: {:.4f} sec'.format(t_stat))

    # --- ret ---
    print('[info] Running ret meshfield ...')
    enorm_r, e2_r, t_ret = run_ret(x, y, z)
    save_csv(x, z, enorm_r, e2_r, 'ret')
    print('[info] Python ret meshfield time: {:.4f} sec'.format(t_ret))

    # Timing CSV
    df_time = pd.DataFrame({
        'solver': ['stat', 'ret'],
        'time_sec': [t_stat, t_ret]
    })
    df_time.to_csv(os.path.join(DATA_DIR, 'python_timing.csv'), index = False)

    print('[info] Python nearfield validation complete.')

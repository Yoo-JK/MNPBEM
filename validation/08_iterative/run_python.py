"""
Iterative BEM Solver Validation -- Python

Tests:
  1. BEMStatIter vs BEMStat       (400-800nm, 41pt, 20nm Au sphere)
  2. BEMRetIter  vs BEMRet        (400-800nm, 41pt, 20nm Au sphere)
  3. BEMRetLayerIter vs BEMRetLayer (450-750nm, 11pt, 20nm Au sphere on glass)

Each test:
  - runs direct solver, records extinction + wall time
  - runs iterative solver, records extinction + wall time
  - saves CSVs to data/
  - saves figures (individual + comparison) to figures/
"""

import gc
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMStat, BEMRet, BEMRetLayer
from mnpbem.bem import BEMStatIter, BEMRetIter, BEMRetLayerIter
from mnpbem.simulation import PlaneWaveStat, PlaneWaveRet, PlaneWaveRetLayer


BASE_DIR = '/home/yoojk20/workspace/MNPBEM/validation/08_iterative'
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


# =========================================================================
# Utilities
# =========================================================================

def save_csv(filepath, enei, ext):
    header = 'wavelength_nm,extinction'
    data = np.column_stack([enei, ext])
    np.savetxt(filepath, data, delimiter = ',', header = header, comments = '')


def plot_single(enei, ext_d, ext_i, label, t_d, t_i, savepath):
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(enei, ext_d, 'b-', linewidth = 1.5, label = 'direct')
    ax.plot(enei, ext_i, 'r--', linewidth = 1.5, label = 'iterative')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Extinction (nm$^2$)')
    ax.set_title('Python {} -- direct({:.3f}s) vs iter({:.3f}s)'.format(label, t_d, t_i))
    ax.legend(loc = 'best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


def plot_comparison(enei, ext_d, ext_i, ml_ext_d, ml_ext_i,
        label, t_py_d, t_py_i, t_ml_d, t_ml_i, savepath):
    fig, axes = plt.subplots(2, 1, figsize = (8, 9))

    ax = axes[0]
    ax.plot(enei, ml_ext_d, 'b-', linewidth = 1.5, label = 'MATLAB direct')
    ax.plot(enei, ml_ext_i, 'b--', linewidth = 1.5, label = 'MATLAB iter')
    ax.plot(enei, ext_d, 'r-', linewidth = 1.5, label = 'Python direct')
    ax.plot(enei, ext_i, 'r--', linewidth = 1.5, label = 'Python iter')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Extinction (nm$^2$)')
    ax.set_title('{} | ML: d={:.3f}s i={:.3f}s, PY: d={:.3f}s i={:.3f}s'.format(
        label, t_ml_d, t_ml_i, t_py_d, t_py_i))
    ax.legend(loc = 'best')
    ax.grid(True)

    # Relative error (direct-vs-direct, iter-vs-iter)
    ax = axes[1]
    rel_d = np.abs(ext_d - ml_ext_d) / (np.abs(ml_ext_d) + 1e-30)
    rel_i = np.abs(ext_i - ml_ext_i) / (np.abs(ml_ext_i) + 1e-30)
    ax.semilogy(enei, rel_d, 'b-', linewidth = 1.5, label = 'direct rel err')
    ax.semilogy(enei, rel_i, 'r--', linewidth = 1.5, label = 'iter rel err')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relative error')
    ax.set_title('max rel err: direct={:.2e}, iter={:.2e}'.format(
        np.max(rel_d), np.max(rel_i)))
    ax.legend(loc = 'best')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(savepath, dpi = 150)
    plt.close(fig)


# =========================================================================
# 1. Stat: BEMStat (direct) vs BEMStatIter
# =========================================================================

def run_stat(enei):
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    exc = PlaneWaveStat([1, 0, 0])
    n = len(enei)

    # --- direct ---
    bem_d = BEMStat(p)
    ext_d = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sig, bem_d = bem_d.solve(exc(p, enei[i]))
        ext_d[i] = exc.extinction(sig)
    t_d = time.time() - t0

    # --- iterative ---
    bem_i = BEMStatIter(p)
    ext_i = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sig, bem_i = bem_i.solve(exc(p, enei[i]))
        ext_i[i] = exc.extinction(sig)
    t_i = time.time() - t0

    return ext_d, ext_i, t_d, t_i


# =========================================================================
# 2. Ret: BEMRet (direct) vs BEMRetIter
# =========================================================================

def run_ret(enei):
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])
    n = len(enei)

    # --- direct ---
    bem_d = BEMRet(p)
    ext_d = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sig, bem_d = bem_d.solve(exc(p, enei[i]))
        ext_val = exc.extinction(sig)
        ext_d[i] = np.real(ext_val) if np.isscalar(ext_val) else np.real(ext_val[0])
    t_d = time.time() - t0

    # --- iterative ---
    bem_i = BEMRetIter(p)
    ext_i = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sig, bem_i = bem_i.solve(exc(p, enei[i]))
        ext_val = exc.extinction(sig)
        ext_i[i] = np.real(ext_val) if np.isscalar(ext_val) else np.real(ext_val[0])
    t_i = time.time() - t0

    return ext_d, ext_i, t_d, t_i


# =========================================================================
# 3. RetLayer: BEMRetLayer (direct) vs BEMRetLayerIter
# =========================================================================

def run_retlayer(enei):
    epstab = [EpsConst(1), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])

    # Use 64 faces (fewer than 144) to reduce memory for layer Green function
    sphere = trisphere(64, 20)
    z_min = sphere.pos[:, 2].min()
    sphere.shift([0, 0, -z_min + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    pol = np.array([1.0, 0.0, 0.0])
    dir_vec = np.array([0.0, 0.0, -1.0])
    n = len(enei)

    # --- direct ---
    exc_d = PlaneWaveRetLayer(pol, dir_vec, layer)
    bem_d = BEMRetLayer(p, layer)
    ext_d = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sys.stdout.write('  direct [{}/{}] lambda={:.0f}nm\n'.format(i + 1, n, enei[i]))
        sys.stdout.flush()
        exc_pot = exc_d(p, enei[i])
        sig, _ = bem_d.solve(exc_pot)
        ext_val = exc_d.extinction(sig)
        ext_d[i] = np.real(ext_val) if np.isscalar(ext_val) else np.real(ext_val[0])
    t_d = time.time() - t0

    # Free memory before iterative run
    del bem_d, exc_d
    gc.collect()

    # --- iterative ---
    exc_i = PlaneWaveRetLayer(pol, dir_vec, layer)
    bem_i = BEMRetLayerIter(p, layer)
    ext_i = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sys.stdout.write('  iter   [{}/{}] lambda={:.0f}nm\n'.format(i + 1, n, enei[i]))
        sys.stdout.flush()
        exc_pot = exc_i(p, enei[i])
        sig, _ = bem_i.solve(exc_pot)
        ext_val = exc_i.extinction(sig)
        ext_i[i] = np.real(ext_val) if np.isscalar(ext_val) else np.real(ext_val[0])
    t_i = time.time() - t0

    return ext_d, ext_i, t_d, t_i


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    timings = {}

    # ----- 1. Stat -----
    print('=' * 60)
    print('[1/3] Stat: BEMStat vs BEMStatIter  (400-800nm, 41pt)')
    print('=' * 60)
    enei_stat = np.linspace(400, 800, 41)
    ext_sd, ext_si, t_sd, t_si = run_stat(enei_stat)
    timings['stat_direct'] = t_sd
    timings['stat_iter'] = t_si
    print('  direct  : {:.4f} s'.format(t_sd))
    print('  iterative: {:.4f} s'.format(t_si))

    save_csv(os.path.join(DATA_DIR, 'python_stat_direct.csv'), enei_stat, ext_sd)
    save_csv(os.path.join(DATA_DIR, 'python_stat_iter.csv'), enei_stat, ext_si)
    plot_single(enei_stat, ext_sd, ext_si, 'Stat', t_sd, t_si,
        os.path.join(FIG_DIR, 'stat_python.png'))
    print('  saved stat_python.png')

    # self-consistency check
    rel_stat = np.max(np.abs(ext_sd - ext_si) / (np.abs(ext_sd) + 1e-30))
    print('  max rel diff (direct vs iter): {:.2e}'.format(rel_stat))

    # ----- 2. Ret -----
    print('=' * 60)
    print('[2/3] Ret: BEMRet vs BEMRetIter  (400-800nm, 41pt)')
    print('=' * 60)
    enei_ret = np.linspace(400, 800, 41)
    ext_rd, ext_ri, t_rd, t_ri = run_ret(enei_ret)
    timings['ret_direct'] = t_rd
    timings['ret_iter'] = t_ri
    print('  direct  : {:.4f} s'.format(t_rd))
    print('  iterative: {:.4f} s'.format(t_ri))

    save_csv(os.path.join(DATA_DIR, 'python_ret_direct.csv'), enei_ret, ext_rd)
    save_csv(os.path.join(DATA_DIR, 'python_ret_iter.csv'), enei_ret, ext_ri)
    plot_single(enei_ret, ext_rd, ext_ri, 'Ret', t_rd, t_ri,
        os.path.join(FIG_DIR, 'ret_python.png'))
    print('  saved ret_python.png')

    rel_ret = np.max(np.abs(ext_rd - ext_ri) / (np.abs(ext_rd) + 1e-30))
    print('  max rel diff (direct vs iter): {:.2e}'.format(rel_ret))

    # ----- 3. RetLayer -----
    print('=' * 60)
    print('[3/3] RetLayer: BEMRetLayer vs BEMRetLayerIter  (450-750nm, 11pt)')
    print('=' * 60)
    enei_rl = np.linspace(450, 750, 11)
    ext_rld, ext_rli, t_rld, t_rli = run_retlayer(enei_rl)
    timings['retlayer_direct'] = t_rld
    timings['retlayer_iter'] = t_rli
    print('  direct  : {:.4f} s'.format(t_rld))
    print('  iterative: {:.4f} s'.format(t_rli))

    save_csv(os.path.join(DATA_DIR, 'python_retlayer_direct.csv'), enei_rl, ext_rld)
    save_csv(os.path.join(DATA_DIR, 'python_retlayer_iter.csv'), enei_rl, ext_rli)
    plot_single(enei_rl, ext_rld, ext_rli, 'RetLayer', t_rld, t_rli,
        os.path.join(FIG_DIR, 'retlayer_python.png'))
    print('  saved retlayer_python.png')

    rel_rl = np.max(np.abs(ext_rld - ext_rli) / (np.abs(ext_rld) + 1e-30))
    print('  max rel diff (direct vs iter): {:.2e}'.format(rel_rl))

    # ----- Timing CSV -----
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('solver,time_seconds\n')
        for key in ['stat_direct', 'stat_iter',
                     'ret_direct', 'ret_iter',
                     'retlayer_direct', 'retlayer_iter']:
            f.write('{},{:.6f}\n'.format(key, timings[key]))

    # ----- Comparison with MATLAB (if data exists) -----
    tests = [
        ('stat', enei_stat, ext_sd, ext_si, t_sd, t_si),
        ('ret', enei_ret, ext_rd, ext_ri, t_rd, t_ri),
        ('retlayer', enei_rl, ext_rld, ext_rli, t_rld, t_rli),
    ]

    ml_timing = {}
    ml_timing_path = os.path.join(DATA_DIR, 'matlab_timing.csv')
    if os.path.exists(ml_timing_path):
        with open(ml_timing_path, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    ml_timing[parts[0]] = float(parts[1])

    for name, enei, py_d, py_i, t_py_d, t_py_i in tests:
        ml_d_path = os.path.join(DATA_DIR, 'matlab_{}_direct.csv'.format(name))
        ml_i_path = os.path.join(DATA_DIR, 'matlab_{}_iter.csv'.format(name))

        if os.path.exists(ml_d_path) and os.path.exists(ml_i_path):
            print('[info] Loading MATLAB {} data for comparison ...'.format(name))
            ml_d_data = np.genfromtxt(ml_d_path, delimiter = ',', skip_header = 1)
            ml_i_data = np.genfromtxt(ml_i_path, delimiter = ',', skip_header = 1)
            ml_ext_d = ml_d_data[:, 1]
            ml_ext_i = ml_i_data[:, 1]

            t_ml_d = ml_timing.get('{}_direct'.format(name), 0.0)
            t_ml_i = ml_timing.get('{}_iter'.format(name), 0.0)

            plot_comparison(enei, py_d, py_i, ml_ext_d, ml_ext_i,
                name, t_py_d, t_py_i, t_ml_d, t_ml_i,
                os.path.join(FIG_DIR, '{}_comparison.png'.format(name)))
            print('  saved {}_comparison.png'.format(name))

            rel_d = np.max(np.abs(py_d - ml_ext_d) / (np.abs(ml_ext_d) + 1e-30))
            rel_i = np.max(np.abs(py_i - ml_ext_i) / (np.abs(ml_ext_i) + 1e-30))
            print('  max rel err: direct={:.2e}, iter={:.2e}'.format(rel_d, rel_i))
        else:
            print('[info] MATLAB {} data not found, skipping comparison.'.format(name))

    # ----- Summary -----
    print('\n' + '=' * 60)
    print('Timing summary')
    print('=' * 60)
    print('  stat    : direct={:.3f}s  iter={:.3f}s'.format(t_sd, t_si))
    print('  ret     : direct={:.3f}s  iter={:.3f}s'.format(t_rd, t_ri))
    print('  retlayer: direct={:.3f}s  iter={:.3f}s'.format(t_rld, t_rli))
    print('\n[info] Python validation complete.')


if __name__ == '__main__':
    main()

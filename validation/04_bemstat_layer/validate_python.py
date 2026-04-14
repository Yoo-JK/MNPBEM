import os
import sys
import time

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMStatLayer
from mnpbem.simulation import PlaneWaveStatLayer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')


def setup_particle() -> Tuple[List[Any], LayerStructure, ComParticle]:

    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]

    layer = LayerStructure(epstab, [1, 3], [0.0])

    sphere = trisphere(144, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1])

    p = ComParticle(epstab, [sphere], [[2, 1]], [1])

    return epstab, layer, p


def run_normal(
        p: ComParticle,
        layer: LayerStructure,
        enei_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:

    exc = PlaneWaveStatLayer(np.array([1.0, 0.0, 0.0]), layer)
    # dir is already [0,0,-1] by default (normal incidence downward)

    bem = BEMStatLayer(p, layer)

    n_wl = len(enei_arr)
    ext = np.zeros(n_wl)
    sca = np.zeros(n_wl)

    print('[info] Normal incidence (theta=0)')
    t0 = time.time()

    for i, enei in enumerate(enei_arr):
        exc_pot = exc(p, enei)
        sig, _ = bem.solve(exc_pot)
        ext[i] = exc.extinction(sig)
        sca_val = exc.scattering(sig)
        sca[i] = float(np.real(sca_val[0] if isinstance(sca_val, tuple) else sca_val))
        print('  [{}/{}] lambda = {:.1f} nm, ext = {:.4e}, sca = {:.4e}'.format(
            i + 1, n_wl, enei, ext[i], sca[i]))

    elapsed = time.time() - t0
    print('[info] Normal incidence done in {:.2f} sec'.format(elapsed))

    return ext, sca, elapsed


def run_oblique(
        p: ComParticle,
        layer: LayerStructure,
        enei_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:

    theta = np.pi / 4.0
    pol = np.array([np.cos(theta), 0.0, np.sin(theta)])
    dir_vec = np.array([np.sin(theta), 0.0, -np.cos(theta)])

    exc = PlaneWaveStatLayer(pol, layer)
    # Override dir for oblique incidence (Python API only accepts pol in constructor)
    exc.dir = dir_vec.reshape(1, -1)

    bem = BEMStatLayer(p, layer)

    n_wl = len(enei_arr)
    ext = np.zeros(n_wl)
    sca = np.zeros(n_wl)

    print('[info] Oblique incidence (theta=45, TM)')
    t0 = time.time()

    for i, enei in enumerate(enei_arr):
        exc_pot = exc(p, enei)
        sig, _ = bem.solve(exc_pot)
        ext[i] = exc.extinction(sig)
        sca_val = exc.scattering(sig)
        sca[i] = float(np.real(sca_val[0] if isinstance(sca_val, tuple) else sca_val))
        print('  [{}/{}] lambda = {:.1f} nm, ext = {:.4e}, sca = {:.4e}'.format(
            i + 1, n_wl, enei, ext[i], sca[i]))

    elapsed = time.time() - t0
    print('[info] Oblique incidence done in {:.2f} sec'.format(elapsed))

    return ext, sca, elapsed


def save_csv(
        filepath: str,
        enei_arr: np.ndarray,
        ext: np.ndarray,
        sca: np.ndarray) -> None:

    with open(filepath, 'w') as f:
        f.write('wavelength_nm,extinction,scattering\n')
        for i in range(len(enei_arr)):
            f.write('{:.6f},{:.10e},{:.10e}\n'.format(enei_arr[i], ext[i], sca[i]))


def plot_results(
        enei_arr: np.ndarray,
        ext: np.ndarray,
        sca: np.ndarray,
        title_str: str,
        save_path: str) -> None:

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    ax.plot(enei_arr, ext, 'b-o', markersize = 3, label = 'Extinction')
    ax.plot(enei_arr, sca, 'r-s', markersize = 3, label = 'Scattering')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Cross section (nm^2)')
    ax.set_title(title_str)
    ax.legend()
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150)
    plt.close(fig)
    print('[info] Figure saved: {}'.format(save_path))


def main() -> None:

    os.makedirs(DATA_DIR, exist_ok = True)
    os.makedirs(FIG_DIR, exist_ok = True)

    enei_arr = np.linspace(400, 800, 41)

    epstab, layer, p = setup_particle()

    print('=' * 60)
    print('Python BEMStatLayer validation')
    print('  20nm Au sphere, 1nm gap, glass substrate (eps=2.25)')
    print('  Wavelength: 400-800nm, 41 points')
    print('=' * 60)

    # Normal incidence
    ext_n, sca_n, time_n = run_normal(p, layer, enei_arr)
    save_csv(os.path.join(DATA_DIR, 'python_normal.csv'), enei_arr, ext_n, sca_n)
    plot_results(enei_arr, ext_n, sca_n,
        'Python BEMStatLayer: Normal incidence',
        os.path.join(FIG_DIR, 'python_normal.png'))

    print()

    # Oblique incidence
    ext_o, sca_o, time_o = run_oblique(p, layer, enei_arr)
    save_csv(os.path.join(DATA_DIR, 'python_oblique.csv'), enei_arr, ext_o, sca_o)
    plot_results(enei_arr, ext_o, sca_o,
        'Python BEMStatLayer: Oblique incidence (theta=45, TM)',
        os.path.join(FIG_DIR, 'python_oblique.png'))

    # Save timing
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('case,time_sec\n')
        f.write('normal,{:.4f}\n'.format(time_n))
        f.write('oblique,{:.4f}\n'.format(time_o))
        f.write('total,{:.4f}\n'.format(time_n + time_o))

    # Summary
    print()
    print('=' * 60)
    print('Python BEMStatLayer validation complete')
    print('  Normal:  {:.2f} sec'.format(time_n))
    print('  Oblique: {:.2f} sec'.format(time_o))
    print('  Total:   {:.2f} sec'.format(time_n + time_o))
    print('=' * 60)

    idx_n = np.argmax(ext_n)
    print('Normal  - peak ext at {:.1f} nm: {:.4e} nm^2'.format(enei_arr[idx_n], ext_n[idx_n]))
    idx_o = np.argmax(ext_o)
    print('Oblique - peak ext at {:.1f} nm: {:.4e} nm^2'.format(enei_arr[idx_o], ext_o[idx_o]))


if __name__ == '__main__':
    main()

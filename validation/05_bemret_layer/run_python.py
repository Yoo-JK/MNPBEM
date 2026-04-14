import os
import sys
import time

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMRetLayer
from mnpbem.simulation import PlaneWaveRetLayer
from mnpbem.greenfun import GreenTabLayer


# ---- Setup ----

epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]

# Layer: vacuum(1) above z=0, glass(3) below
layer = LayerStructure(epstab, [1, 3], [0.0])

# 20nm diameter gold sphere, 1nm above substrate
sphere = trisphere(144, 20.0)
z_min = sphere.pos[:, 2].min()
sphere.shift([0, 0, -z_min + 1.0])

p = ComParticle(epstab, [sphere], [[2, 1]], [1])

# Tabulated Green function
tab = layer.tabspace(p)
gt = GreenTabLayer(layer, tab = tab)
gt.set(np.linspace(350, 800, 5))

# BEM solver with greentab
bem = BEMRetLayer(p, layer, greentab = gt)

# Normal-incidence planewave from above
pol = np.array([[1.0, 0.0, 0.0]])
dir_vec = np.array([[0.0, 0.0, -1.0]])
exc = PlaneWaveRetLayer(pol, dir_vec, layer)

# Wavelength grid (16 points)
enei_arr = np.linspace(450, 750, 16)
n_wl = len(enei_arr)

sca = np.zeros(n_wl)
ext = np.zeros(n_wl)
ab = np.zeros(n_wl)

# ---- Wavelength loop ----

print('[info] Python BEMRetLayer simulation starting ({} wavelengths)'.format(n_wl))
t0 = time.time()

for i, enei in enumerate(enei_arr):
    exc_pot = exc(p, enei)
    sig, _ = bem.solve(exc_pot)
    sca_val, _ = exc.scattering(sig)
    ext_val = exc.extinction(sig)
    sca[i] = sca_val
    ext[i] = ext_val
    ab[i] = ext_val - sca_val
    print('  [{}/{}] lambda = {:.1f} nm, ext = {:.4e}, sca = {:.4e}'.format(
        i + 1, n_wl, enei, ext_val, sca_val))

elapsed = time.time() - t0

# ---- Save results ----

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(output_dir, exist_ok = True)

# Cross sections
csv_path = os.path.join(output_dir, 'python_retlayer.csv')
with open(csv_path, 'w') as f:
    f.write('wavelength_nm,scattering,extinction,absorption\n')
    for i in range(n_wl):
        f.write('{:.6f},{:.10e},{:.10e},{:.10e}\n'.format(
            enei_arr[i], sca[i], ext[i], ab[i]))

# Timing
timing_path = os.path.join(output_dir, 'python_retlayer_timing.csv')
with open(timing_path, 'w') as f:
    f.write('total_sec,n_wavelengths,per_wavelength_sec\n')
    f.write('{:.6f},{},{:.6f}\n'.format(elapsed, n_wl, elapsed / n_wl))

print('\n[info] Python BEMRetLayer complete in {:.1f} sec ({:.2f} sec/wl)'.format(
    elapsed, elapsed / n_wl))
print('[info] Results saved to {}'.format(output_dir))

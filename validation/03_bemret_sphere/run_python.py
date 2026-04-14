"""
BEMRet Sphere Validation - Python

BEMRet + PlaneWaveRet([1,0,0],[0,0,1]), 20nm Au sphere, trisphere(144,20)
Computes extinction, scattering, absorption cross sections (400-800nm, 41pt)
Also computes MieRet analytical reference and timing.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet
from mnpbem.mie import MieRet

output_dir = '/home/yoojk20/workspace/MNPBEM/validation/03_bemret_sphere/data'

# Parameters
diameter = 20
nface = 144
enei = np.linspace(400, 800, 41)

# Setup
epstab = [EpsConst(1.0), EpsTable('gold.dat')]
sp = trisphere(nface, diameter)
p = ComParticle(epstab, [sp], [[2, 1]], [1])

# BEMRet simulation
print('Running BEMRet simulation...')
t0 = time.time()

bem = BEMRet(p)
pol = np.array([[1.0, 0.0, 0.0]])
dir_vec = np.array([[0.0, 0.0, 1.0]])
exc_obj = PlaneWaveRet(pol, dir_vec)

ext_bem = np.zeros(len(enei))
sca_bem = np.zeros(len(enei))

for i, e in enumerate(enei):
    exc = exc_obj.potential(p, e)
    sig, _ = bem.solve(exc)
    ext_bem[i] = np.real(exc_obj.extinction(sig))
    sca_val = exc_obj.scattering(sig)
    sca_bem[i] = np.real(sca_val[0] if isinstance(sca_val, tuple) else sca_val)

abs_bem = ext_bem - sca_bem
time_bem = time.time() - t0
print(f'  BEMRet done in {time_bem:.3f} s')

# MieRet simulation (analytical reference)
print('Running MieRet simulation...')
t0 = time.time()

mie = MieRet(EpsTable('gold.dat'), EpsConst(1.0), diameter)

ext_mie = mie.extinction(enei)
sca_mie = mie.scattering(enei)
abs_mie = ext_mie - sca_mie

time_mie = time.time() - t0
print(f'  MieRet done in {time_mie:.3f} s')

# Save BEMRet CSV
with open(os.path.join(output_dir, 'python_bemret.csv'), 'w') as f:
    f.write('wavelength,extinction,scattering,absorption\n')
    for i in range(len(enei)):
        f.write(f'{enei[i]:.6f},{ext_bem[i]:.10e},{sca_bem[i]:.10e},{abs_bem[i]:.10e}\n')

# Save MieRet CSV
with open(os.path.join(output_dir, 'python_mie.csv'), 'w') as f:
    f.write('wavelength,extinction,scattering,absorption\n')
    for i in range(len(enei)):
        f.write(f'{enei[i]:.6f},{ext_mie[i]:.10e},{sca_mie[i]:.10e},{abs_mie[i]:.10e}\n')

# Save timing CSV
with open(os.path.join(output_dir, 'python_timing.csv'), 'w') as f:
    f.write('method,time_seconds\n')
    f.write(f'bemret,{time_bem:.6f}\n')
    f.write(f'mieret,{time_mie:.6f}\n')

print(f'\nPython results saved to {output_dir}')
print('  python_bemret.csv')
print('  python_mie.csv')
print('  python_timing.csv')

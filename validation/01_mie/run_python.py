"""
Mie Theory Validation Tests (Python)

Compute extinction, scattering, absorption for 3 sub-tests:
  1. MieStat: 20nm Au sphere (quasistatic)
  2. MieRet: 100nm Au sphere (retarded)
  3. MieGans: [20,10,10]nm ellipsoid, x-pol and z-pol

Saves results to data/ directory as CSV files.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials import EpsTable, EpsConst
from mnpbem.mie import MieStat, MieRet, MieGans

# Common parameters
enei = np.linspace(400, 800, 41)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

timings = {}

# =============================================================================
# 1. MieStat: 20nm Au sphere
# =============================================================================
print('=== MieStat: 20nm Au sphere ===')
t0 = time.time()

epsin = EpsTable('gold.dat')
epsout = EpsConst(1.0)
mie_s = MieStat(epsin, epsout, diameter=20)

ext_s = mie_s.extinction(enei)
sca_s = mie_s.scattering(enei)
abs_s = mie_s.absorption(enei)

t_miestat = time.time() - t0
timings['miestat'] = t_miestat
print(f'  Time: {t_miestat:.4f} s')

# Save CSV
with open(os.path.join(data_dir, 'miestat_python.csv'), 'w') as f:
    f.write('wavelength_nm,extinction,scattering,absorption\n')
    for i in range(len(enei)):
        f.write(f'{enei[i]:.6f},{ext_s[i]:.15e},{sca_s[i]:.15e},{abs_s[i]:.15e}\n')
print('  Saved: miestat_python.csv')

# =============================================================================
# 2. MieRet: 100nm Au sphere
# =============================================================================
print('=== MieRet: 100nm Au sphere ===')
t0 = time.time()

epsin = EpsTable('gold.dat')
epsout = EpsConst(1.0)
mie_r = MieRet(epsin, epsout, diameter=100)

ext_r = mie_r.extinction(enei)
sca_r = mie_r.scattering(enei)
abs_r = mie_r.absorption(enei)

t_mieret = time.time() - t0
timings['mieret'] = t_mieret
print(f'  Time: {t_mieret:.4f} s')

# Save CSV
with open(os.path.join(data_dir, 'mieret_python.csv'), 'w') as f:
    f.write('wavelength_nm,extinction,scattering,absorption\n')
    for i in range(len(enei)):
        f.write(f'{enei[i]:.6f},{ext_r[i]:.15e},{sca_r[i]:.15e},{abs_r[i]:.15e}\n')
print('  Saved: mieret_python.csv')

# =============================================================================
# 3. MieGans: [20,10,10]nm ellipsoid
# =============================================================================
print('=== MieGans: [20,10,10]nm ellipsoid ===')
t0 = time.time()

epsin = EpsTable('gold.dat')
epsout = EpsConst(1.0)
mie_g = MieGans(epsin, epsout, ax=np.array([20.0, 10.0, 10.0]))

# x-polarization [1,0,0]
ext_gx = mie_g.extinction(enei, pol=np.array([1.0, 0.0, 0.0]))

# z-polarization [0,0,1]
ext_gz = mie_g.extinction(enei, pol=np.array([0.0, 0.0, 1.0]))

t_miegans = time.time() - t0
timings['miegans'] = t_miegans
print(f'  Time: {t_miegans:.4f} s')

# Save CSV (extinction only, two polarizations)
with open(os.path.join(data_dir, 'miegans_python.csv'), 'w') as f:
    f.write('wavelength_nm,extinction_xpol,extinction_zpol\n')
    for i in range(len(enei)):
        f.write(f'{enei[i]:.6f},{ext_gx[i]:.15e},{ext_gz[i]:.15e}\n')
print('  Saved: miegans_python.csv')

# =============================================================================
# Save timing
# =============================================================================
with open(os.path.join(data_dir, 'python_timing.csv'), 'w') as f:
    f.write('test,time_seconds\n')
    for name, t in timings.items():
        f.write(f'{name},{t:.6f}\n')

print(f'\n=== Timing saved to python_timing.csv ===')
print(f'  miestat: {t_miestat:.4f} s')
print(f'  mieret:  {t_mieret:.4f} s')
print(f'  miegans: {t_miegans:.4f} s')
print(f'  total:   {sum(timings.values()):.4f} s')
print('\nDone.')

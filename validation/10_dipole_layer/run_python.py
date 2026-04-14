import os
import sys
import time
import json
import warnings

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, ComPoint, LayerStructure
from mnpbem.bem import BEMStatLayer, BEMRetLayer
from mnpbem.simulation import DipoleStatLayer, DipoleRetLayer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok = True)
os.makedirs(FIG_DIR, exist_ok = True)


def log(msg: str) -> None:
    print(msg, flush = True)


# ============================================================
# Setup
# ============================================================

epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
layer = LayerStructure(epstab, [1, 3], [0.0])

# 20nm Au sphere, 1nm above substrate
sphere = trisphere(144, 20.0)
z_min = sphere.pos[:, 2].min()
sphere.shift([0, 0, -z_min + 1.0])
p = ComParticle(epstab, [sphere], [[2, 1]], [1])

# Dipole at z=25nm, z-oriented (single dipole -> ndip=1, npt=1)
pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))

# Wavelength grid: 500-700nm, 21 points
enei_arr = np.linspace(500, 700, 21)

results = {}
timings = {}


# ============================================================
# 1. DipoleStatLayer
# ============================================================

log('=' * 60)
log('1. DipoleStatLayer')
log('=' * 60)

dip_s = DipoleStatLayer(pt, layer, dip = np.array([0, 0, 1]))
bem_s = BEMStatLayer(p, layer)

tot_s = np.zeros(len(enei_arr))
rad_s = np.zeros(len(enei_arr))

t0 = time.time()
for i, enei in enumerate(enei_arr):
    exc = dip_s(p, enei)
    sig, _ = bem_s.solve(exc)
    tot, rad, rad0 = dip_s.decayrate(sig)
    tot_s[i] = tot.ravel()[0]
    rad_s[i] = rad.ravel()[0]
    log('  [{}/{}] lambda={:.1f} tot={:.6f} rad={:.6f}'.format(
        i + 1, len(enei_arr), enei, tot_s[i], rad_s[i]))
timings['statlayer'] = time.time() - t0
log('DipoleStatLayer time: {:.3f} s'.format(timings['statlayer']))

results['statlayer'] = {
    'enei': enei_arr.tolist(),
    'tot': tot_s.tolist(),
    'rad': rad_s.tolist()}


# ============================================================
# 2. DipoleRetLayer
# ============================================================

log('')
log('=' * 60)
log('2. DipoleRetLayer')
log('=' * 60)

dip_r = DipoleRetLayer(pt, layer, dip = np.array([0, 0, 1]))
bem_r = BEMRetLayer(p, layer)

tot_r = np.zeros(len(enei_arr))
rad_r = np.zeros(len(enei_arr))

t0 = time.time()
for i, enei in enumerate(enei_arr):
    exc = dip_r(p, enei)
    sig, _ = bem_r.solve(exc)
    tot, rad, rad0 = dip_r.decayrate(sig)
    tot_r[i] = tot.ravel()[0]
    rad_r[i] = rad.ravel()[0]
    log('  [{}/{}] lambda={:.1f} tot={:.6f} rad={:.6f}'.format(
        i + 1, len(enei_arr), enei, tot_r[i], rad_r[i]))
timings['retlayer'] = time.time() - t0
log('DipoleRetLayer time: {:.3f} s'.format(timings['retlayer']))

results['retlayer'] = {
    'enei': enei_arr.tolist(),
    'tot': tot_r.tolist(),
    'rad': rad_r.tolist()}


# ============================================================
# 3. decayrate0 (no particle, just layer)
# ============================================================

log('')
log('=' * 60)
log('3. decayrate0')
log('=' * 60)

tot0_s = np.zeros(len(enei_arr))
rad0_s = np.zeros(len(enei_arr))
tot0_r = np.zeros(len(enei_arr))
rad0_r = np.zeros(len(enei_arr))

t0 = time.time()
for i, enei in enumerate(enei_arr):
    t0s, r0s, _ = dip_s.decayrate0(enei)
    tot0_s[i] = t0s.ravel()[0]
    rad0_s[i] = r0s.ravel()[0]

    t0r, r0r, _ = dip_r.decayrate0(enei)
    tot0_r[i] = t0r.ravel()[0]
    rad0_r[i] = r0r.ravel()[0]

    log('  [{}/{}] lambda={:.1f} stat_tot={:.6f} ret_tot={:.6f}'.format(
        i + 1, len(enei_arr), enei, tot0_s[i], tot0_r[i]))
timings['decayrate0'] = time.time() - t0
log('decayrate0 time: {:.3f} s'.format(timings['decayrate0']))

results['decayrate0'] = {
    'enei': enei_arr.tolist(),
    'stat_tot': tot0_s.tolist(),
    'stat_rad': rad0_s.tolist(),
    'ret_tot': tot0_r.tolist(),
    'ret_rad': rad0_r.tolist()}


# ============================================================
# Save results
# ============================================================

with open(os.path.join(DATA_DIR, 'python_statlayer.csv'), 'w') as f:
    f.write('wavelength_nm,tot,rad\n')
    for i in range(len(enei_arr)):
        f.write('{:.6f},{:.10e},{:.10e}\n'.format(enei_arr[i], tot_s[i], rad_s[i]))

with open(os.path.join(DATA_DIR, 'python_retlayer.csv'), 'w') as f:
    f.write('wavelength_nm,tot,rad\n')
    for i in range(len(enei_arr)):
        f.write('{:.6f},{:.10e},{:.10e}\n'.format(enei_arr[i], tot_r[i], rad_r[i]))

with open(os.path.join(DATA_DIR, 'python_decayrate0.csv'), 'w') as f:
    f.write('wavelength_nm,stat_tot,stat_rad,ret_tot,ret_rad\n')
    for i in range(len(enei_arr)):
        f.write('{:.6f},{:.10e},{:.10e},{:.10e},{:.10e}\n'.format(
            enei_arr[i], tot0_s[i], rad0_s[i], tot0_r[i], rad0_r[i]))

with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
    f.write('test,time_sec\n')
    for k, v in timings.items():
        f.write('{},{:.6f}\n'.format(k, v))

with open(os.path.join(DATA_DIR, 'python_results.json'), 'w') as f:
    json.dump({'results': results, 'timings': timings}, f, indent = 2)


# ============================================================
# Plotting (9 figures)
# ============================================================

def load_matlab(name: str) -> Optional[Dict[str, np.ndarray]]:
    path = os.path.join(DATA_DIR, 'matlab_{}.csv'.format(name))
    if not os.path.exists(path):
        return None
    data = {}
    with open(path) as f:
        header = f.readline().strip().split(',')
        cols = {h: [] for h in header}
        for line in f:
            vals = line.strip().split(',')
            for h, v in zip(header, vals):
                cols[h].append(float(v))
    for h in header:
        data[h] = np.array(cols[h])
    return data


def plot_comparison(py_enei: np.ndarray,
        py_data: np.ndarray,
        py_label: str,
        m_data: Optional[Dict[str, np.ndarray]],
        m_key: str,
        title: str,
        ylabel: str,
        filename: str) -> None:

    fig, axes = plt.subplots(2, 1, figsize = (8, 6), height_ratios = [3, 1],
        gridspec_kw = {'hspace': 0.3})

    ax = axes[0]
    ax.plot(py_enei, py_data, 'b-o', markersize = 3, label = 'Python')
    if m_data is not None:
        m_enei = m_data['wavelength_nm']
        m_vals = m_data[m_key]
        ax.plot(m_enei, m_vals, 'r--s', markersize = 3, label = 'MATLAB')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha = 0.3)

    ax2 = axes[1]
    if m_data is not None:
        m_enei = m_data['wavelength_nm']
        m_vals = m_data[m_key]
        if len(m_enei) == len(py_enei) and np.allclose(m_enei, py_enei):
            rel_err = np.abs(py_data - m_vals) / np.maximum(np.abs(m_vals), 1e-30)
            ax2.semilogy(py_enei, rel_err, 'k-o', markersize = 3)
            ax2.set_ylabel('Relative Error')
            ax2.set_xlabel('Wavelength (nm)')
            rms = np.sqrt(np.mean(rel_err ** 2))
            ax2.set_title('RMS relative error: {:.4e}'.format(rms))
        else:
            ax2.text(0.5, 0.5, 'Wavelength grids differ',
                transform = ax2.transAxes, ha = 'center')
    else:
        ax2.text(0.5, 0.5, 'No MATLAB data available',
            transform = ax2.transAxes, ha = 'center')
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi = 150)
    plt.close()
    log('[info] Saved {}'.format(filename))


m_stat = load_matlab('statlayer')
m_ret = load_matlab('retlayer')
m_dr0 = load_matlab('decayrate0')

# 1. statlayer_tot
plot_comparison(enei_arr, tot_s, 'Python (tot)', m_stat, 'tot',
    'DipoleStatLayer: Total Decay Rate (z-dipole, 20nm Au/glass)',
    'Total Decay Rate', 'statlayer_tot.png')

# 2. statlayer_rad
plot_comparison(enei_arr, rad_s, 'Python (rad)', m_stat, 'rad',
    'DipoleStatLayer: Radiative Decay Rate (z-dipole, 20nm Au/glass)',
    'Radiative Decay Rate', 'statlayer_rad.png')

# 3. statlayer_comp
fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(enei_arr, tot_s, 'b-o', markersize = 3, label = 'Python tot')
ax.plot(enei_arr, rad_s, 'b--s', markersize = 3, label = 'Python rad')
if m_stat is not None:
    ax.plot(m_stat['wavelength_nm'], m_stat['tot'], 'r-o', markersize = 3, label = 'MATLAB tot')
    ax.plot(m_stat['wavelength_nm'], m_stat['rad'], 'r--s', markersize = 3, label = 'MATLAB rad')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Decay Rate')
ax.set_title('DipoleStatLayer: Tot+Rad Comparison')
ax.legend()
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'statlayer_comp.png'), dpi = 150)
plt.close()
log('[info] Saved statlayer_comp.png')

# 4. retlayer_tot
plot_comparison(enei_arr, tot_r, 'Python (tot)', m_ret, 'tot',
    'DipoleRetLayer: Total Decay Rate (z-dipole, 20nm Au/glass)',
    'Total Decay Rate', 'retlayer_tot.png')

# 5. retlayer_rad
plot_comparison(enei_arr, rad_r, 'Python (rad)', m_ret, 'rad',
    'DipoleRetLayer: Radiative Decay Rate (z-dipole, 20nm Au/glass)',
    'Radiative Decay Rate', 'retlayer_rad.png')

# 6. retlayer_comp
fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(enei_arr, tot_r, 'b-o', markersize = 3, label = 'Python tot')
ax.plot(enei_arr, rad_r, 'b--s', markersize = 3, label = 'Python rad')
if m_ret is not None:
    ax.plot(m_ret['wavelength_nm'], m_ret['tot'], 'r-o', markersize = 3, label = 'MATLAB tot')
    ax.plot(m_ret['wavelength_nm'], m_ret['rad'], 'r--s', markersize = 3, label = 'MATLAB rad')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Decay Rate')
ax.set_title('DipoleRetLayer: Tot+Rad Comparison')
ax.legend()
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'retlayer_comp.png'), dpi = 150)
plt.close()
log('[info] Saved retlayer_comp.png')

# 7. decayrate0_stat
plot_comparison(enei_arr, tot0_s, 'Python stat (tot)', m_dr0, 'stat_tot',
    'decayrate0: Stat Total (z-dipole, no particle, layer only)',
    'Total Decay Rate', 'decayrate0_stat.png')

# 8. decayrate0_ret
plot_comparison(enei_arr, tot0_r, 'Python ret (tot)', m_dr0, 'ret_tot',
    'decayrate0: Ret Total (z-dipole, no particle, layer only)',
    'Total Decay Rate', 'decayrate0_ret.png')

# 9. decayrate0_comp
fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(enei_arr, tot0_s, 'b-o', markersize = 3, label = 'Python stat tot')
ax.plot(enei_arr, rad0_s, 'b--s', markersize = 3, label = 'Python stat rad')
ax.plot(enei_arr, tot0_r, 'g-o', markersize = 3, label = 'Python ret tot')
ax.plot(enei_arr, rad0_r, 'g--s', markersize = 3, label = 'Python ret rad')
if m_dr0 is not None:
    ax.plot(m_dr0['wavelength_nm'], m_dr0['stat_tot'], 'r-o', markersize = 3, label = 'MATLAB stat tot')
    ax.plot(m_dr0['wavelength_nm'], m_dr0['stat_rad'], 'r--s', markersize = 3, label = 'MATLAB stat rad')
    ax.plot(m_dr0['wavelength_nm'], m_dr0['ret_tot'], 'm-o', markersize = 3, label = 'MATLAB ret tot')
    ax.plot(m_dr0['wavelength_nm'], m_dr0['ret_rad'], 'm--s', markersize = 3, label = 'MATLAB ret rad')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Decay Rate')
ax.set_title('decayrate0: Stat+Ret Comparison (no particle)')
ax.legend(fontsize = 8)
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'decayrate0_comp.png'), dpi = 150)
plt.close()
log('[info] Saved decayrate0_comp.png')


# ============================================================
# Summary
# ============================================================

log('')
log('=' * 60)
log('SUMMARY')
log('=' * 60)
total_time = sum(timings.values())
for k, v in timings.items():
    log('  {} : {:.3f} s'.format(k, v))
log('  total : {:.3f} s'.format(total_time))

if m_stat is not None or m_ret is not None or m_dr0 is not None:
    log('')
    log('MATLAB Comparison:')
    for name, py_vals, m_ref, m_keys in [
            ('statlayer_tot', tot_s, m_stat, 'tot'),
            ('statlayer_rad', rad_s, m_stat, 'rad'),
            ('retlayer_tot', tot_r, m_ret, 'tot'),
            ('retlayer_rad', rad_r, m_ret, 'rad'),
            ('decayrate0_stat_tot', tot0_s, m_dr0, 'stat_tot'),
            ('decayrate0_stat_rad', rad0_s, m_dr0, 'stat_rad'),
            ('decayrate0_ret_tot', tot0_r, m_dr0, 'ret_tot'),
            ('decayrate0_ret_rad', rad0_r, m_dr0, 'ret_rad')]:
        if m_ref is not None:
            m_vals = m_ref[m_keys]
            rel_err = np.abs(py_vals - m_vals) / np.maximum(np.abs(m_vals), 1e-30)
            rms = np.sqrt(np.mean(rel_err ** 2))
            maxerr = np.max(rel_err)
            status = 'OK' if rms < 0.10 else 'BAD'
            log('  {} : RMS={:.4e} MAX={:.4e} [{}]'.format(name, rms, maxerr, status))
        else:
            log('  {} : no MATLAB data'.format(name))
else:
    log('No MATLAB reference data found. Run run_matlab.m first.')

log('')
log('9 figures saved to {}'.format(FIG_DIR))
log('Data saved to {}'.format(DATA_DIR))

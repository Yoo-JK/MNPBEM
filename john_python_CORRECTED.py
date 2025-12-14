#!/usr/bin/env python
"""MNPBEM Full Test - Python (CORRECTED)
Tests: stat/ret x gold_table/drude
Spectrum: 400-800nm, 80nm Au sphere in water

FIXES APPLIED:
1. trisphere(144, diameter) instead of trisphere(144, radius)
2. PlaneWaveRet initialized with pinfty for spectrum calculations
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Path setup
mnpbem_path = os.path.join(os.getcwd(), 'MNPBEM')
sys.path.insert(0, mnpbem_path)
print(f"Added: {mnpbem_path}\n")

from mnpbem import (
    EpsConst, EpsTable, EpsDrude,
    trisphere, ComParticle,
    BEMStat, BEMRet,
    PlaneWaveStat, PlaneWaveRet,
    SpectrumRet
)

# Logging
log_file = open('python_test_corrected.log', 'w')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("=== MNPBEM Python Test (CORRECTED) ===")
log(f"Date: {datetime.now()}\n")

# Setup
diameter = 80  # nm
wavelengths = np.linspace(400, 800, 41)

# Materials
eps_water = EpsConst(1.33**2)
eps_au_table = EpsTable('gold.dat')
eps_au_drude = EpsDrude.gold()

# ✅ FIX #1: trisphere takes diameter, not radius
log(f"Creating: {diameter} nm Au sphere")
sphere = trisphere(144, diameter)  # ✅ CORRECTED: was trisphere(144, radius)

# Test 1: Quasistatic + Gold Table
log("\n--- Test 1: Quasistatic + Gold Table ---")
p1 = ComParticle([eps_water, eps_au_table], [sphere], [[2, 1]], 1)
bem1 = BEMStat(p1)
exc1 = PlaneWaveStat(pol=np.array([1, 0, 0]))

log("Computing...")
sca1 = []
for wl in wavelengths:
    sig, _ = bem1.solve(exc1(p1, wl))
    sca1.append(exc1.scattering(sig))
sca1 = np.array(sca1)
idx1 = np.argmax(sca1)
log(f"Peak: {sca1[idx1]:.3e} nm^2 at {wavelengths[idx1]:.0f} nm")

# Test 2: Retarded + Gold Table
log("\n--- Test 2: Retarded + Gold Table ---")
p2 = ComParticle([eps_water, eps_au_table], [sphere], [[2, 1]], 1)
bem2 = BEMRet(p2)

# ✅ FIX #2: Initialize spectrum for scattering calculation
pinfty = trisphere(256, 2)
exc2 = PlaneWaveRet(
    pol=np.array([1, 0, 0]),
    dir=np.array([0, 0, 1]),
    pinfty=pinfty,  # ✅ CORRECTED: added pinfty for spectrum
    medium=1
)

log("Computing...")
sca2 = []
for wl in wavelengths:
    sig, _ = bem2.solve(exc2(p2, wl))
    sca_val, _ = exc2.scattering(sig)
    sca2.append(sca_val)
sca2 = np.array(sca2)
idx2 = np.argmax(sca2)
log(f"Peak: {sca2[idx2]:.3e} nm^2 at {wavelengths[idx2]:.0f} nm")

# Test 3: Quasistatic + Drude
log("\n--- Test 3: Quasistatic + Drude ---")
p3 = ComParticle([eps_water, eps_au_drude], [sphere], [[2, 1]], 1)
bem3 = BEMStat(p3)
exc3 = PlaneWaveStat(pol=np.array([1, 0, 0]))

log("Computing...")
sca3 = []
for wl in wavelengths:
    sig, _ = bem3.solve(exc3(p3, wl))
    sca3.append(exc3.scattering(sig))
sca3 = np.array(sca3)
idx3 = np.argmax(sca3)
log(f"Peak: {sca3[idx3]:.3e} nm^2 at {wavelengths[idx3]:.0f} nm")

# Test 4: Retarded + Drude
log("\n--- Test 4: Retarded + Drude ---")
p4 = ComParticle([eps_water, eps_au_drude], [sphere], [[2, 1]], 1)
bem4 = BEMRet(p4)
exc4 = PlaneWaveRet(
    pol=np.array([1, 0, 0]),
    dir=np.array([0, 0, 1]),
    pinfty=pinfty,  # ✅ CORRECTED: added pinfty for spectrum
    medium=1
)

log("Computing...")
sca4 = []
for wl in wavelengths:
    sig, _ = bem4.solve(exc4(p4, wl))
    sca_val, _ = exc4.scattering(sig)
    sca4.append(sca_val)
sca4 = np.array(sca4)
idx4 = np.argmax(sca4)
log(f"Peak: {sca4[idx4]:.3e} nm^2 at {wavelengths[idx4]:.0f} nm")

# Summary
log("\n=== Summary ===")
log(f"Test 1: {wavelengths[idx1]:.0f} nm, {sca1[idx1]:.3e} nm^2")
log(f"Test 2: {wavelengths[idx2]:.0f} nm, {sca2[idx2]:.3e} nm^2")
log(f"Test 3: {wavelengths[idx3]:.0f} nm, {sca3[idx3]:.3e} nm^2")
log(f"Test 4: {wavelengths[idx4]:.0f} nm, {sca4[idx4]:.3e} nm^2")

# Plot
log("\n--- Plotting ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(wavelengths, sca1, 'b-', lw=2)
axes[0,0].set_title('Test 1: Quasistatic + Gold Table')
axes[0,0].set_xlabel('Wavelength (nm)'); axes[0,0].set_ylabel('Scattering (nm²)')
axes[0,0].grid(True)

axes[0,1].plot(wavelengths, sca2, 'r-', lw=2)
axes[0,1].set_title('Test 2: Retarded + Gold Table')
axes[0,1].set_xlabel('Wavelength (nm)'); axes[0,1].set_ylabel('Scattering (nm²)')
axes[0,1].grid(True)

axes[1,0].plot(wavelengths, sca3, 'g-', lw=2)
axes[1,0].set_title('Test 3: Quasistatic + Drude')
axes[1,0].set_xlabel('Wavelength (nm)'); axes[1,0].set_ylabel('Scattering (nm²)')
axes[1,0].grid(True)

axes[1,1].plot(wavelengths, sca4, 'm-', lw=2)
axes[1,1].set_title('Test 4: Retarded + Drude')
axes[1,1].set_xlabel('Wavelength (nm)'); axes[1,1].set_ylabel('Scattering (nm²)')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('python_results_corrected.png', dpi=150)
log("Saved: python_results_corrected.png")

log("\n=== Complete ===")
log_file.close()

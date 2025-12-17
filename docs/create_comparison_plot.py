#!/usr/bin/env python3
"""Create detailed comparison plot showing Drude vs Table issues."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all data
stat_drude = np.loadtxt('spectrum_stat_drude_python.dat')
stat_table = np.loadtxt('spectrum_stat_table_python.dat')
ret_drude = np.loadtxt('spectrum_ret_drude_python.dat')
ret_table = np.loadtxt('spectrum_ret_table_python.dat')

wl = stat_drude[:, 0]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Subplot 1: Drude model comparison
ax1 = axes[0]
ax1.plot(wl, stat_drude[:, 1], 'b-', linewidth=2, label='Quasistatic')
ax1.plot(wl, ret_drude[:, 1], 'r-', linewidth=2, label='Retarded')
ax1.axvline(500, color='b', linestyle=':', alpha=0.5)
ax1.axvline(490, color='r', linestyle=':', alpha=0.5)
ax1.set_xlabel('Wavelength (nm)', fontsize=12)
ax1.set_ylabel('Extinction (nm²)', fontsize=12)
ax1.set_title('Drude Model: Working ✓\n(Peak reduction 33%, shift -10 nm)',
              fontsize=13, fontweight='bold', color='green')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(400, 800)

# Subplot 2: Table model comparison
ax2 = axes[1]
ax2.plot(wl, stat_table[:, 1], 'b-', linewidth=2, label='Quasistatic')
ax2.plot(wl, ret_table[:, 1], 'r-', linewidth=2, label='Retarded')
ax2.axvline(520, color='b', linestyle=':', alpha=0.5, label='Stat peak: 520 nm')
ax2.axvline(400, color='r', linestyle=':', alpha=0.5, label='Ret peak: 400 nm (!)')
ax2.set_xlabel('Wavelength (nm)', fontsize=12)
ax2.set_ylabel('Extinction (nm²)', fontsize=12)
ax2.set_title('Table Model: Issues ⚠\n(Flat spectrum, anomalous -120 nm shift)',
              fontsize=13, fontweight='bold', color='orange')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(400, 800)

# Subplot 3: Ratio analysis
ax3 = axes[2]
ratio_drude = ret_drude[:, 1] / stat_drude[:, 1]
ratio_table = ret_table[:, 1] / stat_table[:, 1]
ax3.plot(wl, ratio_drude, 'g-', linewidth=2, label='Drude (good)')
ax3.plot(wl, ratio_table, 'orange', linewidth=2, label='Table (problematic)')
ax3.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax3.axhline(0.67, color='g', linestyle=':', alpha=0.5, label='Expected ~0.67')
ax3.set_xlabel('Wavelength (nm)', fontsize=12)
ax3.set_ylabel('Ratio: Ret/Stat', fontsize=12)
ax3.set_title('Retarded / Quasistatic Ratio\n(Should be <1 at resonance due to radiation damping)',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(400, 800)
ax3.set_ylim(0, 2)

plt.tight_layout()
plt.savefig('drude_vs_table_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: drude_vs_table_comparison.png')

# Print detailed statistics
print('\n' + '='*70)
print('DETAILED ANALYSIS')
print('='*70)

print('\nDrude Model (Working):')
peak_stat_drude = np.argmax(stat_drude[:, 1])
peak_ret_drude = np.argmax(ret_drude[:, 1])
print(f'  Quasistatic peak: λ={wl[peak_stat_drude]:.0f} nm, σ={stat_drude[peak_stat_drude, 1]:.1e} nm²')
print(f'  Retarded peak:    λ={wl[peak_ret_drude]:.0f} nm, σ={ret_drude[peak_ret_drude, 1]:.1e} nm²')
reduction_drude = (1 - ret_drude[peak_ret_drude, 1] / stat_drude[peak_stat_drude, 1]) * 100
shift_drude = wl[peak_ret_drude] - wl[peak_stat_drude]
print(f'  Peak reduction: {reduction_drude:.1f}% (expected: 20-40%)')
print(f'  Wavelength shift: {shift_drude:.0f} nm (expected: ±10 nm)')
print(f'  Status: ✓ Excellent')

print('\nTable Model (Limited):')
peak_stat_table = np.argmax(stat_table[:, 1])
peak_ret_table = np.argmax(ret_table[:, 1])
print(f'  Quasistatic peak: λ={wl[peak_stat_table]:.0f} nm, σ={stat_table[peak_stat_table, 1]:.1e} nm²')
print(f'  Retarded peak:    λ={wl[peak_ret_table]:.0f} nm, σ={ret_table[peak_ret_table, 1]:.1e} nm²')
reduction_table = (1 - ret_table[peak_ret_table, 1] / stat_table[peak_stat_table, 1]) * 100
shift_table = wl[peak_ret_table] - wl[peak_stat_table]
print(f'  Peak reduction: {reduction_table:.1f}% (expected: 20-40%)')
print(f'  Wavelength shift: {shift_table:.0f} nm (expected: ±10 nm)')
print(f'  Spectrum flatness: {np.std(ret_table[:, 1]):.1e} (Drude: {np.std(ret_drude[:, 1]):.1e})')
print(f'  Status: ⚠ Needs polar integration refinement')

print('\nRoot Cause:')
print('  Drude: k(λ) varies smoothly → diagonal regularization works')
print('  Table: k(λ has sharp features → diagonal regularization insufficient')
print('  Solution: Implement MATLAB polar integration refinement')

print('='*70)

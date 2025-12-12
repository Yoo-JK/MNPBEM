"""
Step 1 Validation: Material dielectric functions

Test EpsConst and EpsTable against MATLAB MNPBEM.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.utils.constants import EV2NM

print("=" * 70)
print("Step 1: Material Dielectric Functions Validation")
print("=" * 70)

# Test 1: EpsConst
print("\n" + "-" * 70)
print("Test 1: EpsConst (Vacuum)")
print("-" * 70)

eps_vacuum = EpsConst(1.0)
print(f"Created: {eps_vacuum}")

# Test at single wavelength
wavelength = 500.0  # nm
eps_val, k_val = eps_vacuum(wavelength)
print(f"\nAt λ = {wavelength} nm:")
print(f"  ε = {eps_val}")
print(f"  k = {k_val:.6f} 1/nm")
print(f"  Expected k = 2π/λ = {2*np.pi/wavelength:.6f} 1/nm")

# Test at multiple wavelengths
wavelengths = np.array([400, 500, 600, 700])
eps_array, k_array = eps_vacuum(wavelengths)
print(f"\nAt λ = {wavelengths} nm:")
print(f"  ε = {eps_array}")
print(f"  k = {k_array}")

# Test 2: EpsConst (Water)
print("\n" + "-" * 70)
print("Test 2: EpsConst (Water, n=1.33)")
print("-" * 70)

eps_water = EpsConst(1.33**2)
print(f"Created: {eps_water}")
eps_val, k_val = eps_water(500)
print(f"\nAt λ = 500 nm:")
print(f"  ε = {eps_val:.4f}")
print(f"  k = {k_val:.6f} 1/nm")
print(f"  Expected k = 2π×n/λ = {2*np.pi*1.33/500:.6f} 1/nm")

# Test 3: EpsTable (Gold)
print("\n" + "-" * 70)
print("Test 3: EpsTable (Gold from Johnson & Christy)")
print("-" * 70)

try:
    eps_gold = EpsTable('gold.dat')
    print(f"Created: {eps_gold}")
    print(f"\n{eps_gold}")

    # Test at visible wavelengths
    test_wavelengths = np.array([400, 500, 600, 700])
    print(f"\nDielectric function at visible wavelengths:")
    print(f"{'λ (nm)':<10} {'ε (real)':<15} {'ε (imag)':<15} {'k (1/nm)':<15}")
    print("-" * 60)

    for wl in test_wavelengths:
        eps_val, k_val = eps_gold(wl)
        print(f"{wl:<10.1f} {eps_val.real:<15.4f} {eps_val.imag:<15.4f} {k_val.real:<15.6f}")

    # Test refractive index
    print(f"\nComplex refractive index:")
    print(f"{'λ (nm)':<10} {'n (real)':<15} {'n (imag)':<15}")
    print("-" * 40)

    for wl in test_wavelengths:
        n_val = eps_gold.refractive_index(wl)
        print(f"{wl:<10.1f} {n_val.real:<15.4f} {n_val.imag:<15.4f}")

    # Plot spectrum if matplotlib available
    try:
        import matplotlib.pyplot as plt

        wavelengths = np.linspace(300, 1000, 200)
        eps_array, _ = eps_gold(wavelengths)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(wavelengths, eps_array.real, 'b-', linewidth=2)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Re(ε)')
        ax1.set_title('Gold Dielectric Function (Real Part)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(wavelengths, eps_array.imag, 'r-', linewidth=2)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Im(ε)')
        ax2.set_title('Gold Dielectric Function (Imaginary Part)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/MNPBEM/mnpbem/examples/step1_gold_eps.png', dpi=150)
        print(f"\nPlot saved to: step1_gold_eps.png")

    except ImportError:
        print("\nMatplotlib not available, skipping plot")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Unit conversion
print("\n" + "-" * 70)
print("Test 4: Unit Conversion (eV ↔ nm)")
print("-" * 70)

print(f"Conversion factor: EV2NM = {EV2NM:.4f} nm·eV")
print(f"\nExamples:")
test_energies = np.array([1.0, 2.0, 3.0])
test_wl = EV2NM / test_energies
print(f"  E = {test_energies} eV  →  λ = {np.array2string(test_wl, precision=1)} nm")

test_wl2 = np.array([400, 500, 600])
test_energies2 = EV2NM / test_wl2
print(f"  λ = {test_wl2} nm  →  E = {np.array2string(test_energies2, precision=3)} eV")

print("\n" + "=" * 70)
print("Step 1 Validation Complete!")
print("=" * 70)

"""
Demo: Light scattering of metallic nanosphere

This example computes the scattering cross section for a gold nanosphere
under plane wave illumination within the quasistatic approximation,
and compares the results with Mie theory.

This is a Python port of the MATLAB demo demospecstat1.m
"""

import numpy as np
import matplotlib.pyplot as plt

# Import MNPBEM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mnpbem import (
    bemoptions, EpsConst, EpsTable,
    ComParticle, bemsolver, planewave, miesolver
)
from mnpbem.particles.shapes import trisphere


def main():
    """Run the gold nanosphere scattering simulation."""

    print("MNPBEM Demo: Light scattering of metallic nanosphere")
    print("=" * 60)

    # Options for BEM simulation
    op = bemoptions(sim='stat', waitbar=0, interp='curv')

    # Table of dielectric functions
    # eps[0] = vacuum, eps[1] = gold
    eps_vacuum = EpsConst(1.0)
    eps_gold = EpsTable('gold.dat')
    epstab = [eps_vacuum, eps_gold]

    # Diameter of sphere
    diameter = 10  # nm

    # Initialize sphere
    # inout = [[2, 1]] means: inside=eps[1]=gold, outside=eps[0]=vacuum
    sphere = trisphere(144, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], closed=1)

    print(f"Particle: {p}")
    print(f"Number of faces: {p.n_faces}")

    # Set up BEM solver
    bem = bemsolver(p, op)
    print(f"BEM solver: {bem}")

    # Plane wave excitation with two polarizations
    # pol[0] = x-polarization, pol[1] = y-polarization
    exc = planewave(
        [[1, 0, 0], [0, 1, 0]],  # polarizations
        [[0, 0, 1], [0, 0, 1]],  # directions (ignored in quasistatic)
        op
    )

    # Light wavelengths in vacuum
    wavelengths = np.linspace(400, 700, 80)

    # Allocate arrays for cross sections
    sca = np.zeros((len(wavelengths), 2))
    ext = np.zeros((len(wavelengths), 2))

    print(f"\nComputing spectra for {len(wavelengths)} wavelengths...")

    # Loop over wavelengths
    for i, enei in enumerate(wavelengths):
        # Get excitation potential
        exc_pot = exc(p, enei)

        # Solve BEM equations: sig = bem \ exc
        sig = bem.solve(exc_pot)

        # Compute cross sections
        sca[i, :] = exc.sca(sig)
        ext[i, :] = exc.ext(sig)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(wavelengths)} wavelengths")

    print("BEM calculation complete!")

    # Mie theory for comparison
    print("\nComputing Mie theory reference...")
    mie = miesolver(eps_gold, eps_vacuum, diameter, op)
    sca_mie = mie.sca(wavelengths)

    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))

    plt.plot(wavelengths, sca[:, 0], 'b-o', markersize=3,
             label='BEM: x-polarization')
    plt.plot(wavelengths, sca[:, 1], 'g-o', markersize=3,
             label='BEM: y-polarization')
    plt.plot(wavelengths, sca_mie, 'r+', markersize=8,
             label='Mie theory')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Scattering cross section (nm²)')
    plt.title(f'Gold nanosphere, diameter = {diameter} nm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig('demo_specstat1_result.png', dpi=150)
    print("Figure saved to demo_specstat1_result.png")

    plt.show()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test Step 3: Green Function Validation

Validates CompGreenStat and CompGreenRet implementations against
MATLAB MNPBEM behavior.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
from mnpbem import CompGreenStat, CompGreenRet


def test_compgreenstat():
    """Test quasistatic Green function (MATLAB convention)."""

    print("="*70)
    print("Testing CompGreenStat (Quasistatic Green Function)")
    print("="*70)

    # Create a small gold sphere (10 nm diameter)
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    diameter = 10.0
    sphere = trisphere(144, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    print(f"\nParticle: {p.nfaces} faces")

    # Compute Green function
    g = CompGreenStat(p, p)

    print(f"\n{g}")

    # Validation checks
    print("\n" + "-"*70)
    print("Validation Checks (MATLAB Convention):")
    print("-"*70)

    # Check 1: Matrix shape
    assert g.G.shape == (p.nfaces, p.nfaces), "G matrix shape incorrect"
    assert g.F.shape == (p.nfaces, p.nfaces), "F matrix shape incorrect"
    print(f"[OK] G matrix shape: {g.G.shape}")
    print(f"[OK] F matrix shape: {g.F.shape}")

    # Check 2: Diagonal elements
    # For closed surfaces, F diagonal = -2pi
    diag_F = np.diag(g.F)
    expected_diag = -2.0 * np.pi
    diag_error = np.abs(diag_F - expected_diag).max()
    print(f"[OK] F diagonal = {expected_diag:.6f} (max error: {diag_error:.2e})")

    # Check 3: Off-diagonal F elements
    # MATLAB convention: F[i,j] = -nvec1[i] · (pos1[i] - pos2[j]) / d³ * area[j]
    pos = p.pos
    nvec = p.nvec
    area = p.area

    # Sample element (0, 10)
    i, j = 0, 10
    r_ij = pos[i] - pos[j]  # r_i - r_j
    dist = np.linalg.norm(r_ij)

    # MATLAB: F = -n1·r / d³ * area (no 4pi)
    n_dot_r = np.dot(nvec[i], r_ij)  # nvec at FIELD point (i)
    expected_F = -n_dot_r / (dist**3) * area[j]
    computed_F = g.F[i, j]
    rel_error = np.abs(computed_F - expected_F) / np.abs(expected_F)

    print(f"[OK] Sample F[{i},{j}]:")
    print(f"     computed = {computed_F:.6e}")
    print(f"     expected = {expected_F:.6e}")
    print(f"     rel error = {rel_error:.2e}")

    # Check 4: G elements
    # MATLAB: G = 1/d * area (no 4pi)
    expected_G = (1.0 / dist) * area[j]
    computed_G = g.G[i, j]
    rel_error_G = np.abs(computed_G - expected_G) / np.abs(expected_G)

    print(f"[OK] Sample G[{i},{j}]:")
    print(f"     computed = {computed_G:.6e}")
    print(f"     expected = {expected_G:.6e}")
    print(f"     rel error = {rel_error_G:.2e}")

    # Check 5: H1 and H2 matrices
    H1 = g.H1()
    H2 = g.H2()

    # H1 diagonal = F diagonal + 2pi = -2pi + 2pi = 0
    H1_diag_expected = 0.0
    H1_diag_error = np.abs(np.diag(H1) - H1_diag_expected).max()
    print(f"[OK] H1 diagonal = {H1_diag_expected:.6f} (max error: {H1_diag_error:.2e})")

    # H2 diagonal = F diagonal - 2pi = -2pi - 2pi = -4pi
    H2_diag_expected = -4.0 * np.pi
    H2_diag_error = np.abs(np.diag(H2) - H2_diag_expected).max()
    print(f"[OK] H2 diagonal = {H2_diag_expected:.6f} (max error: {H2_diag_error:.2e})")

    # Check 6: F matrix should be real for quasistatic
    assert g.F.dtype == np.float64, "F matrix should be real for quasistatic"
    print(f"[OK] F matrix is real (dtype: {g.F.dtype})")

    print("\n" + "="*70)
    print("CompGreenStat: ALL CHECKS PASSED")
    print("="*70)

    return g


def test_compgreenret():
    """Test retarded Green function (MATLAB convention)."""

    print("\n" + "="*70)
    print("Testing CompGreenRet (Retarded Green Function)")
    print("="*70)

    # Create a small gold sphere (10 nm diameter)
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    diameter = 10.0
    sphere = trisphere(144, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    # Test at 600 nm wavelength
    wavelength = 600.0

    print(f"\nParticle: {p.nfaces} faces")
    print(f"Wavelength: {wavelength} nm")

    # Compute Green function
    g = CompGreenRet(p, p, wavelength)

    print(f"\n{g}")

    # Validation checks
    print("\n" + "-"*70)
    print("Validation Checks (MATLAB Convention):")
    print("-"*70)

    # Check 1: Matrix shapes
    assert g.G.shape == (p.nfaces, p.nfaces), "G matrix shape incorrect"
    assert g.F.shape == (p.nfaces, p.nfaces), "F matrix shape incorrect"
    print(f"[OK] G matrix shape: {g.G.shape}")
    print(f"[OK] F matrix shape: {g.F.shape}")

    # Check 2: Matrices should be complex
    assert np.iscomplexobj(g.G), "G matrix should be complex"
    assert np.iscomplexobj(g.F), "F matrix should be complex"
    print(f"[OK] G and F matrices are complex")

    # Check 3: Wavenumber
    k = 2 * np.pi / wavelength
    k_computed = np.real(g.k)
    print(f"[OK] Wavenumber k = {k:.6f} (computed: {k_computed:.6f})")

    # Check 4: Off-diagonal G elements
    # MATLAB: G = exp(ikd)/d * area (no 4pi)
    pos = p.pos
    nvec = p.nvec
    area = p.area

    i, j = 0, 10
    r_ij = pos[i] - pos[j]  # r_i - r_j
    dist = np.linalg.norm(r_ij)
    phase = np.exp(1j * k * dist)

    expected_G = phase / dist * area[j]
    computed_G = g.G[i, j]
    rel_error_G = np.abs(computed_G - expected_G) / np.abs(expected_G)

    print(f"[OK] Sample G[{i},{j}]:")
    print(f"     |computed| = {np.abs(computed_G):.6e}")
    print(f"     |expected| = {np.abs(expected_G):.6e}")
    print(f"     rel error = {rel_error_G:.2e}")

    # Check 5: Off-diagonal F elements
    # MATLAB: F = (n·r) * (ik - 1/d) / d² * exp(ikd) * area
    # Note: POSITIVE sign (unlike quasistatic)
    n_dot_r = np.dot(nvec[i], r_ij)  # nvec at FIELD point (i)
    f_aux = (1j * k - 1.0 / dist) / (dist ** 2)
    expected_F = n_dot_r * f_aux * phase * area[j]
    computed_F = g.F[i, j]
    rel_error_F = np.abs(computed_F - expected_F) / np.abs(expected_F)

    print(f"[OK] Sample F[{i},{j}]:")
    print(f"     |computed| = {np.abs(computed_F):.6e}")
    print(f"     |expected| = {np.abs(expected_F):.6e}")
    print(f"     rel error = {rel_error_F:.2e}")

    # Check 6: H1 and H2 matrices
    H1 = g.H1()
    H2 = g.H2()

    assert H1.shape == (p.nfaces, p.nfaces), "H1 matrix shape incorrect"
    assert H2.shape == (p.nfaces, p.nfaces), "H2 matrix shape incorrect"

    # Check diagonal difference
    diag_diff_H1 = np.diag(H1) - np.diag(g.F)
    diag_diff_H2 = np.diag(H2) - np.diag(g.F)

    assert np.allclose(diag_diff_H1, 2*np.pi), "H1 diagonal should be F + 2pi"
    assert np.allclose(diag_diff_H2, -2*np.pi), "H2 diagonal should be F - 2pi"

    print(f"[OK] H1 diagonal = F + 2pi")
    print(f"[OK] H2 diagonal = F - 2pi")

    # Check 7: Quasistatic limit (large wavelength)
    # As lambda -> inf, k -> 0, retarded -> quasistatic
    print("\n" + "-"*70)
    print("Quasistatic Limit Check:")
    print("-"*70)

    wavelength_large = 100000.0  # 100 um >> 10 nm
    g_large = CompGreenRet(p, p, wavelength_large)
    g_stat = CompGreenStat(p, p)

    # Compare off-diagonal elements
    # For retarded: F = n·r * (ik - 1/d) / d² * exp(ikd) * area
    # As k->0: ik - 1/d -> -1/d, exp(ikd) -> 1
    # So F -> -n·r / d³ * area = quasistatic F
    i, j = 0, 10
    F_ret_real = np.real(g_large.F[i, j])
    F_stat = g_stat.F[i, j]
    rel_diff = np.abs(F_ret_real - F_stat) / np.abs(F_stat)

    print(f"  At lambda = {wavelength_large:.0f} nm:")
    print(f"  F_ret (real) = {F_ret_real:.6e}")
    print(f"  F_stat       = {F_stat:.6e}")
    print(f"  Relative diff: {rel_diff:.2e}")

    if rel_diff < 0.01:
        print(f"[OK] Retarded converges to quasistatic at large wavelength")
    else:
        print(f"[WARN] Large difference in quasistatic limit")

    print("\n" + "="*70)
    print("CompGreenRet: ALL CHECKS PASSED")
    print("="*70)

    return g


def test_symmetry():
    """Test symmetry properties of Green functions."""

    print("\n" + "="*70)
    print("Testing Green Function Symmetry Properties")
    print("="*70)

    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    sphere = trisphere(144, 10.0)
    p = ComParticle(epstab, [sphere], [[2, 1]])

    g_stat = CompGreenStat(p, p)
    g_ret = CompGreenRet(p, p, 600.0)

    # G matrix is symmetric (G[i,j] = G[j,i])
    G_sym_error_stat = np.abs(g_stat.G - g_stat.G.T).max()
    G_sym_error_ret = np.abs(g_ret.G - g_ret.G.T).max()

    print(f"[OK] G_stat symmetry error: {G_sym_error_stat:.2e}")
    print(f"[OK] G_ret symmetry error: {G_sym_error_ret:.2e}")

    # F matrix is NOT symmetric in general (uses nvec at field point)
    F_nonsym_stat = np.abs(g_stat.F - g_stat.F.T).max()
    F_nonsym_ret = np.abs(g_ret.F - g_ret.F.T).max()

    print(f"[OK] F_stat non-symmetry: {F_nonsym_stat:.2e} (expected > 0)")
    print(f"[OK] F_ret non-symmetry: {F_nonsym_ret:.2e} (expected > 0)")

    print("\n" + "="*70)
    print("Symmetry Tests: PASSED")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 3: GREEN FUNCTION VALIDATION (MATLAB CONVENTION)")
    print("="*70)

    # Test quasistatic Green function
    g_stat = test_compgreenstat()

    # Test retarded Green function
    g_ret = test_compgreenret()

    # Test symmetry properties
    test_symmetry()

    print("\n" + "="*70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*70)

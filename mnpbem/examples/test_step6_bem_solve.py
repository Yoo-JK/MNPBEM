"""
Test Step 6: BEM Equation Solving

This test validates the BEM solving workflow matching MATLAB MNPBEM exactly:
1. BEMStat.solve() for quasistatic approximation
2. BEMRet.solve() for retarded/full Maxwell equations
3. field() and potential() computation from solutions
4. Integration with excitation classes from Step 5

MATLAB equivalents:
    sig = bem \ exc(p, enei)  % Solve BEM equations
    field = field(bem, sig, inout)  % Compute fields
    pot = potential(bem, sig, inout)  % Compute potentials
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/MNPBEM')


def test_bemstat_solve():
    """Test BEMStat.solve() for quasistatic approximation."""
    print("\n=== Test: BEMStat.solve() ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat

    # Create gold sphere (simplified with constant eps for testing)
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)  # Simplified gold-like
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create BEM solver
    bem = BEMStat(p)

    # Create plane wave excitation
    exc = PlaneWaveStat([1, 0, 0])  # x-polarized

    # Get excitation at wavelength
    enei = 500.0  # nm
    exc_dict = exc.potential(p, enei)

    print(f"Excitation phip shape: {exc_dict['phip'].shape}")
    print(f"phip range: {exc_dict['phip'].min():.4f} to {exc_dict['phip'].max():.4f}")

    # Solve BEM equations
    sig = bem.solve(exc_dict)

    print(f"Solution sig shape: {sig['sig'].shape}")
    print(f"sig range: {sig['sig'].min():.4e} to {sig['sig'].max():.4e}")

    # Verify solution is non-zero
    assert np.any(sig['sig'] != 0), "Surface charge should be non-zero"

    # Verify solution is complex (includes imaginary part due to lossy metal)
    assert np.any(np.imag(sig['sig']) != 0), "Surface charge should have imaginary part"

    print("PASSED: BEMStat.solve()")
    return sig, bem


def test_bemstat_field():
    """Test BEMStat.field() for electric field computation."""
    print("\n=== Test: BEMStat.field() ===")

    sig, bem = test_bemstat_solve()

    # Compute field outside particle
    field_out = bem.field(sig, inout=2)

    print(f"E-field shape: {field_out['e'].shape}")
    print(f"E-field magnitude range: {np.abs(field_out['e']).min():.4e} to {np.abs(field_out['e']).max():.4e}")

    # Verify field is non-zero and has correct shape
    assert field_out['e'].shape[0] == bem.p.nfaces, "Field should have nfaces rows"
    assert field_out['e'].shape[1] == 3, "Field should have 3 components (xyz)"
    assert np.any(field_out['e'] != 0), "Electric field should be non-zero"

    # Compute field inside particle
    field_in = bem.field(sig, inout=1)
    print(f"Inside E-field magnitude range: {np.abs(field_in['e']).min():.4e} to {np.abs(field_in['e']).max():.4e}")

    print("PASSED: BEMStat.field()")


def test_bemstat_potential():
    """Test BEMStat.potential() for potential computation."""
    print("\n=== Test: BEMStat.potential() ===")

    sig, bem = test_bemstat_solve()

    # Compute potential outside
    pot_out = bem.potential(sig, inout=2)

    print(f"phi2 shape: {pot_out['phi2'].shape}")
    print(f"phi2p shape: {pot_out['phi2p'].shape}")
    print(f"phi2 range: {np.abs(pot_out['phi2']).min():.4e} to {np.abs(pot_out['phi2']).max():.4e}")

    assert 'phi2' in pot_out, "Should have phi2 key"
    assert 'phi2p' in pot_out, "Should have phi2p key"
    assert np.any(pot_out['phi2'] != 0), "Potential should be non-zero"

    # Compute potential inside
    pot_in = bem.potential(sig, inout=1)
    assert 'phi1' in pot_in, "Should have phi1 key"
    assert 'phi1p' in pot_in, "Should have phi1p key"

    print("PASSED: BEMStat.potential()")


def test_bemret_solve():
    """Test BEMRet.solve() for retarded approximation."""
    print("\n=== Test: BEMRet.solve() ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet

    # Create gold sphere (simplified)
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create BEM solver
    bem = BEMRet(p)

    # Create plane wave excitation
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])  # x-polarized, z-propagating

    # Get excitation at wavelength
    enei = 500.0  # nm
    exc_dict = exc.potential(p, enei)

    print(f"Excitation keys: {list(exc_dict.keys())}")
    print(f"a1 shape: {exc_dict['a1'].shape}")
    print(f"a2p shape: {exc_dict['a2p'].shape}")

    # Solve BEM equations
    sig = bem.solve(exc_dict)

    print(f"Solution keys: {list(sig.keys())}")
    print(f"sig1 shape: {sig['sig1'].shape}")
    print(f"sig2 shape: {sig['sig2'].shape}")
    print(f"h1 shape: {sig['h1'].shape}")
    print(f"h2 shape: {sig['h2'].shape}")

    # Verify solution shapes
    nfaces = p.nfaces
    assert sig['sig1'].shape == (nfaces,), f"sig1 should have shape ({nfaces},)"
    assert sig['sig2'].shape == (nfaces,), f"sig2 should have shape ({nfaces},)"
    assert sig['h1'].shape == (nfaces, 3), f"h1 should have shape ({nfaces}, 3)"
    assert sig['h2'].shape == (nfaces, 3), f"h2 should have shape ({nfaces}, 3)"

    # Verify solution is non-zero
    assert np.any(sig['sig1'] != 0) or np.any(sig['sig2'] != 0), "Surface charges should be non-zero"
    assert np.any(sig['h1'] != 0) or np.any(sig['h2'] != 0), "Surface currents should be non-zero"

    print(f"sig2 norm: {np.linalg.norm(sig['sig2']):.4e}")
    print(f"h2 norm: {np.linalg.norm(sig['h2']):.4e}")

    print("PASSED: BEMRet.solve()")
    return sig, bem


def test_bemret_field():
    """Test BEMRet.field() for electric/magnetic field computation."""
    print("\n=== Test: BEMRet.field() ===")

    sig, bem = test_bemret_solve()

    # Compute field outside particle
    field_out = bem.field(sig, inout=2)

    print(f"E-field shape: {field_out['e'].shape}")
    print(f"H-field shape: {field_out['h'].shape}")
    print(f"E-field magnitude range: {np.abs(field_out['e']).min():.4e} to {np.abs(field_out['e']).max():.4e}")

    # Verify field is non-zero and has correct shape
    assert field_out['e'].shape[0] == bem.p.nfaces, "Field should have nfaces rows"
    assert field_out['e'].shape[1] == 3, "Field should have 3 components (xyz)"
    assert np.any(field_out['e'] != 0), "Electric field should be non-zero"

    # Compute field inside particle
    field_in = bem.field(sig, inout=1)
    print(f"Inside E-field magnitude range: {np.abs(field_in['e']).min():.4e} to {np.abs(field_in['e']).max():.4e}")

    print("PASSED: BEMRet.field()")


def test_bemret_potential():
    """Test BEMRet.potential() for potential computation."""
    print("\n=== Test: BEMRet.potential() ===")

    sig, bem = test_bemret_solve()

    # Compute potential outside
    pot_out = bem.potential(sig, inout=2)

    print(f"phi2 shape: {pot_out['phi2'].shape}")
    print(f"a2 shape: {pot_out['a2'].shape}")
    print(f"phi2 range: {np.abs(pot_out['phi2']).min():.4e} to {np.abs(pot_out['phi2']).max():.4e}")

    assert 'phi2' in pot_out, "Should have phi2 key"
    assert 'phi2p' in pot_out, "Should have phi2p key"
    assert 'a2' in pot_out, "Should have a2 key"
    assert 'a2p' in pot_out, "Should have a2p key"
    assert np.any(pot_out['phi2'] != 0) or np.any(pot_out['a2'] != 0), "Potential should be non-zero"

    # Compute potential inside
    pot_in = bem.potential(sig, inout=1)
    assert 'phi1' in pot_in, "Should have phi1 key"
    assert 'a1' in pot_in, "Should have a1 key"

    print("PASSED: BEMRet.potential()")


def test_bemstat_scattering_workflow():
    """Test complete quasistatic scattering workflow."""
    print("\n=== Test: Complete Quasistatic Workflow ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat

    # Create particle
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create BEM solver and excitation
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    # Solve for a wavelength
    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute cross sections
    sca = exc.scattering(sig)
    abs_cs = exc.absorption(sig)
    ext = exc.extinction(sig)

    print(f"Scattering cross section: {sca[0]:.4e} nm^2")
    print(f"Absorption cross section: {abs_cs[0]:.4e} nm^2")
    print(f"Extinction cross section: {ext[0]:.4e} nm^2")

    # Verify energy conservation: ext = sca + abs
    error = np.abs(ext[0] - (sca[0] + abs_cs[0])) / np.abs(ext[0])
    print(f"Energy conservation error: {error:.2e}")

    assert error < 1e-10, "Energy conservation should hold"

    print("PASSED: Complete Quasistatic Workflow")


def test_bemret_scattering_workflow():
    """Test complete retarded scattering workflow."""
    print("\n=== Test: Complete Retarded Workflow ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet

    # Create particle
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create BEM solver and excitation
    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    # Solve for a wavelength
    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute scattering cross section using SpectrumRet
    sca, dsca = exc.scattering(sig)

    print(f"Scattering cross section: {sca:.4e} nm^2")
    print(f"dsca shape: {dsca.shape}")

    # Verify cross section is positive
    assert sca > 0, "Scattering cross section should be positive"

    print("PASSED: Complete Retarded Workflow")


def test_multiple_polarizations():
    """Test BEM solving with multiple polarizations."""
    print("\n=== Test: Multiple Polarizations ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat

    # Create particle
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create BEM solver and excitation with 2 polarizations
    bem = BEMStat(p)
    exc = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])  # x and y polarizations

    # Solve
    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    print(f"Multi-pol sig shape: {sig['sig'].shape}")

    # Should have (nfaces, 2) for 2 polarizations
    assert sig['sig'].shape == (p.nfaces, 2), "Should have shape (nfaces, npol)"

    # Compute cross sections
    sca = exc.scattering(sig)
    print(f"Scattering (x-pol): {sca[0]:.4e} nm^2")
    print(f"Scattering (y-pol): {sca[1]:.4e} nm^2")

    # For sphere, x and y polarizations should give similar results
    rel_diff = np.abs(sca[0] - sca[1]) / np.abs(sca[0])
    print(f"Relative difference: {rel_diff:.4f}")

    assert rel_diff < 0.01, "x and y polarizations should give similar results for sphere"

    print("PASSED: Multiple Polarizations")


def test_dipole_excitation_solve():
    """Test BEM solving with dipole excitation."""
    print("\n=== Test: Dipole Excitation Solve ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import DipoleStat

    # Create particle
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Create dipole outside sphere
    pt_pos = np.array([[0.0, 0.0, 20.0]])  # 20nm from center
    dip = DipoleStat(pt_pos, eps=eps_tab)

    # Create BEM solver
    bem = BEMStat(p)

    # Get excitation
    enei = 500.0
    exc_dict = dip.potential(p, enei)

    print(f"Dipole excitation phip shape: {exc_dict['phip'].shape}")

    # Solve
    sig = bem.solve(exc_dict)

    print(f"Dipole solution sig shape: {sig['sig'].shape}")
    print(f"sig norm: {np.linalg.norm(sig['sig']):.4e}")

    # DipoleStat creates 3 dipole orientations (x, y, z) by default
    # Shape is (nfaces, npt, ndip) = (nfaces, 1, 3)
    assert sig['sig'].shape == (p.nfaces, 1, 3), "Should have shape (nfaces, npt=1, ndip=3)"

    print("PASSED: Dipole Excitation Solve")


def run_all_tests():
    """Run all Step 6 tests."""
    print("=" * 60)
    print("STEP 6: BEM EQUATION SOLVING TESTS")
    print("=" * 60)

    test_bemstat_solve()
    test_bemstat_field()
    test_bemstat_potential()
    test_bemret_solve()
    test_bemret_field()
    test_bemret_potential()
    test_bemstat_scattering_workflow()
    test_bemret_scattering_workflow()
    test_multiple_polarizations()
    test_dipole_excitation_solve()

    print("\n" + "=" * 60)
    print("ALL STEP 6 TESTS PASSED!")
    print("=" * 60)
    print("\nImplemented BEM solving functionality:")
    print("  - BEMStat.solve(): Quasistatic surface charge calculation")
    print("  - BEMRet.solve(): Retarded surface charge/current calculation")
    print("  - BEMStat.field(): Electric field from surface charges")
    print("  - BEMStat.potential(): Potential from surface charges")
    print("  - BEMRet.field(): Electric/magnetic field from surface charges/currents")
    print("  - BEMRet.potential(): Scalar/vector potential from surface charges/currents")
    print("  - Complete workflow: excitation -> solve -> cross sections")


if __name__ == '__main__':
    run_all_tests()

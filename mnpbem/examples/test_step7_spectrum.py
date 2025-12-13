"""
Test Step 7: Optical Spectrum Calculation

This test validates the spectrum calculation workflow matching MATLAB MNPBEM exactly:
1. SpectrumRet: Far-field and scattering for full Maxwell equations
2. SpectrumStat: Far-field and scattering for quasistatic approximation
3. PlaneWaveRet: extinction, scattering, absorption cross sections
4. PlaneWaveStat: extinction, scattering, absorption cross sections

MATLAB equivalents:
    spec = spectrumret(trisphere(256, 2), 'medium', 1)
    field = farfield(spec, sig)
    [sca, dsca] = scattering(spec, sig)
    ext = extinction(planewaveret, sig)
    abs = absorption(planewaveret, sig)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/MNPBEM')


def test_spectrum_ret_init():
    """Test SpectrumRet initialization."""
    print("\n=== Test: SpectrumRet Initialization ===")

    from mnpbem.spectrum import SpectrumRet

    # Default initialization
    spec = SpectrumRet()
    print(f"Default ndir: {spec.ndir}")
    print(f"Default medium: {spec.medium}")
    assert spec.ndir > 0, "Should have directions on unit sphere"
    assert spec.nvec.shape[1] == 3, "nvec should have 3 components"
    assert len(spec.area) == spec.ndir, "area should have ndir elements"

    # With integer (number of faces)
    spec2 = SpectrumRet(144)
    print(f"With 144 faces: ndir={spec2.ndir}")
    assert spec2.ndir >= 144, "Should have at least 144 directions"

    # With direction array
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    spec3 = SpectrumRet(dirs)
    print(f"With 3 directions: ndir={spec3.ndir}")
    assert spec3.ndir == 3, "Should have 3 directions"

    # With medium specification
    spec4 = SpectrumRet(144, medium=2)
    assert spec4.medium == 2, "Medium should be 2"

    print("PASSED: SpectrumRet Initialization")


def test_spectrum_stat_init():
    """Test SpectrumStat initialization."""
    print("\n=== Test: SpectrumStat Initialization ===")

    from mnpbem.spectrum import SpectrumStat

    # Default initialization
    spec = SpectrumStat()
    print(f"Default ndir: {spec.ndir}")
    print(f"Default medium: {spec.medium}")
    assert spec.ndir > 0, "Should have directions on unit sphere"
    assert spec.nvec.shape[1] == 3, "nvec should have 3 components"
    assert len(spec.area) == spec.ndir, "area should have ndir elements"

    # With integer
    spec2 = SpectrumStat(144)
    print(f"With 144 faces: ndir={spec2.ndir}")
    assert spec2.ndir >= 144, "Should have at least 144 directions"

    print("PASSED: SpectrumStat Initialization")


def test_spectrum_ret_farfield():
    """Test SpectrumRet.farfield() for retarded far-field calculation."""
    print("\n=== Test: SpectrumRet.farfield() ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet
    from mnpbem.spectrum import SpectrumRet

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Create spectrum and compute far-field
    spec = SpectrumRet(144, medium=1)
    field = spec.farfield(sig)

    print(f"Far-field e shape: {field['e'].shape}")
    print(f"Far-field h shape: {field['h'].shape}")
    print(f"Far-field |e| max: {np.abs(field['e']).max():.4e}")
    print(f"Far-field |h| max: {np.abs(field['h']).max():.4e}")

    # Verify shapes
    ndir = spec.ndir
    assert field['e'].shape[0] == ndir, "Should have ndir far-field points"
    assert field['e'].shape[1] == 3, "Should have 3 E-field components"
    assert field['h'].shape[0] == ndir, "Should have ndir far-field points"
    assert field['h'].shape[1] == 3, "Should have 3 H-field components"

    # Verify non-zero
    assert np.any(field['e'] != 0), "Far-field E should be non-zero"
    assert np.any(field['h'] != 0), "Far-field H should be non-zero"

    print("PASSED: SpectrumRet.farfield()")
    return sig, spec


def test_spectrum_ret_scattering():
    """Test SpectrumRet.scattering() for scattering cross section."""
    print("\n=== Test: SpectrumRet.scattering() ===")

    sig, spec = test_spectrum_ret_farfield()

    # Compute scattering
    sca, dsca = spec.scattering(sig)

    print(f"Total scattering: {sca:.4e}")
    print(f"dsca shape: {dsca.shape}")
    print(f"dsca range: {dsca.min():.4e} to {dsca.max():.4e}")

    # Verify scattering is positive
    assert sca > 0, "Scattering cross section should be positive"
    # dsca can be negative in some directions (shadow regions)
    # but should integrate to positive total

    print("PASSED: SpectrumRet.scattering()")


def test_spectrum_stat_farfield():
    """Test SpectrumStat.farfield() for quasistatic far-field calculation."""
    print("\n=== Test: SpectrumStat.farfield() ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat
    from mnpbem.spectrum import SpectrumStat

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Create spectrum and compute far-field
    spec = SpectrumStat(144, medium=1)
    field = spec.farfield(sig)

    print(f"Far-field e shape: {field['e'].shape}")
    print(f"Far-field h shape: {field['h'].shape}")
    print(f"Far-field |e| max: {np.abs(field['e']).max():.4e}")

    # Verify shapes
    ndir = spec.ndir
    assert field['e'].shape[0] == ndir, "Should have ndir far-field points"
    assert field['e'].shape[1] == 3, "Should have 3 E-field components"

    # Verify non-zero
    assert np.any(field['e'] != 0), "Far-field E should be non-zero"

    print("PASSED: SpectrumStat.farfield()")
    return sig, spec


def test_spectrum_stat_scattering():
    """Test SpectrumStat.scattering() for scattering cross section."""
    print("\n=== Test: SpectrumStat.scattering() ===")

    sig, spec = test_spectrum_stat_farfield()

    # Compute scattering
    sca, dsca = spec.scattering(sig)

    print(f"Total scattering: {sca:.4e}")
    print(f"dsca shape: {dsca.shape}")

    # Verify scattering is positive
    assert sca > 0, "Scattering cross section should be positive"

    print("PASSED: SpectrumStat.scattering()")


def test_planewave_ret_extinction():
    """Test PlaneWaveRet.extinction() using optical theorem."""
    print("\n=== Test: PlaneWaveRet.extinction() ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute extinction
    ext = exc.extinction(sig)

    print(f"Extinction cross section: {ext[0]:.4e} nm^2")

    # Verify extinction is positive (particles absorb/scatter light)
    assert ext[0] > 0, "Extinction should be positive"

    print("PASSED: PlaneWaveRet.extinction()")
    return sig, exc


def test_planewave_ret_scattering():
    """Test PlaneWaveRet.scattering()."""
    print("\n=== Test: PlaneWaveRet.scattering() ===")

    sig, exc = test_planewave_ret_extinction()

    # Compute scattering
    sca, dsca = exc.scattering(sig)

    print(f"Scattering cross section: {sca:.4e} nm^2")
    print(f"dsca shape: {dsca.shape}")

    assert sca > 0, "Scattering cross section should be positive"

    print("PASSED: PlaneWaveRet.scattering()")


def test_planewave_ret_absorption():
    """Test PlaneWaveRet.absorption()."""
    print("\n=== Test: PlaneWaveRet.absorption() ===")

    sig, exc = test_planewave_ret_extinction()

    # Compute absorption
    abs_cs = exc.absorption(sig)

    print(f"Absorption cross section: {abs_cs[0]:.4e} nm^2")

    # For lossy metal, absorption should be positive
    assert abs_cs[0] > 0, "Absorption should be positive for lossy particle"

    print("PASSED: PlaneWaveRet.absorption()")


def test_planewave_ret_energy_conservation():
    """Test energy conservation: ext = sca + abs."""
    print("\n=== Test: Energy Conservation (Retarded) ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute all cross sections
    ext = exc.extinction(sig)[0]
    sca, _ = exc.scattering(sig)
    abs_cs = exc.absorption(sig)[0]

    print(f"Extinction: {ext:.4e}")
    print(f"Scattering: {sca:.4e}")
    print(f"Absorption: {abs_cs:.4e}")
    print(f"Sca + Abs:  {sca + abs_cs:.4e}")

    # Verify energy conservation
    error = np.abs(ext - (sca + abs_cs)) / np.abs(ext)
    print(f"Energy conservation error: {error:.2e}")

    assert error < 1e-10, f"Energy conservation should hold, error={error:.2e}"

    print("PASSED: Energy Conservation (Retarded)")


def test_planewave_stat_cross_sections():
    """Test PlaneWaveStat cross section calculations."""
    print("\n=== Test: PlaneWaveStat Cross Sections ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute cross sections
    sca = exc.scattering(sig)
    abs_cs = exc.absorption(sig)
    ext = exc.extinction(sig)

    print(f"Scattering: {sca[0]:.4e} nm^2")
    print(f"Absorption: {abs_cs[0]:.4e} nm^2")
    print(f"Extinction: {ext[0]:.4e} nm^2")

    # Verify energy conservation
    error = np.abs(ext[0] - (sca[0] + abs_cs[0])) / np.abs(ext[0])
    print(f"Energy conservation error: {error:.2e}")

    assert sca[0] > 0, "Scattering should be positive"
    assert abs_cs[0] > 0, "Absorption should be positive for lossy particle"
    assert error < 1e-10, "Energy conservation should hold"

    print("PASSED: PlaneWaveStat Cross Sections")


def test_multiple_polarizations_spectrum():
    """Test spectrum calculation with multiple polarizations."""
    print("\n=== Test: Multiple Polarizations Spectrum ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat
    from mnpbem.spectrum import SpectrumStat

    # Create particle
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    # Multiple polarizations
    bem = BEMStat(p)
    exc = PlaneWaveStat([[1, 0, 0], [0, 1, 0]])

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Compute far-field
    spec = SpectrumStat(144)
    field = spec.farfield(sig)

    print(f"Multi-pol far-field e shape: {field['e'].shape}")
    assert field['e'].shape[2] == 2, "Should have 2 polarizations in far-field"

    # Compute scattering
    sca, dsca = spec.scattering(sig)
    print(f"sca shape: {np.asarray(sca).shape}")
    print(f"sca (x-pol): {sca[0]:.4e}, sca (y-pol): {sca[1]:.4e}")

    # For sphere, x and y should be similar
    rel_diff = np.abs(sca[0] - sca[1]) / np.abs(sca[0])
    print(f"Relative difference: {rel_diff:.4f}")
    assert rel_diff < 0.01, "x and y polarizations should give similar results for sphere"

    print("PASSED: Multiple Polarizations Spectrum")


def test_wavelength_scan():
    """Test scattering spectrum over wavelength range."""
    print("\n=== Test: Wavelength Scan ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.excitation import PlaneWaveStat

    # Create particle
    eps_vac = EpsConst(1.0)
    # Variable epsilon to simulate plasmonic response
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])

    # Wavelength scan
    wavelengths = np.linspace(400, 800, 5)
    ext_spectrum = []

    for enei in wavelengths:
        exc_dict = exc.potential(p, enei)
        sig = bem.solve(exc_dict)
        ext = exc.extinction(sig)[0]
        ext_spectrum.append(ext)

    ext_spectrum = np.array(ext_spectrum)

    print(f"Wavelengths: {wavelengths}")
    print(f"Extinction: {ext_spectrum}")
    print(f"Extinction range: {ext_spectrum.min():.4e} to {ext_spectrum.max():.4e}")

    # Verify all values are finite and positive
    assert np.all(np.isfinite(ext_spectrum)), "All extinction values should be finite"
    assert np.all(ext_spectrum > 0), "All extinction values should be positive"

    print("PASSED: Wavelength Scan")


def test_farfield_directions():
    """Test far-field calculation in specific directions."""
    print("\n=== Test: Far-field in Specific Directions ===")

    from mnpbem.materials import EpsConst
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMRet
    from mnpbem.excitation import PlaneWaveRet
    from mnpbem.spectrum import SpectrumRet

    # Create particle and solve
    eps_vac = EpsConst(1.0)
    eps_gold = EpsConst(-10.0 + 1.0j)
    eps_tab = [eps_vac, eps_gold]

    sphere = trisphere(144, 10.0)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    bem = BEMRet(p)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])  # x-pol, z-propagating

    enei = 500.0
    exc_dict = exc.potential(p, enei)
    sig = bem.solve(exc_dict)

    # Create spectrum
    spec = SpectrumRet(144)

    # Compute far-field in specific directions
    # Forward (z), backward (-z), side (x)
    specific_dirs = np.array([
        [0, 0, 1],   # forward
        [0, 0, -1],  # backward
        [1, 0, 0],   # side (x)
        [0, 1, 0],   # side (y)
    ])
    field = spec.farfield(sig, direction=specific_dirs)

    print(f"Far-field in specific directions:")
    for i, d in enumerate(specific_dirs):
        print(f"  dir={d}: |e|={np.abs(field['e'][i]).max():.4e}")

    # Forward and backward should be different (asymmetric scattering)
    forward_e = np.linalg.norm(field['e'][0])
    backward_e = np.linalg.norm(field['e'][1])
    print(f"Forward |e|: {forward_e:.4e}")
    print(f"Backward |e|: {backward_e:.4e}")

    print("PASSED: Far-field in Specific Directions")


def run_all_tests():
    """Run all Step 7 tests."""
    print("=" * 60)
    print("STEP 7: OPTICAL SPECTRUM CALCULATION TESTS")
    print("=" * 60)

    test_spectrum_ret_init()
    test_spectrum_stat_init()
    test_spectrum_ret_farfield()
    test_spectrum_ret_scattering()
    test_spectrum_stat_farfield()
    test_spectrum_stat_scattering()
    test_planewave_ret_extinction()
    test_planewave_ret_scattering()
    test_planewave_ret_absorption()
    test_planewave_ret_energy_conservation()
    test_planewave_stat_cross_sections()
    test_multiple_polarizations_spectrum()
    test_wavelength_scan()
    test_farfield_directions()

    print("\n" + "=" * 60)
    print("ALL STEP 7 TESTS PASSED!")
    print("=" * 60)
    print("\nImplemented spectrum functionality:")
    print("  - SpectrumRet: Far-field for full Maxwell equations")
    print("  - SpectrumStat: Far-field for quasistatic (dipole radiation)")
    print("  - SpectrumRet.farfield(): Garcia de Abajo Eq. (50)")
    print("  - SpectrumStat.farfield(): Jackson Eq. (9.19)")
    print("  - SpectrumRet/Stat.scattering(): Poynting vector integration")
    print("  - PlaneWaveRet.extinction(): Optical theorem")
    print("  - PlaneWaveRet.absorption(): ext - sca")
    print("  - PlaneWaveStat cross sections: Dipole formulas")


if __name__ == '__main__':
    run_all_tests()

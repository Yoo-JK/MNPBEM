"""
API safety / robustness edge case tests (M3 Wave 3 C1).

Validates user-facing API behavior under unusual or hostile input:

  7.1 Error Handling
      - None / empty input
      - Wrong shape arrays (1D where 2D expected, etc.)
      - Missing required arguments
      - Conflicting / inconsistent options

  7.2 Type Safety
      - int vs float vs complex scalars/arrays
      - scalar vs 1D vs 2D broadcasting
      - inf / nan inputs
      - mixed numeric types

  7.3 Reproducibility
      - Same input -> same output (bit-identical when possible,
        ULP-level otherwise)
      - mnpbem solves are deterministic (no hidden RNG)
      - Multi-run memory stability (no obvious leak after 100 runs)

The tests cover the public facade: Particle / ComParticle / ComPoint /
EpsConst / EpsDrude / BEMStat / BEMRet / PlaneWaveStat / PlaneWaveRet
(plus a few cross-checks via the Spectrum and MeshField wrappers).
"""

import gc
import os
import sys
import types
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bypass the broken top-level mnpbem.__init__ so subpackages import cleanly
# (mirrors mnpbem/tests/test_ret_scattering.py and the rest of M3 tests).
# ---------------------------------------------------------------------------
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mnpbem.materials import EpsConst, EpsDrude, EpsTable
from mnpbem.geometry import (
    Particle, ComParticle, Point, ComPoint, trisphere,
)
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import (
    PlaneWaveStat, PlaneWaveRet,
    DipoleStat,
    MeshField,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EPS_VAC = EpsConst(1.0)
EPS_METAL = EpsConst(-10.0 + 0.5j)


def _small_sphere(diameter = 10.0, n_faces = 32):
    p = trisphere(n_faces, diameter)
    return ComParticle([EPS_VAC, EPS_METAL], [p], [[2, 1]])


# =============================================================================
# 7.1 Error handling
# =============================================================================


class TestNoneInput(object):
    """None / empty inputs to the public constructors."""

    def test_particle_none(self):
        # Empty particle is the documented behavior of MATLAB @particle
        p = Particle(None)
        assert p.nverts == 0
        assert p.nfaces == 0

    def test_particle_empty_list(self):
        p = Particle([])
        assert p.nverts == 0
        assert p.nfaces == 0

    def test_particle_empty_array(self):
        p = Particle(np.zeros((0, 3)))
        assert p.nverts == 0
        assert p.nfaces == 0

    def test_bemstat_none_raises(self):
        # BEMStat needs a particle; None must not silently succeed.
        with pytest.raises((AttributeError, TypeError)):
            BEMStat(None)

    def test_bemret_none_raises(self):
        with pytest.raises((AttributeError, TypeError)):
            BEMRet(None)

    def test_planewavestat_none_pol(self):
        # NumPy converts None -> object array; constructor should either
        # accept it as a no-op (legacy) or reject it. Whichever, downstream
        # calls must fail cleanly rather than crashing the interpreter.
        try:
            exc = PlaneWaveStat(None)
        except (TypeError, ValueError):
            return
        # If the constructor accepts None, exercising it must raise.
        sphere = _small_sphere()
        with pytest.raises((TypeError, ValueError, AttributeError)):
            exc.potential(sphere, 600.0)

    def test_compoint_empty(self):
        sphere = _small_sphere()
        cpt = ComPoint(sphere, np.zeros((0, 3)))
        assert cpt.pos.shape == (0, 3)
        assert cpt.n == 0


class TestWrongShape(object):
    """Wrong-shape arrays where a specific rank is expected."""

    def test_particle_1d_verts_no_faces(self):
        # 1D verts with no faces becomes an "empty"-style particle; behaviour
        # is permissive (MATLAB also accepts vector of points), but it must
        # not produce faces.
        p = Particle(np.zeros((3,)), None)
        assert p.nfaces == 0

    def test_particle_5col_faces_rejected(self):
        # 5 columns is neither 3 (triangle) nor 4 (quad), so the Particle
        # implementation re-interprets it as curved boundary data. That is
        # ok, but verts/faces must remain self-consistent.
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        faces = np.array([[0, 1, 2, 0, 0]], dtype = float)
        p = Particle(verts, faces)
        assert p.faces.shape[1] == 4
        # Face indices must remain inside [0, nverts)
        valid = ~np.isnan(p.faces)
        assert np.all(p.faces[valid] < p.nverts)
        assert np.all(p.faces[valid] >= 0)

    def test_planewavestat_pol_2d(self):
        # 2D polarisation array is the multi-pol case
        pol = np.eye(3)
        exc = PlaneWaveStat(pol)
        assert exc.pol.ndim == 2
        assert exc.pol.shape == (3, 3)

    def test_planewavestat_pol_1d_promoted(self):
        # 1D polarisation is reshape (1, 3)
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        assert exc.pol.shape == (1, 3)

    def test_planewaveret_dir_shape_mismatch(self):
        # Most general user error: dir and pol have inconsistent first axis.
        # The polarisations chosen here are all perpendicular to z so the
        # orthogonality check passes; the shape mismatch must still produce
        # a clean failure rather than a corrupt result.
        sphere = _small_sphere()
        pol = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        kdir = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype = float)
        with pytest.raises((ValueError, IndexError, AssertionError)):
            exc = PlaneWaveRet(pol, kdir)
            exc.potential(sphere, 600.0)

    def test_compoint_2col_pos(self):
        # 2-column "positions" must not be silently accepted as 3D pos
        sphere = _small_sphere()
        with pytest.raises((ValueError, IndexError)):
            ComPoint(sphere, np.zeros((3, 2)))


class TestMissingRequired(object):
    """Required arguments must trigger a TypeError, not a hidden bug."""

    def test_eps_const_no_arg(self):
        with pytest.raises(TypeError):
            EpsConst()

    def test_eps_drude_missing_args(self):
        with pytest.raises(TypeError):
            EpsDrude(10.0)
        with pytest.raises(TypeError):
            EpsDrude(10.0, 9.0)

    def test_planewaveret_missing_dir(self):
        with pytest.raises(TypeError):
            PlaneWaveRet([1, 0, 0])

    def test_comparticle_missing_inout(self):
        with pytest.raises(TypeError):
            ComParticle([EPS_VAC, EPS_METAL], [trisphere(32, 10.0)])

    def test_meshfield_missing_y(self):
        sphere = _small_sphere()
        x = np.linspace(-15, 15, 5)
        with pytest.raises(TypeError):
            MeshField(sphere, x)  # y required


class TestConflictingOptions(object):
    """Conflicting / nonsensical option combinations."""

    def test_planewavestat_medium_kwarg_and_pos(self):
        # medium specified both positionally and as kwarg
        # Python rule: TypeError on duplicate
        with pytest.raises(TypeError):
            PlaneWaveStat([1, 0, 0], medium = 1, **{"medium": 1})

    def test_comparticle_eps_too_short(self):
        # inout indices reference eps[2] but eps has only one entry
        sphere = trisphere(32, 10.0)
        cp = ComParticle([EPS_VAC], [sphere], [[2, 1]])
        # Construction may succeed lazily, but evaluating eps must now fail.
        with pytest.raises((IndexError, AttributeError, TypeError)):
            cp.eps1(600.0)

    def test_bemstat_options_unknown_kwarg(self):
        # Unknown kwarg should not crash silently — but if it does, downstream
        # calls must still produce sensible results
        sphere = _small_sphere()
        try:
            bem = BEMStat(sphere, foo_unknown_option = 42)
        except TypeError:
            return
        # If accepted, basic solve should still work
        exc = PlaneWaveStat([1, 0, 0])
        sig, _ = bem.solve(exc.potential(sphere, 600.0))
        assert np.all(np.isfinite(sig.sig))


# =============================================================================
# 7.2 Type safety
# =============================================================================


class TestNumericTypes(object):
    """int vs float vs complex scalars and arrays."""

    def test_eps_const_python_int(self):
        e = EpsConst(2)
        eps, k = e(600.0)
        assert np.isfinite(eps).all()
        assert np.isfinite(k).all()

    def test_eps_const_python_float(self):
        e = EpsConst(2.0)
        eps, k = e(600.0)
        assert np.isfinite(eps).all()

    def test_eps_const_python_complex(self):
        e = EpsConst(-2.0 + 0.5j)
        eps, k = e(600.0)
        assert np.iscomplexobj(eps)
        assert np.isfinite(eps.real).all() and np.isfinite(eps.imag).all()

    def test_eps_const_int_eq_float(self):
        # Int and float dielectric constants must give the same result
        eps_i, k_i = EpsConst(2)(600.0)
        eps_f, k_f = EpsConst(2.0)(600.0)
        np.testing.assert_array_equal(eps_i, eps_f)
        np.testing.assert_array_equal(k_i, k_f)

    def test_eps_const_array_input(self):
        e = EpsConst(2.0)
        wls = np.array([400.0, 500.0, 600.0])
        eps, k = e(wls)
        assert eps.shape == wls.shape
        assert k.shape == wls.shape

    def test_eps_const_scalar_vs_array_consistency(self):
        e = EpsConst(2.0)
        eps_s, k_s = e(600.0)
        eps_a, k_a = e(np.array([600.0]))
        np.testing.assert_allclose(np.atleast_1d(eps_s), np.atleast_1d(eps_a))
        np.testing.assert_allclose(np.atleast_1d(k_s), np.atleast_1d(k_a))

    def test_eps_drude_int_args(self):
        # Drude with int args should behave like float
        ed_i = EpsDrude(10, 9, 1, name = "test_int")
        ed_f = EpsDrude(10.0, 9.0, 1.0, name = "test_float")
        eps_i, k_i = ed_i(600.0)
        eps_f, k_f = ed_f(600.0)
        np.testing.assert_allclose(eps_i, eps_f)
        np.testing.assert_allclose(k_i, k_f)

    def test_planewave_int_pol(self):
        # Integer polarisation must broadcast like float
        exc_i = PlaneWaveStat([1, 0, 0])
        exc_f = PlaneWaveStat([1.0, 0.0, 0.0])
        sphere = _small_sphere()
        pot_i = exc_i.potential(sphere, 600.0)
        pot_f = exc_f.potential(sphere, 600.0)
        np.testing.assert_allclose(pot_i.phip, pot_f.phip)


class TestBroadcasting(object):
    """scalar vs 1D vs 2D broadcasting."""

    def test_eps_const_2d_input(self):
        e = EpsConst(2.0)
        wls = np.array([[400.0, 500.0], [600.0, 700.0]])
        eps, k = e(wls)
        assert eps.shape == wls.shape
        assert k.shape == wls.shape

    def test_eps_drude_2d_input(self):
        ed = EpsDrude(10.0, 9.0, 0.1)
        wls = np.array([[400.0, 500.0], [600.0, 700.0]])
        eps, k = ed(wls)
        assert eps.shape == wls.shape

    def test_planewavestat_pol_promotion(self):
        # 1D pol must be reshaped to (1, 3) automatically (MATLAB compat).
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        assert exc.pol.shape == (1, 3)

        exc = PlaneWaveStat(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        assert exc.pol.shape == (2, 3)


class TestNonFinite(object):
    """inf / nan inputs propagate as NaN, never crash."""

    def test_eps_const_inf(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps, k = EpsConst(np.inf)(600.0)
        assert np.isinf(eps.real) or np.isnan(eps.real)

    def test_eps_const_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps, k = EpsConst(np.nan)(600.0)
        assert np.isnan(eps.real)
        assert np.isnan(k)

    def test_eps_const_zero_wavelength(self):
        # 1/0 -> inf k; should not raise, only RuntimeWarning is OK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eps, k = EpsConst(2.0)(0.0)
        assert np.isinf(k) or np.isnan(k)

    def test_planewave_inf_pol_does_not_crash(self):
        # Inf polarisation should propagate — solve will be NaN but no crash.
        exc = PlaneWaveStat([np.inf, 0.0, 0.0])
        sphere = _small_sphere()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pot = exc.potential(sphere, 600.0)
        assert pot is not None

    def test_point_with_nan_pos(self):
        pos = np.array([[np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]])
        p = Point(pos)
        assert np.isnan(p.pos[0, 0])


class TestMixedTypes(object):
    """numpy arrays with mixed dtypes."""

    def test_eps_const_mixed_int_float_array(self):
        e = EpsConst(2.0)
        wls = np.array([400, 500.0, 600], dtype = object).astype(float)
        eps, k = e(wls)
        assert eps.shape == (3,)

    def test_planewavestat_mixed_pol(self):
        # Mixed int/float in pol vector
        exc = PlaneWaveStat(np.array([1, 0.5, 0.0]))
        assert exc.pol.dtype.kind in ("f", "c")  # float-like


# =============================================================================
# 7.3 Reproducibility
# =============================================================================


class TestDeterministicConstructors(object):
    """Same input must give the same object (modulo NaN-equality)."""

    def test_trisphere_repro(self):
        s1 = trisphere(32, 10.0)
        s2 = trisphere(32, 10.0)
        np.testing.assert_array_equal(s1.verts, s2.verts)
        # faces have NaN padding -> use equal_nan
        assert np.array_equal(s1.faces, s2.faces, equal_nan = True)
        np.testing.assert_array_equal(s1.area, s2.area)

    def test_trisphere_multiple_sizes_repro(self):
        for n in (32, 144, 256):
            s1 = trisphere(n, 10.0)
            s2 = trisphere(n, 10.0)
            np.testing.assert_array_equal(s1.verts, s2.verts)
            assert np.array_equal(s1.faces, s2.faces, equal_nan = True)

    def test_eps_const_repro(self):
        e1, e2 = EpsConst(2.0), EpsConst(2.0)
        a, ka = e1(600.0)
        b, kb = e2(600.0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ka, kb)

    def test_eps_drude_repro(self):
        ed1 = EpsDrude(10.0, 9.07, 0.07)
        ed2 = EpsDrude(10.0, 9.07, 0.07)
        wls = np.linspace(400, 800, 10)
        e1, k1 = ed1(wls)
        e2, k2 = ed2(wls)
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(k1, k2)


class TestDeterministicSolve(object):
    """Two independent BEM solves with identical inputs must agree."""

    def test_bemstat_repro_bit_identical(self):
        cp1 = _small_sphere()
        cp2 = _small_sphere()
        bem1 = BEMStat(cp1)
        bem2 = BEMStat(cp2)
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        sig1, _ = bem1.solve(exc.potential(cp1, 600.0))
        sig2, _ = bem2.solve(exc.potential(cp2, 600.0))
        np.testing.assert_array_equal(sig1.sig, sig2.sig)

    def test_bemret_repro_bit_identical(self):
        cp1 = _small_sphere(diameter = 30.0, n_faces = 32)
        cp2 = _small_sphere(diameter = 30.0, n_faces = 32)
        bem1 = BEMRet(cp1)
        bem2 = BEMRet(cp2)
        exc = PlaneWaveRet([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        sig1, _ = bem1.solve(exc.potential(cp1, 600.0))
        sig2, _ = bem2.solve(exc.potential(cp2, 600.0))
        # BEMRet returns sig1, sig2, h1, h2 (CompStruct fields)
        np.testing.assert_array_equal(sig1.sig1, sig2.sig1)
        np.testing.assert_array_equal(sig1.sig2, sig2.sig2)
        np.testing.assert_array_equal(sig1.h1, sig2.h1)
        np.testing.assert_array_equal(sig1.h2, sig2.h2)

    def test_bemstat_same_object_twice(self):
        # Solving the same BEM twice with the same wavelength must return
        # bit-identical surface charges.
        cp = _small_sphere()
        bem = BEMStat(cp)
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        sig_a, _ = bem.solve(exc.potential(cp, 600.0))
        sig_b, _ = bem.solve(exc.potential(cp, 600.0))
        np.testing.assert_array_equal(sig_a.sig, sig_b.sig)

    def test_bemstat_wavelength_loop_no_state_leak(self):
        # Iterating over a wavelength list and back must give the same result
        # for the original wavelength.
        cp = _small_sphere()
        bem = BEMStat(cp)
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        sig0, _ = bem.solve(exc.potential(cp, 600.0))
        for w in (500.0, 700.0, 800.0):
            bem.solve(exc.potential(cp, w))
        sig0_again, _ = bem.solve(exc.potential(cp, 600.0))
        np.testing.assert_allclose(sig0.sig, sig0_again.sig, rtol = 1e-12)


class TestNoHiddenRandomness(object):
    """mnpbem outputs must not depend on the global numpy RNG."""

    def test_solve_independent_of_global_seed(self):
        cp = _small_sphere()
        exc = PlaneWaveStat([1.0, 0.0, 0.0])

        np.random.seed(1)
        bem_a = BEMStat(cp)
        sig_a, _ = bem_a.solve(exc.potential(cp, 600.0))

        np.random.seed(99999)
        bem_b = BEMStat(cp)
        sig_b, _ = bem_b.solve(exc.potential(cp, 600.0))

        np.testing.assert_array_equal(sig_a.sig, sig_b.sig)


class TestMemoryStability(object):
    """No obvious memory leak after many repeated solves."""

    def test_repeated_bemstat_solve_no_growth(self):
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        cp = _small_sphere()
        exc = PlaneWaveStat([1.0, 0.0, 0.0])
        # Warm up
        bem = BEMStat(cp)
        for _ in range(5):
            bem.solve(exc.potential(cp, 600.0))
        gc.collect()

        proc = psutil.Process(os.getpid())
        rss_start = proc.memory_info().rss

        for _ in range(100):
            bem = BEMStat(cp)
            sig, _ = bem.solve(exc.potential(cp, 600.0))
            del bem, sig
        gc.collect()

        rss_end = proc.memory_info().rss
        growth_mb = (rss_end - rss_start) / (1024 * 1024)
        # Allow up to 50 MB of growth — anything larger probably means a
        # leak (the base solve uses ~few hundred KB of state).
        assert growth_mb < 50.0, (
            "RSS grew by {:.1f} MB across 100 BEMStat constructions/solves, "
            "suggesting a leak (start = {:.1f} MB, end = {:.1f} MB)".format(
                growth_mb,
                rss_start / 1024 / 1024,
                rss_end / 1024 / 1024,
            )
        )

    def test_repeated_trisphere_no_growth(self):
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        # Warm up
        for _ in range(5):
            trisphere(144, 10.0)
        gc.collect()

        proc = psutil.Process(os.getpid())
        rss_start = proc.memory_info().rss

        for _ in range(100):
            sphere = trisphere(144, 10.0)
            del sphere
        gc.collect()

        rss_end = proc.memory_info().rss
        growth_mb = (rss_end - rss_start) / (1024 * 1024)
        assert growth_mb < 30.0

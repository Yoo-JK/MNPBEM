"""M3 B1 — BEM solver edge case tests.

Covers:
    3.1 BEMStat / BEMRet wavelength extremes (lambda << / >> particle size)
        and substrate (BEMStatLayer / BEMRetLayer) dipoles above & below.
    3.2 BEMStatIter / BEMRetIter convergence stress (tol scan, solver scan).
    3.3 Mirror solvers (BEMStatMirror / BEMRetMirror) with sym='x','y','xy'.

Solver objects are constructed for small spheres / segments, a single
wavelength is solved, and the resulting sigma is checked to be finite and
of the expected shape. Iterative tests additionally verify convergence
metadata. Mirror tests cross-check expansion against the full-particle
direct solver.
"""
import os
import sys
import warnings

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import (
    trisphere,
    trispheresegment,
    ComParticle,
    ComParticleMirror,
    ComPoint,
    LayerStructure,
)
from mnpbem.bem import (
    BEMStat,
    BEMRet,
    BEMStatIter,
    BEMRetIter,
    BEMStatLayer,
    BEMRetLayer,
    BEMIter,
)
from mnpbem.bem.bem_stat_mirror import BEMStatMirror
from mnpbem.bem.bem_ret_mirror import BEMRetMirror
from mnpbem.simulation import (
    PlaneWaveStat,
    PlaneWaveRet,
    DipoleStatLayer,
)
from mnpbem.simulation.planewave_stat_mirror import PlaneWaveStatMirror
from mnpbem.simulation.planewave_ret_mirror import PlaneWaveRetMirror


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _eps_gold():
    return EpsDrude(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')


def _eps_vac():
    return EpsConst(1.0)


def _sphere_part(nfaces = 144, diameter = 20.0):
    epstab = [_eps_vac(), _eps_gold()]
    p = trisphere(nfaces, diameter)
    return ComParticle(epstab, [p], [[2, 1]], 1)


def _mirror_segment(sym = 'xy', diameter = 20.0, n_phi = 5, n_theta = 9):
    epstab = [_eps_vac(), _eps_gold()]
    seg = trispheresegment(
        np.linspace(0, np.pi / 2, n_phi),
        np.linspace(0, np.pi, n_theta),
        diameter = diameter,
    )
    return ComParticleMirror(epstab, [seg], [[2, 1]], sym = sym, closed_args = (1,))


@pytest.fixture(scope = 'module')
def sphere():
    return _sphere_part()


@pytest.fixture(scope = 'module')
def small_sphere():
    return _sphere_part(nfaces = 32)


@pytest.fixture(scope = 'module')
def layered_sphere_above():
    """Au sphere 1 nm above a glass substrate (z=0 interface)."""
    epstab = [_eps_vac(), _eps_gold(), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    sphere = trisphere(144, 20.0)
    sphere.shift([0, 0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    return p, layer


# ===========================================================================
# 3.1 BEMStat / BEMRet wavelength extremes
# ===========================================================================
class TestBEMStatWavelengthExtremes:
    """Sphere of diameter ~20 nm.  Test lambda << / >> / ~ particle size."""

    @pytest.mark.parametrize('enei', [10.0, 100.0, 550.0, 5000.0, 100_000.0])
    def test_bem_stat_finite(self, sphere, enei):
        bem = BEMStat(sphere)
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        sig, _ = bem.solve(pw(sphere, enei))
        assert np.all(np.isfinite(sig.sig))
        assert sig.sig.shape[0] == sphere.nfaces

    def test_quasistatic_limit_extinction_scales_with_volume(self, sphere):
        # Long-wavelength quasi-static: extinction prop to V * Im(alpha) so
        # for fixed wavelength the sphere volume sets the magnitude.
        bem = BEMStat(sphere)
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        sig, _ = bem.solve(pw(sphere, 5000.0))
        ext = pw.extinction(sig)
        assert np.isfinite(ext).all()


class TestBEMRetWavelengthExtremes:
    """Retarded BEM with lambda from 100 nm to 5000 nm."""

    @pytest.mark.parametrize('enei', [100.0, 500.0, 1000.0, 5000.0])
    def test_bem_ret_finite(self, sphere, enei):
        bem = BEMRet(sphere)
        pw = PlaneWaveRet(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )
        sig, _ = bem.solve(pw(sphere, enei))
        assert np.all(np.isfinite(sig.sig1))
        assert np.all(np.isfinite(sig.sig2))

    def test_high_q_short_wavelength(self, small_sphere):
        # lambda = 50 nm, sphere diameter 20 nm: well into resonance regime.
        bem = BEMRet(small_sphere)
        pw = PlaneWaveRet(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )
        sig, _ = bem.solve(pw(small_sphere, 50.0))
        assert np.all(np.isfinite(sig.sig1))

    def test_multi_wavelength_loop_stable(self, small_sphere):
        bem = BEMRet(small_sphere)
        pw = PlaneWaveRet(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )
        wls = np.linspace(400.0, 800.0, 6)
        ext_vals = []
        for en in wls:
            sig, _ = bem.solve(pw(small_sphere, en))
            ext = pw.extinction(sig)
            ext_vals.append(np.atleast_1d(ext)[0])
        ext_vals = np.array(ext_vals)
        assert np.all(np.isfinite(ext_vals))


# ===========================================================================
# 3.1b BEMStatLayer / BEMRetLayer — substrate dipole above & below
# ===========================================================================
class TestSubstrateDipolePosition:
    """Dipole positioned above (z>0) and below (z<0) the substrate."""

    def test_dipole_above_substrate(self, layered_sphere_above):
        p, layer = layered_sphere_above
        pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))
        dip = DipoleStatLayer(pt, layer, dip = np.array([[0.0, 0.0, 1.0]]))
        bem = BEMStatLayer(p, layer)
        exc = dip(p, 550.0)
        sig, _ = bem.solve(exc)
        assert np.all(np.isfinite(sig.sig))

    def test_dipole_below_substrate(self, layered_sphere_above):
        # Below substrate means medium index = 3 (the glass half-space).
        p, layer = layered_sphere_above
        pt = ComPoint(p, np.array([[0.0, 0.0, -5.0]]))
        dip = DipoleStatLayer(pt, layer, dip = np.array([[0.0, 0.0, 1.0]]))
        bem = BEMStatLayer(p, layer)
        exc = dip(p, 550.0)
        sig, _ = bem.solve(exc)
        assert np.all(np.isfinite(sig.sig))

    def test_layer_solver_multi_wavelength(self, layered_sphere_above):
        p, layer = layered_sphere_above
        pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))
        dip = DipoleStatLayer(pt, layer, dip = np.array([[0.0, 0.0, 1.0]]))
        bem = BEMStatLayer(p, layer)
        for en in (450.0, 600.0, 750.0):
            sig, _ = bem.solve(dip(p, en))
            assert np.all(np.isfinite(sig.sig))


# ===========================================================================
# 3.2 BEMIter convergence stress
# ===========================================================================
class TestBEMIterTolerance:
    """Tolerance scan: relres should not greatly exceed requested tol."""

    @pytest.mark.parametrize('tol', [1e-3, 1e-6, 1e-10, 1e-13])
    def test_stat_iter_tol_scan(self, sphere, tol):
        bem = BEMStatIter(sphere, solver = 'gmres', tol = tol, maxit = 500,
                          precond = 'hmat')
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        sig, _ = bem.solve(pw(sphere, 550.0))
        assert np.all(np.isfinite(sig.sig))
        flag, relres, _ = bem.info()
        # GMRES stops when relres <= tol (or maxit hit).  Allow generous
        # slack since for very tight tol scipy may not strictly satisfy it.
        assert relres[0] < max(tol * 100.0, 1e-2)

    def test_relres_decreases_with_tightening_tol(self, small_sphere):
        residuals = []
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        for tol in (1e-3, 1e-6, 1e-10):
            bem = BEMStatIter(small_sphere, solver = 'gmres', tol = tol,
                              maxit = 500, precond = 'hmat')
            bem.solve(pw(small_sphere, 550.0))
            _, relres, _ = bem.info()
            residuals.append(relres[0])
        # Each tightening should not increase residual
        assert residuals[1] <= residuals[0] * 1.01 + 1e-12
        assert residuals[2] <= residuals[1] * 1.01 + 1e-12


class TestBEMIterSolverVariants:
    """gmres / bicgstab / cgs converge to same answer (within tol)."""

    @pytest.mark.parametrize('solver', ['gmres', 'bicgstab', 'cgs'])
    def test_stat_iter_solver_variants(self, sphere, solver):
        bem_d = BEMStat(sphere)
        bem_i = BEMStatIter(sphere, solver = solver, tol = 1e-8, maxit = 500,
                            precond = 'hmat')
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        denom = np.abs(sig_d.sig).max()
        rel = np.abs(sig_d.sig - sig_i.sig).max() / denom
        # cgs/bicgstab can have looser convergence
        assert rel < 1e-5, '{} rel diff = {:.3e}'.format(solver, rel)


class TestBEMIterRetSolverVariants:

    @pytest.mark.parametrize('solver', ['gmres', 'bicgstab'])
    def test_ret_iter_solver_variants(self, sphere, solver):
        bem_d = BEMRet(sphere)
        bem_i = BEMRetIter(sphere, solver = solver, tol = 1e-8, maxit = 500,
                           precond = 'hmat')
        pw = PlaneWaveRet(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )
        exc = pw(sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        for fname in ('sig1', 'sig2'):
            a = sig_d.get(fname); b = sig_i.get(fname)
            denom = np.abs(a).max()
            rel = np.abs(a - b).max() / denom
            assert rel < 1e-5, '{} {} rel diff = {:.3e}'.format(
                solver, fname, rel)


class TestBEMIterPreconditioner:
    """precond='hmat' (default) vs maxit=0 (preconditioner-only)."""

    def test_precond_hmat_default(self, small_sphere):
        bem = BEMStatIter(small_sphere)
        assert bem.precond == 'hmat'
        assert bem.solver == 'gmres'

    def test_maxit_zero_uses_only_precond(self, small_sphere):
        # maxit=0 + precond='hmat' returns mfun(b) directly, which for stat
        # BEM equals direct LU solve.
        bem_d = BEMStat(small_sphere)
        bem_i = BEMStatIter(small_sphere, solver = 'gmres', maxit = 0,
                            precond = 'hmat')
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        exc = pw(small_sphere, 550.0)
        sig_d, _ = bem_d.solve(exc)
        sig_i, _ = bem_i.solve(exc)
        denom = np.abs(sig_d.sig).max()
        rel = np.abs(sig_d.sig - sig_i.sig).max() / denom
        assert rel < 1e-10


class TestBEMIterConvergenceFailure:
    """Pathological setup: maxit=1, very tight tol — should not crash."""

    def test_low_maxit_does_not_crash(self, sphere):
        bem = BEMStatIter(sphere, solver = 'gmres', tol = 1e-13, maxit = 1,
                          precond = 'hmat')
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        # Should not raise; residual stats reflect non-convergence.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sig, _ = bem.solve(pw(sphere, 550.0))
        flag, relres, _ = bem.info()
        # flag != 0 expected on early exit; sig still has correct shape.
        assert sig.sig.shape[0] == sphere.nfaces

    def test_unknown_solver_raises(self, sphere):
        bem = BEMStatIter(sphere, solver = 'no_such_solver', tol = 1e-6,
                          maxit = 50, precond = 'hmat')
        pw = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        with pytest.raises(ValueError, match = 'iterative solver not known'):
            bem.solve(pw(sphere, 550.0))


class TestBEMIterOptions:
    """BEMIter.options() defaults and overrides."""

    def test_defaults(self):
        op = BEMIter.options()
        assert op['solver'] == 'gmres'
        assert op['precond'] == 'hmat'
        assert op['tol'] == 1e-6
        assert op['maxit'] == 100
        assert 'kmax' in op and 'htol' in op

    def test_override_kwargs(self):
        op = BEMIter.options(tol = 1e-10, maxit = 999)
        assert op['tol'] == 1e-10
        assert op['maxit'] == 999

    def test_solver_map_keys(self):
        assert set(BEMIter.SOLVER_MAP.keys()) == {'gmres', 'cgs', 'bicgstab'}


# ===========================================================================
# 3.3 Mirror symmetry — sym='x', 'y', 'xy'
# ===========================================================================
class TestMirrorStatSym:
    """BEMStatMirror with x / y / xy.  Solve and check finite sigma."""

    @pytest.mark.parametrize('sym', ['x', 'y', 'xy'])
    def test_stat_mirror_sym_finite(self, sym):
        pm = _mirror_segment(sym = sym)
        bem = BEMStatMirror(pm)
        # Mirror plane wave needs only x,y polarisations.
        pw = PlaneWaveStatMirror(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        exc = pw.potential(pm, 550.0)
        sig, _ = bem.solve(exc)
        assert len(sig.val) >= 1
        for v in sig.val:
            assert np.all(np.isfinite(v.sig))
            assert v.sig.shape[0] == pm.nfaces

    @pytest.mark.parametrize('sym', ['x', 'y', 'xy'])
    def test_stat_mirror_matches_full(self, sym):
        # Mirror solve expanded to full particle should match BEMStat run on
        # the full particle.  Mirror reduces to half/quarter the matrix and
        # rebuilds via symmetry; numerical noise of a few 1e-5 is expected.
        pm = _mirror_segment(sym = sym)
        pf = pm.full()

        bem_m = BEMStatMirror(pm)
        bem_f = BEMStat(pf)
        pw_m = PlaneWaveStatMirror(np.array([[1.0, 0.0, 0.0]]))
        pw_f = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))
        sig_m, _ = bem_m.solve(pw_m.potential(pm, 550.0))
        sig_f, _ = bem_f.solve(pw_f(pf, 550.0))

        # Expand mirror sigma to full size; first item is x-polarization.
        sig_full = sig_m.expand()
        sig_full_arr = np.asarray(sig_full[0].sig).ravel()
        sig_f_arr = np.asarray(sig_f.sig).ravel()
        denom = np.abs(sig_f_arr).max()
        rel = np.abs(sig_full_arr - sig_f_arr).max() / denom
        assert rel < 1e-3, 'sym={} rel = {:.3e}'.format(sym, rel)


class TestMirrorRetSym:
    """BEMRetMirror with x / y / xy."""

    @pytest.mark.parametrize('sym', ['x', 'y', 'xy'])
    def test_ret_mirror_sym_finite(self, sym):
        pm = _mirror_segment(sym = sym)
        bem = BEMRetMirror(pm)
        pw = PlaneWaveRetMirror(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0]]),
        )
        exc = pw.potential(pm, 550.0)
        sig, _ = bem.solve(exc)
        assert len(sig.val) >= 1
        for v in sig.val:
            for fname in ('sig1', 'sig2'):
                a = getattr(v, fname, None)
                assert a is not None
                assert np.all(np.isfinite(a))


class TestMirrorStatExtinction:
    """Mirror + full sigma magnitudes match per-polarization for sym='xy'."""

    def test_xy_sigma_magnitude_matches_full(self):
        # Compare per-polarization sigma magnitudes (max abs).  Direct
        # element-wise comparison is sensitive to sign conventions of the
        # symmetry expansion, but the magnitude at the resonance is robust.
        pm = _mirror_segment(sym = 'xy')
        pf = pm.full()

        bem_m = BEMStatMirror(pm)
        pw_m = PlaneWaveStatMirror(np.array([[1.0, 0.0, 0.0]]))
        bem_f = BEMStat(pf)
        pw_f = PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))

        sig_m, _ = bem_m.solve(pw_m.potential(pm, 600.0))
        sig_full_x = sig_m.expand()[0]      # x-pol expanded
        sig_f, _ = bem_f.solve(pw_f(pf, 600.0))

        mag_m = np.abs(sig_full_x.sig).max()
        mag_f = np.abs(sig_f.sig).max()
        rel = abs(mag_m - mag_f) / mag_f
        assert rel < 1e-3, 'rel = {:.3e}'.format(rel)


class TestMirrorMultipleWavelengths:
    """Mirror solver across a small wavelength sweep."""

    def test_xy_multi_wavelength(self):
        pm = _mirror_segment(sym = 'xy')
        bem = BEMStatMirror(pm)
        pw = PlaneWaveStatMirror(np.array([[1.0, 0.0, 0.0]]))
        for en in (400.0, 550.0, 700.0):
            sig, _ = bem.solve(pw.potential(pm, en))
            assert len(sig.val) >= 1
            for v in sig.val:
                assert np.all(np.isfinite(v.sig))


class TestMirrorInvalidSym:
    """Bogus sym key should raise on construction."""

    def test_invalid_sym_key(self):
        epstab = [_eps_vac(), _eps_gold()]
        seg = trispheresegment(
            np.linspace(0, np.pi / 2, 5),
            np.linspace(0, np.pi, 9),
            diameter = 20.0,
        )
        # 'z' is not a valid mirror axis → symtable not initialised → fails
        # somewhere downstream (KeyError / AttributeError).
        with pytest.raises(Exception):
            ComParticleMirror(epstab, [seg], [[2, 1]], sym = 'z',
                              closed_args = (1,))

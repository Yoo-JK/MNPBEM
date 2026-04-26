"""M3 A3 — Numerical robustness tests.

Covers mesh quality (aspect ratio extremes, very fine / very coarse meshes),
refinement / quadrature option scans, and h-/p-/tolerance-convergence
checks for sphere extinction (compared to Mie theory where appropriate).
"""
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import trisphere, trirod, ComParticle
from mnpbem.bem import BEMRet, BEMRetIter
from mnpbem.simulation import PlaneWaveRet
from mnpbem.mie import MieRet


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _gold_eps():
    return EpsDrude(eps0=10.0, wp=9.065, gammad=0.0708, name='gold')


def _vac_eps():
    return EpsConst(1.0)


def _sphere(nverts, diameter=20.0, **kw):
    p = trisphere(nverts, diameter, **kw)
    cp = ComParticle([_vac_eps(), _gold_eps()], [p], [[2, 1]], 1, **kw)
    return cp, p


def _solve_ret_ext(cp, enei=550.0, iter_kwargs=None):
    pw = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                      np.array([[0.0, 0.0, 1.0]]))
    if iter_kwargs is None:
        bem = BEMRet(cp)
    else:
        bem = BEMRetIter(cp, **iter_kwargs)
    sig, _ = bem.solve(pw(cp, enei))
    ext = float(np.atleast_1d(pw.extinction(sig))[0])
    sca, _ = pw.scattering(sig)
    sca = float(np.atleast_1d(sca)[0])
    return ext, sca, bem


def _aspect_ratios(particle):
    """Per-face longest-edge / shortest-edge ratio for triangular faces."""
    faces = particle.faces
    verts = particle.verts
    ratios = []
    for f in faces:
        mask = ~np.isnan(f)
        idx = f[mask].astype(int)
        if len(idx) < 3:
            continue
        v = verts[idx]
        n = len(idx)
        edges = [np.linalg.norm(v[(i + 1) % n] - v[i]) for i in range(n)]
        emin = min(edges)
        if emin <= 0:
            continue
        ratios.append(max(edges) / emin)
    return np.array(ratios)


# ---------------------------------------------------------------------------
# 6.1 Mesh quality
# ---------------------------------------------------------------------------

class TestMeshQuality:
    """Aspect ratio extremes, very fine / very coarse meshes."""

    def test_sphere_aspect_ratio_well_conditioned(self):
        # Trisphere uses energy-minimised vertex distribution → near-uniform.
        for n in (32, 144, 484):
            p = trisphere(n, 10.0)
            ar = _aspect_ratios(p)
            assert ar.size > 0
            # Sphere meshes should have aspect ratio < ~3 (well conditioned)
            assert ar.max() < 3.0, \
                'sphere({}): max aspect ratio = {:.3f}'.format(n, ar.max())
            assert ar.min() >= 1.0 - 1e-9

    def test_rod_aspect_ratio_extreme_elongation(self):
        # Highly elongated rod → cylinder cells stretch along z and produce
        # large aspect ratios. h/d = 1.5 is degenerate (no cylinder), so
        # compare h/d = 3 vs h/d = 20.
        p_short = trirod(10.0, 30.0, n=[15, 10, 8])    # h/d = 3
        p_long = trirod(10.0, 200.0, n=[15, 10, 8])   # h/d = 20
        ar_short = _aspect_ratios(p_short).max()
        ar_long = _aspect_ratios(p_long).max()
        assert ar_long > ar_short, \
            'long rod aspect ratio not larger: short={:.2f}, long={:.2f}'.format(
                ar_short, ar_long)
        # Long rod should have aspect ratio > 5 (cylinder cells dominate).
        assert ar_long > 5.0

    def test_high_aspect_rod_solvable(self):
        # Even high-aspect-ratio rod is solvable: extinction is finite + > 0.
        eps_v, eps_g = _vac_eps(), _gold_eps()
        p = trirod(10.0, 100.0, n=[15, 10, 6])  # h/d = 10
        cp = ComParticle([eps_v, eps_g], [p], [[2, 1]], 1)
        ext, sca, _ = _solve_ret_ext(cp, enei=700.0)
        assert np.isfinite(ext) and np.isfinite(sca)
        assert ext > 0
        assert sca >= 0

    def test_very_coarse_mesh_solvable(self):
        # Smallest available sphere data is 32 verts (60 faces); solver should
        # still produce a finite, non-zero extinction.
        cp, p = _sphere(32, diameter=10.0)
        assert p.nfaces == 60
        ext, sca, _ = _solve_ret_ext(cp, enei=600.0)
        assert np.isfinite(ext) and np.isfinite(sca)
        assert ext > 0
        assert sca >= 0
        assert sca <= ext + 1e-8 * abs(ext)

    @pytest.mark.slow
    def test_very_fine_mesh_solvable(self):
        # 1444 verts ~ 2884 faces, > 2k mesh — confirm it runs and converges
        # towards a small-particle limit (still close to Mie).
        cp, p = _sphere(1444, diameter=20.0)
        assert p.nfaces > 2500
        ext_bem, _, _ = _solve_ret_ext(cp, enei=550.0)
        mie = MieRet(_gold_eps(), _vac_eps(), 20.0, lmax=10)
        ext_mie = float(np.atleast_1d(mie.extinction(550.0))[0])
        rel = abs(ext_bem - ext_mie) / abs(ext_mie)
        assert rel < 5e-3, 'fine mesh ext disagrees with Mie: rel={:.3e}'.format(rel)

    def test_quadrature_options_give_close_results(self):
        # Same sphere, different quadrature options → results within ~1%.
        results = []
        for opts in ({'rule': 7, 'npol': 5},
                     {'rule': 18, 'npol': 5},
                     {'rule': 18, 'npol': (7, 5)}):
            cp, _ = _sphere(144, diameter=20.0, **opts)
            ext, _, _ = _solve_ret_ext(cp, enei=550.0)
            results.append(ext)
        results = np.array(results)
        spread = (results.max() - results.min()) / results.mean()
        assert spread < 1e-2, 'quadrature spread too large: {:.3e}'.format(spread)


# ---------------------------------------------------------------------------
# 6.2 Refinement / quadrature scans
# ---------------------------------------------------------------------------

class TestRefinementScan:
    """refine / npol scans → results stable & finite."""

    @pytest.mark.parametrize('refine', [None, 1, 2, 3, 4])
    def test_refine_scan_finite(self, refine):
        cp, _ = _sphere(144, diameter=20.0, refine=refine)
        ext, sca, _ = _solve_ret_ext(cp, enei=550.0)
        assert np.isfinite(ext) and np.isfinite(sca)
        assert ext > 0

    def test_refine_scan_results_consistent(self):
        # Different refine values → result spread small.
        exts = []
        for r in (None, 1, 2, 3):
            cp, _ = _sphere(144, diameter=20.0, refine=r)
            ext, _, _ = _solve_ret_ext(cp, enei=550.0)
            exts.append(ext)
        exts = np.array(exts)
        # refine=None / refine=1 should be identical (refine>1 only triggers).
        assert abs(exts[0] - exts[1]) < 1e-12 * abs(exts[0])
        spread = (exts.max() - exts.min()) / exts.mean()
        assert spread < 5e-3, 'refine spread too large: {:.3e}'.format(spread)

    @pytest.mark.parametrize('npol', [5, 10, 20])
    def test_npol_scan_finite(self, npol):
        cp, _ = _sphere(144, diameter=20.0, npol=npol)
        ext, sca, _ = _solve_ret_ext(cp, enei=550.0)
        assert np.isfinite(ext) and np.isfinite(sca)
        assert ext > 0

    def test_npol_scan_converges(self):
        # As npol grows, results approach a limit.
        exts = []
        for npol in (5, 10, 20):
            cp, _ = _sphere(144, diameter=20.0, npol=npol)
            ext, _, _ = _solve_ret_ext(cp, enei=550.0)
            exts.append(ext)
        # Successive differences should not grow (rough monotone convergence).
        d1 = abs(exts[1] - exts[0])
        d2 = abs(exts[2] - exts[1])
        assert d2 <= d1 + 1e-6 * abs(exts[0]), \
            'npol diffs non-monotone: d1={:.3e} d2={:.3e}'.format(d1, d2)
        # Final spread is small.
        spread = (max(exts) - min(exts)) / np.mean(exts)
        assert spread < 1e-2

    def test_mixed_refine_multiparticle(self):
        # Two spheres with different refine levels → solve still works.
        eps_v, eps_g = _vac_eps(), _gold_eps()
        p1 = trisphere(144, 10.0)
        p2 = trisphere(144, 10.0)
        # Apply different refine levels via individual particle quad.
        from mnpbem.utils.quadface import QuadFace as _QF
        p1.quad = _QF(rule=18, npol=(7, 5), refine=2)
        p2.quad = _QF(rule=18, npol=(7, 5), refine=4)
        # Shift p2 so they don't overlap.
        p2 = p2.shift([30.0, 0.0, 0.0])
        cp = ComParticle([eps_v, eps_g], [p1, p2], [[2, 1], [2, 1]], 1)
        ext, sca, _ = _solve_ret_ext(cp, enei=550.0)
        assert np.isfinite(ext) and ext > 0


# ---------------------------------------------------------------------------
# 6.3 Convergence
# ---------------------------------------------------------------------------

class TestHConvergence:
    """h-convergence: mesh refinement → result approaches Mie limit."""

    def test_h_convergence_monotone_to_mie(self):
        diameter = 20.0
        enei = 550.0
        mie = MieRet(_gold_eps(), _vac_eps(), diameter, lmax=10)
        ext_mie = float(np.atleast_1d(mie.extinction(enei))[0])

        ns = [144, 256, 484]
        errs = []
        for n in ns:
            cp, _ = _sphere(n, diameter=diameter)
            ext, _, _ = _solve_ret_ext(cp, enei=enei)
            errs.append(abs(ext - ext_mie) / abs(ext_mie))

        # h-convergence: monotone non-increasing error.
        for i in range(len(errs) - 1):
            assert errs[i + 1] <= errs[i] * 1.05, \
                'h-convergence not monotone: errs={}'.format(errs)
        # Finest mesh within 1% of Mie.
        assert errs[-1] < 1e-2, \
            'finest h-mesh err = {:.3e}'.format(errs[-1])


class TestPConvergence:
    """p-convergence: integration order increase → result stabilises."""

    def test_p_convergence_via_npol(self):
        # Vary polar quadrature order; result spread shrinks.
        cp_list_results = []
        for npol in (5, 10, 20):
            cp, _ = _sphere(144, diameter=20.0, npol=npol)
            ext, _, _ = _solve_ret_ext(cp, enei=550.0)
            cp_list_results.append(ext)

        diffs = np.abs(np.diff(cp_list_results))
        # diffs should shrink (Cauchy-style convergence).
        assert diffs[1] <= diffs[0] + 1e-6 * abs(cp_list_results[0])

    def test_p_convergence_via_rule(self):
        # Vary triangle rule (ngauss).
        exts = []
        for rule in (1, 4, 7, 18):
            cp, _ = _sphere(144, diameter=20.0, rule=rule)
            ext, _, _ = _solve_ret_ext(cp, enei=550.0)
            exts.append(ext)
        # Highest-rule and second-highest agree closely.
        rel = abs(exts[-1] - exts[-2]) / abs(exts[-1])
        assert rel < 5e-3, \
            'p-convergence rule spread: exts={}'.format(exts)


class TestIterativeToleranceScan:
    """Iterative tol = 1e-3, 1e-6, 1e-10 → error decreases roughly with tol."""

    def test_tol_scan_monotone(self):
        cp, _ = _sphere(144, diameter=20.0)

        bem_d = BEMRet(cp)
        pw = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                          np.array([[0.0, 0.0, 1.0]]))
        sig_d, _ = bem_d.solve(pw(cp, 550.0))
        ext_d = float(np.atleast_1d(pw.extinction(sig_d))[0])

        prev_err = np.inf
        for tol in (1e-3, 1e-6, 1e-10):
            bem_i = BEMRetIter(cp, solver='gmres', tol=tol, maxit=500,
                               precond=None)
            sig_i, _ = bem_i.solve(pw(cp, 550.0))
            ext_i = float(np.atleast_1d(pw.extinction(sig_i))[0])
            err = abs(ext_d - ext_i) / abs(ext_d)
            # Error should decrease as tol tightens.
            assert err <= prev_err + 1e-8, \
                'tol={:.0e} err={:.3e} > prev={:.3e}'.format(tol, err, prev_err)
            prev_err = err
        # Tightest tol → near machine precision.
        assert prev_err < 1e-7

    def test_tol_with_hmat_precond_machine_precision(self):
        # With hmat preconditioner, even loose tol gives machine-precision.
        cp, _ = _sphere(144, diameter=20.0)
        bem_d = BEMRet(cp)
        pw = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                          np.array([[0.0, 0.0, 1.0]]))
        sig_d, _ = bem_d.solve(pw(cp, 550.0))
        ext_d = float(np.atleast_1d(pw.extinction(sig_d))[0])

        bem_i = BEMRetIter(cp, solver='gmres', tol=1e-3, maxit=300,
                           precond='hmat')
        sig_i, _ = bem_i.solve(pw(cp, 550.0))
        ext_i = float(np.atleast_1d(pw.extinction(sig_i))[0])
        rel = abs(ext_d - ext_i) / abs(ext_d)
        assert rel < 1e-8, 'hmat-precond rel={:.3e}'.format(rel)

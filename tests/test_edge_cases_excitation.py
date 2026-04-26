"""
Edge case tests for excitation sources (M3 Wave 2 B2).

Validates PlaneWave / Dipole / ElectronBeam excitation classes under
extreme but physically meaningful inputs:

- PlaneWave: incidence angles 0 deg .. 89.99 deg, TM/TE/RHC/LHC/custom
  polarisations, multi-pol simultaneous evaluation, layer combinations.
- PointDipole: positions inside / on surface / far / very near a small
  sphere; orientations along x/y/z and arbitrary unit vectors; multi-
  dipole arrays; layer / mirror combinations.
- ElectronBeam (EELS): aloof / penetrating / grazing impact parameters,
  multi-impact, several kinetic energies (60 keV / 200 keV).

Each test only runs the excitation construction and a single forward
evaluation (potential / field) on a small sphere and checks that the
output has finite values with the expected shape - no full BEM solve
is performed, keeping the suite fast.
"""

import sys
import os
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bypass the broken top-level mnpbem.__init__ so subpackages import cleanly
# (mirrors mnpbem/tests/test_ret_scattering.py).
# ---------------------------------------------------------------------------
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry import (
    trisphere, ComParticle, ComPoint, ComParticleMirror, LayerStructure,
)
from mnpbem.simulation.planewave_stat import PlaneWaveStat
from mnpbem.simulation.planewave_ret import PlaneWaveRet
from mnpbem.simulation.planewave_ret_layer import PlaneWaveRetLayer
from mnpbem.simulation.dipole_stat import DipoleStat
from mnpbem.simulation.dipole_ret import DipoleRet
from mnpbem.simulation.dipole_ret_layer import DipoleRetLayer
from mnpbem.simulation.dipole_stat_mirror import DipoleStatMirror
from mnpbem.simulation.dipole_ret_mirror import DipoleRetMirror
from mnpbem.simulation.eels_stat import EELSStat
from mnpbem.simulation.eels_ret import EELSRet
from mnpbem.simulation.eels_base import EELSBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS_VAC = EpsConst(1.0)
EPS_METAL = EpsConst(-10.0 + 0.5j)
WL = 600.0


def _sphere(diameter=20.0, n_faces=32, z_center=0.0):
    """Create a small metallic sphere ComParticle with optional z-shift."""
    p = trisphere(n_faces, diameter)
    if z_center != 0.0:
        p.verts = p.verts.copy()
        p.verts[:, 2] += z_center
        p.pos = p.pos.copy()
        p.pos[:, 2] += z_center
    cp = ComParticle([EPS_VAC, EPS_METAL], [p], [[2, 1]])
    return cp


def _sphere_mirror(diameter=20.0, n_faces=32, sym='x'):
    """Mirror-symmetric quarter sphere."""
    p = trisphere(n_faces, diameter)
    cpm = ComParticleMirror([EPS_VAC, EPS_METAL], [p], [[2, 1]], sym=sym)
    return cpm


def _flat_layer():
    """Single z=0 vacuum/glass interface."""
    epstab = [EPS_VAC, EpsConst(2.25)]
    return LayerStructure(epstab, [1, 2], [0.0])


def _planewave_dirs(angle_deg, plane='xz'):
    """Build (pol_TM, pol_TE, dir) for a given incidence angle.

    angle is measured from the -z propagation axis; angle=0 -> normal
    incidence along -z.  TM polarisation lies in the plane of incidence,
    TE perpendicular to it.
    """
    th = np.deg2rad(angle_deg)
    if plane == 'xz':
        dir_vec = np.array([np.sin(th), 0.0, -np.cos(th)])
        pol_TM = np.array([np.cos(th), 0.0, np.sin(th)])
        pol_TE = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(plane)
    return pol_TM, pol_TE, dir_vec


def _finite(arr):
    return np.all(np.isfinite(np.asarray(arr).view(float)))


# ---------------------------------------------------------------------------
# 4.1 PlaneWave edge cases
# ---------------------------------------------------------------------------

class TestPlaneWaveAngles:
    """Various incidence angles for PlaneWaveRet (full Maxwell)."""

    @pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 89.99])
    def test_TM_potential(self, angle):
        cp = _sphere()
        pol_TM, _, dir_vec = _planewave_dirs(angle)
        pw = PlaneWaveRet(pol_TM, dir_vec)
        exc = pw.potential(cp, WL)
        assert exc.a1.shape[:2] == (cp.nfaces, 3)
        assert _finite(exc.a1)
        assert _finite(exc.a1p)

    @pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 89.99])
    def test_TE_potential(self, angle):
        cp = _sphere()
        _, pol_TE, dir_vec = _planewave_dirs(angle)
        pw = PlaneWaveRet(pol_TE, dir_vec)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1)
        assert _finite(exc.a2)

    @pytest.mark.xfail(
        reason="planewave_ret.field uses p.index[i] (per-face subobject id, "
               "not per-subobject face indices) and lacks the dtype=int empty "
               "fallback that potential() has -- see /tmp/m3_b2_bugs.md BUG#1.",
        strict=True,
    )
    def test_normal_incidence_field(self):
        cp = _sphere()
        pw = PlaneWaveRet([1, 0, 0], [0, 0, -1])
        f = pw.field(cp, WL)
        assert f.e.shape == (cp.nfaces, 3)
        assert _finite(f.e)
        assert _finite(f.h)

    @pytest.mark.xfail(
        reason="Same bug as test_normal_incidence_field (BUG#1).",
        strict=True,
    )
    def test_grazing_field(self):
        cp = _sphere()
        pol_TM, _, dir_vec = _planewave_dirs(89.99)
        pw = PlaneWaveRet(pol_TM, dir_vec)
        f = pw.field(cp, WL)
        assert _finite(f.e)


class TestPlaneWavePolarisations:
    """TM / TE / RHC / LHC / custom polarisations."""

    def test_RHC_polarisation(self):
        cp = _sphere()
        # Right-hand circular: pol = (x + i y) / sqrt(2), dir = +z
        pol = np.array([1.0, 1j, 0.0]) / np.sqrt(2)
        dir_vec = np.array([0.0, 0.0, 1.0])
        pw = PlaneWaveRet(pol, dir_vec)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1)

    def test_LHC_polarisation(self):
        cp = _sphere()
        pol = np.array([1.0, -1j, 0.0]) / np.sqrt(2)
        dir_vec = np.array([0.0, 0.0, 1.0])
        pw = PlaneWaveRet(pol, dir_vec)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1)

    def test_custom_polarisation(self):
        cp = _sphere()
        # (1, 1, 0) / sqrt(2) - in xy plane, with +z propagation: orthogonal
        pol = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        dir_vec = np.array([0.0, 0.0, 1.0])
        pw = PlaneWaveRet(pol, dir_vec)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1)

    def test_stat_polarisations(self):
        cp = _sphere()
        for pol in [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 1, 1]]:
            pol = np.asarray(pol, dtype=float)
            pol = pol / np.linalg.norm(pol)
            pw = PlaneWaveStat(pol)
            exc = pw.potential(cp, WL)
            # quasistatic potential exposes the surface derivative as 'phip'
            assert _finite(exc.phip)


class TestPlaneWaveSimultaneous:
    """Multi-polarisation simultaneous evaluation."""

    def test_two_pols_orthogonal(self):
        cp = _sphere()
        pol = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])
        dir_vec = np.array([[0.0, 0.0, -1.0],
                            [0.0, 0.0, -1.0]])
        pw = PlaneWaveRet(pol, dir_vec)
        exc = pw.potential(cp, WL)
        assert exc.a1.shape == (cp.nfaces, 3, 2)
        assert _finite(exc.a1)

    def test_three_pols(self):
        cp = _sphere()
        pol = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0] / np.sqrt(2)])
        dir_vec = np.tile(np.array([0.0, 0.0, -1.0]), (3, 1))
        pw = PlaneWaveRet(pol, dir_vec)
        exc = pw.potential(cp, WL)
        assert exc.a1.shape[2] == 3

    def test_stat_two_pols(self):
        cp = _sphere()
        pol = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pw = PlaneWaveStat(pol)
        exc = pw.potential(cp, WL)
        assert exc.phip.shape[-1] == 2


class TestPlaneWaveLayer:
    """PlaneWaveRetLayer at several incidence angles."""

    @pytest.mark.parametrize("angle", [0.0, 30.0, 60.0])
    def test_planewave_ret_layer(self, angle):
        layer = _flat_layer()
        # Particle above the layer
        cp = _sphere(z_center=30.0)
        pol_TM, _, dir_vec = _planewave_dirs(angle)
        pw = PlaneWaveRetLayer(pol_TM, dir_vec, layer)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1) and _finite(exc.a2)

    def test_planewave_ret_layer_TE(self):
        layer = _flat_layer()
        cp = _sphere(z_center=30.0)
        _, pol_TE, dir_vec = _planewave_dirs(45.0)
        pw = PlaneWaveRetLayer(pol_TE, dir_vec, layer)
        exc = pw.potential(cp, WL)
        assert _finite(exc.a1)


# ---------------------------------------------------------------------------
# 4.2 PointDipole / DipoleRet edge cases
# ---------------------------------------------------------------------------

class TestDipolePosition:
    """Various dipole positions relative to a small (R = 10nm) sphere."""

    @pytest.mark.parametrize("z", [50.0, 30.0, 11.0, 10.0001])
    def test_outside(self, z):
        cp = _sphere(diameter=20.0)
        pt = ComPoint(cp, np.array([[0.0, 0.0, z]]))
        d = DipoleRet(pt, np.array([0, 0, 1]))
        exc = d.potential(cp, WL)
        assert _finite(exc.a1)
        assert _finite(exc.phi1)

    def test_far(self):
        cp = _sphere(diameter=20.0)
        pt = ComPoint(cp, np.array([[0.0, 0.0, 1000.0]]))
        d = DipoleRet(pt, np.array([0, 0, 1]))
        exc = d.potential(cp, WL)
        assert _finite(exc.a1)

    def test_very_near_surface(self):
        # 1e-5 nm above sphere surface (radius=10nm => z=10 + 1e-5)
        cp = _sphere(diameter=20.0)
        pt = ComPoint(cp, np.array([[0.0, 0.0, 10.0 + 1e-5]]))
        d = DipoleRet(pt, np.array([0, 0, 1]))
        exc = d.potential(cp, WL)
        # Near-field can be huge but must remain finite
        assert _finite(exc.a1)
        assert _finite(exc.phi1)


class TestDipoleOrientation:
    """Dipole moment orientations."""

    @pytest.mark.parametrize("dip", [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        np.array([1.0, -2.0, 3.0]) / np.linalg.norm([1.0, -2.0, 3.0]),
    ])
    def test_orientation(self, dip):
        cp = _sphere()
        pt = ComPoint(cp, np.array([[0.0, 0.0, 30.0]]))
        d = DipoleRet(pt, dip)
        exc = d.potential(cp, WL)
        assert _finite(exc.a1)

    def test_default_eye3(self):
        # No dip argument -> 3 orthogonal dipoles (npt, 3, 3)
        cp = _sphere()
        pt = ComPoint(cp, np.array([[0.0, 0.0, 30.0]]))
        d = DipoleRet(pt)
        assert d.dip.shape == (1, 3, 3)
        exc = d.potential(cp, WL)
        # last axis is ndip = 3
        assert exc.a1.shape[-1] == 3

    def test_stat_orientation(self):
        cp = _sphere()
        pt = ComPoint(cp, np.array([[0.0, 0.0, 30.0]]))
        for dip in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            d = DipoleStat(pt, np.asarray(dip, dtype=float))
            exc = d.potential(cp, WL)
            assert _finite(exc.phip)


class TestDipoleMultiple:
    """Multiple dipoles evaluated together."""

    def test_two_positions(self):
        cp = _sphere()
        positions = np.array([
            [0.0, 0.0, 30.0],
            [0.0, 0.0, -30.0],
        ])
        pt = ComPoint(cp, positions)
        d = DipoleRet(pt, np.array([0, 0, 1]))
        exc = d.potential(cp, WL)
        # shape (nfaces, 3, npt, ndip)
        assert exc.a1.shape[2] == 2

    def test_three_positions(self):
        cp = _sphere()
        positions = np.array([
            [30.0, 0.0, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 30.0],
        ])
        pt = ComPoint(cp, positions)
        d = DipoleRet(pt, np.eye(3))
        exc = d.potential(cp, WL)
        # potential output flattens (npos, ndip) into the trailing axis,
        # so 3 positions x 3 dipole orientations -> 9
        assert exc.a1.shape == (cp.nfaces, 3, 9)
        assert _finite(exc.a1)


class TestDipoleLayer:
    """DipoleRetLayer with various dipole positions."""

    @pytest.mark.parametrize("z", [10.0, 30.0, 100.0])
    def test_dipole_above_layer(self, z):
        layer = _flat_layer()
        cp = _sphere(z_center=30.0)
        pt = ComPoint(cp, np.array([[0.0, 0.0, z]]), layer=layer)
        d = DipoleRetLayer(pt, layer, np.array([0, 0, 1]))
        exc = d.potential(cp, WL)
        assert _finite(exc.a1)


class TestDipoleMirror:
    """DipoleStatMirror / DipoleRetMirror smoke construction."""

    def test_dipole_stat_mirror_construction(self):
        cpm = _sphere_mirror(sym='x')
        pt = ComPoint(cpm, np.array([[0.0, 0.0, 30.0]]))
        d = DipoleStatMirror(pt, np.array([0, 0, 1]))
        # potential evaluates via internal DipoleStat; just check construction
        assert d.dip is not None

    def test_dipole_ret_mirror_construction(self):
        cpm = _sphere_mirror(sym='x')
        pt = ComPoint(cpm, np.array([[0.0, 0.0, 30.0]]))
        d = DipoleRetMirror(pt, np.array([0, 0, 1]))
        assert d.dip is not None


# ---------------------------------------------------------------------------
# 4.3 ElectronBeam (EELS) edge cases
# ---------------------------------------------------------------------------

class TestElectronBeamImpact:
    """Aloof / penetrating / grazing impact parameters."""

    def test_aloof(self):
        cp = _sphere(diameter=20.0)
        # impact 30 nm from sphere center, radius=10nm => clearly aloof
        impact = np.array([[30.0, 0.0]])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.a1)
        assert _finite(exc.phi1)

    def test_penetrating(self):
        cp = _sphere(diameter=20.0)
        # impact at center -> beam pierces particle
        impact = np.array([[0.0, 0.0]])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.a1)

    def test_grazing(self):
        cp = _sphere(diameter=20.0)
        # impact essentially at the sphere boundary (radius=10nm)
        impact = np.array([[10.001, 0.0]])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.a1)

    def test_stat_aloof(self):
        cp = _sphere(diameter=20.0)
        impact = np.array([[30.0, 0.0]])
        eb = EELSStat(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.phi)
        assert _finite(exc.phip)

    def test_stat_penetrating(self):
        cp = _sphere(diameter=20.0)
        impact = np.array([[0.0, 0.0]])
        eb = EELSStat(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.phi)
        assert _finite(exc.phip)


class TestElectronBeamMultiImpact:
    """Multiple impact parameters at once."""

    def test_three_aloof_impacts(self):
        cp = _sphere(diameter=20.0)
        impact = np.array([
            [15.0, 0.0],
            [20.0, 0.0],
            [25.0, 0.0],
        ])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        # last axis indexes impacts
        assert exc.a1.shape[-1] == 3
        assert _finite(exc.a1)

    def test_mixed_impacts(self):
        cp = _sphere(diameter=20.0)
        impact = np.array([
            [0.0, 0.0],   # penetrating
            [10.001, 0.0],  # grazing
            [30.0, 0.0],  # aloof
        ])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=0.69, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert exc.a1.shape[-1] == 3
        assert _finite(exc.a1)


class TestElectronBeamEnergies:
    """Velocity from kinetic energy at common STEM voltages."""

    @pytest.mark.parametrize("kev", [60.0, 100.0, 200.0, 300.0])
    def test_kev_to_velocity(self, kev):
        v = EELSBase.ene2vel(kev * 1e3)
        # v/c must lie in (0, 1)
        assert 0.0 < v < 1.0

    def test_60kev_excitation(self):
        cp = _sphere(diameter=20.0)
        v = EELSBase.ene2vel(60.0e3)
        impact = np.array([[20.0, 0.0]])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=v, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.a1)

    def test_200kev_excitation(self):
        cp = _sphere(diameter=20.0)
        v = EELSBase.ene2vel(200.0e3)
        impact = np.array([[20.0, 0.0]])
        eb = EELSRet(cp, impact=impact, width=0.5, vel=v, cutoff=20.0)
        exc = eb.potential(cp, WL)
        assert _finite(exc.a1)

    def test_velocity_increases_with_energy(self):
        v60 = EELSBase.ene2vel(60.0e3)
        v200 = EELSBase.ene2vel(200.0e3)
        v300 = EELSBase.ene2vel(300.0e3)
        assert v60 < v200 < v300

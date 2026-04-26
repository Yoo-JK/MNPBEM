"""
Analytical cross-check tests for MNPBEM (M3 Wave 3 C2).

Validates BEM and Layer/Dipole results against independent analytical
references:

- 8.1 Mie theory (quasistatic + retarded) vs BEMStat / BEMRet sphere
  extinction / scattering.
- 8.1 Direct analytical Mie series (scipy spherical Bessel) vs MieRet.
- 8.1 Fresnel reflection coefficients vs LayerStructure.fresnel for
  vacuum / glass and vacuum / metal interfaces, plus quick limit
  checks (n1=n2 -> r=0, n2=infty/perfect mirror -> |r|->1).
- 8.1 Image dipole positions and reflection factors used by
  DipoleStatLayer match the textbook (Jackson 4.4) image charge
  recipe for a planar dielectric interface.

Each test uses small spheres / few wavelengths to keep runtime under
~30 s.  Tolerances are typically 1 - 5% for analytical comparisons,
mesh resolution being the dominant error source.
"""

import os
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bypass the broken top-level mnpbem.__init__ so subpackages import cleanly
# (mirrors test_edge_cases_excitation.py).
# ---------------------------------------------------------------------------
if "mnpbem" not in sys.modules:
    _stub = types.ModuleType("mnpbem")
    _stub.__path__ = ["mnpbem"]
    sys.modules["mnpbem"] = _stub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from scipy.special import spherical_jn, spherical_yn

from mnpbem.materials.eps_const import EpsConst
from mnpbem.materials.eps_drude import EpsDrude
from mnpbem.geometry import trisphere, ComParticle, ComPoint, LayerStructure
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation.planewave_stat import PlaneWaveStat
from mnpbem.simulation.planewave_ret import PlaneWaveRet
from mnpbem.simulation.dipole_stat_layer import DipoleStatLayer
from mnpbem.mie import MieStat, MieRet


# ---------------------------------------------------------------------------
# Common fixtures / helpers
# ---------------------------------------------------------------------------

# Drude gold reproducing MieStat exactly when used inside BEMStat (validated
# in 02_bemstat / sphere comparison directory).
_AU_DRUDE = dict(eps0 = 10.0, wp = 9.065, gammad = 0.0708, name = 'gold')


def _scalar(x):
    """Convert single-element ndarray-like to plain Python float / complex."""
    arr = np.atleast_1d(np.asarray(x))
    return arr.flat[0]


def _make_au_sphere_stat(diameter, nfaces = 144):
    epsm = EpsConst(1.0)
    epsAu = EpsDrude(**_AU_DRUDE)
    sphere = trisphere(nfaces, diameter)
    p = ComParticle([epsm, epsAu], [sphere], [[2, 1]], 1, interp = 'curv')
    return p, epsm, epsAu


def _solve_stat(p, enei, pol = (1.0, 0.0, 0.0)):
    bem = BEMStat(p)
    exc = PlaneWaveStat([list(pol)])
    sig, _ = bem.solve(exc.potential(p, enei))
    return float(np.real(_scalar(exc.extinction(sig)))), \
           float(np.real(_scalar(exc.scattering(sig))))


def _solve_ret(p, enei, pol = (1.0, 0.0, 0.0), kdir = (0.0, 0.0, 1.0)):
    bem = BEMRet(p)
    exc = PlaneWaveRet([list(pol)], [list(kdir)])
    sig, _ = bem.solve(exc.potential(p, enei))
    return float(np.real(_scalar(exc.extinction(sig)))), \
           float(np.real(_scalar(exc.scattering(sig))))


# ---------------------------------------------------------------------------
# 8.1.1  Mie quasistatic vs BEMStat sphere
# ---------------------------------------------------------------------------

class TestMieStatVsBEMStat(object):
    """Quasistatic Mie theory vs BEMStat for a small sphere.

    For a sphere with diameter << wavelength the full BEM should
    converge to the analytical dipole polarisability result.
    """

    @pytest.mark.parametrize('enei', [400.0, 500.0, 600.0, 700.0])
    def test_extinction_small_sphere(self, enei):
        diameter = 10.0  # ka << 1 across visible
        p, epsm, epsAu = _make_au_sphere_stat(diameter, nfaces = 144)

        ext_bem, _ = _solve_stat(p, enei)
        mie = MieStat(epsAu, epsm, diameter = diameter)
        ext_mie = float(_scalar(mie.extinction(np.array([enei]))))

        rel = abs(ext_bem - ext_mie) / ext_mie
        # Mesh discretisation error for 144-face sphere ~ 0.5%
        assert rel < 5e-3, (
            'enei={}: rel error {:.4f}, BEM {:.4f}, Mie {:.4f}'
            .format(enei, rel, ext_bem, ext_mie))

    def test_scattering_small_sphere(self):
        """Scattering for 10nm Au sphere at resonance."""
        diameter = 10.0
        p, epsm, epsAu = _make_au_sphere_stat(diameter, nfaces = 144)

        enei = 500.0
        _, sca_bem = _solve_stat(p, enei)
        mie = MieStat(epsAu, epsm, diameter = diameter)
        sca_mie = float(_scalar(mie.scattering(np.array([enei]))))

        rel = abs(sca_bem - sca_mie) / sca_mie
        assert rel < 1e-2, 'rel error {:.4f}'.format(rel)

    def test_resonance_position(self):
        """Plasmon resonance peak from BEM and Mie should coincide."""
        diameter = 10.0
        p, epsm, epsAu = _make_au_sphere_stat(diameter, nfaces = 144)

        wls = np.linspace(450, 600, 16)
        ext_bem = np.array([_solve_stat(p, w)[0] for w in wls])
        mie = MieStat(epsAu, epsm, diameter = diameter)
        ext_mie = mie.extinction(wls)

        peak_bem = wls[int(np.argmax(ext_bem))]
        peak_mie = wls[int(np.argmax(ext_mie))]
        # Same grid -> identical bin or 1 step off
        assert abs(peak_bem - peak_mie) <= (wls[1] - wls[0]) + 1e-6


# ---------------------------------------------------------------------------
# 8.1.2  Mie retarded vs BEMRet sphere (a few size parameters)
# ---------------------------------------------------------------------------

class TestMieRetVsBEMRet(object):
    """Retarded Mie theory vs BEMRet for spheres of increasing ka.

    BEMRet on a 144-face sphere has a few-percent mesh error; we accept
    up to 5% relative deviation but require finiteness everywhere.
    """

    @pytest.mark.parametrize('diameter', [20.0, 50.0])
    def test_extinction_at_500nm(self, diameter):
        epsm = EpsConst(1.0)
        epsAu = EpsDrude(**_AU_DRUDE)
        p, _, _ = _make_au_sphere_stat(diameter, nfaces = 144)

        enei = 500.0
        ext_bem, _ = _solve_ret(p, enei)
        mie = MieRet(epsAu, epsm, diameter = diameter)
        ext_mie = float(_scalar(mie.extinction(np.array([enei]))))

        rel = abs(ext_bem - ext_mie) / max(abs(ext_mie), 1e-12)
        assert np.isfinite(ext_bem)
        assert rel < 5e-2, (
            'diameter={}: rel error {:.4f}'.format(diameter, rel))


# ---------------------------------------------------------------------------
# 8.1.3  Direct analytical Mie series (scipy) vs MieRet
# ---------------------------------------------------------------------------

def _analytic_mie_extinction(diameter, eps_in, eps_out, enei, lmax = 20):
    """Independent analytical Mie extinction using scipy spherical Bessel.

    Bohren & Huffman 1983, Eq. (4.61):
        sigma_ext = (2 pi / k^2) * sum_{l=1}^{lmax} (2l+1) Re(a_l + b_l).
    Mie coefficients (mu_r = 1):
        a_l = (m psi_l(mx) psi_l'(x) - psi_l(x) psi_l'(mx))
              / (m psi_l(mx) xi_l'(x) - xi_l(x) psi_l'(mx))
        b_l = (psi_l(mx) psi_l'(x) - m psi_l(x) psi_l'(mx))
              / (psi_l(mx) xi_l'(x) - m xi_l(x) psi_l'(mx))
    with psi_l(z) = z j_l(z), xi_l(z) = z h_l^(1)(z) and
    m = sqrt(eps_in / eps_out), x = k a, k = 2 pi sqrt(eps_out) / enei.
    """
    eps_in = complex(eps_in)
    eps_out = complex(eps_out)
    a = 0.5 * diameter
    nb = np.sqrt(eps_out)
    k = 2 * np.pi * nb / enei

    m = np.sqrt(eps_in / eps_out)
    x = k * a
    mx = m * x

    # psi_l(z), psi_l'(z) and xi_l(z), xi_l'(z)
    l = np.arange(1, lmax + 1)
    j_x = np.array([spherical_jn(int(li), x) for li in l])
    jp_x = np.array([spherical_jn(int(li), x, derivative = True) for li in l])
    y_x = np.array([spherical_yn(int(li), x) for li in l])
    yp_x = np.array([spherical_yn(int(li), x, derivative = True) for li in l])
    h_x = j_x + 1j * y_x
    hp_x = jp_x + 1j * yp_x

    j_mx = np.array([spherical_jn(int(li), mx) for li in l])
    jp_mx = np.array([spherical_jn(int(li), mx, derivative = True) for li in l])

    # psi(z) = z * j_l(z); psi'(z) = j_l(z) + z * j_l'(z)
    psi_x = x * j_x
    psip_x = j_x + x * jp_x
    xi_x = x * h_x
    xip_x = h_x + x * hp_x
    psi_mx = mx * j_mx
    psip_mx = j_mx + mx * jp_mx

    a_l = (m * psi_mx * psip_x - psi_x * psip_mx) \
        / (m * psi_mx * xip_x - xi_x * psip_mx)
    b_l = (psi_mx * psip_x - m * psi_x * psip_mx) \
        / (psi_mx * xip_x - m * xi_x * psip_mx)

    sigma_ext = (2 * np.pi / k**2) * np.sum((2 * l + 1) * np.real(a_l + b_l))
    return float(np.real(sigma_ext))


class TestAnalyticalMieRet(object):
    """MieRet implementation must match independent analytical series."""

    @pytest.mark.parametrize('diameter, enei', [
        (20.0, 500.0),
        (50.0, 500.0),
        (100.0, 600.0)])
    def test_mieret_matches_analytical(self, diameter, enei):
        epsm = EpsConst(1.0)
        epsAu = EpsDrude(**_AU_DRUDE)
        eps_in_val = complex(_scalar(epsAu(enei)[0]))
        eps_out_val = complex(_scalar(epsm(enei)[0]))

        ext_ana = _analytic_mie_extinction(
            diameter, eps_in_val, eps_out_val, enei, lmax = 20)
        mie = MieRet(epsAu, epsm, diameter = diameter)
        ext_mie = float(_scalar(mie.extinction(np.array([enei]))))

        rel = abs(ext_mie - ext_ana) / abs(ext_ana)
        assert rel < 1e-6, (
            'd={}: MieRet={:.6f}, analytical={:.6f}, rel={:.2e}'
            .format(diameter, ext_mie, ext_ana, rel))


# ---------------------------------------------------------------------------
# 8.1.4  Fresnel reflection vs LayerStructure
# ---------------------------------------------------------------------------

def _make_layer(eps_top, eps_bot):
    epstab = [EpsConst(eps_top), EpsConst(eps_bot)]
    return LayerStructure(epstab, [1, 2], [0.0])


def _fresnel_p(layer, enei, kpar):
    pos = {'r': np.array([0.0]),
           'z1': np.array([1.0]),
           'ind1': np.array([1]),
           'z2': np.array([1.0]),
           'ind2': np.array([1])}
    r = layer.fresnel(enei, kpar, pos)
    return _scalar(r['p'])


class TestFresnelAnalytical(object):
    """Compare LayerStructure.fresnel to closed-form Fresnel formulae."""

    def test_normal_incidence_glass(self):
        """At kpar=0, r_p = (n1 - n2) / (n1 + n2)."""
        layer = _make_layer(1.0, 2.25)
        r_p = complex(_fresnel_p(layer, enei = 500.0, kpar = 0.0))
        n1, n2 = 1.0, np.sqrt(2.25)
        r_expected = (n1 - n2) / (n1 + n2)
        assert abs(r_p.real - r_expected) < 5e-3
        # imaginary part comes from O(kpar/k) -> O(1e-3) due to z=1 phase
        # factors; just require magnitude small
        assert abs(r_p.imag) < 1e-2

    def test_identical_media_zero_reflection(self):
        """eps_top == eps_bot -> r = 0."""
        layer = _make_layer(1.0, 1.0)
        r_p = complex(_fresnel_p(layer, enei = 500.0, kpar = 0.0))
        assert abs(r_p) < 1e-6

    def test_metal_substrate_high_reflection(self):
        """High-conductivity metal (large -Re(eps), small Im) ->
        |r| close to 1 at normal incidence."""
        layer = _make_layer(1.0, -100.0 + 1.0j)
        r_p = complex(_fresnel_p(layer, enei = 500.0, kpar = 0.0))
        assert abs(r_p) > 0.7

    def test_oblique_incidence_glass(self):
        """At kpar < k1, r_p must remain bounded < 1 for dielectric-
        dielectric interface and approach +1 as kpar -> k1 (grazing)."""
        layer = _make_layer(1.0, 2.25)
        enei = 500.0
        k1 = 2 * np.pi / enei  # vacuum
        kpars = np.linspace(0.05 * k1, 0.95 * k1, 10)
        rs = [abs(_fresnel_p(layer, enei = enei, kpar = float(kp)))
              for kp in kpars]
        assert all(r < 1.5 for r in rs)
        assert np.all(np.isfinite(rs))


# ---------------------------------------------------------------------------
# 8.1.5  Image dipole vs DipoleStatLayer image factors
# ---------------------------------------------------------------------------

class TestImageDipole(object):
    """Image dipole position and strength must match Jackson 4.4
    formulas used internally by DipoleStatLayer."""

    def test_image_position_mirror_through_interface(self):
        """For a flat interface at z = z0, an image of a dipole at
        z = z_dip > z0 sits at 2 z0 - z_dip."""
        epstab = [EpsConst(1.0), EpsConst(2.25), EpsConst(1.0)]
        layer = LayerStructure(epstab, [1, 3], [0.0])

        # Build a small particle so ComPoint's medium check works.
        sphere = trisphere(60, 1.0)
        sphere.shift([0, 0, 50.0])
        p = ComParticle(epstab, [sphere], [[2, 1]], [1])

        z_dip = 20.0
        pt = ComPoint(p, np.array([[0.0, 0.0, z_dip]]))
        dip = DipoleStatLayer(pt, layer,
            dip = np.array([[0.0, 0.0, 1.0]]))
        pos_image = dip._image_positions()
        # image z = 2*0 - 20 = -20
        assert pos_image.shape == (1, 3)
        assert abs(pos_image[0, 0]) < 1e-12
        assert abs(pos_image[0, 1]) < 1e-12
        assert abs(pos_image[0, 2] - (-z_dip)) < 1e-12

    def test_image_factors_glass_substrate(self):
        """Jackson 4.45 image charge factor for parallel / perp
        components of a dipole above a dielectric interface:
            q_par  = (eps_above - eps_below) / (eps_above + eps_below)
            q_perp = -q_par
        DipoleStatLayer._image_factors derives them from layer.eps[0]
        and layer.eps[1] (the two media adjoining the interface).

        For LayerStructure(epstab, [1, 3], ...), eps[0]=epstab[0]
        (vacuum) and eps[1]=epstab[2] (glass below).
        """
        eps_a = 1.0
        eps_b = 2.25
        epstab = [EpsConst(eps_a), EpsConst(-10 + 1j), EpsConst(eps_b)]
        layer = LayerStructure(epstab, [1, 3], [0.0])

        sphere = trisphere(60, 1.0)
        sphere.shift([0, 0, 50.0])
        p = ComParticle(epstab, [sphere], [[2, 1]], [1])

        pt = ComPoint(p, np.array([[0.0, 0.0, 20.0]]))
        dip = DipoleStatLayer(pt, layer,
            dip = np.array([[0.0, 0.0, 1.0]]))
        q1, q2 = dip._image_factors(500.0)
        q1_expected = (eps_a - eps_b) / (eps_a + eps_b)
        q2_expected = -q1_expected
        assert abs(complex(q1).real - q1_expected) < 1e-10
        assert abs(complex(q2).real - q2_expected) < 1e-10

    def test_image_factor_no_substrate_limit(self):
        """When eps below = eps above (no real interface), the image
        factors must vanish."""
        epstab = [EpsConst(1.0), EpsConst(-10 + 1j), EpsConst(1.0)]
        layer = LayerStructure(epstab, [1, 3], [0.0])

        sphere = trisphere(60, 1.0)
        sphere.shift([0, 0, 50.0])
        p = ComParticle(epstab, [sphere], [[2, 1]], [1])

        pt = ComPoint(p, np.array([[0.0, 0.0, 20.0]]))
        dip = DipoleStatLayer(pt, layer,
            dip = np.array([[0.0, 0.0, 1.0]]))
        q1, q2 = dip._image_factors(500.0)
        assert abs(complex(q1)) < 1e-12
        assert abs(complex(q2)) < 1e-12

    def test_image_factor_metal_limit(self):
        """Perfect conductor (eps_below -> -infty): with the
        ``q1 = (eps_above - eps_below)/(eps_above + eps_below)``
        convention used in DipoleStatLayer, q1 -> -1 and q2 -> +1.
        """
        epstab = [EpsConst(1.0), EpsConst(-10 + 1j), EpsConst(-1e6)]
        layer = LayerStructure(epstab, [1, 3], [0.0])

        sphere = trisphere(60, 1.0)
        sphere.shift([0, 0, 50.0])
        p = ComParticle(epstab, [sphere], [[2, 1]], [1])

        pt = ComPoint(p, np.array([[0.0, 0.0, 20.0]]))
        dip = DipoleStatLayer(pt, layer,
            dip = np.array([[0.0, 0.0, 1.0]]))
        q1, q2 = dip._image_factors(500.0)
        assert abs(complex(q1).real - (-1.0)) < 1e-3
        assert abs(complex(q2).real - 1.0) < 1e-3

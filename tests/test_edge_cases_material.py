"""Edge case tests for material dielectric functions and layer structures.

M3 Wave 1 A2: covers EpsConst (real/complex/zero/negative), EpsTable (gold/
silver/custom user .dat), EpsFun (lambdas, lorentz), highly absorbing media,
Brewster-like pairs, and LayerStructure with various ztab/epstab combos
(single interface, 3-layer air/dielectric/metal, 5+ layer thin-film stack,
imaginary-k absorbing layer, negative epsilon layer, very thin <1nm layers,
layer between metals).

BEM solve is intentionally skipped — these tests focus on object construction,
eps evaluation, wavenumber computation, sanity (finite, complex consistency)
and layer indexing. Bugs encountered during writing are reported in
/tmp/m3_a2_bugs.md.
"""
import os
import tempfile

from typing import Tuple

import numpy as np
import pytest

from mnpbem.materials import EpsConst, EpsTable, EpsDrude, EpsFun, epsfun
from mnpbem.geometry.layer_structure import LayerStructure


TOL = 1e-12
WAVELENGTHS = np.array([400.0, 500.0, 600.0, 700.0, 800.0])


# ============================================================
# 2.1 Dielectric Functions — EpsConst
# ============================================================
def test_eps_const_real_positive():
    eps_obj = EpsConst(2.25)
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(eps_val).all()
    assert np.isfinite(k_val).all()
    assert np.isclose(np.real(eps_val), 2.25, atol = TOL)
    assert np.isclose(np.imag(eps_val), 0.0, atol = TOL)


def test_eps_const_real_one_vacuum():
    eps_obj = EpsConst(1.0)
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert np.allclose(np.real(eps_val), 1.0, atol = TOL)
    expected_k = 2.0 * np.pi / WAVELENGTHS
    assert np.allclose(np.real(k_val), expected_k, atol = TOL)


def test_eps_const_complex():
    eps_in = -10.0 + 1.5j
    eps_obj = EpsConst(eps_in)
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(np.real(eps_val))
    assert np.isfinite(np.imag(eps_val))
    assert np.isclose(eps_val, eps_in, atol = TOL)
    # k = 2pi / lambda * sqrt(eps); imag part should be > 0 (decaying wave)
    assert np.isfinite(k_val)


def test_eps_const_zero():
    eps_obj = EpsConst(0.0)
    eps_val, k_val = eps_obj(500.0)
    assert np.isclose(np.real(eps_val), 0.0, atol = TOL)
    assert np.isclose(np.real(k_val), 0.0, atol = TOL)
    assert np.isfinite(k_val)


@pytest.mark.xfail(
    reason = 'BUG: EpsConst(real<0) returns k=nan because np.sqrt is called '
            'on the raw real scalar instead of a complex value. '
            'Workaround: pass complex eps (e.g. EpsConst(-9.0+0j)).',
    strict = True)
def test_eps_const_negative_real():
    # Negative epsilon — typical for metals at certain wavelengths
    eps_obj = EpsConst(-9.0)
    eps_val, k_val = eps_obj(500.0)
    assert np.isclose(np.real(eps_val), -9.0, atol = TOL)
    # sqrt(-9) = 3j, so k must be purely imaginary (evanescent)
    k_arr = np.atleast_1d(k_val)
    assert np.all(np.isfinite(k_arr))
    assert np.isclose(np.real(k_arr[0]), 0.0, atol = 1e-10)
    assert np.imag(k_arr[0]) > 0  # decaying


def test_eps_const_negative_real_complex_workaround():
    # With explicit complex input, sqrt becomes complex and k is finite
    eps_obj = EpsConst(-9.0 + 0.0j)
    eps_val, k_val = eps_obj(500.0)
    assert np.isclose(np.real(eps_val), -9.0, atol = TOL)
    assert np.isfinite(k_val)
    assert np.isclose(np.real(k_val), 0.0, atol = 1e-10)
    assert np.imag(k_val) > 0


def test_eps_const_highly_absorbing():
    eps_in = -10.0 + 50.0j
    eps_obj = EpsConst(eps_in)
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert np.all(np.isfinite(eps_val))
    assert np.all(np.isfinite(k_val))
    assert np.allclose(eps_val, eps_in, atol = TOL)
    # Im(k) > 0 indicates absorption (decaying wave in propagation direction)
    assert np.all(np.imag(k_val) > 0.0)


def test_eps_const_array_input():
    eps_obj = EpsConst(2.25)
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert eps_val.shape == WAVELENGTHS.shape
    assert k_val.shape == WAVELENGTHS.shape
    # Wavenumber should scale as 1/lambda
    ratio = k_val[0] / k_val[-1]
    expected = WAVELENGTHS[-1] / WAVELENGTHS[0]
    assert np.isclose(np.real(ratio), expected, atol = 1e-10)


def test_eps_const_wavenumber_method():
    eps_obj = EpsConst(4.0)
    k1 = eps_obj.wavenumber(500.0)
    _, k2 = eps_obj(500.0)
    assert np.isclose(k1, k2, atol = TOL)


# ============================================================
# 2.1 Dielectric Functions — EpsTable
# ============================================================
def test_eps_table_gold():
    eps_obj = EpsTable('gold.dat')
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(eps_val)
    assert np.isfinite(k_val)
    # Gold at 500 nm: eps ~ -3 to -5 + small Im (Johnson & Christy)
    assert np.real(eps_val) < 0
    assert np.imag(eps_val) > 0


def test_eps_table_silver():
    eps_obj = EpsTable('silver.dat')
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(eps_val)
    assert np.isfinite(k_val)
    # Silver at 500 nm: highly negative real part
    assert np.real(eps_val) < 0


def test_eps_table_palik_variants():
    for fname in ['goldpalik.dat', 'silverpalik.dat', 'copperpalik.dat']:
        eps_obj = EpsTable(fname)
        eps_val, k_val = eps_obj(WAVELENGTHS)
        assert np.all(np.isfinite(eps_val)), fname
        assert np.all(np.isfinite(k_val)), fname


def test_eps_table_user_dat_file(tmp_path):
    # Write a small user .dat in the documented "energy(eV) n k" format
    custom = tmp_path / 'custom.dat'
    lines = [
        '% custom user material',
        '1.0  1.5  0.05',
        '1.5  1.6  0.07',
        '2.0  1.7  0.10',
        '2.5  1.8  0.15',
        '3.0  1.9  0.20',
        '3.5  2.0  0.25',
        '4.0  2.1  0.30',
    ]
    custom.write_text('\n'.join(lines))

    eps_obj = EpsTable(str(custom))
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(eps_val)
    assert np.isfinite(k_val)
    # n + ik = ~1.7 + 0.1j at 2.0 eV (~620 nm). At 500 nm (2.48 eV):
    # interpolated n approx 1.78, k approx 0.14 → eps approx 3.15 + 0.5j
    assert np.real(eps_val) > 0
    assert np.imag(eps_val) > 0


def test_eps_table_out_of_range_raises():
    eps_obj = EpsTable('gold.dat')
    # Way out of the J&C range (J&C covers ~190-2000 nm)
    with pytest.raises(ValueError):
        eps_obj(50.0)


def test_eps_table_array_input():
    eps_obj = EpsTable('gold.dat')
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert eps_val.shape == WAVELENGTHS.shape
    assert k_val.shape == WAVELENGTHS.shape
    assert np.all(np.isfinite(eps_val))


def test_eps_table_refractive_index_consistency():
    eps_obj = EpsTable('gold.dat')
    n_complex = eps_obj.refractive_index(500.0)
    eps_val, _ = eps_obj(500.0)
    # eps must equal n^2
    assert np.isclose(eps_val, n_complex ** 2, atol = 1e-10)


# ============================================================
# 2.1 Dielectric Functions — EpsFun (user-defined)
# ============================================================
def test_eps_fun_simple_lambda():
    eps_obj = EpsFun(lambda enei: 1.0 + 0.5j * enei / 500.0)
    eps_val, k_val = eps_obj(500.0)
    assert np.isfinite(eps_val)
    assert np.isfinite(k_val)
    assert np.isclose(np.real(eps_val), 1.0, atol = TOL)
    assert np.isclose(np.imag(eps_val), 0.5, atol = TOL)


def test_eps_fun_lorentz():
    def lorentz(enei):
        w = 1240.0 / enei
        return 1.0 + 1.0 / (3.0 - w**2 - 0.1j * w)
    eps_obj = EpsFun(lorentz)
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert np.all(np.isfinite(eps_val))
    assert np.all(np.isfinite(k_val))


def test_eps_fun_ev_input_key():
    # When key='eV', input "enei" (in nm) is converted to eV before passing to fun
    eps_obj = EpsFun(lambda ev: 2.0 + 0.0j * ev, key = 'eV')
    eps_val, k_val = eps_obj(500.0)
    assert np.isclose(np.real(eps_val), 2.0, atol = TOL)


def test_eps_fun_invalid_callable_raises():
    with pytest.raises(TypeError):
        EpsFun(42)  # not callable


def test_eps_fun_invalid_key_raises():
    with pytest.raises(ValueError):
        EpsFun(lambda x: 1.0, key = 'angstrom')


# ============================================================
# 2.1 — epsfun factory
# ============================================================
def test_epsfun_factory_dispatch():
    assert isinstance(epsfun(1.0), EpsConst)
    assert isinstance(epsfun(2.25 + 0.1j), EpsConst)
    assert isinstance(epsfun('gold.dat'), EpsTable)
    assert isinstance(epsfun('drude:gold'), EpsDrude)
    assert isinstance(epsfun('drude:silver'), EpsDrude)
    assert isinstance(epsfun('drude:aluminum'), EpsDrude)
    assert isinstance(epsfun(lambda x: 1.0), EpsFun)


def test_epsfun_factory_invalid_drude():
    with pytest.raises(ValueError):
        epsfun('drude:platinum')


def test_epsfun_factory_invalid_type():
    with pytest.raises(TypeError):
        epsfun([1, 2, 3])


# ============================================================
# 2.1 — Highly absorbing & Brewster-like
# ============================================================
def test_eps_const_brewster_pair():
    # eps1 * eps2 ~ -1 — surface plasmon condition
    eps1_obj = EpsConst(1.0)
    eps2_obj = EpsConst(-1.0 + 0.01j)
    eps1_val, _ = eps1_obj(500.0)
    eps2_val, _ = eps2_obj(500.0)
    product = eps1_val * eps2_val
    assert np.isclose(np.real(product), -1.0, atol = 1e-2)


def test_eps_const_high_imaginary_dominant():
    eps_in = -1.0 + 100.0j
    eps_obj = EpsConst(eps_in)
    eps_val, k_val = eps_obj(WAVELENGTHS)
    assert np.all(np.isfinite(eps_val))
    assert np.all(np.isfinite(k_val))
    assert np.all(np.imag(k_val) > 0)


# ============================================================
# 2.2 Layer Structures — Single interface
# ============================================================
def test_layer_single_interface():
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    assert layer.n == 1
    assert len(layer.eps) == 2
    assert layer.z[0] == 0.0


def test_layer_indlayer_single_interface():
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    # z>0 -> upper half; z<0 -> lower half
    z_test = np.array([10.0, -10.0, 5.0, -5.0])
    ind, in_layer = layer.indlayer(z_test)
    assert ind.shape == z_test.shape
    assert in_layer.shape == z_test.shape
    # Points well away from z=0 must not be tagged "in layer"
    assert not np.any(in_layer)


def test_layer_mindist_single_interface():
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    z_test = np.array([0.0, 1.0, -2.0, 5.0])
    zmin, ind = layer.mindist(z_test)
    assert np.allclose(zmin, np.array([0.0, 1.0, 2.0, 5.0]), atol = TOL)
    assert np.all(ind == 1)


# ============================================================
# 2.2 — 3-layer: air / dielectric / metal
# ============================================================
def test_layer_3layer_air_diel_metal():
    epstab = [EpsConst(1.0), EpsConst(2.25), EpsConst(-9.0 + 0.5j)]
    # 3 indices, 2 interfaces (z=0, z=-50)
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -50.0])
    assert layer.n == 2
    assert len(layer.eps) == 3


def test_layer_3layer_indlayer():
    epstab = [EpsConst(1.0), EpsConst(2.25), EpsConst(-9.0 + 0.5j)]
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -50.0])
    z_test = np.array([10.0, -25.0, -100.0])
    ind, _ = layer.indlayer(z_test)
    # MATLAB convention: layers ordered by decreasing z. Three regions exist.
    unique_inds = set(int(i) for i in ind)
    assert len(unique_inds) == 3


# ============================================================
# 2.2 — 5+ layer thin-film stack
# ============================================================
def test_layer_5layer_thinfilm():
    # air | SiO2 | TiO2 | SiO2 | metal
    epstab = [
        EpsConst(1.0),
        EpsConst(2.13),       # SiO2
        EpsConst(6.25),       # TiO2
        EpsConst(-9.0 + 0.5j), # metal
    ]
    # 5 layers, 4 interfaces — repeat SiO2 (idx=2) for layer 4
    z_interfaces = [0.0, -10.0, -30.0, -40.0]
    layer = LayerStructure(epstab, [1, 2, 3, 2, 4], z_interfaces)
    assert layer.n == 4
    assert len(layer.eps) == 5


def test_layer_6layer_stack():
    epstab = [EpsConst(1.0 + 0.0j),
              EpsConst(2.0),
              EpsConst(3.0),
              EpsConst(4.0),
              EpsConst(5.0),
              EpsConst(-2.0 + 0.1j)]
    z_interfaces = [10.0, 0.0, -5.0, -15.0, -30.0]
    layer = LayerStructure(epstab, list(range(1, 7)), z_interfaces)
    assert layer.n == 5
    z_test = np.array([20.0, 5.0, -2.0, -10.0, -20.0, -50.0])
    ind, _ = layer.indlayer(z_test)
    assert ind.shape == z_test.shape


# ============================================================
# 2.2 — Imaginary k layer (absorbing)
# ============================================================
def test_layer_imaginary_absorbing():
    epstab = [EpsConst(1.0), EpsConst(-2.0 + 30.0j)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    eps_val, k_val = layer.eps[1](500.0)
    assert np.isfinite(eps_val)
    assert np.isfinite(k_val)
    assert np.imag(k_val) > 0  # decaying


# ============================================================
# 2.2 — Negative epsilon layer
# ============================================================
def test_layer_negative_epsilon():
    # Use complex eps to avoid the EpsConst(real<0) NaN bug (see xfail above).
    epstab = [EpsConst(1.0), EpsConst(-9.0 + 0.0j)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    eps_val, k_val = layer.eps[1](500.0)
    assert np.isclose(np.real(eps_val), -9.0, atol = TOL)
    # purely imaginary k (evanescent)
    assert np.isclose(np.real(k_val), 0.0, atol = 1e-10)
    assert np.imag(k_val) > 0


# ============================================================
# 2.2 — Very thin layer (<1 nm)
# ============================================================
def test_layer_very_thin_sub_nm():
    epstab = [EpsConst(1.0), EpsConst(2.25), EpsConst(-9.0)]
    # Thin film of thickness 0.5 nm
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -0.5])
    assert layer.n == 2
    thickness = layer.z[0] - layer.z[1]
    assert np.isclose(thickness, 0.5, atol = TOL)


def test_layer_very_thin_picometer():
    epstab = [EpsConst(1.0), EpsConst(2.25), EpsConst(-9.0)]
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -1e-3])
    assert layer.n == 2
    z_test = np.array([10.0, -5e-4, -10.0])
    ind, _ = layer.indlayer(z_test)
    assert ind.shape == z_test.shape


# ============================================================
# 2.2 — Layer between metals
# ============================================================
def test_layer_metal_dielectric_metal():
    epstab = [EpsConst(-9.0 + 0.5j), EpsConst(2.25), EpsConst(-12.0 + 1.0j)]
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -20.0])
    assert layer.n == 2
    eps0, _ = layer.eps[0](500.0)
    eps2, _ = layer.eps[2](500.0)
    assert np.real(eps0) < 0
    assert np.real(eps2) < 0


# ============================================================
# 2.2 — Round_z and tolerance handling
# ============================================================
def test_layer_round_z_pushes_outside_zmin():
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0], zmin = 1.0)
    z_in = np.array([0.5, -0.5, 5.0, -10.0])
    (z_out,) = layer.round_z(z_in)
    # Points within zmin should be pushed to ±zmin
    assert abs(z_out[0]) >= 1.0 - 1e-12
    assert abs(z_out[1]) >= 1.0 - 1e-12
    # Far points unchanged
    assert np.isclose(z_out[2], 5.0, atol = TOL)
    assert np.isclose(z_out[3], -10.0, atol = TOL)


def test_layer_indlayer_in_layer_flag():
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0], ztol = 0.1)
    z_test = np.array([0.0, 0.05, -0.05, 1.0])
    _, in_layer = layer.indlayer(z_test)
    # Within ztol=0.1
    assert in_layer[0]
    assert in_layer[1]
    assert in_layer[2]
    assert not in_layer[3]


# ============================================================
# 2.2 — Mixed eps types in epstab
# ============================================================
def test_layer_mixed_eps_types():
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsDrude.silver()]
    layer = LayerStructure(epstab, [1, 2, 3], [0.0, -10.0])
    assert len(layer.eps) == 3
    # Each should evaluate cleanly at 500 nm
    for eps in layer.eps:
        eps_val, k_val = eps(500.0)
        assert np.isfinite(eps_val)
        assert np.isfinite(k_val)

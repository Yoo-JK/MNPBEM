import os
import sys
from typing import Any, Tuple

import numpy as np
import pytest

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem.geometry import (
        Polygon,
        Polygon3,
        EdgeProfile,
        Particle,
        trisphere,
        trirod,
        tricube,
        tritorus,
        tripolygon,
        fvgrid)


# ===========================================================================
# Helpers
# ===========================================================================

def _check_particle_sane(p: Any, label: str = '') -> None:
    assert p is not None, '[error] particle is None ({})'.format(label)
    assert p.nverts > 0, '[error] no vertices ({})'.format(label)
    assert p.nfaces > 0, '[error] no faces ({})'.format(label)
    assert p.verts.shape[1] == 3, '[error] verts not 3D ({})'.format(label)
    assert np.all(np.isfinite(p.verts)), '[error] non-finite verts ({})'.format(label)
    assert hasattr(p, 'area'), '[error] missing area ({})'.format(label)
    assert p.area.shape == (p.nfaces,), '[error] bad area shape ({})'.format(label)
    assert np.all(np.isfinite(p.area)), '[error] non-finite area ({})'.format(label)
    assert np.all(p.area > 0), '[error] non-positive area ({})'.format(label)


def _check_polygon_sane(poly: Any, label: str = '') -> None:
    assert poly is not None, '[error] polygon is None ({})'.format(label)
    assert poly.pos.ndim == 2 and poly.pos.shape[1] == 2, \
            '[error] polygon pos not 2D ({})'.format(label)
    assert poly.n_verts >= 3, '[error] polygon too few verts ({})'.format(label)
    assert np.all(np.isfinite(poly.pos)), '[error] polygon pos non-finite ({})'.format(label)


# ===========================================================================
# 1.1 Polygon 자유 정의
# ===========================================================================

@pytest.mark.parametrize('n', [3, 4, 5, 7, 12, 50, 100, 1000])
def test_ngon_stress(n: int) -> None:
    poly = Polygon(n, size=[10.0, 10.0])
    _check_polygon_sane(poly, label='n={}'.format(n))
    assert poly.n_verts == n


def test_polygon_irregular_random() -> None:
    rng = np.random.default_rng(seed=42)
    n = 12
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 1.5, n) * 5.0
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    poly = Polygon(pos)
    _check_polygon_sane(poly, label='irregular_random')
    assert poly.n_verts == n


def test_polygon_star_5point() -> None:
    n_pts = 5
    angles = np.linspace(0, 2 * np.pi, 2 * n_pts, endpoint=False) - np.pi / 2
    radii = np.tile([10.0, 4.0], n_pts)
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    poly = Polygon(pos)
    _check_polygon_sane(poly, label='star5')
    assert poly.n_verts == 10


def test_polygon_C_shape() -> None:
    pos = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 3.0],
        [3.0, 3.0],
        [3.0, 7.0],
        [10.0, 7.0],
        [10.0, 10.0],
        [0.0, 10.0]])
    poly = Polygon(pos)
    _check_polygon_sane(poly, label='C_shape')
    assert poly.n_verts == 8


@pytest.mark.parametrize('size', [0.1, 1.0, 100.0, 10000.0, 100000.0])
def test_polygon_extreme_size(size: float) -> None:
    poly = Polygon(8, size=[size, size])
    _check_polygon_sane(poly, label='size={}'.format(size))
    s = poly.size_
    assert np.isclose(s[0], size, rtol=1e-9)
    assert np.isclose(s[1], size, rtol=1e-9)


def test_polygon_donut_via_polymesh2d() -> None:
    outer = Polygon(20, size=[20.0, 20.0])
    inner = Polygon(12, size=[6.0, 6.0])
    inner.dir = -1
    verts, faces = outer.polymesh2d(inner)
    assert verts.shape[1] == 2
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert np.all(np.isfinite(verts))


def test_polygon_rect_with_size_tuple() -> None:
    poly = Polygon(4, size=[20.0, 5.0])
    s = poly.size_
    assert np.isclose(s[0], 20.0)
    assert np.isclose(s[1], 5.0)


def test_polygon_normals_outward() -> None:
    poly = Polygon(8, size=[10.0, 10.0])
    nvec = poly.compute_normals()
    assert nvec.shape == (8, 2)
    assert np.all(np.isfinite(nvec))
    norms = np.linalg.norm(nvec, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-6)


def test_polygon_round_corners() -> None:
    poly = Polygon(4, size=[10.0, 10.0])
    n_before = poly.n_verts
    poly.round_(rad=1.0, nrad=5)
    assert poly.n_verts > n_before
    _check_polygon_sane(poly, label='rounded')


# ===========================================================================
# 1.2 3D Particle 생성
# ===========================================================================

@pytest.mark.parametrize('nsav', [32, 60, 144, 256, 1444])
def test_trisphere_nsav_scan(nsav: int) -> None:
    p = trisphere(nsav, 10.0)
    _check_particle_sane(p, label='nsav={}'.format(nsav))
    assert p.nverts == nsav
    radii = np.linalg.norm(p.verts, axis=1)
    assert np.all(np.abs(radii - 5.0) < 1e-6)


def test_trisphere_169_alternate() -> None:
    p = trisphere(169, 1.0)
    _check_particle_sane(p, label='trisphere169')
    assert p.nverts == 169


@pytest.mark.parametrize('aspect', [1.0, 2.0, 5.0, 10.0])
def test_trirod_aspect_ratio(aspect: float) -> None:
    diameter = 10.0
    height = diameter * aspect
    if aspect <= 1.0:
        height = diameter * 1.5
    p = trirod(diameter, height)
    _check_particle_sane(p, label='aspect={}'.format(aspect))
    z = p.verts[:, 2]
    assert np.isclose(z.max(), height / 2.0, atol=1e-6)
    assert np.isclose(z.min(), -height / 2.0, atol=1e-6)


def test_trirod_extreme_aspect_100() -> None:
    p = trirod(1.0, 100.0)
    _check_particle_sane(p, label='aspect=100')


def test_tricube_default() -> None:
    p = tricube(8, length=10.0, e=0.25)
    _check_particle_sane(p, label='tricube_default')
    bbox_max = p.verts.max(axis=0)
    bbox_min = p.verts.min(axis=0)
    extent = bbox_max - bbox_min
    assert np.allclose(extent, [10.0, 10.0, 10.0], atol=0.5)


@pytest.mark.parametrize('e', [0.1, 0.25, 0.5])
def test_tricube_rounding_param(e: float) -> None:
    p = tricube(8, length=10.0, e=e)
    _check_particle_sane(p, label='tricube_e={}'.format(e))


def test_tritorus_default() -> None:
    # major radius = diameter/2 = 10, minor radius rad = 5
    # → torus annulus: 5 <= sqrt(x^2+y^2) <= 15, |z| <= 5
    p = tritorus(20.0, 5.0)
    _check_particle_sane(p, label='tritorus')
    radii_xy = np.sqrt(p.verts[:, 0] ** 2 + p.verts[:, 1] ** 2)
    assert radii_xy.max() <= 10.0 + 5.0 + 1e-6
    assert radii_xy.min() >= 10.0 - 5.0 - 1e-6
    assert np.abs(p.verts[:, 2]).max() <= 5.0 + 1e-6


def test_tripolygon_polygon_input() -> None:
    poly = Polygon(8, size=[20.0, 20.0])
    edge = EdgeProfile(5.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='tripolygon_basic')


def test_tripolygon_thin_disk() -> None:
    poly = Polygon(20, size=[100.0, 100.0])
    edge = EdgeProfile(1.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='thin_disk')


def test_tripolygon_with_rounded_polygon() -> None:
    poly = Polygon(4, size=[20.0, 20.0])
    poly.round_(rad=2.0, nrad=4)
    edge = EdgeProfile(5.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='rounded_polygon_extruded')


def test_fvgrid_basic() -> None:
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    verts, faces = fvgrid(x, y)
    assert verts.shape[1] == 3
    assert faces.shape[0] > 0
    assert np.all(np.isfinite(verts))


def test_fvgrid_triangles_mode() -> None:
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    verts, faces = fvgrid(x, y, triangles=True)
    assert verts.shape[1] == 3
    assert faces.shape[0] > 0


# ===========================================================================
# 1.3 Multi-particle (multi assemblies via Particle vertcat / sums)
# ===========================================================================

def test_dimer() -> None:
    p1 = trisphere(60, 10.0)
    p2 = trisphere(60, 10.0).shift([15.0, 0.0, 0.0])
    p_combined = p1 + p2
    _check_particle_sane(p_combined, label='dimer')
    assert p_combined.nfaces == p1.nfaces + p2.nfaces


def test_trimer_equilateral_triangle() -> None:
    r = 10.0
    sep = 25.0
    parts = []
    for k in range(3):
        ang = k * 2 * np.pi / 3
        sphere = trisphere(60, 2 * r)
        sphere.shift([sep * np.cos(ang), sep * np.sin(ang), 0.0])
        parts.append(sphere)
    p_all = parts[0]
    for q in parts[1:]:
        p_all = p_all + q
    _check_particle_sane(p_all, label='trimer')
    assert p_all.nfaces == sum(q.nfaces for q in parts)


def test_tetramer_square() -> None:
    parts = []
    for dx, dy in [(0, 0), (20, 0), (20, 20), (0, 20)]:
        sphere = trisphere(60, 8.0)
        sphere.shift([dx, dy, 0.0])
        parts.append(sphere)
    p_all = parts[0]
    for q in parts[1:]:
        p_all = p_all + q
    _check_particle_sane(p_all, label='tetramer')


@pytest.mark.parametrize('nmer', [6, 8])
def test_nmer_ring(nmer: int) -> None:
    sep = 25.0
    parts = []
    for k in range(nmer):
        ang = k * 2 * np.pi / nmer
        sphere = trisphere(60, 8.0)
        sphere.shift([sep * np.cos(ang), sep * np.sin(ang), 0.0])
        parts.append(sphere)
    p_all = parts[0]
    for q in parts[1:]:
        p_all = p_all + q
    _check_particle_sane(p_all, label='{}-mer'.format(nmer))


def test_grid_array_3x3() -> None:
    parts = []
    for i in range(3):
        for j in range(3):
            sphere = trisphere(32, 5.0)
            sphere.shift([i * 12.0, j * 12.0, 0.0])
            parts.append(sphere)
    p_all = parts[0]
    for q in parts[1:]:
        p_all = p_all + q
    _check_particle_sane(p_all, label='grid_3x3')
    assert p_all.nfaces == sum(q.nfaces for q in parts)


def test_grid_array_5x5() -> None:
    parts = []
    for i in range(5):
        for j in range(5):
            sphere = trisphere(32, 4.0)
            sphere.shift([i * 10.0, j * 10.0, 0.0])
            parts.append(sphere)
    p_all = parts[0]
    for q in parts[1:]:
        p_all = p_all + q
    _check_particle_sane(p_all, label='grid_5x5')


# ===========================================================================
# 1.4 Particle 변환 chain
# ===========================================================================

def test_chain_shift_rot_scale_flip() -> None:
    p = trisphere(60, 10.0)
    n_before = p.nverts
    q = p.shift([5.0, 0.0, 0.0]).rot(45.0).scale(2.0).flip(0)
    _check_particle_sane(q, label='chain_full')
    assert q.nverts == n_before


def test_chain_repeat_shift() -> None:
    p = trisphere(60, 5.0)
    centroid_before = p.verts.mean(axis=0)
    p.shift([1.0, 0.0, 0.0]).shift([0.0, 1.0, 0.0]).shift([0.0, 0.0, 1.0])
    centroid_after = p.verts.mean(axis=0)
    assert np.allclose(centroid_after - centroid_before, [1.0, 1.0, 1.0], atol=1e-9)


def test_chain_rot_z_360_identity() -> None:
    p = trisphere(60, 10.0)
    verts0 = p.verts.copy()
    p.rot(360.0)
    assert np.allclose(p.verts, verts0, atol=1e-8)


def test_chain_rot_x_axis() -> None:
    p = trisphere(60, 10.0)
    p.rot(90.0, dir=[1, 0, 0])
    _check_particle_sane(p, label='rot_x')


def test_chain_scale_anisotropic() -> None:
    # trisphere uses energy-minimized vertex distribution → bbox does not
    # exactly equal diameter. Compare scaled bbox ratios to original instead.
    p = trisphere(144, 10.0)
    bbox0 = p.verts.max(axis=0) - p.verts.min(axis=0)
    p.scale([2.0, 1.0, 0.5])
    bbox1 = p.verts.max(axis=0) - p.verts.min(axis=0)
    assert np.isclose(bbox1[0] / bbox0[0], 2.0, atol=1e-9)
    assert np.isclose(bbox1[1] / bbox0[1], 1.0, atol=1e-9)
    assert np.isclose(bbox1[2] / bbox0[2], 0.5, atol=1e-9)


def test_chain_flip_axis() -> None:
    p = trisphere(60, 10.0).shift([5.0, 0.0, 0.0])
    q = p.flip(0)
    assert np.isclose(q.verts[:, 0].mean(), -p.verts[:, 0].mean(), atol=1e-6)


def test_select_subset_carfun() -> None:
    p = trisphere(256, 10.0)
    obj1, obj2 = p.select(carfun=lambda x, y, z: z > 0)
    if obj1.nfaces > 0:
        _check_particle_sane(obj1, label='select_top_half')
        assert np.all(obj1.pos[:, 2] > 0)
    if obj2.nfaces > 0:
        _check_particle_sane(obj2, label='select_bottom_half')


def test_select_index_first_half() -> None:
    p = trisphere(256, 10.0)
    half = p.nfaces // 2
    obj1, obj2 = p.select(index=np.arange(half))
    assert obj1.nfaces == half


def test_vertcat_three_sphere() -> None:
    p1 = trisphere(60, 5.0)
    p2 = trisphere(60, 5.0).shift([15.0, 0.0, 0.0])
    p3 = trisphere(60, 5.0).shift([0.0, 15.0, 0.0])
    p = p1.vertcat(p2, p3)
    assert p.nfaces == p1.nfaces + p2.nfaces + p3.nfaces


def test_clean_after_concat() -> None:
    p1 = trisphere(60, 10.0)
    p2 = trisphere(60, 10.0)
    combined = p1 + p2
    cleaned = combined.clean()
    assert cleaned.nverts <= combined.nverts
    _check_particle_sane(cleaned, label='clean_after_concat')


def test_clean_no_op_for_clean_sphere() -> None:
    p = trisphere(144, 10.0)
    pclean = p.clean()
    assert pclean.nfaces == p.nfaces


def test_chain_select_then_shift() -> None:
    p = trisphere(256, 10.0)
    obj_top, _ = p.select(carfun=lambda x, y, z: z > 0)
    if obj_top.nfaces > 0:
        obj_top.shift([0.0, 0.0, 100.0])
        assert np.all(obj_top.verts[:, 2] > 50.0)


def test_chain_rot_then_select() -> None:
    p = trisphere(256, 10.0).rot(30.0)
    obj1, obj2 = p.select(carfun=lambda x, y, z: x > 0)
    total = (obj1.nfaces if obj1.nfaces > 0 else 0) + (obj2.nfaces if obj2.nfaces > 0 else 0)
    assert total == p.nfaces


# ===========================================================================
# Extra: combined edge cases
# ===========================================================================

def test_polygon_negative_dir() -> None:
    poly = Polygon(8, size=[10.0, 10.0], dir=-1)
    _check_polygon_sane(poly, label='neg_dir')
    assert poly.dir == -1


def test_polygon_symmetry_x() -> None:
    poly = Polygon(8, size=[10.0, 10.0], sym='x')
    _check_polygon_sane(poly, label='sym_x')


def test_polygon_symmetry_xy() -> None:
    poly = Polygon(8, size=[10.0, 10.0], sym='xy')
    _check_polygon_sane(poly, label='sym_xy')


def test_polygon_shift_then_normals() -> None:
    poly = Polygon(8, size=[10.0, 10.0])
    poly.shift([100.0, 100.0])
    nvec = poly.compute_normals()
    norms = np.linalg.norm(nvec, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-6)


def test_polygon_scale_anisotropic() -> None:
    poly = Polygon(8, size=[10.0, 10.0])
    poly.scale([2.0, 0.5])
    s = poly.size_
    assert np.isclose(s[0], 20.0)
    assert np.isclose(s[1], 5.0)


def test_polygon_rot_360_identity() -> None:
    poly = Polygon(8, size=[10.0, 10.0])
    pos0 = poly.pos.copy()
    poly.rot(360.0)
    assert np.allclose(poly.pos, pos0, atol=1e-8)


def test_polygon_flip_axis() -> None:
    poly = Polygon(8, size=[10.0, 10.0])
    pos0 = poly.pos.copy()
    poly.flip(axis=0)
    assert np.allclose(poly.pos[:, 0], -pos0[:, 0])


def test_edgeprofile_default() -> None:
    edge = EdgeProfile(5.0)
    assert edge.zmin <= edge.zmax
    assert np.isfinite(edge.zmin)
    assert np.isfinite(edge.zmax)


def test_edgeprofile_sharp_mode() -> None:
    edge = EdgeProfile(5.0, mode='11')
    assert np.isclose(edge.zmax - edge.zmin, 5.0, atol=1e-6)


def test_tripolygon_multi_polygon() -> None:
    p1 = Polygon(8, size=[10.0, 10.0])
    p1.shift([-12.0, 0.0])
    p2 = Polygon(8, size=[10.0, 10.0])
    p2.shift([12.0, 0.0])
    edge = EdgeProfile(5.0)
    p = tripolygon([p1, p2], edge)
    _check_particle_sane(p, label='tripolygon_multi')


def test_tripolygon_star_shape() -> None:
    n_pts = 5
    angles = np.linspace(0, 2 * np.pi, 2 * n_pts, endpoint=False) - np.pi / 2
    radii = np.tile([10.0, 4.0], n_pts)
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    poly = Polygon(pos)
    edge = EdgeProfile(2.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='star_extruded')


def test_tripolygon_C_shape_extrusion() -> None:
    pos = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 3.0],
        [3.0, 3.0],
        [3.0, 7.0],
        [10.0, 7.0],
        [10.0, 10.0],
        [0.0, 10.0]])
    poly = Polygon(pos)
    edge = EdgeProfile(2.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='C_extruded')


def test_tripolygon_extreme_huge() -> None:
    poly = Polygon(8, size=[1e5, 1e5])
    edge = EdgeProfile(1e4)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='huge_extrusion')


def test_tripolygon_extreme_tiny() -> None:
    poly = Polygon(8, size=[1e-2, 1e-2])
    edge = EdgeProfile(1e-3)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='tiny_extrusion')


def test_trirod_thin_diameter() -> None:
    p = trirod(0.1, 100.0)
    _check_particle_sane(p, label='trirod_thin')


def test_trirod_high_resolution() -> None:
    p = trirod(10.0, 30.0, n=[40, 40, 40])
    _check_particle_sane(p, label='trirod_high_res')
    assert p.nverts > 1000


def test_tritorus_thin_tube() -> None:
    p = tritorus(20.0, 0.5)
    _check_particle_sane(p, label='tritorus_thin')


def test_trisphere_n_below_min_clamps_to_32() -> None:
    p = trisphere(2, 1.0)
    assert p.nverts == 32


def test_trisphere_n_above_max_clamps_to_1444() -> None:
    p = trisphere(10000, 1.0)
    assert p.nverts == 1444


def test_polygon_collinear_vertices_extrude() -> None:
    pos = np.array([[0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    poly = Polygon(pos)
    edge = EdgeProfile(2.0)
    p = tripolygon(poly, edge)
    _check_particle_sane(p, label='collinear_extruded')

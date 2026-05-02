# API Reference

This document lists every public symbol exported from `mnpbem`. The
package mirrors the MATLAB MNPBEM17 class layout — if a name exists in
both, the calling sequence is kept compatible (see
[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for line-by-line mapping).

All examples below assume:

```python
import numpy as np
from mnpbem.materials import EpsConst, EpsTable, EpsDrude
from mnpbem.geometry  import trisphere, trirod, tricube, ComParticle
from mnpbem.bem       import BEMStat, BEMRet, BEMRetIter, BEMRetLayer
from mnpbem.simulation import PlaneWaveRet, DipoleRet, EELSRet
from mnpbem.spectrum  import SpectrumRet
```

---

## Table of contents

1. [Materials](#1-materials)
2. [Geometry](#2-geometry)
3. [Green functions](#3-green-functions)
4. [BEM solvers](#4-bem-solvers)
5. [Simulation (excitations)](#5-simulation-excitations)
6. [Spectrum (far-field / cross sections)](#6-spectrum-far-field--cross-sections)
7. [Mie theory](#7-mie-theory)
8. [Misc utilities](#8-misc-utilities)
9. [Parallel utilities](#9-parallel-utilities)

---

## 1. Materials

`from mnpbem.materials import ...`

Dielectric functions. All return `eps(enei)` and `k(enei)` (vacuum
wavenumber, units 1/nm) when called.

| Class / Function | Signature | Description |
|---|---|---|
| `EpsConst(eps)` | `eps : complex` | Constant (frequency-independent) dielectric. Use `EpsConst(1.0)` for vacuum. |
| `EpsTable(filename)` | `filename : str` | Tabulated `eps(lambda)` from a `.dat` file. Bundled: `gold.dat`, `silver.dat`, `goldpalik.dat`, `silverpalik.dat`, `copperpalik.dat`. |
| `EpsDrude(eps0, wp, gammad, name=None)` | `eps0, wp, gammad : float` | Drude model `eps = eps0 - wp^2 / (omega(omega + i gammad))`. |
| `EpsFun(fun)` / `epsfun(fun)` | `fun : callable` | Wraps any user function `fun(enei) -> eps`. |
| `EpsNonlocal(eps_metal, eps_embed, delta_d=0.05, ...)` | see below | Hydrodynamic Drude nonlocal cover-layer permittivity (Yu Luo et al., PRL 111, 093901). |

**Common methods** (all five classes):

- `__call__(enei) -> (eps, k)` where `enei` is a wavelength in nm and
  `k = 2*pi*sqrt(eps)/enei`.

**Example:**

```python
gold = EpsTable("gold.dat")
eps, k = gold(550.0)        # at lambda = 550 nm
```

### EpsNonlocal

**Hydrodynamic Drude nonlocal dielectric function** for nano-gap (< 1 nm)
and sub-5 nm particle simulations.

Cover-layer formulation: a thin shell (δ ≈ 0.05 nm) with effective
ε_t represents the nonlocal correction. Used together with
`coverlayer.shift` + `coverlayer.refine` for the 2-layer geometry.

**Constructor**:

```python
EpsNonlocal(eps_metal, eps_embed, delta_d=0.05, eps_inf=None,
            omega_p=None, gamma=None, beta=None, name=None)
```

Returns the shell `eps_t(enei)` based on Yu Luo et al. (PRL 111, 093901
(2013)).

The closed-form expression is

```
              eps_m * eps_b
eps_t(omega) = ---------------- * q_L(omega) * delta_d
              eps_m - eps_b

q_L(omega) = sqrt(omega_p^2 / eps_inf - omega * (omega + i * gamma)) / beta
```

Parameters: `eps_inf`, `omega_p`, `gamma` are pulled from `eps_metal`
(when `eps_metal` is `EpsDrude`) and otherwise must be supplied
explicitly. `beta` defaults to `sqrt(3/5) * v_F * hbar` from a built-in
Fermi-velocity table for `Au`/`Ag`/`Al`.

**Factory methods**:

- `EpsNonlocal.gold(eps_embed=None, delta_d=0.05, beta=None)` — Au
  (default β ≈ 0.714 eV·nm).
- `EpsNonlocal.silver(eps_embed=None, delta_d=0.05, beta=None)` — Ag.
- `EpsNonlocal.aluminum(eps_embed=None, delta_d=0.05, beta=None)` — Al.
- `EpsNonlocal.from_table(eps_table, eps_embed, eps_inf, omega_p, gamma, beta, delta_d=0.05, name=None)`
  — custom Drude + tabulated metal (e.g. Johnson-Christy gold).

**Helper**:

```python
from mnpbem.materials import make_nonlocal_pair

eps_core, eps_shell = make_nonlocal_pair(
    metal_name='gold',
    eps_embed=eps_embed,
    delta_d=0.05,
    beta=None,            # None → table default
    eps_metal=None,       # override core (e.g. EpsTable)
)
# eps_core  : EpsDrude (or override via eps_metal=)
# eps_shell : EpsNonlocal
```

**Usage example** (5 nm Au sphere, δ = 0.05 nm shell, vacuum):

```python
from mnpbem.materials import EpsConst, make_nonlocal_pair
from mnpbem.geometry  import trisphere, ComParticle
from mnpbem.greenfun  import coverlayer
from mnpbem.bem       import BEMStat

eps_embed = EpsConst(1.0)
eps_core, eps_shell = make_nonlocal_pair(
    'gold', eps_embed=eps_embed, delta_d=0.05,
)

p_core  = trisphere(144, 5.0 - 2 * 0.05)
p_shell = coverlayer.shift(p_core, 0.05)

p = ComParticle(
    [eps_embed, eps_core, eps_shell],
    [p_shell, p_core],
    [[3, 1], [2, 3]],
    closed=[1, 2],
)
refun = coverlayer.refine(p, [[1, 2]])

bem = BEMStat(p, refun=refun)        # or BEMRet(p, refun=refun)
```

For nonlocal modelling guidance and pitfalls (shell thickness,
β values, mesh face count) see
[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) section "Nonlocal hydrodynamic
Drude".

---

## 2. Geometry

`from mnpbem.geometry import ...`

### Mesh generators

| Function | Signature | Description |
|---|---|---|
| `trisphere(n, diameter=1.0)` | `n : int` (faces ~ n) | Triangulated sphere. `n` must be one of `{144, 256, 484, 1024, ...}` (precomputed meshes). |
| `trirod(diameter, height, n=None, triangles=False)` | | Triangulated rod (capsule) of given diameter and total height. |
| `tricube(n, length=1.0, e=0.25)` | `n : int` per edge, `e` edge rounding | Rounded cube. |
| `tritorus(diameter, rad, n=None)` | `rad`: tube radius | Triangulated torus. |
| `trispheresegment(phi, theta, diameter=1.0, triangles=False)` | `phi, theta : array` | Sphere segment. |
| `trispherescale(p, scale, unit=False)` | scales sphere into ellipsoid | Returns new `Particle`. |
| `tripolygon(poly, edge, **opts)` | `poly : Polygon`, `edge : EdgeProfile` | 3D extrusion from 2D polygon. |
| `fvgrid(x, y, triangles=False)` | | Convert parametric `(x, y)` surface to face/vertex. |
| `connect(p1, p2=None, ind=None)` | | Connect two meshes / find connectivity tables. |

### Particle classes

| Class | Signature | Description |
|---|---|---|
| `Particle(verts, faces=None, interp='flat', norm='on')` | `verts : (N,3)`, `faces : (M,3)` | Single triangulated surface. |
| `ComParticle(eps, particles, inout, *closed_args, **kwargs)` | `eps : list[Eps]`, `particles : list[Particle]`, `inout : (M,2) int` | Compound particle: composes multiple `Particle`s with a dielectric environment. `inout[i]` = `[outside_idx, inside_idx]` into `eps`. |
| `ComParticleMirror(eps, particles, inout, sym='x', closed_args=None)` | | `ComParticle` with `x`/`y`/`xy` mirror symmetry. |
| `CompStructMirror(p, enei, fun=None)` | | Container for mirror-symmetry-aware fields. |
| `Compound(eps, p, inout)` | | Lightweight compound particle (no closed-surface bookkeeping). |
| `Point(pos, nvec=None, area=None)` | `pos : (N,3)` | Pure point cloud (used as observation set in `MeshField`). |
| `ComPoint(p_or_eps, pos_or_points, inout_or_medium=None, mindist=None, ...)` | | `Point` placed inside a dielectric environment. |
| `EdgeProfile(*args, e=0.4, dz=0.15, mode='00', ...)` | | Edge rounding profile for `tripolygon`. |
| `Polygon(n_or_verts, mode='size', dir=1, sym=None, size=None)` | | 2D polygon (regular polygon by side count, or by explicit vertices). |
| `Polygon3(poly, z, edge=None, refun=None)` | | 3D polygon at height `z`, optional edge rounding. |
| `LayerStructure(epstab, ind, z)` | `epstab : list[Eps]`, `ind : list[int]`, `z : list[float]` | Stratified medium definition (`z[k]` = interface, `ind[k]` = dielectric index). |

**Common attributes** (`Particle`, `ComParticle`):
- `.verts` (V,3) vertices.
- `.faces` (F,3) triangle indices.
- `.pos`   (F,3) face centroids.
- `.area`  (F,) face areas.
- `.nvec`  (F,3) outward face normals.
- `.nfaces`, `.nverts` integer counts.

---

## 3. Green functions

`from mnpbem.greenfun import ...`

Most users do not construct Green objects directly — `BEMStat`, `BEMRet`,
... build them internally. They are exposed so you can plug in your own
solver, or call `MeshField` to evaluate fields off-mesh.

| Class | Signature | Description |
|---|---|---|
| `GreenStat(p1, p2, **opts)` | | Quasistatic free-space Green tensor. |
| `CompGreenStat(p1, p2, **opts)` | | Composite Green for `ComParticle`. |
| `CompGreenRet(p1, p2, **opts)` | | Retarded composite Green. |
| `CompGreenStatMirror(...)`, `CompGreenRetMirror(...)` | | Mirror-symmetry variants. |
| `CompGreenStatLayer(...)`, `CompGreenRetLayer(p1, p2, layer, **opts)` | | Layered-substrate variants. |
| `CompGreenTabLayer(p1, p2, layer, tab=None, **opts)` | | Tabulated/precomputed substrate Green. |
| `GreenRetLayer(p1, p2, layer, tab=None, deriv='cart')` | | Direct retarded layer Green. |
| `GreenTabLayer(layer, tab=None)` | | Builds the tabulation used by `*TabLayer`. |
| `CompStruct` | runtime container | Holds `(phi, sig1, sig2, ...)` returned by Green and BEM calls. |

### ACA / H-matrix (large-mesh accelerator)

| Class | Signature | Description |
|---|---|---|
| `ClusterTree(pos, cleaf=32, ipart_arr=None)` | | Hierarchical clustering of mesh positions. |
| `HMatrix(tree=None, htol=1e-6, kmax=100, fadmiss=None)` | | Block-low-rank matrix used by ACA. |
| `ACACompGreenStat(p, htol=1e-6, kmax=100, cleaf=32)` | | ACA-compressed quasistatic Green. |
| `ACACompGreenRet(p, htol=1e-6, kmax=100, cleaf=32, eta=2.5)` | | ACA-compressed retarded Green. |
| `ACACompGreenRetLayer(p, layer, ...)` | | ACA-compressed retarded substrate Green. |

### Function

- `greenfunction(p1, p2, op=None, **kwargs)` — convenience factory that
  picks `GreenStat`, `GreenRet`, or `GreenRetLayer` depending on `op`.

---

## 4. BEM solvers

`from mnpbem.bem import ...`

All solvers share the same calling pattern:

```python
bem = BEMRet(p)                  # build
sig, bem = bem.solve(exc.potential(p, enei))   # solve at one wavelength
```

| Class | Signature | Description |
|---|---|---|
| `BemBase(p, **opts)` | abstract | Common base; not instantiated directly. |
| `BEMStat(p, enei=None, **opts)` | | Quasistatic dense solver. |
| `BEMRet(p, enei=None)` | | Retarded dense solver (full Maxwell). |
| `BEMStatMirror(p, ...)`, `BEMRetMirror(p, ...)` | | Mirror-symmetry variants (solve only the irreducible block). |
| `BEMStatEig(p, ...)` | | Quasistatic eigenmode-expansion solver. |
| `BEMStatEigMirror(...)` | | Eigenmode + mirror symmetry. |
| `BEMStatLayer(p, layer, ...)` | | Quasistatic on a substrate. |
| `BEMRetLayer(p, layer, enei=None, greentab=None, **opts)` | | Retarded on a substrate. |
| `BEMLayerMirror(...)` | | Layer + mirror symmetry. |
| `BEMIter(...)` | abstract | Base for iterative solvers. |
| `BEMStatIter(p, enei=None, **opts)` | | Quasistatic iterative (ACA+GMRES). |
| `BEMRetIter(p, enei=None, **opts)` | | Retarded iterative (ACA+GMRES) — recommended for >5 k faces. |
| `BEMRetLayerIter(p, layer, ...)` | | Retarded iterative on a substrate. |

### Common methods

- `.init(enei) -> self` — refresh internal matrices for a new wavelength
  (called automatically by `.solve`).
- `.solve(exc) -> (sig, bem)` — solve `bem * sig = exc` for surface
  charges/currents `sig`. Returns the solver itself for chaining.
- `.field(sig, point) -> field_struct` — evaluate `(E, H, phi, ...)` at
  `point : Point` or `ComPoint`.
- `.potential(sig, point)` — evaluate scalar/vector potential at `point`.
- `.clear()` — drop cached factors / matrices to free memory.
- `.name`, `.needs` — metadata (excitation/observable kinds the solver
  supports).

### Iterative-solver options

`BEMStatIter` / `BEMRetIter` accept (kwargs or `op` dict):

- `aca={"htol": 1e-6, "kmax": 100, "cleaf": 32, "eta": 2.5}` — ACA tolerance.
- `iter={"tol": 1e-6, "restart": 30, "maxit": 200}` — GMRES parameters.
- `precond="diag"` — Jacobi preconditioner (default).

### Schur complement (v1.2.0)

`BEMStat` / `BEMRet` 가 `schur=True` 옵션으로 cover-layer 변수 자동 소거를
지원한다. nonlocal cover-layer 시뮬레이션의 메모리를 약 4× → ~2× 로
줄이고, LU 풀이 단계를 약 30% 가속한다. 결과는 standard formulation 과
수학적으로 동등 (rel `< 1e-12`).

```python
from mnpbem.materials import EpsConst, make_nonlocal_pair
from mnpbem.geometry  import trisphere, ComParticle
from mnpbem.greenfun  import coverlayer
from mnpbem.bem       import BEMStat

eps_embed = EpsConst(1.0)
eps_core, eps_shell = make_nonlocal_pair(
    'gold', eps_embed=eps_embed, delta_d=0.05,
)

p_core  = trisphere(144, 5.0 - 2 * 0.05)
p_shell = coverlayer.shift(p_core, 0.05)

p = ComParticle(
    [eps_embed, eps_core, eps_shell],
    [p_shell, p_core],
    [[3, 1], [2, 3]],
    closed=[1, 2],
)
refun = coverlayer.refine(p, [[1, 2]])

# Schur 적용 — cover layer 변수 소거하여 reduced matrix 풀이
bem = BEMStat(p, refun=refun, schur=True)
# 동일하게:
# bem = BEMRet(p, refun=refun, schur=True)
```

옵션 값:

- `schur=True` — cover layer 가 감지되면 schur 소거 강제 적용.
- `schur=False` (default) — standard formulation, v1.1.0 동작.
- `schur='auto'` — cover layer 가 자동 감지되면 schur 적용, 아니면 standard.

`pymnpbem_simulation` wrapper 는 `schur='auto'` 를 기본값으로 사용한다.

### `plasmonmode(bem, n=10, ...)`

Compute the `n` lowest plasmon eigenmodes of a `BEMStat` solver.

---

## 5. Simulation (excitations)

`from mnpbem.simulation import ...`

Excitation sources. Pattern:

```python
exc = PlaneWaveRet(np.array([[1, 0, 0]]), np.array([[0, 0, 1]]))
pot = exc.potential(p, enei)        # build excitation at one wavelength
sig, bem = bem.solve(pot)
ext = exc.extinction(sig)           # cross sections
sca, _ = exc.scattering(sig)
abs_ = exc.absorption(sig)
```

### Plane wave

| Class | Signature |
|---|---|
| `PlaneWaveStat(pol, medium=1, **opts)` | quasistatic plane wave |
| `PlaneWaveRet(pol, dir, medium=1, **opts)` | retarded plane wave |
| `PlaneWaveStatMirror(pol, medium=1, sym=None, **opts)` | + mirror symmetry |
| `PlaneWaveRetMirror(pol, dir, medium=1, sym=None, **opts)` | + mirror symmetry |
| `PlaneWaveStatLayer(pol, medium=1, **opts)` | + layer / substrate |
| `PlaneWaveRetLayer(pol, dir, medium=1, layer=None, **opts)` | + layer / substrate |

`pol`, `dir` accept either `(3,)` or `(N,3)` arrays for batched
polarizations.

### Dipole

| Class | Signature |
|---|---|
| `DipoleStat(pt, dip=None, full=False, **opts)` | quasistatic |
| `DipoleRet(pt, dip=None, full=False, medium=1, pinfty=None, **opts)` | retarded |
| `DipoleStatMirror(...)`, `DipoleRetMirror(...)` | mirror |
| `DipoleStatLayer(...)`, `DipoleRetLayer(...)` | layered |

`pt : ComPoint` is the dipole position; `dip : (N,3)` are dipole moments
(default = canonical Cartesian basis if `None` and `full=True`).

Methods: `.decayrate(sig)` returns total / radiative decay rates;
`.field(sig, p)`; `.potential(p, enei)`.

### EELS (electron energy-loss spectroscopy)

| Class | Signature |
|---|---|
| `EELSStat(p, impact, width, vel, cutoff=None, phiout=0.01, **opts)` | quasistatic |
| `EELSRet(p, impact, width, vel, cutoff=None, phiout=0.01, pinfty=None, medium=1, **opts)` | retarded |
| `EELSBase` | abstract base |

`impact : (N,2)` are 2D impact parameters; `vel` electron speed in units
of `c`; `width` Gaussian beam width.

Methods: `.loss(sig)` returns the EELS probability spectrum.

### Field probes

- `MeshField(p, x, y, z=None, nmax=None, mindist=None, sim='stat', **opts)` —
  evaluate fields on a grid `(x, y, z)` outside the particle.

### Convenience factories (MATLAB-style dispatch)

- `dipole(pt, op=None, **kwargs)` → returns `DipoleStat` or `DipoleRet`.
- `planewave(pol, dir=None, op=None, **kwargs)` → likewise.
- `electronbeam(p, op=None, **kwargs)` → returns `EELSStat` / `EELSRet`.

---

## 6. Spectrum (far-field / cross sections)

`from mnpbem.spectrum import ...`

| Class | Signature | Description |
|---|---|---|
| `SpectrumStat(pinfty=None, medium=1)` | | Far-field for quasistatic. |
| `SpectrumRet(pinfty=None, medium=1)` | | Far-field for retarded. |
| `SpectrumStatLayer(pinfty=None, layer=None, medium=None)` | | Far-field on substrate (quasistatic). |
| `SpectrumRetLayer(pinfty=None, layer=None, medium=1)` | | Far-field on substrate (retarded). |
| `spectrum(op=None, **kwargs)` | factory | Returns the right `Spectrum*` for `op`. |

`pinfty` is a far-field collection mesh (sphere of unit radius). The
package ships `pinfty256.bin` (256-face unit sphere). Construct with:

```python
from mnpbem.geometry import trisphere
pinfty = trisphere(256, 2.0)        # diameter 2 -> radius 1
spec   = SpectrumRet(pinfty)
```

Methods: `.scattering(sig)`, `.farfield(sig)`.

---

## 7. Mie theory

`from mnpbem.mie import ...`

Analytical references for sphere validation.

| Class / Function | Signature | Description |
|---|---|---|
| `MieStat(epsin, epsout, diameter, lmax=20)` | | Quasistatic Mie. |
| `MieRet(epsin, epsout, diameter, lmax=20)` | | Retarded Mie (full multipole). |
| `MieGans(epsin, epsout, ax)` | `ax : (3,)` half-axes | Gans theory for ellipsoid (quasistatic). |
| `mie_solver(epsin, epsout, diameter, sim='stat', lmax=20)` | factory | Returns `MieStat` or `MieRet`. |
| `spharm(ltab, mtab, theta, phi)` | | Scalar spherical harmonic. |
| `vecspharm(ltab, mtab, theta, phi)` | | Vector spherical harmonic. |
| `sphtable(lmax, key='z')` | | (l, m) ordering convention. |

`MieRet` methods: `.extinction(enei)`, `.scattering(enei)`,
`.absorption(enei)`, `.field(enei, point)`.

---

## 8. Misc utilities

`from mnpbem.misc import ...`

### Math helpers

| Symbol | Description |
|---|---|
| `matmul(a, b)` | Batched 3x3 matrix-matrix multiply over leading dims. |
| `inner(a, b)`, `outer(a, b)` | Batched inner / outer products of 3-vectors. |
| `matcross(a, b)` | Cross product. |
| `vec_norm(v)`, `vec_normalize(v)` | Vector norm / unit. |
| `spdiag(v)` | Sparse diagonal. |

### Distance helpers

| Symbol | Description |
|---|---|
| `pdist2(x, y)` | All-pair squared distance (mirrors MATLAB `pdist2`). |
| `bradius(p)` | Bounding-sphere radius of a particle. |
| `bdist2(p, q)` | Bounding-sphere distance squared. |
| `distmin3(...)` | Minimum 3D distance helper. |

### Constants

| Symbol | Value | Unit |
|---|---|---|
| `EV2NM` | 1239.8414... | nm·eV (`enei[nm] * E[eV] = EV2NM`) |
| `BOHR`  | 0.05292      | nm  |
| `HARTREE` | 27.211     | eV  |
| `FINE`  | 0.00729735... | fine-structure constant |

### Options

- `bemoptions(op=None, **kwargs)` — build a default `op` dict.
- `getbemoptions(*args, **kwargs)` — extract options for a given solver.
- `getfields(struct, *names)` — pull fields from a `CompStruct`.

### Quadrature & shapes

- `Tri`, `Quad`, `triangle_unit_set`, `trisubdivide` — triangle and
  unit-element helpers.
- `lglnodes`, `lgwt` — Gauss-Legendre nodes / weights.
- `IGrid2`, `IGrid3` — integer-indexed regular grids.
- `ValArray`, `VecArray` — typed array containers.
- `QuadFace` — quadrilateral face element.

### Plotting

- `BemPlot(**kwargs)` — particle / field viewer (matplotlib).
- `arrowplot`, `coneplot`, `coneplot2` — vector field plots.
- `mycolormap` — divergent colormap matching MATLAB's default.
- `particlecursor` — interactive face inspector.

### Other

- `nettable`, `patchcurvature`, `memsize`, `round_left`, `Mem`,
  `multi_waitbar` — miscellaneous helpers ported from MATLAB.

---

## 9. Parallel utilities

`from mnpbem.utils import ...`

| Function | Description |
|---|---|
| `compute_spectrum(bem, exc, enei, kind='ext', spectrum=None)` | Serial spectrum sweep. Returns `np.ndarray` with one entry per `enei`. |
| `compute_spectrum_parallel(bem_factory, exc_factory, enei, n_workers=None, ...)` | Multi-process sweep using a process pool. |

For multi-GPU and multi-node MPI sweeps see
[examples/07_gpu_multigpu.py](../examples/07_gpu_multigpu.py) and
the wrappers in `mnpbem_simulation` (companion repo).

### VRAM share — multi-GPU LU (v1.2.0)

큰 mesh (25k+ face) 의 dense LU 풀이를 multi-GPU 메모리 풀로 처리한다.
단일 GPU VRAM 한계 (예: RTX A6000 48 GB) 를 초과하는 경우 2 GPU pool
(96 GB), 4 GPU pool (192 GB) 로 fit 시킨다.

환경변수 인터페이스 (가장 간단):

```python
import os
os.environ['MNPBEM_VRAM_SHARE_GPUS']    = '4'           # 4 GPU 메모리 합쳐 사용
os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = 'cusolvermg'  # default

from mnpbem.bem import BEMRet
bem = BEMRet(p)   # 자동으로 cuSolverMg multi-GPU LU 활용
```

직접 호출 인터페이스:

```python
from mnpbem.utils.gpu import lu_factor_dispatch, lu_solve_dispatch

lu = lu_factor_dispatch(A, n_gpus=4)        # cuSolverMg block-cyclic LU
x  = lu_solve_dispatch(lu, b, n_gpus=4)
```

지원 backend (`MNPBEM_VRAM_SHARE_BACKEND`):

- `cusolvermg` — NVIDIA cuSOLVER MG (권장). NVLink/PCIe 자동 최적화.
- `magma` — ICL Magma multi-GPU (예정).
- `nccl` — 사용자 정의 block-distributed LU (예정).

`pymnpbem_simulation` wrapper 의 `compute.n_gpus_per_worker > 1` 설정이
자동으로 `MNPBEM_VRAM_SHARE_GPUS` 를 설정한다. wavelength 분배
(Lane D, multi-worker) 와 결합 가능 — 예: 8 GPU 환경에서 2-GPU pool 4개
(`n_workers=4`, `n_gpus_per_worker=2`).

상세는 `docs/ARCHITECTURE.md` "Key design decisions #12" 참조.

---

## Versioning & deprecations

- `mnpbem.__version__` follows SemVer.
- v1.x will keep the public API listed above stable. Any signature
  change requires a major-version bump.
- Internal modules (`mnpbem._*`, anything not in `__all__`) are *not*
  part of the public API and may change without notice.

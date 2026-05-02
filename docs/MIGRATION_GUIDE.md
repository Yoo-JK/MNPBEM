# Migration from MATLAB MNPBEM to Python

This guide is for users with an existing MATLAB MNPBEM17 script who want
to port it to the Python toolbox. Calling sequences and class names are
intentionally preserved, so the translation is mostly mechanical. Below
are the most common patterns plus a list of pitfalls that catch new
users.

## Quick mapping table

### Materials

| MATLAB | Python |
|---|---|
| `epsconst(1)` | `from mnpbem.materials import EpsConst; EpsConst(1.0)` |
| `epstable('gold.dat')` | `from mnpbem.materials import EpsTable; EpsTable('gold.dat')` |
| `epsdrude('Au')` | `from mnpbem.materials import EpsDrude; EpsDrude(eps0, wp, gammad)` |
| `[eps,k]=epstab(enei)` | `eps, k = epstab(enei)` |

### Geometry

| MATLAB | Python |
|---|---|
| `p = trisphere(144, 20)` | `from mnpbem.geometry import trisphere; p = trisphere(144, 20)` |
| `p = trirod(20, 50)` | `from mnpbem.geometry import trirod; p = trirod(20, 50)` |
| `p = tricube(11, 20)` | `from mnpbem.geometry import tricube; p = tricube(11, 20)` |
| `comparticle({p}, {eps1,eps2}, [2,1], 1, op)` | `ComParticle([eps1, eps2], [p], [[2, 1]], 1, **op)` |
| `polygon(8, 'size', [10 20])` | `Polygon(8, mode='size', size=[10, 20])` |
| `polygon3(poly, 5, edge)` | `Polygon3(poly, 5, edge)` |
| `edgeprofile(0.4)` | `EdgeProfile(e=0.4)` |
| `tripolygon(poly3, edge)` | `tripolygon(poly3, edge)` |
| `layerstructure(epstab, [1,2], 0)` | `LayerStructure(epstab, [1, 2], [0.0])` |

### BEM solvers

| MATLAB | Python |
|---|---|
| `bem = bemstat(p)` | `from mnpbem.bem import BEMStat; bem = BEMStat(p)` |
| `bem = bemret(p)` | `from mnpbem.bem import BEMRet; bem = BEMRet(p)` |
| `bem = bemstatlayer(p, layer)` | `from mnpbem.bem import BEMStatLayer; bem = BEMStatLayer(p, layer)` |
| `bem = bemretlayer(p, layer)` | `from mnpbem.bem import BEMRetLayer; bem = BEMRetLayer(p, layer)` |
| `bem = bemretiter(p, op)` | `from mnpbem.bem import BEMRetIter; bem = BEMRetIter(p, **op)` |
| `bem = bem.init(enei)` | `bem = bem.init(enei)`  *(usually implicit in `.solve`)* |
| `sig = bem \ exc` | `sig, bem = bem.solve(exc)` |
| `sig = bem \ exc` (mirror) | `sig, bem = bem.solve(exc)`  *(same call)* |
| `clear(bem)` | `bem.clear()` |

### Excitations

| MATLAB | Python |
|---|---|
| `exc = planewave([1 0 0])` (stat) | `from mnpbem.simulation import PlaneWaveStat; exc = PlaneWaveStat([[1, 0, 0]])` |
| `exc = planewaveret(...)` | shorthand: `from mnpbem.simulation import planewave; exc = planewave(pol, dir, op)` |
| `exc = planewave([1 0 0], [0 0 1])` (ret) | `PlaneWaveRet([[1,0,0]], [[0,0,1]])` |
| `exc = dipole(pt)` | `from mnpbem.simulation import DipoleRet, DipoleStat; DipoleRet(pt)` (or `DipoleStat`) |
| `exc = electronbeam(p, impact, w, vel)` | `from mnpbem.simulation import EELSRet; EELSRet(p, impact, w, vel)` |

### Calling convention

| MATLAB pattern | Python pattern |
|---|---|
| `exc_struct = exc(p, enei)` | `pot = exc.potential(p, enei)` |
| `sig = bem \ exc_struct` | `sig, bem = bem.solve(pot)` |
| `[ext, dipole] = exc.extinction(sig)` | `ext = exc.extinction(sig)` |
| `[sca, dsig] = exc.scattering(sig)` | `sca, dsig = exc.scattering(sig)` |
| `abs = exc.absorption(sig)` | `abs_ = exc.absorption(sig)` |

### Far-field / spectrum

| MATLAB | Python |
|---|---|
| `pinfty = trisphere(256, 2)` | `pinfty = trisphere(256, 2)` |
| `spec = spectrum(pinfty, op)` | `from mnpbem.spectrum import SpectrumRet; spec = SpectrumRet(pinfty)` |
| `[sca, dsig] = spec.scattering(sig)` | `sca, dsig = spec.scattering(sig)` |
| `field = spec.farfield(sig)` | `field = spec.farfield(sig)` |

### Mie reference

| MATLAB | Python |
|---|---|
| `mie = miesolver(epsin, epsout, d, op)` | `from mnpbem.mie import mie_solver; mie = mie_solver(epsin, epsout, d, sim='ret')` |
| `[sca, ext, abs] = mie.cross(enei)` | `sca = mie.scattering(enei); ext = mie.extinction(enei); abs_ = ext - sca` |

### Iterative solver / ACA

| MATLAB | Python |
|---|---|
| `op = bemoptions('sim','ret','interp','curv','RelCutoff',2)` | `op = dict(sim='ret', interp='curv', RelCutoff=2)` |
| `op.iter = struct('tol',1e-6,'restart',30)` | pass `iter={'tol':1e-6,'restart':30}` to solver |
| `op.aca = struct('htol',1e-6,'kmax',100)` | pass `aca={'htol':1e-6,'kmax':100}` to solver |

---

## Side-by-side worked example

### MATLAB (sphere extinction, retarded)

```matlab
op       = bemoptions('sim','ret','interp','curv');
epstab   = {epsconst(1), epstable('gold.dat')};
p        = comparticle(epstab, {trisphere(144,20)}, [2,1], 1, op);
bem      = bemret(p);
exc      = planewave([1 0 0], [0 0 1], op);

enei     = linspace(400,800,41);
ext      = zeros(size(enei));
for i = 1:length(enei)
    sig    = bem \ exc(p, enei(i));
    ext(i) = exc.extinction(sig);
end
```

### Python equivalent

```python
import numpy as np
from mnpbem.materials  import EpsConst, EpsTable
from mnpbem.geometry   import trisphere, ComParticle
from mnpbem.bem        import BEMRet
from mnpbem.simulation import PlaneWaveRet

epstab = [EpsConst(1.0), EpsTable("gold.dat")]
p      = ComParticle(epstab, [trisphere(144, 20)], [[2, 1]], 1, interp="curv")
bem    = BEMRet(p)
exc    = PlaneWaveRet(np.array([[1, 0, 0]]), np.array([[0, 0, 1]]))

enei   = np.linspace(400, 800, 41)
ext    = np.zeros_like(enei)
for i, e in enumerate(enei):
    sig, bem = bem.solve(exc.potential(p, e))
    ext[i]   = float(np.real(np.ravel(exc.extinction(sig))[0]))
```

---

## Common pitfalls

### 1. `comparticle` outside/inside convention

MATLAB:

```matlab
comparticle({p}, {eps_out, eps_in}, [2, 1; 1, 2], 1, op)
```

is read as: face `i` has dielectric `eps(2)` on the *outside* and
`eps(1)` on the *inside* in the first column-pair, and the reverse for
the second. The trailing `1` is the index of which particle is
**closed** (a watertight surface).

Python keeps the same convention:

```python
ComParticle(eps, [p], [[2, 1], [1, 2]], 1)        # closed = 1 (1-based, like MATLAB)
ComParticle(eps, [p], [[2, 1]], 1)                # only one face / one row
```

If `closed_args` is omitted, no surface is treated as closed (which
breaks the static problem). When in doubt, pass `closed=[1]` (or
`closed=[1, 2]` for two surfaces).

### 2. 1-based vs 0-based indexing

- **`inout` entries**: 1-based, like MATLAB. `[[2, 1]]` means
  `eps[1]` outside, `eps[0]` inside (Python lists).
- **`closed` argument**: 1-based, like MATLAB.
- **`faces`** array: stored 0-based internally, like every NumPy array.
  When you write `p.faces` you get 0-based indices into `p.verts`.
  MATLAB's `p.faces` is 1-based — adjust by `-1` if you compare both.

### 3. Polarization / direction shape

MATLAB lets you write `planewave([1 0 0], [0 0 1])` — a 1×3 row vector.
NumPy keeps the same shape but typed:

```python
PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),         # shape (1, 3)
             np.array([[0.0, 0.0, 1.0]]))
```

A 1-D `(3,)` array is also accepted, but the `(1, 3)` form makes it
explicit that you are passing a single polarization (vs. a batch
of `(N, 3)`).

### 4. Retarded vs static

The MATLAB option `sim` does not exist in Python; you choose by
importing the right class:

| MATLAB `op.sim` | Python class |
|---|---|
| `'stat'` | `BEMStat`, `PlaneWaveStat`, `DipoleStat`, `EELSStat`, `SpectrumStat` |
| `'ret'`  | `BEMRet`,  `PlaneWaveRet`,  `DipoleRet`,  `EELSRet`,  `SpectrumRet`  |

The convenience factories `planewave(...)`, `dipole(...)`,
`electronbeam(...)`, `spectrum(...)` accept `op={'sim':'ret'}` and
return the right class.

### 5. ODE / quadrature tolerance

MATLAB's `bemoptions` uses `'AbsCutoff'`, `'RelCutoff'`, `'refine'`.
The Python solver accepts the same names as keyword arguments:

```python
ComParticle(eps, [p], [[2, 1]], 1, interp="curv", AbsCutoff=1e-3, RelCutoff=2)
```

If you previously tuned `op.RelCutoff` for accuracy, keep the same
value in Python — the integration scheme is bit-identical for
`interp='flat'` and ULP-close for `interp='curv'`.

### 6. Closed surfaces & EELS

EELS in MATLAB defaults to `closed=p.closed` from the `comparticle`.
In Python, build the particle with `closed=[1]` (or whichever is
closed) and pass the same `p` to `EELSRet(p, impact, w, vel)`:

```python
p   = ComParticle(eps, [sphere], [[2, 1]], 1)         # 1 = sphere is closed
exc = EELSRet(p, impact=np.array([[0, 0]]), width=0.5, vel=0.7)
```

### 7. `iter` and `aca` options

MATLAB:

```matlab
op.iter = struct('tol', 1e-6, 'restart', 30);
op.aca  = struct('htol', 1e-6, 'kmax', 100);
bem     = bemretiter(p, op);
```

Python:

```python
bem = BEMRetIter(p,
                 iter={"tol": 1e-6, "restart": 30, "maxit": 200},
                 aca ={"htol": 1e-6, "kmax": 100, "cleaf": 32, "eta": 2.5})
```

The default `htol=1e-6` matches MATLAB. For very small particles
(<2 nm) consider `htol=1e-8`.

### 8. `clear(bem)` vs `bem.clear()`

MATLAB: `clear(bem)` releases cached factors. Python:
`bem.clear()`. Useful inside a wavelength loop if memory is tight.

### 9. Cell arrays → Python lists

Whenever MATLAB uses `{ ... }` for a cell array, Python uses `[ ... ]`.

```matlab
{epsconst(1), epstable('gold.dat')}
```
becomes
```python
[EpsConst(1.0), EpsTable("gold.dat")]
```

### 10. `struct` returns → `CompStruct`

MATLAB returns nested structs from `bem \ exc`, `exc.potential(...)`,
etc. Python returns a `CompStruct` object whose fields are accessible
both as attributes and via `getfields(s, 'phi')`:

```python
sig.phi          # scalar potential
sig.sig1         # surface charge on side 1
getfields(sig, "phi", "sig1")     # tuple
```

### 11. Plotting

MATLAB `plot(p)` opens a figure. Python uses `BemPlot`:

```python
from mnpbem.misc import BemPlot
fig = BemPlot()
fig.plot(p)
fig.show()
```

For matplotlib non-interactive use (CI, headless), set the backend
before importing:

```python
import matplotlib
matplotlib.use("Agg")
```

### 12. Far-field collection mesh

In MATLAB you sometimes pass `[]` for `pinfty`. In Python, pass
`None` or omit:

```python
spec = SpectrumRet()                # uses default 256-face unit sphere
```

### 13. `bem.solve` return signature

MATLAB: `sig = bem \ exc` (single return).
Python: `sig, bem = bem.solve(exc)` (two returns: the second is the
solver itself, with cached factors). Always unpack both — if you do
`sig = bem.solve(exc)` you'll get a tuple by accident.

### 14. Wavelength sweep performance

In MATLAB, `bem.init(enei)` is sometimes called explicitly. In Python,
`bem.solve(exc.potential(p, enei))` does it implicitly. If you sweep
many wavelengths for the same particle, the dense-matrix factor is
re-built each call — use `BEMRetIter` (ACA + GMRES) which scales much
better, or use `compute_spectrum_parallel` for embarrassingly-parallel
wavelength sweeps.

### 15. GPU acceleration

Set `MNPBEM_GPU=1` (and ensure `cupy` is installed) to run dense
operations on the GPU. The Python API is otherwise identical:

```bash
MNPBEM_GPU=1 python my_script.py
```

For multi-GPU wavelength dispatch see `examples/07_gpu_multigpu.py`.

---

## What does **not** map cleanly

A small number of MATLAB features are not yet ported — usually because
their Python equivalent is materially different. If you rely on these,
file an issue.

| MATLAB feature | Status |
|---|---|
| `nonlocal.m` (Pendry-style nonlocal cover layer) | partial — only the simplest local-cover mode is wrapped |
| GUI (`MNPBEM_GUI`) | not ported (use `BemPlot` for static viewing) |
| `makemnpbemhelp.m` (HTML help generator) | replaced by this `docs/` directory |
| `compound.norm`, `compound.union` (set-algebra helpers) | partial — see `docs/API_REFERENCE.md`, `Compound` |

For everything in the standard `Demo/` directory (72 demos, including
EELS, layered substrate, dipole decay, plane wave, mirror symmetry),
the Python port reproduces the MATLAB output to machine precision in 55
of 72 cases (see `docs/PERFORMANCE.md` for the rest).

---

## Quick reference: where to find each MATLAB file

| MATLAB directory | Python module |
|---|---|
| `Particles/` | `mnpbem.geometry` |
| `Material/` | `mnpbem.materials` |
| `Greenfun/` | `mnpbem.greenfun` |
| `BEM/` | `mnpbem.bem` |
| `Simulation/` | `mnpbem.simulation`, `mnpbem.spectrum` |
| `Mie/` | `mnpbem.mie` |
| `Misc/` | `mnpbem.misc` |
| `Mesh2d/` | `mnpbem.geometry.mesh2d` (used by `tripolygon`) |
| `Demo/` | `examples/` (selected) + `validation/` (full regression suite) |

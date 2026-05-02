# MNPBEM Python Port — Architecture

This document is aimed at contributors and maintainers (including future-self).
It describes how the Python port is laid out, *why* the major design choices
were made, and where to look when something behaves differently from the
MATLAB MNPBEM toolbox.

External users should start with `README.md` and `docs/API_REFERENCE.md`
instead.

## 1. High-level overview

The Python port is a faithful, MATLAB-compatible re-implementation of
Hohenester & Trügler's MNPBEM toolbox. The goals, in priority order, are:

1. **Numerical parity with MATLAB** on the official demo set (50 + 22 demos)
   — bit-identical where the underlying math libraries permit, and within
   floating-point ULP tolerance otherwise.
2. **Pure Python distribution** — no MATLAB runtime required at import or
   call time. MATLAB Engine is only used as an *optional* validation backend
   (opt-in via env var) for cross-checking specific demos.
3. **Performance parity or better** — CPU-only path keeps up with MATLAB on
   small/medium meshes; CuPy GPU path scales beyond what MATLAB can do.
4. **A drop-in API** — class names, method names, and option keywords
   mirror the MATLAB OOP layout so MATLAB users can port scripts mechanically.

The core stack is:

- Python 3.11 / 3.12
- `numpy` + `scipy` (LU, GMRES, Bessel functions, ODE integrators)
- `numba` (JIT for the dense Green-function and meshfield hot loops)
- `cupy` (optional, single- and multi-GPU dense LU / matmul / refinement)
- `mpi4py` (optional, multi-node wavelength dispatch)
- `fmm3dpy` (optional, free-space retarded meshfield acceleration)

## 2. Repository layout

```
MNPBEM/
├── mnpbem/                     # the importable package
│   ├── __init__.py             # re-exports the public API surface
│   ├── geometry/               # particles, polygons, mesh2d, layer structure
│   ├── materials/              # dielectric functions
│   ├── greenfun/               # Green's functions, ACA, H-matrix
│   ├── bem/                    # BEM solvers (direct + iterative)
│   ├── simulation/             # excitations + meshfield evaluation
│   ├── spectrum/               # spectrum / cross-section helpers
│   ├── mie/                    # Mie reference solver
│   ├── misc/                   # math, plotting, options, units
│   └── utils/                  # GPU dispatch, multi-GPU, MPI, parallel
├── docs/                       # ARCHITECTURE / API_REFERENCE / PERF / ...
├── tests/                      # unit + regression tests
├── validation/                 # Mie / sphere / rod / dimer / shapes
└── 72demos_validation/         # MATLAB-vs-Python demo harness
```

### 2.1 `mnpbem.geometry`

| File                     | Role                                                          |
|--------------------------|---------------------------------------------------------------|
| `particle.py`            | `Particle` base class (`verts`, `faces`, `nvec`, `area`, ...) |
| `comparticle.py`         | Multi-particle composite; tracks `inout`, `closed`, `eps`     |
| `comparticle_mirror.py`  | Mirror-symmetric variant; whitelisted `sym ∈ {x,y,z,xy,...}`  |
| `compoint.py`            | `Point` / `ComPoint` — observation-point bag with `inout`     |
| `polygon.py`             | 2D polygon — boolean ops, normalization, plotting             |
| `polygon3.py`            | Lift a 2D polygon into 3D; `plate`, `vribbon`, etc.           |
| `edgeprofile.py`         | Edge profile generator for rounded prism edges                |
| `mesh_generators.py`     | `trisphere`, `trirod`, `tricube`, `tritorus`, `trispheresegment`, `tripolygon`, `fvgrid` |
| `mesh2d.py`              | 2D Delaunay mesher — line-by-line port of MATLAB `mesh2d`     |
| `mesh2d_core.py` / `mesh2d_utils.py` | helpers split out for reuse                       |
| `shape_functions.py`     | Linear/curv face shape functions and quadrature mappings      |
| `connect.py`             | Particle-particle connectivity & edge stitching               |
| `compound.py`            | `@compound` MATLAB OOP class (10 public methods)              |
| `layer_structure.py`     | Stratified layer system + Sommerfeld integrators              |

### 2.2 `mnpbem.materials`

`EpsConst`, `EpsTable` (loads MATLAB `.mat` data files in `materials/data/`),
`EpsDrude`, `EpsFun`, plus the `epsfun(...)` factory.

### 2.3 `mnpbem.greenfun`

Quasistatic and retarded Green functions for free space, mirror symmetry,
and stratified layer systems.

| File                          | Role                                                |
|-------------------------------|-----------------------------------------------------|
| `compgreen_stat.py`           | Quasistatic Green function (G, F, Gp)               |
| `compgreen_ret.py`            | Retarded Green function (G, F, H1/H2 + Cartesian)   |
| `compgreen_stat_mirror.py` / `compgreen_ret_mirror.py` | Mirror-symmetric extensions |
| `compgreen_stat_layer.py` / `compgreen_ret_layer.py`   | Layered-medium extensions  |
| `greenstat.py` / `greenret_layer.py` / `greenret_refined.py` | Lower-level kernel + diagonal/off-diagonal refinement |
| `greentab_layer.py` / `compgreentab_layer.py` | Tabulated layered Green functions; bilinear/trilinear interp |
| `coverlayer.py`               | `+coverlayer` — refinement on layer-interface particles |
| `clustertree.py` / `hmatrix.py` | Cluster tree + hierarchical low-rank matrix      |
| `aca_compgreen_stat.py` / `aca_compgreen_ret.py` / `aca_compgreen_ret_layer.py` | ACA-accelerated Green functions |
| `aca_gpu.py` / `h_matrix_gpu.py` | GPU prototypes of ACA and H-matrix              |
| `_numba_kernels.py` / `_numba_ret_kernels.py` / `_numba_layer.py` | Numba JIT kernels for dense fill / interpolation |

### 2.4 `mnpbem.bem`

| File                   | Role                                               |
|------------------------|----------------------------------------------------|
| `bembase.py`           | Abstract `BemBase` + 5 factory functions           |
| `bem_stat.py`          | Direct quasistatic BEM solver                      |
| `bem_ret.py`           | Direct retarded BEM solver (2x2 block form)        |
| `bem_stat_layer.py`    | Quasistatic BEM with layer Green functions         |
| `bem_ret_layer.py`     | Retarded BEM with layer Green functions            |
| `bem_stat_mirror.py` / `bem_ret_mirror.py` | Mirror-symmetric direct solvers |
| `bem_iter.py`          | Common GMRES/iterative solver scaffolding          |
| `bem_stat_iter.py` / `bem_ret_iter.py` / `bem_ret_layer_iter.py` | Iterative + ACA H-matrix variants |
| `bem_stat_eig.py` / `bem_stat_eig_mirror.py` | Eigenmode-based quasistatic solver |
| `bem_layer_mirror.py`  | Mirror x layer combination                         |
| `plasmonmode.py`       | Plasmon eigenmode extraction (left/right pairing)  |
| `solver_factory.py`    | `bem.solver(...)` dispatcher                       |
| `matlab_bem.py`        | MATLAB Engine adapter (opt-in validation backend)  |

### 2.5 `mnpbem.simulation`

| File                           | Role                                              |
|--------------------------------|---------------------------------------------------|
| `planewave_*.py`               | Stat / Ret / Mirror / Layer plane-wave excitation |
| `dipole_*.py`                  | Stat / Ret / Mirror / Layer dipole excitation     |
| `eels_base.py` / `eels_stat.py` / `eels_ret.py` | EELS excitation + loss      |
| `meshfield.py` / `_meshfield_numba.py` / `meshfield_fmm.py` | Near-field evaluator on grids; Numba + optional FMM3D |
| `electronbeam_factory.py` / `dipole_factory.py` / `planewave_factory.py` | Construction helpers |
| `retarded_utils.py`            | shared helpers for the ret excitation classes     |

### 2.6 `mnpbem.spectrum`

`SpectrumStat`, `SpectrumRet`, `SpectrumStatLayer`, `SpectrumRetLayer`, plus
the `spectrum(...)` factory. The retarded variants embed the MATLAB
`pinfty256.bin` reference far-field for bit-identical extinction/scattering.

### 2.7 `mnpbem.mie`

Reference Mie solver (`MieStat`, `MieRet`, `MieGans`) used by the validation
suite to cross-check spherical-particle results without going through BEM.

### 2.8 `mnpbem.misc`

Math primitives (`matmul`, `vec_norm`, `pdist2`, `bdist2`, ...), Gauss-Legendre
nodes (`lglnodes`, `lgwt`), `QuadFace` polar quadrature, options
(`bemoptions`, `getbemoptions`), units (`EV2NM`, `BOHR`, `HARTREE`, `FINE`),
`BemPlot`/`coneplot` plotting, `IGrid2`/`IGrid3` interpolators,
`ValArray`/`VecArray` containers.

### 2.9 `mnpbem.utils`

| File              | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `matlab_compat.py`| Bit-identical MATLAB primitives (`matan2`, `mexp`, `msqrt`, `mlinspace`, `m_unique`, ...) |
| `gpu.py`          | `lu_factor_dispatch` / `lu_solve_dispatch` / `solve` / `eigh` / `matmul` over CPU and CuPy |
| `multi_gpu.py`    | Wavelength-batched dispatch across local GPUs (subprocess-per-GPU) |
| `mpi_dispatch.py` | Multi-node wavelength dispatch on top of `multi_gpu` (mpi4py) |
| `parallel.py`     | `compute_spectrum`, `compute_spectrum_parallel` (CPU)       |
| `quadface.py` / `quadrature.py` | shared quadrature helpers                     |
| `matlab_ode45.py` | 1:1 re-implementation of MATLAB `ode45` step controller     |
| `constants.py`    | `EV2NM` and friends                                         |

## 3. Key design decisions

### 3.1 `matlab_compat` — why it exists

`np.linspace`, `np.arctan2`, `np.exp`, `np.log`, `np.sqrt` and friends each
differ from MATLAB by up to 1 ULP because of different accumulation order or
fdlibm vs. MKL implementations. A 1 ULP drift in `theta` or `linspace`
propagates through `p @ rot` and `np.unique`, eventually changing the mesh
**topology** (different vertex ordering produces different face adjacency,
which changes BEM matrix sparsity patterns). To eliminate this drift the
port loads MATLAB's own `libmwmathutil.so` via `ctypes` when available and
calls its fdlibm-derived primitives directly. When MATLAB is not installed
the wrappers fall back to numpy.

This is the single most important design choice for parity with MATLAB:
without it most demos regress to "warn" or "BAD" because of mesh-induced
divergence rather than algorithm bugs.

See: `mnpbem/utils/matlab_compat.py`, `docs/MESH2D_FP_LIMIT.md`.

### 3.2 Numba JIT (default ON)

- Activation: env var `MNPBEM_NUMBA` defaults to `1`; set `0` to disable.
- Used for: dense Green-function fill (`compgreen_stat`, `compgreen_ret`,
  per-particle distance kernels), bilinear/trilinear interpolation in
  `greentab_layer`, ACA inner loops, and the `meshfield` per-wavelength
  dense-Green evaluator.
- Kernels are decorated `@njit(cache=True)`. `fastmath` is **off** because
  we observed it breaks IEEE-754 sign-of-zero handling, which in turn
  changes results in off-diagonal panels of `compgreen_ret`.
- This is the M4 N1-N6 work; without it the CPU path lags MATLAB on dense
  meshes.

### 3.3 GPU dispatch (opt-in)

- Activation: env var `MNPBEM_GPU=1` (default OFF — explicit opt-in to avoid
  surprising users without CUDA).
- Threshold: `MNPBEM_GPU_THRESHOLD` (default 1500). Below the threshold,
  scipy CPU LU is faster than the host↔device round trip.
- Layer-Green specialization: `MNPBEM_GPU_LAYER` / `MNPBEM_GPU_LAYER_THRESHOLD`
  control the GEMM fast path inside the layer Sommerfeld batcher.
- Native cupy outputs: `MNPBEM_GPU_NATIVE=1` keeps refined Green tensors on
  the device end-to-end through the BEMRet pipeline (`Sigma1 = H @ G^-1`
  is solved with cuSolver `lu_solve` directly).
- All GPU calls return numpy arrays unless the caller is already cupy-aware,
  so the public API does not change when the env var is set.
- See `mnpbem/utils/gpu.py` for the dispatcher and `bem_*_iter.py`,
  `bem_*_mirror.py`, `bem_stat_eig*.py` for the consumers.

### 3.4 Multi-GPU and multi-node

Wavelength sweeps are embarrassingly parallel for the BEMRet solve: each
λ builds and solves an independent system. This is the only axis the port
parallelises across processes:

- `mnpbem.utils.multi_gpu.solve_spectrum_multi_gpu` — splits λ across local
  GPUs, one subprocess per CUDA device, `CUDA_VISIBLE_DEVICES` pinning,
  results merged through a `multiprocessing.Queue`.
- `mnpbem.utils.mpi_dispatch.solve_spectrum_mpi` — adds an MPI rank axis on
  top: each rank gets a wavelength slice, then internally calls
  `solve_spectrum_multi_gpu`. Falls back to a serial CPU loop if the rank
  has no GPU and to `solve_spectrum_multi_gpu` if `mpi4py` is missing or
  the world has size 1.

### 3.5 ACA (Adaptive Cross Approximation) and H-matrix

Far-field BEM blocks are compressed with rank-revealing ACA on a binary
cluster tree. Defaults match MATLAB:

- `htol=1e-6` (Frobenius approximation tolerance)
- `kmax=[4, 100]` (rank bounds)
- `cleaf=200` (leaf cluster size)
- `ACATOL=1e-10` (cross-pivot abort)
- `fadmiss` is k-aware for retarded problems; for layered Green functions
  it accepts a per-λ override (`make_kaware_fadmiss`).

Consumers: `BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`. Direct solvers
keep the dense path for now; the iterative + ACA path is the recommended
route for meshes above ~5000 faces.

The complex128 ACA inner loops are Numba-JITted (`hmatrix.py`) and
`fadmiss` admissibility takes the wavelength k into account
(`make_kaware_fadmiss`). An experimental GPU port lives in
`aca_gpu.py` / `h_matrix_gpu.py` (see `docs/H_MATRIX_GPU.md`).

### 3.6 Mesh quadrature

Surface integrals use the polar-quadrature scheme of MATLAB MNPBEM
(`QuadFace` + diagonal refinement via `refinematrix` / `refinematrixlayer`).
Triangles and quads are refined separately; refinement points are picked
by MATLAB-compatible `pdist2` (we re-implemented the algorithm because
`scipy.spatial.distance` reorders pairs of equal distance differently and
that reorder propagates through the mesh).

### 3.7 Sommerfeld integrator

The layer-Green tabulation needs the radial Sommerfeld integral over the
complex k-plane. Two backends:

- Default: composite Gauss-Legendre on adaptive panels with batched RHS
  (vectorised across query points), faster than MATLAB `ode113` and
  trivially differentiable.
- ODE backend (opt-in via option): `scipy.solve_ivp` with a Numba-compiled
  RHS, plus a custom `matlab_ode45.py` step controller for cases where
  exact MATLAB step pattern is required.

### 3.8 Direct retarded solver — block matrix form

`BEMRet` rebuilds the MATLAB `initmat.m` 2x2 structured block system rather
than the older single-monolith form. This was required to match MATLAB on
multi-component particles and on layer-coupled cases, and it lets the
matrix assembly stage be eagerly pushed to GPU (Lane A2 in M4).

### 3.9 Why we deviate from MATLAB (intentionally)

| Item | Difference | Reason |
|---|---|---|
| FP roundoff | up to ~ULP in some demos | MATLAB libmwmathutil vs. Python math libs; minimised by `matlab_compat` but not always eliminated (see `MESH2D_FP_LIMIT.md`) |
| ACA tie-break | Slightly different pivot order in degenerate panels | No measurable impact on Green-function accuracy; left as numpy default for clarity |
| Sommerfeld integration | Default is GL panels, not `ode113` | 5-10× faster, same accuracy at default tolerance |
| Direct BEMRet | 2x2 block reformulation | Required for multi-particle layer cases; equivalent to MATLAB |
| `MNPBEM_GPU` default OFF | Opt-in instead of opt-out | Avoids initialising CUDA when the user did not ask for it |
| `pinfty` default | MATLAB `pinfty256.bin` for ret spectra | MATLAB ships a fixed reference far-field; using anything else introduces a 0.5-1% bias on extinction |
| `EV2NM` constant | `1/8.0655477e-4` | Match MATLAB `Misc/units.m` exactly (Wave 28 D) |

### 3.10 Why we *do not* deviate

The single biggest pull on the port has been resisting the temptation to
"fix" MATLAB quirks. Several BAD demos turned out to be Python being more
correct than MATLAB; we matched MATLAB anyway, because parity is the
acceptance criterion. Examples:

- `_minrectangle` tie-break uses MATLAB's strict `<` comparison instead of
  Python's `<=`, even though either is mathematically valid — so that
  regular-N-gon meshes orient identically.
- `dipoleretlayer` keeps MATLAB's `pinfty` bug for compatibility on
  `demodipret*` (Wave 22 B).
- `intbessel` / `inthankel` use MATLAB's specific multiplication order so
  the layer Sommerfeld values agree to ULP (Wave 49).

If you find a MATLAB behaviour that looks wrong, please flag it before
"fixing" it — the test suite will likely regress.

## 4. Performance summary

See `docs/PERFORMANCE.md` for the numbers. The strategy is documented in
`docs/PERFORMANCE_STRATEGY.md` (M4 four-tier roadmap). The short version is:

- Tier 1 (`scipy` `check_finite=False`/`overwrite_a=True`): 10-20% LU gain.
- Tier 2 (multi-RHS wavelength batching, GMRES iterative path).
- Tier 3 (Numba kernels): 5-50× on the dense Green fill.
- Tier 4 (GPU + H-matrix + FMM3D): order-of-magnitude on large meshes.

Headline numbers: CPU geometry-build speedup 2.21×, GPU geometry-build
speedup 3.60×, vs. the pre-M4 baseline.

## 5. Testing

See `tests/regression/README.md` for the regression infrastructure.
The validation hierarchy is:

- `tests/` — unit tests (pytest). 600+ tests covering Mie, EELS, layer,
  mirror, iterative, edge cases.
- `validation/` — sphere-and-rod numerical cross-checks against MATLAB,
  driven from `validation/_common`. Each subdirectory is one validation
  axis (Mie, BEMStat, BEMRet, BEMStatLayer, BEMRetLayer, mirror,
  eigenmode, iterative, dipole, dipole-layer, EELS, near-field, shapes).
- `72demos_validation/` — full MATLAB-vs-Python demo harness driving the
  72 demo scripts and producing the `compare_smart_v3.py` accuracy table.

CI runs unit tests on every commit and regression suites on tagged
releases (see `.github/workflows/`).

## 6. Pointers for new contributors

- Adding a particle shape: extend `mnpbem.geometry.mesh_generators`, mirror
  the MATLAB `+particles/` files line-by-line, and add a Mie cross-check
  if it has a closed-form reference.
- Adding a Green function: subclass the appropriate `CompGreen*` and
  register a factory entry in `mnpbem.greenfun.greenfunction.greenfunction`.
- Adding a BEM solver: extend `BemBase`, mirror the MATLAB `@bem*` class,
  register in `mnpbem.bem.solver_factory.solver`, and add the regression
  hook in `tests/`.
- Adding an excitation: place it under `mnpbem.simulation`, expose it in
  `simulation/__init__.py`, and register the factory.

When in doubt, the rule is: **read the MATLAB source, port it line-by-line,
and only deviate when explicitly justified in this document**.

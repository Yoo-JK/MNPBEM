# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.0] - 2026-05-11

### Fixed

- **GPU correctness audit (17-18 bug)**: results of a 5-agent (A1-A5)
  parallel audit + Phase 1 integration audit. Every (BEM solver ×
  excitation × layer / mirror) combination matches the CPU baseline in
  GPU mode to 1e-7 per face / 1e-9 per cross section.

#### A1 — BEMRet / BEMRetLayer

- Added a regression guard for the disjoint-dimer non-uniform eps edge case.

#### A2 — BEMRetIter / BEMRetLayerIter

- dense path GPU backend mix fix.

#### A3 — BEMStat family

- `clear()` stale-cache + GPU LU leak.

#### A4 — Mirror BEM (4 types)

- Host promotion of `CompGreenRetMirror` / `CompGreenStatMirror` eval cupy results.
- `BEMStatEigMirror` half-mesh index range fix.
- `BEMLayerMirror` dummy assertion.

#### A5 — Excitation runners (17 types)

- Host materialization of the cupy sig members of PlaneWaveStat /
  DipoleStat / DipoleRet / DipoleStatLayer / DipoleRetLayer / EELSStat /
  EELSRet.

#### Phase 1 (integration)

- **CompGreenRet `_matmul` / `_cross`**: added host promotion where a
  cupy operand used to cause a silent zero return.
- **BEM solver `solve()` return**: convert cupy sig members to
  user-facing host (BEMRet, BEMStat, BEMRetLayer, BEMStatLayer,
  BEMRetMirror, BEMStatMirror). Makes `np.asarray(sig.sig1)`-style calls
  in demo scripts safe.
- Added a `lu_solve_native` GPU LU + cupy b residency regression guard.
- New EELS × Layer integration smoke test (EELS × Mirror is
  unsupported — a combination that does not exist in the MATLAB Demo set
  either).

### Verification

- 72 demo regression (`/tmp/mnpbem_demo_comparison`):
  v1.7 GPU mode BAD 0 / 72, perf 65 / 72.

## [1.6.4] - 2026-05-08

### Fixed

- **HMatrix matvec backend/dtype consistency** (Phase 1): when the input
  was numpy but some ACA blocks resided on cupy, there was a path inside
  `mtimes_vec` where the backend of the host result and the GPU blocks
  got mixed. It now sniffs whether the blocks reside on the GPU to unify
  the destination backend, then converts back to the input backend at the
  end, guaranteeing numpy input → numpy output. A defensive
  `_ensure_numpy` wrapper was added to the 6 HMatrix matvec call sites in
  `bem_ret_iter._afun` (a no-op on the normal path, ensuring host
  slice-assignment safety).
- Impact: prevents random failures / slowdowns caused by numpy/cupy mix
  on the GPU + iter+hmat+precond path. On a 1176-face Au@Ag dimer,
  ext = 25759.0950 (CPU/GPU/all modes agree, rel diff 1e-15).

### Added

- **`MNPBEM_AGGRESSIVE_GPU_MFUN=1` environment variable** (Phase 2
  opt-in): uploads the dense Sigma1 / L_diff / L1 matrices of
  `BEMRetIter._mfun` to the GPU so that the Sigma1·v / L·v calls made
  during GMRES iteration are dispatched via cupy matmul. When memory
  capacity is insufficient, it falls back automatically tier by tier
  (Sigma1 only → Sigma1 + L_diff → all). With the flag off (default) it
  is bit-identical to v1.6.3.
- Measurements:
  - 5400-face Au@Ag dimer: 111s → 103s (1.08x speedup, full
    Sigma1+L_diff+L1 resident on GPU).
  - 12672-face Au@Ag dimer: 608s → 620s (flat). At the 12672-face
    scale, GMRES converges in 1 outer iter (mfun call =
    3 times) → mfun GPU dispatch is meaningless. The benefit shows up
    on wider geometry / other sweeps where GMRES takes many iterations.
- Regression guard: `mnpbem/tests/test_mfun_gpu_dispatch.py` (verifies
  flag off / on rel diff < 1e-10 on a 1136-face Au@Ag dimer).

## [1.6.3] - 2026-05-05

### Changed

- **BEMRetIter precond GPU LU hybrid pipeline**:
  the existing preconditioner, which routed the entire N >= 8000 branch
  to host scipy, was restructured into a cuSOLVER getrs LU + host MKL
  matrix-multiply hybrid. The LU factorization of G/Delta/Sigma_mat is
  run on the GPU and preserved with a `('gpu', lu, piv)` tag, so that the
  GMRES iterate of `_mfun` is routed through cuSOLVER getrs (~5 ms /
  call). The G^{-1} / Sigma1 / L1 GEMMs are delegated to host scipy
  because the 132-core MKL pipeline is faster.
- `_gpu_precond_capacity_ok(N)`: attempts the GPU path after checking
  available GPU memory against 7 N² × 16 byte; if insufficient it falls
  back to host automatically, and it also falls back safely to host scipy
  even when `_GpuPrecondOOM` is raised.
- `MNPBEM_GPU_PRECOND_HOST_THRESHOLD` env (default 32768) can force a
  host fallback.

## [1.6.2] - 2026-05-02

### Fixed

- **VRAM share env vars wiring** (found in the C agent v1.6.0/v1.6.1
  Tier-3 benchmark): even with `MNPBEM_VRAM_SHARE_GPUS=N` +
  `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg` set, the BEM solvers
  (`bem_ret_iter.py`, `bem_stat_layer.py`, `bem_ret_layer.py`, etc.) did
  not explicitly pass the `n_gpus` kwarg when calling
  `lu_factor_dispatch` → cuSolverMg failed to activate, and the
  12672-face dense LU hit a 49 GB single-GPU OOM. fix:
  - Added the `mnpbem/utils/gpu.py::_vram_share_env_defaults()` helper.
  - When the `n_gpus` / `backend` / `device_ids` kwargs are not
    specified, `lu_factor_dispatch` / `solve_dispatch` now read the env
    vars (`MNPBEM_VRAM_SHARE_GPUS`, `MNPBEM_VRAM_SHARE_BACKEND`,
    `MNPBEM_VRAM_SHARE_DEVICE_IDS`) automatically. Explicit kwargs always
    take precedence.
  - New support for `MNPBEM_VRAM_SHARE_DEVICE_IDS` (comma-separated).
  - `MNPBEM_VRAM_SHARE=0` master switch can force it off.
  - Compatible with the existing `bem_ret.py::_vram_share_lu_kwargs`
    explicit-pass path (no behavior change, since kwargs take
    precedence).

### Added

- `mnpbem/tests/test_vram_share_env_wiring.py` (15 tests, verifies
  dispatch routing even on a CPU-only host using a mock cuSolverMg
  backend)

### Verified

- 4× RTX A6000 smoke: with just the `MNPBEM_VRAM_SHARE_GPUS=2`
  environment variable, a 256×256 complex LU `lu_factor_dispatch` enters
  the mgpu branch, residual 9.3e-16

### Known issues

- Tier-3 12672-face cuSolverMg formal batch benchmark not yet run (M5+
  follow-up)

## [1.6.1] - 2026-05-02

### Added

- `mnpbem/greenfun/_numba_refine.py` — Numba JIT kernels for BEM assembly hot loops
- `mnpbem/tests/test_compgreen_ret_layer_multi.py` — multi-particle layer Green test

### Fixed

- `compgreen_ret_layer.py:651` shape mismatch (Au@Ag core-shell on substrate)
- BEM assembly single-thread CPU bottleneck (Numba JIT applied)

### Performance

- ~70% reduction in 3k+ face mesh assembly (Numba)

### Known issues

- VRAM share env vars wiring incomplete — v1.6.x follow-up
- Tier-3 timing benchmark not yet formally measured — batch retry
  recommended

## [1.6.0] - 2026-05-02

### Added

- `BEMRetIter` `schur_eps_form='auto'` option — automatic detection of non-uniform eps
- `SchurIterOperator` `eps_form='pointwise'|'operator'` branch + threshold 4096
- `BEMRetLayerIter` operator-form (substrate + iter + multi-material)
- pymnpbem `--str-conf <X.py> --sim-conf <Y.py> --verbose` CLI
- pymnpbem `mesh_density` (nm) takes priority (over n_per_edge)
- New tests: test_b_schur, test_iter_convergence_layer, test_mesh_density_priority, test_cli_str_sim

### Fixed

- 60-face nonlocal+schur+iter+hmat 25 min GMRES hang → converges in 6:51
- BEMRetLayerIter multi-material drift (Au+Ag dimer on glass)

### Performance

- (perf measurements are in PERFORMANCE.md §11.5)

### Known limits / Deferred

- BEM assembly perf (single-thread CPU bound) — v1.6.x follow-up
- compgreen_ret_layer multi-particle indexing — v1.6.x follow-up

## [1.5.2] - 2026-05-02

### Fixed

- **Bug 5 — `HMatrix.full()` numpy/cupy interop** (`mnpbem/greenfun/hmatrix.py:374`).
  In v1.5.1, when `MNPBEM_GPU_NATIVE=1` is active and `CompGreenRet`
  returns a cupy ndarray, `HMatrix.val[i]` becomes cupy, but `full()`
  fails with `TypeError: Implicit conversion to a NumPy array is not
  allowed.` when it implicit-casts a cupy slice into a host numpy buffer.
  fix: `full(xp=None)` auto-detects cupy from val/lhs/rhs → allocates
  `mat` on the cupy backend and unifies the device with a per-block
  numpy ↔ cupy conversion helper. The caller can also force `xp=np` or
  `xp=cupy`.
- **Bug 6 — `_plus_hmat` / `_truncate_block` backend unification**. When
  the region (0,0) val is cupy and the region (1,0) val is numpy (e.g.
  multi-region + cross-connectivity as in Au@Ag), `G11 - G21` fails with
  `Unsupported type <numpy.ndarray>`. fix: the `_same_backend(a, b)`
  helper promotes both sides to cupy if either is cupy, and
  `_truncate_block` QR/SVD also dispatches with `xp=cupy` when lhs is
  cupy.
- **Tier-3 12672-face Au@Ag GPU full validation passed** —
  the `MNPBEM_GPU=1 + iter+hmat+precond + multi-GPU wavelength-split`
  path completed end-to-end successfully for the first time. Resolves the
  BAD grade of v1.5.0/v1.5.1.

### Added

- `mnpbem/tests/test_hmatrix_full_consistency.py` — 8 tests: cupy/numpy
  full() agreement, forced xp argument, mixed blocks, the realistic ACA
  dense=cupy/lhs=numpy scenario, and BEMRetIter._init_matrices GPU
  end-to-end smoke.

### Backward compatibility

100% backward compatible with v1.5.1. Even though the `HMatrix.full()`
signature is extended to `full(xp=None)`, the default is auto-detect, so
no existing caller needs to change.

## [1.5.1] - 2026-05-02

### Fixed

- **4 mnpbem GPU bug fixes** (for Au@Ag GPU full-mesh acceptance)
  - `mnpbem/bem/bem_ret.py` (Bug 1) — unified the `G1i/G2i/Deltai` LU
    backend on the CPU init path.
  - `mnpbem/bem/bem_ret_iter.py:264` (Bug 2) — removed the `Sigma1 = H1 @
    G1i` numpy/cupy mix (dense LU preconditioner build).
  - `mnpbem/greenfun/hmatrix.py:250` (Bug 3) — guarded against
    cupy/numpy index mixing in `_aca_block` `cols[pivot_col_local]`.
  - `mnpbem/utils/multi_gpu.py::_worker` (Bug 4) — extended to receive
    the `BEM` class explicitly via a `bem_class` argument. Previously the
    wavelength-split path forced `BEMRet` (dense) even when
    `simulation.type=ret_iter`.
- **BEMRetIter operator-form eps fix** — Au@Ag (multi-material) iter
  drift 70% → 0%. An iter-formulation algorithm bug in the non-uniform
  eps + cross-connectivity case. Now matches the dense `BEMRet`
  `L1 = G1·diag(eps1)·G1⁻¹`.
- **`pymnpbem_simulation` `simulation.type=ret + iterative=true`
  automatic routing** (Issue A) — in-place rewrite to the `_iter` variant
  in both `dispatch_single_node` and `convert_py_to_yaml`.
- **`pymnpbem_simulation.dispatch.multi_gpu`** —
  the wavelength-split path derives `bem_class` from `simulation.type`
  and passes it to `solve_spectrum_multi_gpu(bem_class=…)`. It also
  propagates `compute.iter.{hmatrix, preconditioner, schur, htol, tol,
  maxit}` to the worker BEM. This is a wrapper-side follow-up to Bug 4,
  where up through v1.5.0 the multi-GPU path always forced `BEMRet`.

### Added

- `mnpbem/tests/test_gpu_cupy_consistency.py` — 14 tests, GPU
  cupy/numpy interop regression.
- `mnpbem/tests/test_iter_convergence.py` — 8 tests, BEMRetIter
  operator-form regression (case_g 1136-face Au@Ag).

### Known issues

- `BEMRetLayerIter` needs the same operator-form eps patch — the
  substrate + iter combined scenario. v1.5.2 or v1.6 follow-up.
- `mnpbem/tests/test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur`
  hangs in the current environment — a separate investigation item
  (isolated on its own in the regression stats; the other 10 schur_iter
  tests PASS).
- **Bug 5 — cupy/numpy mix in `mnpbem/greenfun/hmatrix.py:374`
  `HMatrix.full()`** — on the `BEMRetIter(hmatrix=True,
  preconditioner='auto')` path, during the dense LU preconditioner build,
  `_compress` → `hmat.full()` fails trying to implicit-cast a cupy
  `self.val[i]` into a numpy `mat`. Found in the Tier-3 12672-face Au@Ag
  GPU iter scenario. Same category as the α 4-GPU bug fixes of v1.5.1.
  Needs cleanup via `xp.zeros` / `cupy.asnumpy` dispatch in a follow-up
  (v1.5.2).

## [1.5.0] - 2026-05-03

### Added

- **H-matrix LU preconditioner** for iterative BEM solvers (Lane E2 follow-up)
  - `BEMRetIter(p, hmatrix=True, preconditioner='auto', htol_precond=1e-4)`,
    same for `BEMStatIter`.
  - 256-face sphere GMRES iter 55 → 1 (55× reduction).
  - modes: `auto` (default ON when `hmatrix=True`), `none`, `hlu_dense`,
    `hlu_tree`.
  - Implementation: `mnpbem/bem/preconditioner.py` (`HMatrixLUPreconditioner`).
- **Schur complement × Iterative BEM** integration
  - `BEMRetIter(p, schur=True, hmatrix=True)` (both can be ON; through
    v1.4 this was a `NotImplementedError`).
  - `SchurIterOperator` `LinearOperator`:
    `A_eff(x_c) = A_cc x_c − A_cs · A_ss⁻¹ · A_sc x_c`.
  - `g_ss_solver`: `lu_dense` / `gmres` / `callable` / `auto`.
  - 568-face nano-gap nonlocal: solve 21.17s → 16.65s (21.3% reduction).
  - Implementation: `mnpbem/bem/schur_iter_helpers.py`.
- **51 pre-existing test failures cleanup** (51 → 0)
  - Deleted 11 stale, fixed 38 infra, 1 fix, 1 update.
- **jk-config 3 follow-up issues** fix
  - Issue 2: N-layer generalization of the multi-shell `core_shell` builder.
  - Issue 3: Metal substrate `IndexError` (`LayerStructure._enlarge`
    boundary clip).
  - Issue 4: automatic conversion of field-only config
    (`py_to_yaml._redirect_field_only_simulation`).
- Exposed the `iter.preconditioner`, `iter.schur` options of `pymnpbem_simulation`.

### Changed

- (none — backward compatible with v1.4.0)

### Performance

- 256-face sphere GMRES: iter 55 → 1, wall 1.03s → 0.82s.
- 568-face nonlocal Schur×Iter: 21.3% time reduction.
- 25k face: the true value of alpha-2 H-tree LU requires a Sigma/Delta
  H-matrix reconstruction — v1.6+ scope.

### Known limits

- `BEMRetIter`'s 8N×8N coupled system → limited benefit from G-only
  H-tree LU alone (alpha-2 ≈ alpha-1 dense fallback).
- The truly memory-friendly preconditioner for 25k face = Sigma/Delta
  H-matrix reconstruction = v1.6+.
- `BEMStatIter` tree mode → the diagonal term breaks, so it falls back to
  dense (one-time log).

## [1.4.0] - 2026-05-XX

### Added

- **Split CPU/GPU install** — refined pyproject extras
  (`gpu` / `mpi` / `fmm` / `all` / `dev` / `test` / `docs`).
  - `pip install mnpbem` (CPU only, lightest; no cupy dependency).
  - `pip install mnpbem[gpu]` (includes cupy-cuda12x, NVIDIA GPU acceleration).
  - `pip install mnpbem[all]` (gpu + mpi + fmm, all features).
  - No separate wheel split — a single wheel + extras is the standard
    PyPI pattern.
- **Runtime GPU auto-detection** —
  `mnpbem.utils.gpu.has_gpu_capability(verbose=True)` checks cupy import
  + CUDA driver + GPU device availability and returns a `bool`. Prints a
  friendly fallback message when unavailable.
- **`mnpbem.utils.gpu.get_install_hint()`** — a helper that suggests the
  `pip install mnpbem[gpu]` command appropriate for the user's
  environment.
- **`docs/INSTALL.md`** — a per-scenario install guide (CPU only / GPU /
  multi-GPU / multi-node / development environment).

### Changed

- Simplified the `README.md` `Installation` section — links to
  `docs/INSTALL.md` for details.
- (none breaking — 100% backward compatible with v1.3.0)

### Performance

- (no perf impact — packaging improvement)

## [1.3.0] - 2026-05-XX

### Added

- **H-matrix BEMRetIter integration** (Lane E2 follow-up).
  - New `BEMRetIter(p, hmatrix=True)`, `BEMStatIter(p, hmatrix=True)`
    options. ACA H-tree compression + GMRES resolves the dense LU OOM
    (50+ GB) of large 25k+ face meshes.
  - Both memory and matvec scale as `O(N log N)` — 25k face fits on a
    single GPU. Combined with VRAM share (v1.2.0), 56k+ face becomes
    attainable too.
  - Exposed parameters: `htol` (ACA truncation, default 1e-6),
    `kmax` (ACA rank upper bound, default `[4, 100]`),
    `cleaf` (leaf cluster size, default 200).
  - `BEMRetLayerIter + hmatrix` is unsupported (`NotImplementedError`) —
    no cover-layer + planar substrate combined scenario.
  - Simultaneous `BEM*Iter + Schur (v1.2.0)` is also unsupported —
    H-matrix + Schur integration is follow-up work.
- **`iter.hmatrix: 'auto'` option in the `pymnpbem_simulation` iter
  runner** — automatically activates H-matrix BEMRetIter for 5000+ face
  meshes.

### Changed

- (none — backward compatible with v1.2.0)

### Performance

- 25k face dimer: dense LU OOM (~50+ GB peak) →
  H-matrix BEMRetIter fits on a single GPU (for measured figures see
  `docs/PERFORMANCE.md` §11).
- per-wl time: dense BEMRet vs H-matrix BEMRetIter comparison
  (`docs/PERFORMANCE.md` §11).
- Accuracy: dense vs H-matrix BEMRetIter rel `< 1e-4` (htol-based).

## [1.2.0] - 2026-05-XX

### Added

- **Schur complement** for cover-layer BEM — 50% nonlocal memory
  reduction, 30% faster LU solve.
  - `BEMStat(p, schur=True)`, `BEMRet(p, schur=True)` options.
  - Schur-eliminates the cover layer (`EpsNonlocal`) variables to solve a reduced matrix.
  - The result is mathematically equivalent to the standard formulation (rel < 1e-12).
  - `schur='auto'`, or the wrapper auto-detects the cover layer.
  - Implementation: `mnpbem/bem/schur_helpers.py`.
- **VRAM share** — 1 worker handles a large single computation by pooling multi-GPU memory.
  - cuSolverMg backend (NVIDIA's official multi-GPU LU API).
  - A 25k+ face dense LU (50+ GB) fits in a 2 GPU pool (96 GB).
  - environment variables `MNPBEM_VRAM_SHARE_GPUS=N`,
    `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg`.
  - Supports direct calls to `mnpbem.utils.gpu.lu_factor_dispatch(A, n_gpus=N)`.
  - `compute.n_gpus_per_worker > 1` in `pymnpbem_simulation` activates it automatically.
  - Implementation: `mnpbem/utils/multi_gpu_lu.py`.

### Changed

- (none — backward compatible with v1.1.0)

### Performance

- nonlocal cover-layer simulations: memory 4× → ~2× (when Schur is applied).
- 25k+ face dense LU: single-GPU OOM → possible with a multi-GPU pool.
- (for figures see `docs/PERFORMANCE.md`)

## [1.1.0] - 2026-05-XX

### Added

- `EpsNonlocal` — hydrodynamic Drude nonlocal dielectric function
  (cover-layer formulation).
  - Yu Luo et al., PRL 111, 093901 (2013) effective-layer mapping.
  - Factory methods: `EpsNonlocal.gold()`, `.silver()`, `.aluminum()`,
    `.from_table(path)`.
  - Helper: `make_nonlocal_pair(metal, eps_embed, delta_d, beta)` →
    `(core, shell)` tuple.
  - 18 unit tests; bit-identical to the MATLAB `demospecstat19` reference
    formula at `rtol = 1e-12`.
- `BEMRet` now accepts a `refun` parameter (parity with `BEMStat`) — the
  retarded path can be combined with the cover-layer integration.
- `pymnpbem_simulation` wrapper updated: nonlocal workaround replaced
  with the official `EpsNonlocal` call path.

### Changed

- (none — backward compatible with v1.0.0)

### Performance

- (no performance impact — algorithmic feature only)

## [1.0.0] - 2026-05-XX

First production release of the PyMNPBEM. Pure-Python distribution
of Hohenester & Trügler's MATLAB MNPBEM toolbox, validated against MATLAB on
50 + 22 official demos and on the sphere/rod/dimer cross-checks.

### Milestone 1 — Demo complete

Goal: bring the 50 official MATLAB MNPBEM demos to MATLAB-Python parity, then
extend to the 72-demo extended harness. Reduce the BAD-category demos from
12 to 0 and lift machine-precision matches from 0 to 55 of 72 (76%).

Highlights:

- Quasistatic and retarded BEM solvers ported (`BEMStat`, `BEMRet`).
- Mirror-symmetric solvers (`BEMStatMirror`, `BEMRetMirror`).
- Layered-medium solvers (`BEMStatLayer`, `BEMRetLayer`) including
  Sommerfeld-integrated layer Green functions and tabulated interpolators.
- Iterative solvers (`BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`) with
  GMRES + ACA H-matrix acceleration.
- Eigenmode solver (`BEMStatEig`) with bi-orthogonal pairing.
- Plane-wave / dipole / EELS excitations across all of stat/ret/mirror/layer.
- Mesh generators: `trisphere`, `trirod`, `tricube`, `tritorus`,
  `trispheresegment`, `tripolygon`, plus `Polygon`, `Polygon3`, `EdgeProfile`.
- 2D mesher: line-by-line port of MATLAB `mesh2d`.
- Reference Mie solver (`MieStat`, `MieRet`, `MieGans`).

Representative commits:

- `d8d396e` `merge: apply T1 scipy lu_factor/solve check_finite=False/overwrite flag`
- `0f7637d` `mesh2d._minrectangle: match MATLAB strict < tie-break` (BAD 12 → 8)
- `af69b7d` `matlab_compat: bit-identical port of all MATLAB libmwmathutil transcendental functions`
- `0320f9e` `matlab_compat.matan2: bit-identical implementation via direct MATLAB libmwmathutil call`
- `b8fadd4` `all BEM solvers: replace np.linalg.inv() → scipy.linalg.lu_factor/lu_solve`
- `a371b30` `rewrite the BEMRetLayer solver into the same structured 2x2 block matrix system as MATLAB initmat.m/mldivide.m`
- `ac988d8` `implement all 29 missing methods: achieve 100% feature parity with MATLAB MNPBEM`

### Milestone 2 — Missing API porting

Goal: cover the MATLAB classes/functions that were not part of the demo
critical path but are part of the public surface.

- `ComParticleMirror` mirror whitelist + `sym` validation.
- `CompGreenStatMirror` / `CompGreenRetMirror` full ports.
- `CompGreenStatLayer` Cartesian derivatives (`H1p` / `H2p`).
- `CompGreenTabLayer` multi-tab dispatch (`_MultiGreenTabLayer`).
- `MeshField` near-field evaluator on `IGrid2` / `IGrid3` grids.
- `coverlayer` package — refinement on layer-interface particles.
- `compound` (`@compound`) — 10 public methods.
- `Polygon` boolean / normalize / symmetry helpers.
- `polymesh2d` outer + hole multi-polygon support.
- ACA-accelerated Green functions (`ACACompGreenStat`, `ACACompGreenRet`,
  `ACACompGreenRetLayer`).
- ClusterTree + HMatrix data structures.
- `plasmonmode` left/right eigenvector pairing for complex eigenvalues.
- BemPlot / coneplot / arrowplot visualisation.
- `epsfun` factory and the `eps_table` data files.

Representative commits:

- `0fcd647` `docs: Add comprehensive MNPBEM API audit report (MATLAB → Python)`
- `c510e89` `Implement closed surface regularization in CompGreenRet`
- `7024e3f` `implement MeshField class: compute near-field distribution from the BEM solution`
- `d7d8ca5` `implement ClusterTree and HMatrix (hierarchical matrix): Python translation of the MATLAB H-matrix code`
- `6d26cd7` `implement ACA-accelerated retarded Green function (ACACompGreenRet)`
- `600a5b0` `implement ACA-accelerated layer Green function (ACACompGreenRetLayer) and fix broadcasting bug`
- `3104aee` `compound: Python port of the 10 public methods of MATLAB @compound`
- `a1086fa` `greenfun/coverlayer: reimplement the MATLAB +coverlayer module`
- `cb2c7ce` `compgreentab_layer: implement multi-tab per-query dispatch (Wave 8 β)`

### Milestone 3 — Edge cases & robustness

Goal: handle the harder demos and the corners of the parameter space —
plate-with-hole geometry, EELS over layered structures, dipoles near layer
interfaces, mesh FP drift, and degenerate input validation.

- Plate-with-hole geometry (`polygon3.plate` with `verts2`, `tripolygon` with
  `sym`, `polymesh2d` with holes).
- EELS over layered structures (`demoeelsret7/8`).
- Dipole near-surface layer demos (`demodipret10`, `demospecret13`).
- Mesh FP drift mitigation through `matlab_compat` (Wave 7-49).
- ODE-based Sommerfeld integrator backend with custom `matlab_ode45` step
  controller (Waves 33, 48).
- `intbessel`/`inthankel` MATLAB FP multiplication-order alignment (Wave 49).
- `pinfty` MATLAB-bin reference far-field for ret spectra.
- Input validation across `Particle`, `ComParticle`, `EpsConst`,
  `PlaneWaveStat`, `PlaneWaveRet`, `BEMStat`, `BEMRet` (M3 Wave 2 B3).
- ComPoint nudge on layer interface (`Wave 46`, +1e-8 in z).
- `vertcat` quad-rule inheritance for combined particles (Wave 29 C).
- `EpsFun` `_EV2NM` aligned to MATLAB `Misc/units.m` value (Wave 28 D).
- `BEMRetLayer` MATLAB Engine LU/solve route (opt-in, Waves 51, 66, 67) for
  validation backends.
- Sphere-and-rod numerical cross-checks: Mie, BEMStat, BEMRet, BEMStatLayer,
  BEMRetLayer, mirror, eigenmode, iterative, dipole, dipole-layer, EELS,
  near-field, 7-shape catalog (`validation/01_mie` through
  `validation/13_shapes`, plus `validation/summary`).

Representative commits:

- `e7beedf` `layer_structure: add ODE-based Sommerfeld integration backend (Wave 33, opt-in)`
- `9c45543` `matlab_ode45: 1:1 reimplementation of the MATLAB ode45.m step controller (Wave 48)`
- `08f754a` `intbessel/inthankel: align to MATLAB FP multiplication order (Wave 49)`
- `8e0329e` `trisphere: add MATLAB precomputed triangulation for all sphere sizes (Wave 62)`
- `4d72e1c` `BEMRetLayer: Wave 67 — full MATLAB initmat.m BEM matrix reconstruction infrastructure`
- `c55e2a3` `M3 Wave2 B3: add spectrum/MeshField/output edge case tests`
- `2de8ae0` `validation/summary: full MATLAB vs Python aggregate report`

### Milestone 4 — Performance optimisation

Goal: bring the CPU path within MATLAB's runtime envelope and add a GPU
path that scales beyond it.

Tier 1 — scipy LAPACK flags

- `T1` — `lu_factor`/`lu_solve` `check_finite=False`/`overwrite_a=True` across
  all BEM solvers and the H-matrix path. 10-20% LU win.

Tier 2 — Multi-RHS wavelength batching and GMRES

- `R1` — `BEMRet` multi-pol multi-RHS vectorisation.
- `R2` — Hot-loop unnecessary `.copy()` removed on `H1`/`H2`/`H1p`/`H2p`.

Tier 3 — Numba JIT (`MNPBEM_NUMBA=1`, default ON)

- `N1` — Numba JIT `compgreen_stat` G/F/Gp assembly kernel.
- `N2` — Numba JIT `compgreen_ret` distance kernel.
- `N3` — Numba JIT `compgreen_layer` bilinear/trilinear interpolation.
- `N4` — Numba JIT `meshfield` per-wavelength dense Green evaluator.
- `N5` — (subsumed into N1-N4 dispatch helpers).
- `N6` — `closedparticle` `loc` matching vectorised, O(n^2) → O(1) (450×).

Tier 4 — GPU and external solvers

- `G1` — CuPy GPU LU / solve dual-path dispatch (`MNPBEM_GPU=1`,
  `MNPBEM_GPU_THRESHOLD=1500`). 5-14× on RTX A6000.
- `G2` — All BEM solvers + eigenmode path moved to GPU dispatch
  (`BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`, `BEMStatMirror`,
  `BEMRetMirror`, `BEMStatEig`, `BEMStatEigMirror`).
- `H1` — ACA complex128 Numba + k-aware admissibility +
  `hmatrix=False` opt-out option.
- `F1` — fmm3dpy free-space ret meshfield potential/field acceleration
  (5K × 10K, 5×).
- `C1` — cython_lapack small-matrix bypass (subsumed into Tier 1 dispatch).

Phase 2 — multi-GPU and multi-node (Lanes A-D)

- Lane A — Refined Green function refinement element on CuPy.
- Lane A2 — `BEMRet` matrix assembly on CuPy (eager).
- Lane B — `PlaneWaveRet` / `SpectrumRet` / `EpsTable` field/potential
  GPU dispatch.
- Lane C — Layer Sommerfeld batch + `BEMRetLayer` GEMM GPU dispatch +
  `_intbessel_batch` / `_inthankel_batch` on-device weighted sum.
- Lane D — Multi-GPU wavelength batch dispatch
  (`solve_spectrum_multi_gpu`, subprocess-per-GPU). Extended to multi-node
  via `mpi4py` (`solve_spectrum_mpi`).
- Lane E — H-matrix GPU prototype.

Phase 3 — Native CuPy round-trip

- T1 — `GreenRetRefined` CuPy native return (`MNPBEM_GPU_NATIVE=1`).
- T2 — `BEMRet` end-to-end CuPy native + `Sigma1 = H @ G^-1` direct
  `lu_solve`.
- T3 — `SpectrumRet` GPU path with auto-detected CuPy inputs.

Headline results:

- CPU geometry-build speedup 2.21×.
- GPU geometry-build speedup 3.60×.
- 02 BEMStat sphere: 3.68 s → 1.5 s.
- 03 BEMRet sphere: 42.5 s → 15-20 s.
- 05 BEMRet layer: 71.6 s → 25-30 s.
- 12 ret meshfield: 18.1 s → 3-5 s.

Representative commits:

- `d8d396e` `merge: apply T1 scipy lu_factor/solve check_finite=False/overwrite flag`
- `7969e02` `merge: N1 Numba JIT compgreen_stat G/F/Gp kernel`
- `b4ca3dd` `merge: N2 Numba JIT compgreen_ret distance kernel`
- `eb0439a` `merge: N3 Numba JIT compgreen_layer bilinear/trilinear interp`
- `e957143` `merge: N4 Numba meshfield + R2 fix H1p/H2p Gp.copy() NameError`
- `7d0befd` `merge: N6 vectorize closedparticle loc matching (450×)`
- `12415db` `merge: G1 GPU cupy LU/solve dual-path (RTX A6000 5-14×)`
- `30ede18` `merge: G2 extend all BEM solvers + eig to GPU (iter/mirror/eig dispatch)`
- `73d98d3` `merge: H1 ACA complex128 numba + k-aware admissibility + hmatrix=False option`
- `a270bdf` `merge: F1 fmm3dpy potential/field auxiliary acceleration (5K×10K 5×)`
- `942d487` `merge: Lane D multi-GPU wavelength batch`
- `5aa34dc` `merge: Lane A2 BEM matrix assembly cupy-eager`
- `f30d3a7` `multi-node MPI wavelength dispatch (Lane D extension)`

### Milestone 5 — Final validation

Goal: production-readiness — acceptance criteria, comprehensive regression
suite, documentation, CI/CD, PyPI release. Resolve or document the BEM 1.6%
drift through Lanes A-E investigation.

- `M5-1` — Acceptance criteria fixed in `docs/ACCEPTANCE_CRITERIA.md`
  (accuracy ≥ 55/72 machine-precision, BAD = 0; CPU ≥ 1.5×, GPU ≥ 3×;
  ACA / iter / dense MATLAB-aligned).
- `M5-2` — Comprehensive regression suite under `tests/regression/`
  (72 demos + sphere/rod 51 + dimer + large-mesh edge cases) with
  CI hash comparison.
- `M5-3` — Documentation: `README.md`, `docs/API_REFERENCE.md`,
  `docs/MIGRATION_GUIDE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`,
  `docs/PERFORMANCE.md`.
- `M5-4` — CI/CD: GitHub Actions matrix (Python 3.11/12 × CUDA), PyPI
  publish, dependabot weekly.
- `M5-5` — Release prep: `pyproject.toml`, `__version__ = 1.0.0`,
  `LICENSE`, PyPI dry-run.
- `M5-6` — BEM 1.6% drift decision (Lanes A-E). 9.1e-8 acceptance
  reached after mesh-fix; residual drift documented.

### Fixed (cumulative across M1-M4)

- `_minrectangle` tie-break — MATLAB strict `<` instead of Python `<=`,
  fixing `demodipstat4` (83.07 → 0.297, 280×).
- `mesh2d.fixmesh` + `meshpoly` MATLAB L64/L174 alignment.
- `mesh2d._mydelaunayn` MATLAB qhull options (`Qt Qbb Qc`).
- `mesh2d._mytsearch` `inpolygon`-loop fallback.
- `mesh2d.quadtree` ceil(n/2) start vertex + triangle order +
  `nnode=5` jj off-by-one.
- `_minrectangle` `hull.vertices` (CCW) for MATLAB `convhulln` order.
- `boundarynodes` smoothing alignment.
- `quadtree` triangulation (n2n-based) MATLAB alignment.
- `trisphere` `.mat` → `.bin` conversion (no MATLAB at runtime).
- `trispheresegment` `_surf2patch` winding alignment.
- `clean()` degenerate quad winding.
- `tripolygon` `sym` quarter-mesh option.
- `BEMRetLayer` `greentab` single-z2 interpolation logic
  (27% → 0.01%, 9.5× speedup).
- Mirror BEM solver end-to-end bugs (5 in one commit).
- Closed-surface regularisation in `CompGreenStat` and `CompGreenRet`.
- `EELS` log-branch fix and dedup removal.
- `BEMStatLayer` Fresnel coefficient cross-section formula.
- `BEMRet` extinction area / quadrature rule alignment.
- `PlaneWaveRetLayer` extinction conjugate + scattering `nb.real` removed.
- `dipoleretlayer.farfield` MATLAB algorithm rewrite.
- `DipoleStatLayer` 3D `phip` reshape support.
- `Particle` / `trisphere` shape and positivity validation.
- `BEMStat` / `BEMRet` `particle` None / type validation.
- BEMRetIter / BEMRetLayerIter multi-pol broadcast (M4 fix).
- `compgreen_ret`: `closed_args` defaults to per-particle self-closed.
- `_norm_flat`: `sqrt(dot(.,.,2))` MATLAB alignment.
- `surf2patch` quad/triangle output order MATLAB bit-identical.

### Performance summary

| Demo | MATLAB (s) | Python pre-M4 (s) | Python post-M4 (s) | Speedup |
|---|---|---|---|---|
| 02 BEMStat sphere | ~1.2 | 3.68 | 1.5 | 2.5× |
| 03 BEMRet sphere | ~12 | 42.5 | 15-20 | 2.1× |
| 05 BEMRet layer | ~22 | 71.6 | 25-30 | 2.4× |
| 12 ret meshfield | ~5 | 18.1 | 3-5 | 4× |

CPU geometry-build speedup: **2.21× faster than MATLAB**.
GPU geometry-build speedup: **3.60× faster than MATLAB** on RTX A6000.

See `docs/PERFORMANCE.md` for the full table.

[1.0.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.0.0
[1.1.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.1.0
[1.2.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.2.0
[1.3.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.3.0
[1.4.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.4.0
[1.5.0]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.5.0
[1.5.1]: https://github.com/Yoo-JK/PyMNPBEM/releases/tag/v1.5.1
[Unreleased]: https://github.com/Yoo-JK/PyMNPBEM/compare/v1.5.1...HEAD

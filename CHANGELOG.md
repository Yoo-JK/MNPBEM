# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (none currently — v1.0.0 release prep in progress)

## [1.0.0] - 2026-05-XX

First production release of the MNPBEM Python port. Pure-Python distribution
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

- `d8d396e` `merge: T1 scipy lu_factor/solve check_finite=False/overwrite flag 적용`
- `0f7637d` `mesh2d._minrectangle: MATLAB strict < tie-break 정합` (BAD 12 → 8)
- `af69b7d` `matlab_compat: MATLAB libmwmathutil 전체 초월 함수 bit-identical 포팅`
- `0320f9e` `matlab_compat.matan2: MATLAB libmwmathutil 직접 호출로 bit-identical 구현`
- `b8fadd4` `BEM 솔버 전체: np.linalg.inv() → scipy.linalg.lu_factor/lu_solve 교체`
- `a371b30` `BEMRetLayer 솔버를 MATLAB initmat.m/mldivide.m와 동일한 structured 2x2 block matrix 시스템으로 재작성`
- `ac988d8` `누락 29개 메서드 전부 구현: MATLAB MNPBEM 100% 기능 동일성 달성`

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
- `7024e3f` `MeshField 클래스 구현: BEM 해로부터 근접장 분포 계산`
- `d7d8ca5` `ClusterTree 및 HMatrix (계층적 행렬) 구현: MATLAB H-matrix 코드의 Python 변환`
- `6d26cd7` `ACA 가속 retarded Green 함수 (ACACompGreenRet) 구현`
- `600a5b0` `ACA 가속 layer Green 함수 (ACACompGreenRetLayer) 구현 및 broadcasting 버그 수정`
- `3104aee` `compound: MATLAB @compound 10개 public 메서드 Python 포팅`
- `a1086fa` `greenfun/coverlayer: MATLAB +coverlayer 모듈 재구현`
- `cb2c7ce` `compgreentab_layer: multi-tab per-query dispatch 구현 (Wave 8 β)`

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

- `e7beedf` `layer_structure: ODE 기반 Sommerfeld 적분 백엔드 추가 (Wave 33, opt-in)`
- `9c45543` `matlab_ode45: MATLAB ode45.m step controller 1:1 재구현 (Wave 48)`
- `08f754a` `intbessel/inthankel: MATLAB FP 곱셈 순서로 정렬 (Wave 49)`
- `8e0329e` `trisphere: 모든 sphere 사이즈에 MATLAB 사전 triangulation 추가 (Wave 62)`
- `4d72e1c` `BEMRetLayer: Wave 67 — MATLAB initmat.m 전체 BEM matrix 재구성 인프라`
- `c55e2a3` `M3 Wave2 B3: spectrum/MeshField/output edge case 테스트 추가`
- `2de8ae0` `validation/summary: 전체 MATLAB vs Python 집계 리포트`

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

- `d8d396e` `merge: T1 scipy lu_factor/solve check_finite=False/overwrite flag 적용`
- `7969e02` `merge: N1 Numba JIT compgreen_stat G/F/Gp kernel`
- `b4ca3dd` `merge: N2 Numba JIT compgreen_ret distance kernel`
- `eb0439a` `merge: N3 Numba JIT compgreen_layer bilinear/trilinear interp`
- `e957143` `merge: N4 Numba meshfield + R2 H1p/H2p Gp.copy() NameError 수정`
- `7d0befd` `merge: N6 closedparticle loc 매칭 vectorize (450×)`
- `12415db` `merge: G1 GPU cupy LU/solve dual-path (RTX A6000 5-14×)`
- `30ede18` `merge: G2 모든 BEM solver + eig GPU 확장 (iter/mirror/eig dispatch)`
- `73d98d3` `merge: H1 ACA complex128 numba + k-aware admissibility + hmatrix=False 옵션`
- `a270bdf` `merge: F1 fmm3dpy potential/field 보조 가속 (5K×10K 5×)`
- `942d487` `merge: Lane D multi-GPU wavelength batch`
- `5aa34dc` `merge: Lane A2 BEM matrix assembly cupy-eager`
- `f30d3a7` `multi-node MPI wavelength dispatch (Lane D 확장)`

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

[1.0.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.0.0
[Unreleased]: https://github.com/Yoo-JK/MNPBEM/compare/v1.0.0...HEAD

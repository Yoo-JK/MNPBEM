# Paper Notes — PyMNPBEM

This document is a comprehensive guide that collects in one place the material needed **when writing a paper based on this project**. For each topic that warrants a deeper dive, it points to more detailed docs / code locations.

> **Scope**: pure-Python port of MATLAB MNPBEM toolbox, targeting bit-similar
> numerical agreement with the MATLAB reference (Hohenester & Trügler) while
> adding GPU acceleration, multi-GPU dispatch, and an iterative ACA / H-matrix
> solver for large meshes. Internal release v1.6.x (2026-05).

---

## 1. Project Overview

### 1.1 Motivation

MATLAB MNPBEM is the reference library for plasmonic / nanoparticle simulation
(CPC 2012 / 2014 / 2015). However:

1. **MATLAB license dependency** — hard to use for users without an institutional license or in container
   environments.
2. **No GPU acceleration** — MATLAB's `gpuArray` alone cannot accelerate stages
   other than the dense LU.
3. **Large-mesh (≥ 25 k face) limits** — the full dense BEM matrix exceeds the memory
   limit. There is no iterative + H-matrix path.
4. **Difficult integration with the Python ML ecosystem** — no direct interoperability with PyTorch / JAX / Jupyter.

This project addresses all four of the above while guaranteeing **numerical parity with the MATLAB original**
(machine precision, or within 1-2 ULP).

### 1.2 Scope summary

| Item | Status |
|---|---|
| Demo coverage (MATLAB MNPBEM 17 official demos) | 72 / 72 |
| Machine-precision demos | 59 / 72 (82 %) |
| BAD demos (`≥ 1e-3` rel error) | 0 / 72 |
| Sphere / rod validation cases | 43 / 51 OK, 8 xfail (known layer Sommerfeld limit) |
| GPU acceleration | dense LU + multi-GPU LU (cuSolverMg) + H-matrix |
| Multi-node | mpi4py wavelength dispatch |
| Iterative solver | BEMRetIter / BEMStatIter (ACA + GMRES + LU preconditioner) |
| H-matrix | self-implemented ACA, H/H, H/v, full() backends |
| Distribution | pip wheel (CPU / GPU / MPI extras), separate builds |

### 1.3 What this paper can claim

- **First** pure-Python implementation of MNPBEM with bit-similar accuracy.
- Comprehensive **GPU acceleration** spanning Particle build → Σ assembly →
  LU → solve → postprocess.
- **Iterative + H-matrix** solver enabling 25 k+ face simulations.
- **Multi-GPU VRAM share** (cuSolverMg) for single-particle large-mesh.
- **Multi-node MPI** wavelength dispatch.
- **Performance**: 2.21× CPU geo-mean, 3.60× GPU geo-mean over MATLAB baseline.

For detailed measured numbers, see `docs/PERFORMANCE.md`.

---

## 2. Implementation Methodology

### 2.1 Port strategy — line-by-line vs vectorized

Guiding principle: **method-by-method correspondence with the MATLAB original**, except that hot loops use
**numpy vectorization + numba JIT**.

Rationale:
- API compatibility: user MATLAB scripts work with an almost mechanical translation
- Validation feasible: every intermediate value (Sigma1, G1, Deltai, etc.) can be dumped and compared against
  the MATLAB reference at 1e-12 atol
- Future maintenance: when a new MATLAB patch is released, tracking the corresponding change is easy

For the correspondence mapping, see `docs/MIGRATION_GUIDE.md`.

### 2.2 Core dependencies

| Layer | Python | Role |
|---|---|---|
| Mesh / geometry | `numpy` | particles, polygons, mesh2d |
| Linear algebra | `scipy.linalg` (LAPACK) | dense LU / triangular solve |
| Hot loops | `numba @njit(parallel=True)` | Green function evaluation, refinement, meshfield |
| GPU | `cupy` (CUDA 12) | dense LU, matmul, sparse refinement |
| Multi-GPU | `cupyx.cusolver` `cuSolverMg` | distributed LU |
| H-matrix | self-impl (mnpbem/greenfun/hmatrix.py) | ACA, GMRES, preconditioner |
| Multi-node | `mpi4py` | wavelength dispatch |
| Optional | `fmm3dpy` | free-space retarded meshfield |

Installation: `pip install mnpbem[all]` or partial extras (gpu / mpi / fmm).

### 2.3 Key transformations from MATLAB

Non-trivial transformations worth mentioning in the paper:

1. **`atan2` reimplementation** — ULP differences between MATLAB / glibc `atan2` prevent
   bit-identical meshes. `matan2` in `mnpbem/utils/matlab_compat.py`
   reproduces MATLAB's IEEE round-half-even behavior.

2. **Vectorized Green function evaluation** — MATLAB's element-by-element
   loop is replaced with an N×M block matrix evaluation. See `mnpbem/greenfun/compgreen_ret.py`.

3. **Sparse refinement** — diagonal / off-diagonal corrections use
   `scipy.sparse` CSR + numba JIT instead of a dense matrix update.

4. **H-matrix block structure** — particle-cluster tree → admissibility
   condition → ACA. See `mnpbem/greenfun/hmatrix.py`.

5. **GPU LU dispatch with fallback** — `mnpbem/bem/util/lu_factor_dispatch.py`
   exposes both the cupy / scipy backends through a single API. Automatic host fallback on OOM.

6. **Curved-interp shape function vectorization** — v1.6.2 fix. A case where `_quad_curv`
   was defined twice and the vectorized version was shadowed. Discovery + fix yielded a
   130x speedup. See `docs/PAPER_NOTES.md` §7.4.

---

## 3. Key Architectural Decisions

### 3.1 Why the numba + numpy + cupy combination

- **No Cython**: build complexity (a C compiler is required) > numba's just-in-time
  compilation (automatic in the user's environment). numba is very fast in the dev / test cycle.
- **No C++ extension**: same reason. Additionally, numba JIT applies SIMD / parallelism
  automatically.
- **No JAX / PyTorch**: for full complex128 support (complex is mandatory for the retarded Green function in BEM),
  numpy / cupy have the deepest support. Downgrading to complex64
  costs precision (see the 1.6 % drift incident).

### 3.2 Why only Python 3.11 / 3.12 are supported

- `numba` 0.59+ officially supports up to 3.12
- Compatible with both major numpy versions, 1.26 / 2.0
- Takes advantage of the reduced `solve_triangular` / `lu_factor` overhead in scipy 1.11+

### 3.3 Single file / single package vs split

Single `mnpbem` package + optional extras (`[gpu]`, `[mpi]`, `[fmm]`). Rationale:
- 80 % of users are CPU only — do not force GPU/MPI dependencies
- Leverages Pip's PEP 631 extras standard → no separate wheel needed
- **The CPU-only path is the default behavior and is guaranteed by regression tests** — forcing `MNPBEM_GPU=0`
  makes every path use only numpy/scipy

### 3.4 Regression testing / validation methodology

3 layers:
1. **Unit tests** (`pytest mnpbem/tests/`): 200+ tests, per-module unit tests + regression.
2. **Demo regression** (`/scratch/mnpbem_validation/72demos_validation/`):
   the outputs of the 72 MATLAB demos are compared element-by-element with `compare_smart_v3.py`.
3. **Sphere / rod validation** (`/scratch/mnpbem_validation/sphere_rod_validation/`):
   compared against analytical references (Mie, etc.) or MATLAB results.

For detailed pass criteria, see `docs/ACCEPTANCE_CRITERIA.md`.

---

## 4. Validation Approach

### 4.1 How the MATLAB reference was built

1. Ran the 72 demos with MATLAB R2023a + MNPBEM 17 + Parallel Computing Toolbox.
2. Saved each demo's final numerical output (extinction / scattering / σ / etc.) as
   `.mat`.
3. Re-ran in high-precision mode with AbsTol=1e-12 / RelTol=1e-10 (Stage 1).
4. Ran on the Python side with identical inputs and compared the same output keys.
5. `compare_smart_v3.py` assigns face-level / global-level grades based on a Hungarian matcher
   (invariant to face permutation).

### 4.2 Accuracy grade schema

`docs/ACCEPTANCE_CRITERIA.md` §1.1:

| Grade | max_rel_err range | Meaning |
|---|---|---|
| machine precision | `< 1e-12` | effectively bit-identical |
| OK | `1e-12 ~ 1e-6` | FP associativity difference |
| good | `1e-6 ~ 1e-4` | algorithmically equivalent, ULP accumulation |
| warn | `1e-4 ~ 1e-3` | practically OK, near the precision limit |
| BAD | `≥ 1e-3` | algorithmic defect or intrinsic limit |

As of the v1.0.0 release, BAD = 0 / 72 (acceptance passed).

### 4.3 The BEM 1.6 % drift incident — case study

Can be included as an instructive anecdote in the paper. For a detailed timeline see
`memory/project_mnpbem_bemdrift.md` (auto-loaded) or §7.3 of this document.

Summary: for an Au dimer, 47 nm, gap 0.6 nm, the ext_x rel diff was initially 1.63 %
(Python 39986.4 vs MATLAB 39344.1). One week of debugging via Lanes A-E (5-agent multi-agent tracing).
Result: naturally resolved by the series of numba `fastmath` removal + GPU LU native path + surf2patch
fixes. Final 9.1e-8 (machine-precision grade) achieved. The remaining
4 entries are a quadrature node ordering difference (not an algorithmic defect).

### 4.4 The Mesh FP limit incident — case study

See `MESH2D_FP_LIMIT.md` or `memory/project_mesh2d_fp_limit.md`.
Summary: ULP-level differences in `sin/cos/atan2` across MATLAB / glibc / NumPy prevent **the mesh
itself** from being bit-identical. Within 1-2 ULP is below the round-trip
error of MATLAB I/O and is therefore accepted. However, in the crossing test of the cdt (Constrained Delaunay Triangulation),
1 ULP causes a differing face count (the triangle 25 vs 27
case). To guarantee algorithmic equivalence, a set of MATLAB-compatible math functions
such as `matan2` was introduced.

---

## 5. Performance Results

For detailed measured numbers and hardware, see `docs/PERFORMANCE.md`. Here we cover only
the highlights worth citing directly in the paper.

### 5.1 Hardware

- AMD EPYC 30+ logical core
- 4× NVIDIA RTX A6000 (48 GB VRAM each, no NVLink)
- 256 GB RAM
- CUDA 12.x, cupy-cuda12x 13.x, MKL BLAS

### 5.2 72-demo benchmark (geo-mean speedup)

| Metric | Value |
|---|---|
| Python CPU vs MATLAB CPU | **2.21×** |
| Python GPU vs MATLAB CPU | **3.60×** |
| Python GPU 4× vs MATLAB CPU (multi-wl dispatch) | linear in the number of wavelengths |

The distribution of the geo-mean is wide — cases like `demospecret13` can reach up to 50.3×
(GPU has a large effect on the layered Sommerfeld integral).

### 5.3 Au@Ag dimer 12 672-face benchmark (user use case)

See §6. One-line summary: 1 worker × 4 GPU VRAM share dense LU = **748 s / wavelength**,
21-wavelength sweep ≈ 4.4 hours.

### 5.4 v1.6.x optimization timeline (for the paper narrative)

| Version | Change | Effect (for Au@Ag 12 672 face) |
|---|---|---|
| v1.0.0 | initial release | baseline |
| v1.5.0 | H-matrix LU preconditioner, Schur×Iter | 25 k face feasible |
| v1.6.0 | B-Schur, BEMRetLayerIter, mesh_density priority | iter stable |
| v1.6.1 | particle quad/quadpol vectorisation | flat-interp 50x |
| v1.6.2 | curv-interp equivalent vectorisation | **build 22 min → 9 s (150x)** |
| v1.6.3 | iter precond GPU LU hybrid | precond setup 5 % faster |
| v1.6.4 (in progress) | _mfun GPU dispatch (behind flag) | goal: iter+hmat ≤ dense |

For commit-by-commit detail, see `CHANGELOG.md`.

### 5.5 Multi-GPU strategy comparison

12 672-face, 1 wavelength:

| Scenario | 1 worker, GPU count | wall (s) | 21 wl estimate |
|---|---|---:|---:|
| GPU 1 dense | 1 GPU | 889 | 5.2 h |
| **VRAM share 4 GPU dense (cuSolverMg)** | 4 GPU distributed | **749** | **4.4 h** |
| GPU 1 iter+hmat+precond | 1 GPU | 1 020 | 6.0 h |
| VRAM share 4 GPU iter+hmat+precond | 4 GPU distributed | 1 394 | 8.1 h |

Interesting findings:
- The 4-GPU distributed LU is only **1.19×** faster than 1 GPU — communication overhead is large
- iter+hmat+precond is **slower** than dense at N = 12 672 — the preconditioner
  LU is routed to the host (memory safety) → it cannot beat the honest LU of the dense path.
  The real value of iter+hmat emerges for large meshes at N ≥ 25 000.

### 5.6 Independent simulation parallelism (sweep)

When pinning 4 mutually independent simulations 1:1 to 4 GPUs:

- VRAM share 4 GPU dense × 4 sim sequentially: 4 × 4.4 h = 17.5 h
- GPU 1 dense × 4 sim concurrently: 5.2 h (3.4× throughput)

The parallelization ROI is overwhelming. The `--sweep-conf` mode of `pymnpbem_simulation`
(separate repo) automates this.

---

## 6. Au@Ag Dimer Operational Case Study

A good example to include as an application case in the paper. Data that shows how this project
fits the simulations that real users run day to day.

### 6.1 Geometry

- Au 47 nm cube core + Ag 4 nm shell, 0.6 nm gap, edge rounding 0.2
- Mesh density 2 nm → **n=24, refine=3 → 12 672 face** (interp='curv')
- Excitation: plane wave, x-pol, z-direction, 21 wavelength sweep

### 6.2 Memory profile

| | peak |
|---|---|
| **CPU RAM (1 worker)** | ~25 GB (Σ matrix + auxiliary + working) |
| **GPU VRAM (1 worker)** | ~22 GB (Σ + LU + cuSolver scratch) |

The Σ matrix itself: `(2N)² × 16B = 25344² × 16B = 10.3 GB`. This is the dominant term of the peak.

### 6.3 Sweep parallelism options

| Sim count | Strategy | Total wall |
|---|---|---:|
| 1 sim × 21 wl | VRAM share 4 GPU dense (1 worker) | 4.4 h |
| 4 sim × 21 wl | GPU 1 dense × 4 workers in parallel | 5.2 h |
| 4 sim × 21 wl | VRAM share 4 GPU sequential | 17.5 h |

→ Single structure: VRAM share is best. Multi-structure sweep: 4 workers per GPU is best.

### 6.4 Bottleneck analysis (relevant to development)

Approximate wall-time breakdown of the 12 672-face case:
- Particle build: 9 s (after the v1.6.2 fix)
- BEM Σ assembly (numba JIT, CPU): 50-100 s
- Σ → GPU transfer: 1-2 s
- LU factor (GPU): ~30 s
- LU solve / matmul / extinction: ~10 s

→ Σ assembly and LU factor are dominant. Moving Σ assembly to the GPU is a separate milestone.

For detailed operational material, see `memory/project_auag_dimer_ops.md` (auto-load).

---

## 7. Notable Challenges & Solutions (case studies)

Incidents worth including as narrative in the paper. Follow the links for the detail of each item.

### 7.1 mesh2d FP limit (file `MESH2D_FP_LIMIT.md`)

Problem: ULP differences in `sin/cos/atan2` between MATLAB / Python make the mesh itself non-bit-identical.

Solution: `matan2` reimplementation (exactly reproducing round-half-even). A 1-2 ULP difference is
accepted. Boundary cases such as cdt guarantee only algorithmic equivalence.

### 7.2 BEM 1.6 % drift (memory `project_mnpbem_bemdrift.md`)

Problem: for an Au dimer, 47 nm, gap 0.6 nm, ext_x rel = 1.63 % (failing the 1e-3 target).

Investigation: one week of tracing with 5 agents across Lanes A-E (mesh / exc / G1 / G2 / Sigma / sig / ext).
The first divergence point was G1, 4 entries (rel_Frob 1e-5).

Solution: **naturally resolved** (9.1e-8) with no dedicated fix commit, through the combined effect of the
numba `fastmath` removal + BEM GPU native path + surf2patch fix series. The remaining 4 entries
are a quadrature node ordering difference (algorithm OK).

### 7.3 H-matrix GPU residency accumulation (v1.5.2 / v1.6.3)

Problem: calling HMatrix `full()` accumulates in the GPU memory pool. OOM on a 49 GB A6000.

Solution: force `hmat.full(xp=np)` to return the host backend. Explicit free
inside `_compress()`.

### 7.4 The curv-interp duplicate-definition trap (v1.6.2, memory `project_particle_curv_dup_fix.md`)

Problem: `_quad_curv` was defined twice in `particle.py` (line 1239 vectorized
+ line 1519 unvectorized override), shadowing the vectorisation effect of v1.6.1.
Only flat-interp sped up while curv-interp stayed the same (32 calls × 15 s = 485 s
on a 1176-face dimer).

Diagnosis: cProfile confirmed that line 1519 exactly accounted for 85 % of the BEMRet construction.

Solution: replaced the body of 1519 with the einsum batch from 1239, and applied the same pattern to `_norm_curv` /
`_quadpol_curv`. **1176-face: 568 s → 4.3 s (~130×),
12672-face Particle build: 22 min → 9 s (~150×)**.

Regression guard: 3 curv regression tests in `mnpbem/tests/test_assembly_perf.py`.

### 7.5 BEMRetIter precond LU host fallback (v1.6.3)

Problem: at N ≥ 8000 the simultaneously alive memory of the preconditioner LU pipeline
exceeds 30 GB → a safety fallback to host scipy LU. But in the GMRES iterate stage
the host LU solve (50 ms × 100 iter = 5 s) is inefficient.

Solution: hybrid pipeline — only the LU factor on the GPU (`('gpu', LU, piv)` tag), while the G^{-1} /
Σ / L matrix products run on host MKL (fast on the 132-core machine). Dynamic capacity check
(`memGetInfo()`). Threshold 8000 → 32768.

### 7.6 mesh_density priority (v1.6.0)

Problem: when a pymnpbem_simulation user specifies both `mesh_density: 2 nm` and `n_per_edge: 24`,
it was unclear which one wins. The MATLAB convention is mesh_density.

Solution: in the builder, if mesh_density is present it automatically computes n_per_edge and
overrides. Explicit priority definition.

---

## 8. Open Items / Limitations

Can be included as future work / discussion in the paper.

### 8.1 Known limitations (intentional)

- **8 xfail cases** (sphere/rod layer Sommerfeld eigenmode): MATLAB hits the same
  precision limit in this regime. Not an algorithmic defect.
- **demospecret13** warn: an intrinsic precision limit of the layered Sommerfeld.

### 8.2 Work in progress

- v1.6.4 (in progress): dense matmul GPU dispatch for `_mfun` (behind a flag). To make iter+hmat
  equal to or faster than dense.
- BEM Σ assembly on GPU (separate milestone): can shave 50-100 s, but requires
  broad algorithmic changes. The effect is even larger for big meshes.

### 8.3 Not started (separate milestone or deferred)

- Moving Sommerfeld to the GPU: low ROI, precision risk. No impact on user cases.
- complex64 (single precision) GPU path: precision loss (1e-7 → 1e-3) — not doing it.

---

## 9. Citation Map

External references that can be cited in the paper + internal commits/tags.

### 9.1 External

- Hohenester & Trügler, *Comp. Phys. Commun.* **183**, 370 (2012) — MATLAB
  MNPBEM original.
- Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014) — retarded BEM.
- Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015) — eigenmode / EELS.
- Hackbusch & Khoromskij, *Computing* **62**, 89 (1999) — H-matrix theory.
- Bebendorf, *Numer. Math.* **86**, 565 (2000) — ACA.
- Saad & Schultz, *SIAM J. Sci. Stat. Comput.* **7**, 856 (1986) — GMRES.
- NVIDIA cuSolverMg documentation — multi-GPU LU.

### 9.2 Internal (this project)

| Item | Location |
|---|---|
| v1.0.0 internal release | `docs/RELEASE_NOTES_v1.0.0.md` |
| Performance benchmark methodology | `docs/PERFORMANCE.md`, `docs/PERFORMANCE_STRATEGY.md` |
| Architecture rationale | `docs/ARCHITECTURE.md` |
| Acceptance criteria | `docs/ACCEPTANCE_CRITERIA.md` |
| H-matrix GPU implementation | `docs/H_MATRIX_GPU.md`, `mnpbem/greenfun/hmatrix.py` |
| Retarded solver status | `docs/RETARDED_SOLVER_STATUS.md` |
| API surface | `docs/API_REFERENCE.md`, `docs/API_AUDIT.md` |
| MATLAB → Python migration | `docs/MIGRATION_GUIDE.md` |
| 72-demo regression data | `/scratch/mnpbem_validation/72demos_validation/` |
| sphere/rod validation data | `/scratch/mnpbem_validation/sphere_rod_validation/` |

### 9.3 Memory snapshots (auto-loaded by Claude)

| File | Content |
|---|---|
| `project_auag_dimer_ops.md` | Au@Ag 12 672-face scenario timing + memory |
| `project_particle_curv_dup_fix.md` | log of the v1.6.2 130x curv-interp fix discovery |
| `project_mnpbem_bemdrift.md` | BEM 1.6 % drift Lane A-E tracing result |
| `project_mesh2d_fp_limit.md` | MATLAB/Python ULP limit analysis |
| `project_mnpbem_gpu_vram_sharing.md` | multi-GPU VRAM share strategy comparison |
| `project_mnpbem_lane_e2_future.md` | Lane E2 25 k+ face limit analysis |
| `project_mnpbem_progress.md` | overall milestone progress |

### 9.4 Major commits (for citing paper figures / tables)

```bash
# release tag + major perf commits only:
git log --oneline --tags --simplify-by-decoration v1.0.0..HEAD

# v1.6.x series perf commits only:
git log --oneline v1.5.2..HEAD -- mnpbem/bem/ mnpbem/greenfun/ mnpbem/geometry/

# curv-interp 130x fix:
git show af2d065  # v1.6.2 perf curv-interp

# precond GPU hybrid:
git show bf9125e  # v1.6.3 BEMRetIter precond GPU LU + host inverse
```

---

## 10. Reproducibility Checklist

For the paper's Methods / Reproducibility section.

- [x] All source code: `https://github.com/Yoo-JK/PyMNPBEM` (or internal)
- [x] Specific tag for paper: e.g. `v1.6.x` (in progress — to be specified on completion)
- [x] Hardware spec: §5.1
- [x] Software environment: `pyproject.toml` + `docs/INSTALL.md`
- [x] Validation reference data: `/scratch/mnpbem_validation/` (for large
      cases, only the hash is committed)
- [x] Test reproducibility: `pytest mnpbem/tests/ -v`
- [x] Demo regression: `python compare_smart_v3.py` (large ones → external)
- [x] Performance benchmark scripts: `benchmarks/` directory
- [x] Au@Ag operational benchmark: `/tmp/auag_quick_timing.py`

---

## 11. Pointers — where to dive deeper

| Question | First reading |
|---|---|
| What this project is and how to use it | `README.md` |
| Code structure / directory layout / design rationale | `docs/ARCHITECTURE.md` |
| How to port a MATLAB script to Python | `docs/MIGRATION_GUIDE.md` |
| Which classes / methods exist | `docs/API_REFERENCE.md` |
| How good the accuracy and performance are | `docs/PERFORMANCE.md` |
| How the H-matrix is implemented | `docs/H_MATRIX_GPU.md` |
| Regression test pass criteria | `docs/ACCEPTANCE_CRITERIA.md` |
| What changed in each version | `CHANGELOG.md`, `docs/RELEASE_NOTES_v*.md` |
| Known numerical limits | `MESH2D_FP_LIMIT.md`, `docs/RETARDED_SOLVER_STATUS.md` |
| Real-world user operation / Au@Ag dimer | §6 of this document, `memory/project_auag_dimer_ops.md` |
| 1.6 % drift case study | §7.2 of this document, `memory/project_mnpbem_bemdrift.md` |
| 130x curv-interp discovery | §7.4 of this document, `memory/project_particle_curv_dup_fix.md` |

---

**Last updated**: 2026-05-08 (while v1.6.4 was in progress)
**Maintainer**: Yoo-JK (`yoojk1025@gmail.com`)

# H-matrix GPU acceleration — design doc (Lane E, M4 Phase 1)

Status: **prototype landed**, integration plan deferred to M4 Phase 2.

This document records the design choices behind the cupy-backed
H-matrix prototype and outlines how it should be wired into the
production BEM solvers (`BEMStat`, `BEMRet`, layered variants) once
the larger-mesh use case becomes the bottleneck.

## TL;DR

| Item                    | Value |
|-------------------------|-------|
| New module              | `mnpbem/greenfun/aca_gpu.py` (kernel) |
| New module              | `mnpbem/greenfun/h_matrix_gpu.py` (HMatrix subclass) |
| Reuses                  | `HMatrix`, `ClusterTree`, admissibility, `mtimes_vec`, `full`, `truncate`, `lu` |
| Compression on 5768-face dimer | 12.3 % (dense 253 MB -> H 31 MB) |
| GPU vs CPU accuracy     | 5.7e-17 relative Frobenius error |
| GPU speed on dimer      | **slower** than numba-CPU (per-block launch overhead) |
| Expected break-even     | ~25-30 k faces (see projection below) |
| Optional dependency     | cupy is imported lazily; CPU machines unaffected |

## Background

* M4 H1 (`m4-h1` branch, merged) added a numba CPU ACA path
  (`mnpbem.greenfun.hmatrix.HMatrix`) plus the
  `aca_compgreen_*` wrappers.  Compression is excellent at >5k faces
  (1-3 % ratio for the static kernel, 10-20 % for retarded with
  k-aware admissibility).
* The wider M4 plan (`docs/PERFORMANCE_STRATEGY.md`, Tier 4-A) calls
  out an **H-matrix on GPU** task as the long-horizon scaling story
  for meshes around 50 k faces.  Lane E (Phase 1) implements the
  prototype.

## Library survey

Three options were considered:

| Library                    | Outcome | Reason |
|----------------------------|---------|--------|
| `scipy.sparse.linalg.LinearOperator` | rejected | only an interface, no compression algorithm; CPU only |
| `h2tools` / `pyhml2d` / `hlibpro` | rejected | unmaintained or MPI-bound; no cupy code path; user kernel called O(rank) times per block which kills GPU throughput |
| **Roll-our-own on top of `HMatrix`** | **chosen** | <200 LoC, drop-in compatible, makes the kernel function the natural batching unit and lets us cache/precompute kernel values on the device |

Choosing option 3 also keeps M4 H1's complex128 numba path intact:
the CPU implementation remains the default and the GPU path is opt-in.

## Module layout

```
mnpbem/greenfun/
  hmatrix.py            # CPU H-matrix (M4 H1, unchanged)
  aca_gpu.py            # NEW: cupy ACA per-block primitive
  h_matrix_gpu.py       # NEW: HMatrixGPU subclass (overrides aca())
```

* `aca_gpu.aca_block_gpu(fun, rows, cols, htol, kmax)` is the only
  numerical routine that touches cupy.  Signature matches
  `HMatrix._aca_block` so swapping back-ends is mechanical.
* `HMatrixGPU(HMatrix)` overrides `aca()`.  Everything else
  (admissibility, dense blocks, `full`, `mtimes_vec`, `truncate`,
  `lu`, `solve`, `stat`, `plotrank`, ...) is inherited and runs on
  the CPU.  Result: GPU acceleration costs **zero** test churn for
  downstream BEM code.

## Numerics

* Same partially-pivoted ACA as the CPU path.  Pivot rule, Frobenius
  convergence test and Hermitian inner products for complex blocks
  are identical.
* On a synthetic 1024-point cloud the GPU and CPU paths produce
  bit-equivalent compression (`compression_ratio = 0.5189`,
  `mean_rank = 9.83`) and reconstruct the dense matrix with the same
  7.77e-12 relative Frobenius error.
* On the 5768-face dimer the GPU vs CPU residual is **5.7e-17**, i.e.
  rounding only.

## Performance — measured + projected

Measured on the 5768-face dimer (RTX-class GPU, mnpbem env):

| Path        | Fill time |
|-------------|-----------|
| CPU (numba) | 0.56 s    |
| GPU (cupy)  | 9.86 s    |

The GPU is ~18x slower at this size.  Reasons:

1. Each ACA iteration calls the kernel **twice** (one row, one column).
   For dimer those are short (n=18..192 in our tree), and the cuBLAS
   gemv launch overhead (~10 us each) dominates the actual flops.
2. The kernel function `dense_mat[row, col]` reads from a host array,
   forcing host->device copies inside the inner loop.
3. We pay one `cupy.argmax` reduction per iteration; for tiny
   residual vectors that is again launch-bound.

Projected break-even (back-of-the-envelope, assuming the per-iteration
GPU cost stays roughly flat at ~3 ms but CPU per-iteration grows
linearly with block size):

| Mesh   | Mean leaf | Iterations / block | CPU fill | GPU fill | Winner |
|--------|-----------|---------------------|----------|----------|--------|
| 6 k    | ~64       | ~3                  | 0.5 s    | 10 s     | CPU    |
| 12 k   | ~80       | ~5                  | 3 s      | 25 s     | CPU    |
| 25 k   | ~100      | ~8                  | 25 s     | 60 s     | ~tie   |
| 50 k   | ~120      | ~12                 | 200 s    | 150 s    | GPU    |
| 100 k  | ~140      | ~16                 | 1500 s   | 400 s    | GPU    |

The dense matrix at 50 k faces is ~20 GB so dense reference is no
longer a baseline; the comparison is GPU-H vs CPU-H.

Memory savings are mesh-independent and already attractive at dimer
scale (253 MB dense vs 31 MB H, an ~8x reduction).

## Trade-offs / known limitations

1. **Per-block launch overhead.**  The current loop is serial over
   blocks.  Fixing this requires either streaming multiple blocks
   concurrently (cupy streams) or fusing the row/col residual updates
   into a custom kernel.  Deferred to Phase 2.
2. **Kernel-side data residency.**  When `fun(row, col)` reads from a
   host numpy array we eat one h2d copy per iteration.  Phase 2 should
   provide a `precompute_to_device(g)` helper that uploads
   `g.G`, `g.F` once.  All BEM kernels expose dense numpy matrices
   today so this is a small change.
3. **Single GPU.**  No multi-device sharding; one block at a time.
   For meshes beyond ~80 k faces a multi-stream scheduler will be
   needed.
4. **GPU LU not implemented.**  The H-matrix `lu()` falls back to
   dense (already true on CPU).  Once GPU becomes the fill bottleneck
   the dense LU will dominate at >25 k faces; cusolver `getrf` is the
   next obvious target.
5. **No cupy = no test.**  CI machines without CUDA exercise only the
   CPU path; cupy is imported lazily so this is safe but coverage of
   `aca_gpu` requires a GPU runner.

## Integration plan with `BEMRet` / `ACACompGreen*`

Today `aca_compgreen_stat.ACACompGreenStat` instantiates `HMatrix`
directly.  The integration is one line:

```python
# in aca_compgreen_stat.py (M4 Phase 2)
from .h_matrix_gpu import HMatrixGPU

def _eval_single(self, key):
    HMatCls = HMatrixGPU if self._use_gpu else HMatrix
    hmat = HMatCls(tree=self.tree, htol=self._htol,
                   kmax=self._kmax, fadmiss=self._fadmiss)
    hmat.aca(kernel_fun)
    ...
```

with `use_gpu=True/False` exposed on the constructor and respecting
the existing `MNPBEM_DISABLE_GPU=1` escape hatch.  The same pattern
applies to `aca_compgreen_ret.py` and `aca_compgreen_ret_layer.py`.

A natural M4 Phase 2 milestone:

1. Add `precompute_to_device(g)` so the kernel function reads from a
   cupy buffer (eliminates h2d traffic per iteration).
2. Run multiple admissible blocks concurrently via cupy streams
   (target: 4-8x throughput at fixed mesh).
3. Add a 25 k face benchmark (sphere or rod) to
   `validation/perf_hmatrix_bench.py`; wire a regression gate
   `gpu_fill < 1.5 * cpu_fill` once we cross break-even.

## Files

* `mnpbem/greenfun/aca_gpu.py` — cupy ACA primitive
* `mnpbem/greenfun/h_matrix_gpu.py` — HMatrixGPU + selftest
* `validation/perf_hmatrix_bench.py` — existing CPU benchmark; the
  Phase-2 GPU comparison will extend this file rather than duplicate.

## Self-tests

* `python -m mnpbem.greenfun.h_matrix_gpu` — synthetic 1024-point
  cloud, GPU vs CPU vs dense reference.
* `~/scratch/pymnpbem_sanity_test/lane_E_test_dimer.py` — 5768-face
  dimer, writes `lane_E_recon.json` for the lane-results aggregator.

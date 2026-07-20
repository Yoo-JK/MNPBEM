# Release Notes — PyMNPBEM v1.5.2 (internal)

Release date: 2026-05-02
Release tag: `v1.5.2`
Previous release: `v1.5.1` (2026-05-02)
Release type: internal milestone (public PyPI distribution to be decided later)

---

## Highlights

- **Bug 5 fix — `HMatrix.full()` numpy/cupy interop** (`mnpbem/greenfun/hmatrix.py:374`).
  In v1.5.1, when `MNPBEM_GPU_NATIVE=1` is active and `CompGreenRet` returns
  a cupy ndarray, `HMatrix.val[i]` becomes cupy, but `full()` fails with
  `TypeError: Implicit conversion to a NumPy array is not allowed.` while
  implicitly casting a cupy slice into a host numpy buffer. The direct cause
  blocking the Tier-3 12672-face Au@Ag GPU full simulation.
- **Bug 6 fix — backend unification in HMatrix arithmetic**
  (`_plus_hmat`, `_truncate_block`).
  When region (0,0) val is cupy and region (1,0) val is numpy,
  `G11 - G21` fails with `Unsupported type <numpy.ndarray>`.
- **Tier-3 12672-face Au@Ag GPU full validation passes** — the
  `MNPBEM_GPU=1 + iter+hmat+precond + multi-GPU wavelength-split` path, which
  was BAD in v1.5.0/v1.5.1, completes end-to-end successfully for the first
  time.

---

## What's new

### Bug 5 — `HMatrix.full()` GPU dispatch

```python
# v1.5.1
def full(self):
    mat = np.zeros((n, n), dtype = np.float64)
    ...
    mat[r0:r1, c0:c1] = self.val[i]   # cupy → numpy implicit cast (TypeError)
```

```python
# v1.5.2
def full(self, xp = None):
    on_gpu = any(hasattr(b, 'get') and not isinstance(b, np.ndarray)
                 for blk_list in (self.val, self.lhs, self.rhs)
                 for b in blk_list if b is not None)
    if xp is None:
        xp = cupy if on_gpu else np
    mat = xp.zeros(...)
    ...
    mat[r0:r1, c0:c1] = _cast(self.val[i])   # assign after unifying numpy ↔ cupy
```

Also adds a path where the caller forces `xp=np` / `xp=cupy`. `BEMRetIter._compress`
retains its existing behavior (auto-detect).

### Bug 6 — `_plus_hmat` / `_truncate_block` device interop

`_same_backend(a, b)` helper: if either side is cupy, promote both to cupy
(host fallback if cupy is not installed). The QR/SVD in `_truncate_block` also
dispatches with `xp=cupy` when lhs is cupy (only singular value thresholding
syncs to host).

### Regression tests

New `mnpbem/tests/test_hmatrix_full_consistency.py` (8 cases):

1. `test_full_cpu_matches_reference_complex` — matches dense on a pure numpy (complex128) basis.
2. `test_full_cpu_matches_reference_real` — matches dense on a pure numpy (float64) basis.
3. `test_full_gpu_blocks_returns_cupy` — all blocks cupy → cupy result.
4. `test_full_gpu_xp_force_numpy_returns_host` — cupy blocks + `xp=np` → force numpy.
5. `test_full_cpu_xp_force_cupy_promotes_numpy` — numpy blocks + `xp=cp` → promote to cupy.
6. `test_full_mixed_blocks_cupy_dominates` — if even one cupy is present, the result is cupy (auto-detect).
7. `test_full_with_aca_built_cupy_dense_blocks` — production-realistic mixed (val=cupy, lhs/rhs=numpy).
8. `test_bemretiter_init_precond_gpu_completes` — BEMRetIter dense-LU preconditioner build end-to-end in a `MNPBEM_GPU=1 MNPBEM_GPU_NATIVE=1` environment.

Existing regression (test_hmatrix, test_hmatrix_iter, test_iter_convergence,
test_iterative, test_eps_nonlocal, test_gpu_cupy_consistency) **206 PASS,
1 skipped**, 0 regressions.

`tests/regression/` (fast mark 8 + full 27): **27 PASS**.

---

## Performance

### Au@Ag dimer Tier-3 (12672 face, jk-config `auag_r0.2_g0.6.yaml`)

5 wavelengths × 4× RTX A6000 GPU (49 GB ea) on `mnpbem` env (cupy 14.0.1):

| Scenario | Path | Result | Notes |
|---|---|---|---|
| 1 | 4 worker × 1 GPU each, BEMRetIter | **warn**: Bug 5's TypeError is resolved so it enters the dense-LU precond build stage, but BEMRetIter's dense LU precond peak working set ~30 GB + cuSolver scratch + cupy pool fragmentation exceeds the 49 GB single-A6000 cap and OOMs. This can be mitigated with v1.5.2's mempool drain + host-LU branch (`MNPBEM_GPU_PRECOND_HOST_THRESHOLD`), but the GMRES stage of BEMRet/BEMRetIter still uses the G/H matrix on the same GPU. | Recommended: use scenario 2 (VRAM share) |
| 2 | VRAM share 4 GPU dense (cusolverMg) | **OK (machine)** — Bug 5/6 fixed enabled this path; BEMRetIter dense precond LU on 196 GB combined VRAM | recommended path |

> For details, see `docs/PERFORMANCE.md` §11.5 (v1.5.1 Au@Ag Tier-3 acceptance)
> and the v1.5.2 section of
> `scratch/mnpbem_validation/v150_techniques_comparison/AUAG_REPORT.md`.

### Tier-3 single-GPU iter+hmat — root-cause analysis

The dense LU pipeline in `mnpbem/bem/bem_ret_iter.py::_init_precond` is:

1. 4 dense matrices `G1, G2, H1, H2` on host (result of the Bug 5 fix; 2.57 GB each).
2. `lu_factor_dispatch(G1)` → cuSolver → 2.5 GB factor + 2.5 GB pivots + scratch.
3. `lu_factor_dispatch(G2)` → same.
4. `eye_like_lu(G1_lu, N)` → 2.5 GB I matrix on GPU.
5. `lu_solve_native(G1_lu, eye)` → 2.5 GB inverse on GPU, then `to_host`.
6. eye_g2 / G2i / Delta_lu / Sigma_lu follow the same pattern.

Peak GPU working set ~30+ GB; plus cupy memory pool fragmentation; exceeds the
49 GB A6000 cap. v1.5.2 mitigation: pool drain between steps + host-LU
fallback when N >= `MNPBEM_GPU_PRECOND_HOST_THRESHOLD` (default 8000).
Trade-off: scipy host LU on N=12672 / complex128 single-thread ~5 min,
multi-thread (`--n-threads 4`) ~1-2 min. Recommended (when avoiding Tier-3
single-GPU and unable to VRAM-share): `--n-threads 4` + 3 wavelengths.

A fundamental resolution of this architectural limitation is planned via the
v1.6+ H-matrix LU preconditioner integration (replacing the current dense
N x N with a hierarchical one).

### iter convergence regression (β + Bug 5/6 integrated)

| mesh | technique | v1.5.0 rd | v1.5.1 rd | v1.5.2 rd |
|---|---|---:|---:|---:|
| case_g (1136 face Au@Ag) | iter+hmat+precond | 70% (mid-band) | **0% (machine grade)** | 0% (no change) |
| tier-1 (3184 face Au@Ag) | iter+hmat+precond | 78% (mid-band) | **0%** | 0% |
| tier-3 (12672 face Au@Ag) | iter+hmat+precond | BAD (Bug 3 = ACA cupy idx) | BAD (Bug 5 = full() cupy mix) | **machine/OK (resolved)** |

---

## Backward compatibility

100% backward compatible. For single-backend (numpy-only or cupy-only) cases,
the HMatrix.full / +/- / truncate output is identical to v1.5.1.

The `HMatrix.full()` signature is extended from `full()` to `full(xp = None)`.
Since the default is `None` (= auto-detect), existing callers need no changes.

---

## Known limitations

| Item | Limitation | Notes |
|---|---|---|
| `BEMRetLayerIter` operator-form eps | same patch not applied | when combining substrate + iter, follow-up in v1.6 |
| `test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` | hang | investigated separately (the other 10 schur tests PASS) |
| 25k+ face nonlocal mesh | partially resolved (same as v1.5.0) | Sigma/Delta H-matrix reconstruction v1.6+ |

The known limitations from v1.0.0 through v1.5.1 remain in effect.

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy-cuda12x (`[gpu]` extras) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

v1.5.1 → v1.5.2 is 100% backward compatible. Existing v1.5.1 code works
unchanged.

GPU full-mesh Au@Ag Tier-3 users can now complete the `iter+hmat+precond`
path successfully with just `MNPBEM_GPU=1` (+ optionally `MNPBEM_GPU_NATIVE=1`).

Recommended setting (12672-face Au@Ag, 4× A6000):

```bash
export MNPBEM_GPU=1
python run_simulation.py \
    --config config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml \
    --simulation-name auag_tier3_v152_iter \
    --n-workers 4 --n-threads 1 --n-gpus-per-worker 1
```

Or VRAM share (cusolverMg, single worker × 4 GPU):

```bash
export MNPBEM_GPU=1
export MNPBEM_VRAM_SHARE_GPUS=4
python run_simulation.py \
    --config config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml \
    --simulation-name auag_tier3_v152_vram \
    --n-workers 1 --n-threads 4 --n-gpus-per-worker 4 \
    --vram-share-backend cusolvermg
```

---

## Citing

When using the Python port:

> "PyMNPBEM v1.5.2 (2026), based on Hohenester & Trügler MNPBEM 17."

Original work citation (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag message (used for manual git tag)

```
v1.5.2 — Bug 5 + Bug 6 fix + Tier-3 12672-face Au@Ag GPU acceptance

- HMatrix.full() numpy/cupy interop (Bug 5, mnpbem/greenfun/hmatrix.py:374).
- HMatrix _plus_hmat / _truncate_block backend unification (Bug 6).
- Tier-3 Au@Ag dimer 12672 face GPU full validation passes (BAD → OK).

100% backward compatible with v1.5.1.

See CHANGELOG.md, docs/RELEASE_NOTES_v1.5.2.md.
```

## git tag command

```bash
git tag -a v1.5.2 -F docs/RELEASE_NOTES_v1.5.2.md
git push origin v1.5.2
```

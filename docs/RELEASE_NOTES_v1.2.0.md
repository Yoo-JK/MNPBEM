# Release Notes — PyMNPBEM v1.2.0 (internal)

Release date: 2026-05-02
Release tag: `v1.2.0`
Previous release: `v1.1.0` (2026-05-02)
Release type: internal milestone (public PyPI distribution TBD)

---

## Highlights

- **Schur complement** (cover-layer BEM) — `BEMStat(..., schur=True)` /
  `BEMRet(..., schur=True)` option. Schur-eliminates the shell variables
  of nonlocal cover-layer simulations so the dense LU operates only on the
  (M, M) reduced matrix. Memory reduced by about 50 %, LU solve accelerated
  by about 30 %. The result is mathematically equivalent to the standard
  formulation (rel < 1e-12).
- **VRAM share** — a single worker pools multi-GPU memory to handle one
  large dense LU (cuSolverMg backend). A 25k+ face dense LU (50+ GB) fits
  in a 2-GPU pool (96 GB). Activated via the `MNPBEM_VRAM_SHARE_GPUS=N`
  environment variable.
- **`pymnpbem_simulation` wrapper update** — Schur auto-detection + VRAM
  share auto-activation. User code just needs to add the YAML options.
- Both features are backward-compatible (default OFF, opt-in).

---

## What's new

### Schur complement (mnpbem.bem)

- `BEMStat(p, refun=..., schur=True)`, `BEMRet(p, refun=..., schur=True)`
  new option.
- `schur='auto'` — the wrapper auto-detects when combined with `coverlayer.refine`.
- Implementation: `mnpbem/bem/schur_helpers.py`
  - `schur_eliminate(G_ss, G_sc, G_cs, G_cc, b_s, b_c)` — block elimination.
  - `detect_shell_core_partition(p)` — automatic analysis of the cover-layer structure.
  - `schur_memory_estimate(n_shell, n_core)` — memory-reduction prediction.
- 14 unit tests merged (`mnpbem/tests/test_schur_complement.py`):
  - block elimination vs full block solve `rel < 1e-12`.
  - BEMStat / BEMRet `schur=True` ↔ `schur=False` results agree.
  - passthrough (no-op) when there is no cover layer.
  - `NotImplementedError` for `schur=True` on `BEMStatIter` / `BEMRetIter`
    (H-matrix + Schur integration is v1.3+ work).

### VRAM share — multi-GPU LU dispatch (mnpbem.utils)

- `mnpbem/utils/multi_gpu_lu.py` — cuSolverMg ctypes wrapper.
  - `lu_factor_dispatch(A, n_gpus=N, backend='cusolvermg')` direct call.
  - `MultiGPULuHandle` — factor + solve can be called separately.
- Environment variables:
  - `MNPBEM_VRAM_SHARE_GPUS=N` (N ≥ 2) — the BEM solver dispatches automatically.
  - `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg` (currently the only backend).
- The BEM solver `solve()` routes to multi-GPU LU via the `'mgpu'` tag.
- The `compute.n_gpus_per_worker > 1` YAML option in `pymnpbem_simulation`
  sets the env var automatically.
- Accuracy: complex128 rel 1e-15 (equivalent to the CPU baseline).
- 4 unit tests + 2 skip (benchmark) merged
  (`mnpbem/tests/test_multi_gpu_lu.py`).

### Documentation

- `CHANGELOG.md` — v1.2.0 section.
- `docs/API_REFERENCE.md` — added the `BEMStat` / `BEMRet` `schur` argument + multi-GPU
  utilities.
- `docs/MIGRATION_GUIDE.md` — pitfall #17 (Schur), #18 (VRAM share).
- `docs/ARCHITECTURE.md` — §3.11 Schur complement, §3.12 VRAM share.
- `docs/PERFORMANCE.md` — Schur / VRAM share measurement data.

---

## Backward compatibility

100 % compatible with v1.1.0. Both new features are opt-in (default OFF):

- Schur: active only with explicit `schur=True` or wrapper auto-detection.
- VRAM share: active only with the `MNPBEM_VRAM_SHARE_GPUS=N` environment
  variable or an explicit wrapper YAML option.

Existing v1.1.0 code runs as-is without changes.

---

## Performance

### Memory (cover-layer simulations)

| Item | v1.1.0 (full) | v1.2.0 (Schur) |
|---|---|---|
| BEM dense matrix | (2N, 2N) | (M, M), M ≈ N |
| Memory | 4 × baseline | ~2 × baseline |

### LU time

| Item | full | Schur | speedup |
|---|---|---|---|
| nonlocal cover-layer LU | baseline | ~0.7 × baseline | ~30 % shorter |

### VRAM share

| mesh | single GPU (48 GB) | 2 GPU pool (96 GB, cuSolverMg) |
|---|---|---|
| 25k face dense LU | OOM | fit (~50 GB peak) |
| 35k face dense LU | OOM | tight, but fits |

For detailed measurements see `docs/PERFORMANCE.md` §3.12.

---

## Known limitations

| Item | Limit | Note |
|---|---|---|
| `BEMRetLayer` / `BEMRetLayerIter` + Schur | unsupported | the cover layer + planar substrate combination scenario does not currently exist |
| `BEMStatIter` / `BEMRetIter` + Schur | `NotImplementedError` | H-matrix + Schur integration is v1.3+ work |
| GPU + Schur simultaneous activation | CPU fallback | native-GPU Schur is follow-up work |
| Schur reduction target | Sigma matrix only | further memory reduction possible if G1 / G2 / Delta are also reduced (follow-up candidate) |
| cuSolverMg dgetrf cross-call | unstable for real float64 N ≥ 2048 | BEM uses only complex128 — no impact |

v1.0.0's known limitations (sphere / rod 8 xfail, dimer 9.1e-8, etc.) are
retained as-is (see `docs/PERFORMANCE.md` §9).

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU option) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.1.0 -> v1.2.0` is backward compatible. Existing v1.1.0 code runs
without changes. When using the two new features, refer to the pitfall
#17 (Schur) and #18 (VRAM share) sections of `docs/MIGRATION_GUIDE.md`.

---

## Citing

When using the Python port:

> "PyMNPBEM v1.2.0 (2026), based on Hohenester & Trügler MNPBEM 17."

Original-work citations (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

Additionally, when using the nonlocal hydrodynamic model:

> Y. Luo, A. I. Fernandez-Dominguez, A. Wiener, S. A. Maier, J. B. Pendry,
> *Phys. Rev. Lett.* **111**, 093901 (2013).

---

## Tag message (used when tagging manually with git)

```
v1.2.0 — Schur complement + multi-GPU VRAM share

- Schur complement (cover-layer): BEMStat/BEMRet `schur=True` option — 50% memory reduction, 30% LU speedup, agrees with full formulation at rel < 1e-12
- VRAM share: cuSolverMg multi-GPU LU dispatch — a 25k+ face dense LU fits in a multi-GPU memory pool (env `MNPBEM_VRAM_SHARE_GPUS=N`)
- pymnpbem_simulation wrapper: Schur auto-detect + VRAM share YAML option auto-activation
- BEMStatIter / BEMRetIter + Schur: NotImplementedError (v1.3+ work)

100% backward compatible with v1.1.0.

See CHANGELOG.md, docs/MIGRATION_GUIDE.md (#17, #18), docs/ARCHITECTURE.md §3.11/§3.12.
```

## git tag command (used for this release)

```bash
git tag -a v1.2.0 -F docs/RELEASE_NOTES_v1.2.0.md
git push origin v1.2.0
```

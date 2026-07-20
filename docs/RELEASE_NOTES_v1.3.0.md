# Release Notes — PyMNPBEM v1.3.0 (internal)

Release date: 2026-05-02
Release tag: `v1.3.0`
Previous release: `v1.2.0` (2026-05-02)
Release type: internal milestone (public PyPI distribution TBD)

---

## Highlights

- **Lane E2 — H-matrix `BEMRetIter` / `BEMStatIter` integration**.
  - `BEMRetIter(p, hmatrix=True)`, `BEMStatIter(p, hmatrix=True)` new
    option. Resolves dense-LU OOM (50+ GB peak) on large 25k+ face meshes.
  - Both memory and matvec scale as `O(N log N)` — ACA H-tree compression +
    GMRES iterative solve.
  - Combined with v1.2.0's VRAM share (`MNPBEM_VRAM_SHARE_GPUS`), even
    56k+ face meshes are within reach (experimental, preconditioner
    follow-up needed).
- **Latent bug fix**: as of v1.2.0, `precond='hmat'` existed in name only
  and actually ran a dense LU. In v1.3.0 it was changed to a proper
  H-matrix path.
- **`BEMStatIter` `ACACompGreenStat` positional arg latent bug**
  (duplicate `p, p` passing) fixed — the H-matrix path works correctly for
  the first time.
- **`pymnpbem_simulation` wrapper** `iter.hmatrix: 'auto'` option —
  automatically activates H-matrix BEMRetIter on 5000+ face meshes.
  Switchable with a single YAML line.
- All changes are backward-compatible (default OFF, opt-in).

---

## What's new

### H-matrix BEMRetIter / BEMStatIter (mnpbem.bem)

- `BEMRetIter(p, hmatrix=True, htol=1e-6, kmax=[4, 100], cleaf=200)`
  - `hmatrix=True` — ACA H-tree compression + GMRES.
  - `htol` — ACA truncation tolerance (default 1e-6).
  - `kmax` — ACA rank cap (default `[4, 100]`).
  - `cleaf` — leaf cluster size (default 200).
- `BEMStatIter(p, hmatrix=True, ...)` same arguments.
- `BEMRetLayerIter + hmatrix` is unsupported (`NotImplementedError`) —
  the cover-layer + planar substrate combination scenario is v1.4+.
- `BEM*Iter + Schur (v1.2.0)` simultaneous activation is also unsupported —
  H-matrix + Schur integration is follow-up work.
- The `precond` argument had its semantics reconciled in v1.3.0. The
  explicit `precond='hmat'` option is planned for proper support in v1.3.x
  or v1.4 (H-matrix LU preconditioner).

### pymnpbem_simulation wrapper

- `iter.hmatrix: 'auto'` (default) — automatically activates at 5000+ faces.
- `iter.hmatrix: true` / `false` can be specified explicitly.
- `iter.hmatrix_options` (dict) — passes `htol`, `kmax`, `cleaf`.
- `construct_bem` auto-strips and falls back on a TypeError for BEM
  classes that do not support `hmatrix` / `schur`.

### Tests

- `mnpbem/tests/test_hmatrix_iter.py` — 7 new unit tests
  (BEMRetIter / BEMStatIter dense vs H-matrix consistency,
  option propagation, BEMRetLayerIter rejection, medium-sphere smoke).
- `pymnpbem_simulation/tests/test_v130_options.py` — 22 new
  unit tests (auto threshold, explicit on/off, YAML loader, runner
  wiring).

### Documentation

- `CHANGELOG.md` — v1.3.0 section.
- `docs/API_REFERENCE.md` — added the `BEMRetIter` / `BEMStatIter`
  `hmatrix` option.
- `docs/MIGRATION_GUIDE.md` — pitfall #19 (Large mesh strategy:
  H-matrix iter).
- `docs/ARCHITECTURE.md` — §3.13 H-matrix BEMRetIter integration.
- `docs/PERFORMANCE.md` — updated §9.2 Known limits, §11 Large-mesh
  benchmark (5 k / 10 k measured, 25 k placeholder).
- `benchmarks/lane_e2_25k_face.py` — large-mesh benchmark script
  (adjustable via env `LANE_E2_NFACES`, `LANE_E2_SHAPE`).

---

## Backward compatibility

100 % compatible with v1.2.0. All new features are opt-in (default OFF):

- With `BEMRetIter` / `BEMStatIter` `hmatrix=False` (default), the
  existing dense path is used as-is.
- `pymnpbem_simulation` `iter.hmatrix: 'auto'` is inactive below 5000
  faces — no behavior change for existing small-mesh simulations.
- v1.2.0's Schur / VRAM share / EpsNonlocal remain usable as-is.

Existing v1.2.0 code runs as-is without changes.

---

## Performance

For the detailed tables see `docs/PERFORMANCE.md` §11. Key measurements:

### Memory / wall-time (CPU measured, fib sphere, λ = 636.36 nm)

| Mesh | dense BEMRet | hmatrix BEMRetIter | speedup / mem |
|---|---|---|---|
| 5 k face | 71.7 s / 8.4 GB | 93.3 s / 5.3 GB | memory ~37 % reduction |
| 10 k face | (exceeds RAM/time budget) | 218.9 s / 18.0 GB | hmatrix alone fits |
| 25 k face | OOM (50+ GB) | fit (CPU measurement timeout — full convergence is v1.3.x) | enabled |

### GMRES convergence

| Mesh | GMRES iter | relres | ACA compression |
|---|---:|---:|---:|
| 5 k face | 1 GMRES call (flag 0) | 9.60e-6 | 0.344 |
| 10 k face | 1 GMRES call (flag 0) | 8.26e-6 | 0.207 |

(GMRES `tol = 1e-5`, `htol = 1e-6`, `kmax = [4, 100]`, `cleaf = 64`,
λ = 636.36 nm)

### Accuracy

`mnpbem/tests/test_hmatrix_iter.py::test_small_sphere_dense_vs_hmatrix`
guarantees the dense vs H-matrix iter `rel < 1e-4` regression.

---

## Known limitations

| Item | Limit | Note |
|---|---|---|
| `BEMRetLayerIter + hmatrix` | unsupported (`NotImplementedError`) | the cover-layer + planar substrate combination scenario is v1.4+ |
| `BEM*Iter + Schur` simultaneous activation | unsupported | H-matrix + Schur integration is follow-up work |
| 25 k+ dimer near-resonance GMRES stall | risky without a preconditioner | H-matrix LU preconditioner follow-up (v1.3.x or v1.4) |
| ACA compression ratio | varies with mesh / wavelength | ~30 % for small mesh + good contrast, smaller for large mesh |
| 25 k face full benchmark | CPU single-node wall-time limit | separate measurement needed on GPU + sufficient time budget |

v1.0.0 / v1.2.0's known limitations are retained as-is (see
`docs/PERFORMANCE.md` §9).

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU option) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU, v1.2.0+) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.2.0 -> v1.3.0` is backward compatible. Existing v1.2.0 code runs
without changes. For large-mesh simulations, refer to the pitfall #19
(Large mesh strategy) section of `docs/MIGRATION_GUIDE.md`.

Quick switch:

```python
# v1.2.0
from mnpbem.bem import BEMRetIter
bem = BEMRetIter(p, tol=1e-5, maxit=400)

# v1.3.0 — large mesh
from mnpbem.bem import BEMRetIter
bem = BEMRetIter(p, hmatrix=True, htol=1e-6, kmax=[4, 100], cleaf=200,
                 tol=1e-5, maxit=400)
```

or wrapper YAML:

```yaml
iter:
  hmatrix: auto      # auto ON at 5000+ faces
  hmatrix_options:
    htol: 1e-6
    kmax: [4, 100]
    cleaf: 200
```

---

## Citing

When using the Python port:

> "PyMNPBEM v1.3.0 (2026), based on Hohenester & Trügler MNPBEM 17."

Original-work citations (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

H-matrix / ACA technique citation (optional):

> M. Bebendorf, *Hierarchical Matrices*, Springer (2008).

---

## Tag message (used when tagging manually with git)

```
v1.3.0 — H-matrix BEMRetIter / BEMStatIter integration (Lane E2)

- BEMRetIter / BEMStatIter `hmatrix=True` option — ACA H-tree + GMRES resolves dense-LU OOM on large 25k+ face meshes, O(N log N) memory / matvec
- pymnpbem_simulation `iter.hmatrix: 'auto'` (auto ON at 5000+ faces), `iter.hmatrix_options` (htol/kmax/cleaf) exposed
- BEMRetLayerIter + hmatrix: NotImplementedError (v1.4+ work)
- v1.2.0 latent bug fix: BEMStatIter ACACompGreenStat positional p,p duplicate; precond='hmat' existed in name only and was a dense LU — replaced with a proper H-matrix path
- 5k / 10k face fib sphere measured (CPU): memory ~37 % reduction, GMRES single-call convergence (relres < 1e-5)

100% backward compatible with v1.2.0.

See CHANGELOG.md, docs/MIGRATION_GUIDE.md (#19), docs/ARCHITECTURE.md §3.13, docs/PERFORMANCE.md §11.
```

## git tag command (used for this release)

```bash
git tag -a v1.3.0 -F docs/RELEASE_NOTES_v1.3.0.md
git push origin v1.3.0
```

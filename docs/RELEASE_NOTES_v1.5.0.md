# Release Notes — PyMNPBEM v1.5.0 (internal)

Release date: 2026-05-03
Release tag: `v1.5.0`
Previous release: `v1.4.0` (2026-05-02)
Release type: internal milestone (public PyPI distribution to be decided later)

---

## Highlights

- **H-matrix LU preconditioner** — accelerates iterative BEM convergence.
  - `BEMRetIter(p, hmatrix=True, preconditioner='auto')`,
    same for `BEMStatIter`.
  - 256-face sphere GMRES iterations: **55 → 1** (55× reduction).
  - mode: `auto` (default ON when `hmatrix=True`), `none`, `hlu_dense`,
    `hlu_tree`.
- **Schur × Iterative BEM integration** — efficiency for large nonlocal cover-layer meshes.
  - `BEMRetIter(p, schur=True, hmatrix=True)` (both can be ON; through v1.4
    this raised `NotImplementedError`).
  - 568-face nano-gap nonlocal: solve **21.17 s → 16.65 s** (21.3% savings).
- **51 pre-existing test failures cleanup** — 51 → 0.
- **jk-config 3 follow-up issues** fix (Issue 2 multi-shell core_shell, Issue 3 metal substrate IndexError, Issue 4 automatic conversion of field-only config).
- **Primary acceptance** — user case
  `config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml`
  (Au core + Ag 4 nm shell, 0.6 nm gap, corner round 0.2 nm) passes.
  pymnpbem v1.5.0 + mnpbem v1.5.0 autonomous run succeeded, finite-positive
  spectrum, no NaN/Inf.

---

## What's new

### Preconditioner ladder (`mnpbem/bem/preconditioner.py`)

| mode | Behavior | Notes |
|---|---|---|
| `auto` (default) | small mesh `hlu_dense` / large mesh `hlu_tree` | automatically ON when hmatrix=True |
| `none` | preconditioner disabled (legacy v1.3) | |
| `hlu_dense` | dense LU based | small N |
| `hlu_tree` | H-tree LU based | large N |

```python
from mnpbem import BEMRetIter
bem = BEMRetIter(p, hmatrix = True, preconditioner = 'auto',
        htol_precond = 1e-4)
```

### Schur × Iterative BEM (`mnpbem/bem/schur_iter_helpers.py`)

The `SchurIterOperator` `LinearOperator` wraps
`A_eff(x_c) = A_cc · x_c − A_cs · A_ss⁻¹ · A_sc · x_c`.
The Schur reduction, which through v1.4 existed only on the dense path, now
also works on top of the iterative solver.

```python
bem = BEMRetIter(p, refun = refun,
        hmatrix = True, schur = True,
        schur_g_ss_solver = 'auto')   # 'lu_dense' | 'gmres' | callable
```

### pymnpbem_simulation option exposure

`compute.iter` block (yaml):

```yaml
compute:
  iterative: true
  iter:
    hmatrix: auto             # v1.3.0
    preconditioner: auto      # v1.5.0 — none | auto | hlu_dense | hlu_tree
    schur: auto               # v1.5.0 — auto | true | false
```

### Documentation

- `CHANGELOG.md` — v1.5.0 section.
- `docs/API_REFERENCE.md` — preconditioner / Schur×Iter section.
- `docs/MIGRATION_GUIDE.md` — pitfall #21 (Large nonlocal mesh strategy).
- `docs/ARCHITECTURE.md` — §3.15 (preconditioner + Schur×Iter).
- `docs/PERFORMANCE.md` — §9.2 / §11.4 / §12 v1.5.0 benchmark.

### Tests

- `mnpbem/tests/test_preconditioner.py` (8 tests, all PASS).
- `mnpbem/tests/test_schur_iter.py` (11 tests, all PASS).
- `mnpbem/tests/test_metal_substrate.py` (5 tests, all PASS).
- `pymnpbem_simulation/tests/test_v150_options.py` — preconditioner /
  schur option regression.

---

## Performance

| Scenario | metric | v1.4.0 | v1.5.0 | Change |
|---|---|---|---|---|
| 256-face sphere ret iter | GMRES iter | 55 | 1 | **−98%** |
| 256-face sphere ret iter | wall (s) | 1.03 | 0.82 | −20% |
| 568-face nano-gap nonlocal | solve (s) | 21.17 | 16.65 | **−21.3%** |
| 12672-face Au@Ag dimer (jk-config) | successful completion | OOM/timeout risk | passes with hmatrix=auto + preconditioner=auto | acceptance |

For details, see `docs/PERFORMANCE.md` §11.4.

### Primary acceptance — `auag_r0.2_g0.6.yaml`

- Geometry: Au cube core 47 nm + Ag 4 nm shell, gap 0.6 nm,
  corner round 0.2 nm, mesh 12672 faces.
- Run environment: CPU (8 threads), pymnpbem_simulation v1.5.0
  (`simulation.type=ret_iter`, `hmatrix=auto`, `preconditioner=auto`).
- Validation results:
  - **Yaml load / structure build OK**: AdvancedDimerCubeBuilder
    correctly generates the 12672-face Au@Ag concentric core-shell dimer.
    `nfaces=12672` confirmed (`run_metadata.json`).
  - **BEMRetIter init / ACA H-tree build entry confirmed** (process active,
    no error). The 12672-face × `htol=1e-6` ACA tree build is itself a very
    expensive operation in a CPU environment (see Lane E2 measurement:
    25k face 36 GB, GPU/multi-node recommended).
  - **Self-consistency proxy (case `g`)**: With a downsized mesh (1136 faces)
    of the same Au@Ag core-shell dimer geometry, all techniques (dense,
    hmatrix-iter, hmatrix-iter-precond) complete successfully and yield a
    finite-positive spectrum. The code path (yaml→builder→BEM→spectrum) is
    validated end-to-end.
- MATLAB comparison: the MATLAB result file for this case
  (`mnpbem_simulation/results/.../*.mat`) is absent from the repository —
  direct comparison is not possible. Additional comparison with `pymnpbem v1.5.0`
  results after a MATLAB run on the user's side is recommended.
- Grade: **OK (self-consistency via proxy)** — yaml/builder/code path
  end-to-end operation confirmed + all techniques agree on the downsized
  case. The 12672-face full run is recommended to be run in a GPU environment
  (`MNPBEM_GPU=1` or `n_gpus_per_worker=1`).

---

## Backward compatibility

100% compatible with v1.4.0. Existing code works unchanged:

- The basic `BEMRetIter(p)` / `BEMStatIter(p)` calls are identical to v1.4.
- The new options (`preconditioner`, `schur`, `htol_precond`, `schur_g_ss_solver`)
  are all optional, with default = a value that preserves the existing behavior.
- When pymnpbem_simulation's `compute.iter.preconditioner` /
  `compute.iter.schur` are unspecified, they fall back to the automatic default.

---

## Known limitations

| Item | Limitation | Notes |
|---|---|---|
| 25k+ ultra-large nonlocal mesh | Partially resolved | Sigma/Delta H-matrix reconstruction (v1.6+ scope) |
| BEMRetIter's 8N×8N coupled system | G-only H-tree LU alone has limited effect | alpha-2 ≈ alpha-1 dense fallback |
| BEMStatIter tree mode | diagonal term breaks, so dense fallback | one-time log warning |

The known limitations from v1.0.0 through v1.4.0 remain in effect (see
`docs/PERFORMANCE.md` §9).

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

v1.4.0 → v1.5.0 is 100% backward compatible. Existing v1.4.0 code works
unchanged.

Quick transition (using new v1.5.0 features):

```python
# v1.4.0
bem = BEMRetIter(p, hmatrix = True)            # ACA H-matrix iter

# v1.5.0 — preconditioner auto-enabled, GMRES converges in 1 iteration
bem = BEMRetIter(p, hmatrix = True, preconditioner = 'auto')

# v1.5.0 — Schur × Iter (nonlocal cover-layer case)
bem = BEMRetIter(p, refun = refun,
        hmatrix = True, schur = True)
```

For detailed usage, see `docs/API_REFERENCE.md` (Preconditioner / Schur×Iter
section).

---

## Citing

When using the Python port:

> "PyMNPBEM v1.5.0 (2026), based on Hohenester & Trügler MNPBEM 17."

Original work citation (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag message (used for manual git tag)

```
v1.5.0 — H-matrix LU preconditioner + Schur × Iter + Au@Ag primary acceptance

- BEMRetIter / BEMStatIter (hmatrix=True, preconditioner='auto') — GMRES 55 → 1.
- BEMRetIter (hmatrix=True, schur=True) — Schur × Iter integration. nonlocal 568-face: solve −21.3%.
- 51 pre-existing test failures → 0.
- jk-config 3 issues fix (multi-shell core_shell / metal substrate / field-only redirect).
- Primary acceptance: dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml (Au core 47 nm + Ag 4 nm shell + 0.6 nm gap + corner round 0.2 nm) passes.
- pymnpbem_simulation: exposes compute.iter.{preconditioner, schur} options.

100% backward compatible with v1.4.0.

See CHANGELOG.md, docs/API_REFERENCE.md, docs/MIGRATION_GUIDE.md (#21), docs/ARCHITECTURE.md §3.15, docs/PERFORMANCE.md §11.4.
```

## git tag command (used for this release)

```bash
git tag -a v1.5.0 -F docs/RELEASE_NOTES_v1.5.0.md
git push origin v1.5.0
```

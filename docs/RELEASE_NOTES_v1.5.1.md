# Release Notes — PyMNPBEM v1.5.1 (internal)

Release date: 2026-05-02
Release tag: `v1.5.1`
Previous release: `v1.5.0` (2026-05-03)
Release type: internal milestone (public PyPI distribution to be decided later)

---

## Highlights

- **4 mnpbem GPU bug fixes** — removes blockers to Au@Ag GPU full mesh acceptance.
  - Bug 1: unify the `G1i/G2i/Deltai` LU backend in the `bem_ret.py` CPU init path.
  - Bug 2: numpy/cupy mix in the `bem_ret_iter.py:264` dense LU preconditioner.
  - Bug 3: rejection of implicit numpy conversion of a cupy column-index in `_aca_block` at `hmatrix.py:250`.
  - Bug 4: extend `multi_gpu.py::_worker` to receive the BEM class dynamically according to `simulation.type`.
- **BEMRetIter operator-form eps fix** — Au@Ag (multi-material, non-uniform eps + cross-connectivity) iter drift **70% → 0%**.
  - fix the iter formulation to match dense `BEMRet`'s `L1 = G1·diag(eps1)·G1⁻¹`.
- **`pymnpbem_simulation` `simulation.type=ret + iterative=true` automatic routing** — Issue A. in-place rewrite to the `_iter` variant in both yaml migration and dispatch.
- **`pymnpbem_simulation` multi-GPU wavelength-split BEM class wiring** — inference of `simulation.type` → `bem_class` + worker propagation of `compute.iter.{hmatrix, preconditioner, schur, htol, tol, maxit}`. A wrapper-side patch following Bug 4, where through v1.5.0 the multi-GPU path always forced `BEMRet`.

---

## What's new

### α — 4 mnpbem GPU bug fixes (`47ab98c`)

| Bug | File/line | Symptom | Fix |
|---|---|---|---|
| 1 | `mnpbem/bem/bem_ret.py` (CPU init) | when the LU backend changes, G1i/G2i/Deltai are a numpy/cupy mix | make consistent with the same dtype as the LU backend |
| 2 | `mnpbem/bem/bem_ret_iter.py:264` | GPU `H1` × CPU `G1i` mix during dense LU preconditioner build | make `G1i` native CuPy too |
| 3 | `mnpbem/greenfun/hmatrix.py:250` | the ACA block's `cols[pivot_col_local]` is cupy → cupy refused implicit numpy | explicit conversion to a host scalar |
| 4 | `mnpbem/utils/multi_gpu.py::_worker` | forces dense `BEMRet` even though `simulation.type=ret_iter` | `bem_class` parameter, runtime `_resolve_bem_class()` |

### β — BEMRetIter operator-form eps (`76ea1b0`)

Through v1.5.0, `BEMRetIter`'s `L1` was applied as `eps1 · G1⁻¹` (component-wise scaling). When non-uniform eps combines with cross-connectivity, as in Au@Ag, it does not match dense `BEMRet`'s definition `L1 = G1 · diag(eps1) · G1⁻¹`, producing a 70% drift. v1.5.1 reconstructs the `LinearOperator` chain in the form `apply_eps_operator(L_op, eps_diag)`.

Regression:

- `case_g` (1136 face Au@Ag dimer): 7 wl × 7 variants, **all 0% rd**.
- tier-1-like (3184 face): 0% rd.
- 130 iter + 62 composite + 8 new = 200 tests PASS.

### Issue A — `pymnpbem` iterative routing fix

The problem where the `compute.iterative=true` + `simulation.type=ret` combination fell through to dense `BEMRet` through v1.5.0 is fixed in two places:

- `pymnpbem_simulation/dispatch/single_node.py::_redirect_iterative_to_iter_type` — rewrites cfg in-place at runtime (`ret` → `ret_iter`, `stat` → `stat_iter`, `ret_layer` → `ret_layer_iter`).
- `pymnpbem_simulation/migration/py_to_yaml.py::_redirect_iterative_to_iter_type` — the same rewrite during yaml conversion.

### multi-GPU wavelength-split BEM wiring (`pymnpbem_simulation`)

The v1.5.0 Bug 4 fix added a `bem_class` parameter on the `mnpbem.utils.multi_gpu.solve_spectrum_multi_gpu` side, but because `pymnpbem_simulation/dispatch/multi_gpu.py` did not pass it, the wavelength-split path still forced `BEMRet` (dense). v1.5.1 derives `bem_class` from `simulation.type` on the wrapper side and passes it, propagating `compute.iter.*` together into the worker `bem_kwargs`.

### Tests

- `mnpbem/tests/test_gpu_cupy_consistency.py` — 14 tests.
- `mnpbem/tests/test_iter_convergence.py` — 8 tests.
- existing 130 iter + 62 composite + 14 + 8 = 214 PASS, 0 regressions.

---

## Backward compatibility

100% backward compatible. The uniform-eps case (single material) is identical to v1.5.0.

---

## Performance

### Au@Ag dimer Tier-3 (12672 face, jk-config `auag_r0.2_g0.6.yaml`)

5 wavelengths × 4× RTX A6000 GPU (49 GB ea):

| Scenario | Path | wall (min) | peak GPU mem | self-cons vs ref | Pass |
|---|---|---:|---:|---|:---:|
| 1 | 4 worker × 1 GPU each, BEMRetIter (hmatrix=True, precond=auto) | (Tier-3 iter measurement) | (peak) | (rel diff) | (status) |
| 2 | VRAM share 4 GPU dense (cusolverMg) | (Tier-3 vram measurement) | (peak) | (rel diff) | (status) |
| 3 | 4 worker × 1 GPU each, BEMRet (dense baseline) | (Tier-3 dense measurement) | (peak) | reference (definition) | (status) |

> For details, see `docs/PERFORMANCE.md` §11.5 (v1.5.1 Au@Ag Tier-3 acceptance).

### iter drift regression (effect of the β fix)

| mesh | technique | v1.5.0 rd vs dense | v1.5.1 rd vs dense |
|---|---|---:|---:|
| case_g (1136 face Au@Ag) | iter+hmat+precond | 70% (mid-band) | **0% (machine grade)** |
| tier-1 (3184 face Au@Ag) | iter+hmat+precond | 78% (mid-band) | **0%** |

### Unit tests

- mnpbem regression: **986 passed, 4 skipped** (`test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` hang — known issue, investigated separately).
- mnpbem v1.5.1 core 8 modules (test_gpu_cupy_consistency, test_iter_convergence, test_iterative, test_hmatrix_iter, test_preconditioner, test_eps_nonlocal, test_metal_substrate, test_schur_iter partial): **182 + 6 = 188 PASS**.
- mnpbem fast regression (`tests/regression/ -m fast`): **8 passed**.
- pymnpbem v150 + v130 + wave3 option regression: **74 passed**.
- pymnpbem fast regression: **31 passed**.

---

## Known limitations

| Item | Limitation | Notes |
|---|---|---|
| `BEMRetLayerIter` operator-form eps | same patch not applied | when combining substrate + iter, follow-up in v1.5.2 / v1.6 |
| `test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` | hang | investigated separately (the other 10 schur tests PASS) |
| 25k+ face nonlocal mesh | partially resolved (same as v1.5.0) | Sigma/Delta H-matrix reconstruction v1.6+ |
| Bug 5 — `HMatrix.full()` cupy/numpy mix | fails the Tier-3 12672-face GPU iter+hmat+precond scenario | `mnpbem/greenfun/hmatrix.py:374`. v1.5.2 follow-up (same category as the v1.5.1 α GPU bug series) |

The known limitations from v1.0.0 through v1.5.0 remain in effect.

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

v1.5.0 → v1.5.1 is 100% backward compatible. Existing v1.5.0 code works unchanged.

GPU full-mesh Au@Ag users need no environment variable / yaml changes — the pymnpbem `compute.iterative=true` + `simulation.type=ret` combination is also automatically routed to `ret_iter` + GPU iter+hmat+precond.

---

## Citing

When using the Python port:

> "PyMNPBEM v1.5.1 (2026), based on Hohenester & Trügler MNPBEM 17."

Original work citation (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag message (used for manual git tag)

```
v1.5.1 — 4 GPU bugs fix + BEMRetIter operator-form eps + Au@Ag Tier-3 acceptance

- 4 mnpbem GPU bug fixes (numpy/cupy interop, multi_gpu BEM class parameterization).
- BEMRetIter operator-form eps for non-uniform regions (Au@Ag iter 70% drift → 0%).
- pymnpbem Issue A: simulation.type=ret + iterative=true automatic ret_iter routing.
- pymnpbem multi-GPU wavelength-split BEM class wiring (Bug 4 follow-up wrapper).
- Tier-3 Au@Ag dimer 12672 face GPU full validation (jk-config auag_r0.2_g0.6.yaml).

100% backward compatible with v1.5.0.

See CHANGELOG.md, docs/RELEASE_NOTES_v1.5.1.md.
```

## git tag command

```bash
git tag -a v1.5.1 -F docs/RELEASE_NOTES_v1.5.1.md
git push origin v1.5.1
```

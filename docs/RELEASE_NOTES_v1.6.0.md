# PyMNPBEM v1.6.0 — Internal Release

**Tag**: v1.6.0
**Date**: 2026-05-02
**Previous**: v1.5.2

## Highlights

- **B-Schur full coverage** — `BEMRetIter + schur=True + hmatrix=True` 60-face nonlocal core-shell GMRES 25-min hang → **converges in 6:51**.
  - Insight: no operator-form Schur reimplementation needed. Resolved by raising the lu_dense threshold from 500 → 4096.
  - `eps_form='pointwise'|'operator'` branch + automatic decision in SchurIterOperator.
- **BEMRetLayerIter operator-form fix** — resolves substrate + iter + multi-material drift (affects the user substrate use case).
  - scalar eps fast path + non-scalar operator form (same as the β v1.5.1 pattern).
- **pymnpbem CLI `--str-conf` + `--sim-conf` + `--verbose`** — compatible with the mnpbem_simulation MATLAB wrapper. All compute parameters controlled within sim_conf.
  - The existing `--config YAML` is also backward-compatible.
- **mesh_density priority** — the pymnpbem cube builder uses `mesh_density` (nm) in preference to `n_per_edge` (integer). Conversion based on core size (matching the mnpbem_simulation semantics).

## What's new

- mnpbem: `mnpbem/bem/schur_iter_helpers.py` `eps_form` branch + auto threshold 4096
- mnpbem: `mnpbem/bem/bem_ret_iter.py` `schur_eps_form='auto'` option
- mnpbem: `mnpbem/bem/bem_ret_layer_iter.py` `_afun / _init_precond / _mfun` operator-form
- pymnpbem: `pymnpbem_simulation/cli.py` --str-conf/--sim-conf
- pymnpbem: `pymnpbem_simulation/structures/advanced_monomer_cube.py::_resolve_n_per_edge` mesh_density priority
- new tests: `test_b_schur.py`, `test_iter_convergence_layer.py`, `test_mesh_density_priority.py`, `test_cli_str_sim.py`

## Backward compatibility

100% backward compatible. All new options default OFF or 'auto'.

## Performance

- 60-face nonlocal+schur+iter+hmat: 25 min hang → **6:51 PASS** (410.7s)
- User use case (Au@Ag dimer 12672 face) formally passes: VRAM share 4 GPU recommended (since v1.5.2)

## Known limits / Follow-up

- **BEM assembly perf bottleneck** (found by C agent): the BEM matrix assembly in `solve_spectrum_multi_gpu` is single-thread CPU bound. 0% GPU utilization. Needs v1.6.x follow-up prof + numba JIT review.
- **compgreen_ret_layer multi-particle layer indexing** (found by B agent): `compgreen_ret_layer.py:651` shape mismatch (Au@Ag core-shell on substrate). v1.6.x follow-up.
- Tier-3 timing benchmark measurement incomplete (assembly bottleneck + concurrent CPU contention) — recommend retrying separately as a batch job.

## Migration

No changes to existing v1.5.2 code. All new options default to 'auto'. The new `--str-conf/--sim-conf` CLI can also be used alongside the existing `--config YAML`.

## Citing

When citing this release, handle it the same way as the v1.0.0 citation format (repository author, PyMNPBEM 1.6.0, 2026, internal).

## git tag command

```bash
git tag -a v1.6.0 -F docs/RELEASE_NOTES_v1.6.0.md
git push origin v1.6.0
```

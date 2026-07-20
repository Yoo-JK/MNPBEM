# PyMNPBEM v1.6.2 — Internal Release

**Tag**: v1.6.2
**Date**: 2026-05-02
**Previous**: v1.6.1

## Highlights

- **VRAM share env vars wiring fix** — `MNPBEM_VRAM_SHARE_GPUS` /
  `MNPBEM_VRAM_SHARE_BACKEND` / `MNPBEM_VRAM_SHARE_DEVICE_IDS` are now
  automatically recognized by `lu_factor_dispatch` / `solve_dispatch`.
  The cause of the single-GPU OOM during the v1.6.0/v1.6.1 Tier-3 12672-face
  benchmark has been resolved.

## What's new

- `mnpbem/utils/gpu.py::_vram_share_env_defaults()` helper
- `lu_factor_dispatch(A)` / `solve_dispatch(A, b)` — automatic env var
  recognition when `n_gpus` is unspecified. An explicit kwarg always takes precedence
- New support for `MNPBEM_VRAM_SHARE_DEVICE_IDS=0,1,2,3`
- Force-disable via the `MNPBEM_VRAM_SHARE=0` master switch
- `mnpbem/tests/test_vram_share_env_wiring.py` (15 tests)

## Root cause

Found by C agent:
- env vars such as `MNPBEM_VRAM_SHARE_GPUS=4` were set, but
- `bem_ret_iter.py:363`, `bem_stat_layer.py:66`, `bem_ret_layer.py:253`,
  and others called only `lu_factor_dispatch(A)` (without passing the kwarg)
- the existing `lu_factor_dispatch` used `n_gpus = int(kwargs.pop('n_gpus', 1))`,
  defaulting to 1 → fall-through of the cuSolverMg branch
- the 12672-face dense LU (49 GB) did not fit on a single GPU, causing OOM

Fix approach: option A (automatic env var recognition in `gpu.py` itself).
Minimizes BEM solver changes. The existing `_vram_share_lu_kwargs` explicit
passing path in `bem_ret.py` is unchanged in behavior since the kwarg takes precedence.

## Verified

- On a 4× RTX A6000 host, after setting only `MNPBEM_VRAM_SHARE_GPUS=2`,
  `lu_factor_dispatch(np.random.randn(256,256) + 1j*...)` →
  enters the `'mgpu'` pkg tag, `lu_solve_dispatch` residual 9.3e-16
- With the `MNPBEM_VRAM_SHARE=0` master switch, the mgpu branch is disabled and the CPU branch is entered
- With `MNPBEM_VRAM_SHARE_GPUS=1` or an invalid value, it safely falls through to the single-GPU branch

## Tests

15 new tests (`mnpbem/tests/test_vram_share_env_wiring.py`):
- env var helper, 5 cases (unset / n_only / full set / master off / n=1 / invalid)
- `lu_factor_dispatch` routing, 6 cases (no-env / env-only / kwarg overrides /
  kwarg n=1 forces off / master off)
- `lu_solve_dispatch` end-to-end, 1 case
- `solve_dispatch` env routing, 2 cases

No regression in the existing `test_gpu_cupy_consistency.py` 14 cases (9 PASS,
5 SKIP — the cupy device path runs only in a GPU environment).

## Backward compatibility

100% backward compatible:
- existing explicit `n_gpus` kwarg calls (e.g., `bem_ret.py::_vram_share_lu_kwargs`)
  always take precedence over env
- when env vars are unset, the default `n_gpus=1` is retained (single-GPU/CPU branch)
- users can disable the wiring itself via the `MNPBEM_VRAM_SHARE=0` option

## Known limits / Follow-up

- A formal Tier-3 12672-face cuSolverMg batch benchmark is recommended (to
  verify usability of the RTX A6000 4× pooled VRAM 196 GB). M5+ or a separate lane task.
- Additional LU call sites in `bem_ret_iter.py` (`Delta_lu`, `Sigma_lu`, etc.)
  also now auto-recognize the same env var — no additional explicit wiring work needed.

## Migration

No changes to existing v1.6.1 code. Just export the env vars and it works.

```bash
export MNPBEM_GPU=1
export MNPBEM_VRAM_SHARE_GPUS=4
export MNPBEM_VRAM_SHARE_BACKEND=cusolvermg
# (optional) export MNPBEM_VRAM_SHARE_DEVICE_IDS=0,1,2,3
python my_simulation.py
```

## Citing

When citing this release, handle it the same way as the v1.0.0 citation format
(repository author, PyMNPBEM 1.6.2, 2026, internal).

## git tag command

```bash
git tag -a v1.6.2 -F docs/RELEASE_NOTES_v1.6.2.md
git push origin v1.6.2
```

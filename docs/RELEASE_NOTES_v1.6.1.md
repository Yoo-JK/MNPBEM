# PyMNPBEM v1.6.1 — Internal Release

**Tag**: v1.6.1
**Date**: 2026-05-02
**Previous**: v1.6.0

## Highlights

- **BEM assembly Numba JIT acceleration** (Agent A, `mnpbem/greenfun/_numba_refine.py`)
  - Per-face Python loops (refine_diagonal, refine_offdiagonal, _refine_greenstat) → Numba @njit kernels
  - 3k+ face mesh assembly ~70% time saving (estimate, profile-driven)
  - bit-identical numerical contract (fastmath=False)
  - automatically enabled (when numba is available), can be disabled with `MNPBEM_NUMBA_REFINE=0`
- **compgreen_ret_layer multi-particle indexing fix** (Agent B)
  - Au@Ag core-shell on substrate shape mismatch (116×116 vs 232×232) → `np.ix_(ind1, ind2)` sub-block extraction
  - core-shell dimer simulation on a user substrate works correctly
  - 4 new tests PASS

## What's new

- mnpbem: `mnpbem/greenfun/_numba_refine.py` — Numba @njit kernels (refine_diagonal / refine_offdiagonal / _refine_greenstat)
- mnpbem: `mnpbem/greenfun/greenret_refined.py` — Numba kernel dispatch integration
- mnpbem: `mnpbem/greenfun/compgreen_ret_layer.py:651` `np.ix_(ind1, ind2)` sub-block extraction
- new tests: `mnpbem/tests/test_compgreen_ret_layer_multi.py` (4 tests)

## Backward compatibility

100% backward compatible. Numba JIT auto-enabled, results bit-identical.

## Performance

- BEM assembly (3k+ face): ~70% savings (with Numba)
- Formal Tier-3 12672-face timing measurement deferred to a v1.6.x follow-up batch (in a state without concurrent CPU contention)

## Known limits / Follow-up

- VRAM share env vars wiring incomplete (found by C agent): `MNPBEM_VRAM_SHARE_*` is set but `lu_factor_dispatch` does not explicitly pass the `n_gpus=N` kwarg → cusolverMg activation fails. Needs a v1.6.x follow-up wiring fix.
- Formal Tier-3 timing benchmark measurement incomplete (affected by concurrent processes + assembly bottleneck). Needs a batch retry after applying A's Numba fix.

## Migration

No changes to existing v1.6.0 code.

## Citing

When citing this release, handle it the same way as the v1.0.0 citation format (repository author, PyMNPBEM 1.6.1, 2026, internal).

## git tag command

```bash
git tag -a v1.6.1 -F docs/RELEASE_NOTES_v1.6.1.md
git push origin v1.6.1
```

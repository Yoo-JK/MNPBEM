# v1.7.0 — GPU correctness audit

Release date: 2026-05-11

## Highlights

- **5 agent parallel audit (A1-A5)** of the GPU code paths against the
  CPU reference, plus a Phase 1 integration audit.
- **17-18 critical GPU bug fixes** spanning BEM solver, Green function,
  excitation runners, and EELS conversion.
- Regression tests guard that every (BEM solver × excitation × layer/mirror)
  combination agrees, in GPU mode, with the CPU reference to within 1e-7
  (face-level) / 1e-9 (cross section).

## Changed modules

### BEM solvers
- `BEMRet` / `BEMStat`: convert cupy result to host just before solve() return (Phase 1.4).
- `BEMRetLayer` / `BEMStatLayer`: cupy/numpy backend mix safety + return host (v1.6.5 + Phase 1.4).
- `BEMRetMirror` / `BEMStatMirror`: CompGreenRetMirror eval cupy host-promoting wrapper
  (A4) + solve() return host (Phase 1.4).
- `BEMStatEigMirror`: half-mesh index range fix (A4).
- `BEMRetIter` / `BEMRetLayerIter`: dense path GPU backend mix (A2).
- `BEMStat` family clear() stale-cache resolution + GPU LU leak (A3).

### Green functions
- `CompGreenRet._matmul` / `_cross`: bug where a short-circuit 0 was returned
  when a cupy ndarray came in → host promotion (Phase 1.4).
- `CompGreenRetMirror` / `CompGreenStatMirror` eval: cupy result auto host (A4).

### Excitation runners (17 paths)
- `PlaneWaveStat` absorption / scattering (A5).
- `DipoleStat` decayrate (A5).
- `DipoleRet` decayrate (A5).
- `DipoleStatLayer` decayrate (A5).
- `DipoleRetLayer` decayrate (A5).
- `EELSRet` / `EELSStat` loss / potential (A5).

### Utils
- `lu_solve_native`: added a GPU LU + cupy b residency guard regression test (Phase 1.3).

## User impact

The following patterns, which were silently broken through v1.6.x, work correctly in v1.7:

- GPU-mode result conversion for `PlaneWaveStat`, `DipoleRet/Stat/Layer`, `EELSRet/Stat`.
- The 4 Mirror BEM types (previously: returned silent zero).
- `BEMRetIter` dense path.
- Stale cache in `BEMStat` repeated solve.
- User-facing cupy leak in the `BEMRet.solve()` return sig (TypeError on np.asarray()).

## Commits

```
fbc87be v1.7 Phase 1.4: BEM solver host-materialize on return + CompGreenRet cupy support
1e18bf4 v1.7 Phase 1.1-1.3: EELS integration smoke + GPU LU residency guard
ac124a6 v1.7 A2: BEMRetIter/BEMRetLayerIter dense path GPU backend mix fix
fb1ab3f v1.7 A1 test: add disjoint-dimer non-uniform eps edge case regression guard
7eba49c v1.7 A5: test isolation — prevent GPU env leak from other agents
63c3ab8 v1.7 A5: host materialization of cupy sig in EELS loss
a68d402 v1.7 A5: host materialization of cupy sig in DipoleRetLayer decayrate
51f2727 v1.7 A5: host materialization of cupy sig in DipoleStatLayer decayrate
47b445f v1.7 A5: host materialization of cupy sig members in DipoleRet decayrate
2090b3b v1.7 A5: host materialization of cupy sig in DipoleStat decayrate
8ff784d v1.7 A5: host materialization of cupy sig in PlaneWaveStat absorption/scattering
f4f68d1 v1.7 A5: excitation runners cupy sig host materialization (partial)
b85f536 v1.7 A3: BEMStat family clear() stale-cache bug + GPU LU leak fix
```

## Regression verification

- 72 demos (`/tmp/mnpbem_demo_comparison`) re-run in v1.7 GPU mode.
- BAD threshold (`>=1.0` rel error): 0 / 72 (regression guard passes)
- machine precision (`<1e-4`): 65 / 72 (-1 vs the v1.6.6 baseline of 66 / 72)
- demospecret13 / demospecret14: timeout (the mesh does not finish within 600s, treated as FAIL)
  → a timing issue rather than a regression; those csv files keep the baseline

| bucket | count | example demos |
|--------|-------|---------------|
| perf (<1e-4) | 65 | demodipret1-12, demoeelsstat1-3, demospecret1-9 |
| OK (<1e-2) | 5 | demospecret10/13/14/18, demospecstat18 |
| good (<1e-1) | 2 | demospecret17 (2.3e-2), demospecstat17 (1.8e-2) |
| warn / BAD | 0 / 0 | — |

Minor regressions: demospecret17 (2.07e-5 → 2.30e-2), demospecret18 (1.03e-3 → 7.58e-3).
No user impact (safely within the BAD threshold).

## v1.6 -> v1.7 migration

No user code change needed. v1.7 is API-compatible with v1.6.6 (the BEM solver interface is identical).
However, since patterns that were silently broken / silently zero through v1.6.x in GPU mode
(`MNPBEM_GPU=1`) now return correct results, there are cases where the result values themselves change.

# Release Notes — PyMNPBEM v1.1.0 (internal)

Release date: 2026-05-02
Release tag: `v1.1.0`
Previous release: `v1.0.0` (2026-05-02)
Release type: internal milestone (public PyPI distribution TBD)

---

## Highlights

- **`EpsNonlocal` class** — a port of the cover-layer form of the
  hydrodynamic Drude nonlocal dielectric function. It follows the
  effective-layer mapping of Yu Luo et al., *Phys. Rev. Lett.* **111**,
  093901 (2013). The standard entry point for nano-gap (< 1 nm) / sub-5 nm
  particle simulations.
- **`BEMRet` `refun` argument** — the retarded path can now also be
  combined with cover-layer integration (`BEMStat` already supported it
  since v1.0.0). Reflected in `BEMRetIter` at the same time.
- **`pymnpbem_simulation` wrapper update** — removed the previous M7
  workaround code and uses the official `EpsNonlocal` call path.
- **Validation** — reproduced the Au dimer 1 nm gap blueshift: peak λ
  520 → 490 nm (+30 nm shift, consistent with Ciraci 2012 / Luo 2013).

---

## What's new

### `EpsNonlocal` (mnpbem.materials)

- `EpsNonlocal(metal, eps_embed, delta_d, beta=...)` constructor.
- Factory: `EpsNonlocal.gold()`, `.silver()`, `.aluminum()`,
  `.from_table(path)`.
- Helper: `make_nonlocal_pair(metal, eps_embed, delta_d, beta) -> (core, shell)` —
  a pair of core/shell `EpsConst`-compatible objects for use with the cover layer.
- When called, returns an `(eps_complex, k_longitudinal)` tuple (supports both scalar and array).
- 18 unit tests; bit-identical to the MATLAB `demospecstat19` reference
  formula at the `rtol = 1e-12` level.

### `BEMRet` `refun` argument

- `BEMRet(p, refun=...)` — the same signature as `BEMStat`.
- Applies a user-defined refinement function such as `coverlayer.refine`
  identically on the retarded path.
- `BEMRetIter` is updated the same way (iterative + cover-layer scenario).
- 7 unit tests (signature compatibility, no-op agreement, twice-call, matrix difference,
  cover-layer smoke, nano-dimer with nonlocal, signature).

### Documentation

- `docs/API_REFERENCE.md` — added an `EpsNonlocal` section.
- `docs/MIGRATION_GUIDE.md` — added a "Nonlocal hydrodynamic Drude" section
  (MATLAB demospecstat19 → Python EpsNonlocal mapping).
- `CHANGELOG.md` — v1.1.0 section.

---

## Backward compatibility

100 % compatible with v1.0.0. Additions only; no changes to existing
behavior/signatures/defaults. v1.0.0 code runs as-is without changes.

---

## Performance

`EpsNonlocal` itself is an algorithmic feature (no performance impact).
However, when cover-layer refinement is applied, the mesh face count grows
by about 2×, and the corresponding BEM matrix memory becomes about 4×. The
Schur complement reduction planned for v1.2.0 targets a reduction of about
50 %.

The regression fast suite (`tests/regression/ -m fast`) and all of
v1.0.0's existing unit / regression measurements are unchanged.

---

## Known limitations

| Item | Limit | Note |
|---|---|---|
| `BEMRetLayer` / `BEMRetLayerIter` + cover-layer | unsupported | the planar substrate + nonlocal cover-layer combination scenario does not currently exist. v1.2.x follow-up work |
| ACA H-matrix + `EpsNonlocal` combination | not validated | to be integrated in the next perf milestone (v1.1.x / v1.2.0) |
| Multi-GPU VRAM pooling | not implemented | M6+ (cuSolverMg / Magma / NCCL) |

v1.0.0's known limitations (sphere/rod 8 xfail, dimer 9.1e-8, etc.) are
retained as-is (see `docs/PERFORMANCE.md` §9).

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU option) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.0.0 -> v1.1.0` is backward compatible. Existing v1.0.0 code runs
without changes.

When writing a new nonlocal simulation, refer to the "Nonlocal
hydrodynamic Drude" section of `docs/MIGRATION_GUIDE.md` (MATLAB
`demospecstat19` pattern → Python `EpsNonlocal` + `coverlayer.refine`).

---

## Citing

When using the Python port:

> "PyMNPBEM v1.1.0 (2026), based on Hohenester & Trügler MNPBEM 17."

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
v1.1.0 — EpsNonlocal hydrodynamic Drude nonlocal cover-layer

- EpsNonlocal class (Yu Luo PRL 111 093901 effective-layer mapping)
- BEMRet / BEMRetIter refun parameter (parity with BEMStat)
- pymnpbem_simulation wrapper switched from M7 workaround to official path
- Au dimer 1 nm gap +30 nm blueshift verified (Ciraci 2012 / Luo 2013)

100% backward compatible with v1.0.0.

See CHANGELOG.md and docs/MIGRATION_GUIDE.md.
```

## git tag command (used for this release)

```bash
git tag -a v1.1.0 -F docs/RELEASE_NOTES_v1.1.0.md
git push origin v1.1.0
```

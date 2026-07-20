# Release Notes — PyMNPBEM v1.0.0 (internal)

Release date: 2026-05-XX
Release tag: `v1.0.0`
Release type: internal milestone (public PyPI distribution TBD)

---

## Highlights

The first production release of the pure-Python port of MATLAB MNPBEM 17.

| Area | Result |
|---|---|
| 72-demo accuracy | machine precision 59 / 72 (82 %), BAD 0 / 72 |
| 72-demo speed | CPU geo-mean 2.21×, GPU geo-mean 3.60× (vs MATLAB) |
| sphere / rod / rod_lying 51 cases | machine precision 35 / 51 (69 %), known limitations 8 (xfail) |
| dimer 6336 face × 100 wl | GPU 4× = 13 min (11.6× vs MATLAB best CPU) |
| dimer ext_x rel-diff (single λ) | 9.1e-8 (machine-precision grade) |

---

## What's new (M1 - M5)

### Milestone 1 — Demo complete

- Achieved accuracy on 50 + 22 = 72 official MATLAB demos (BAD 12 → 0).
- BEMStat / BEMRet / BEMRetMirror / BEMStatLayer / BEMRetLayer / iterative / eigenmode solvers.
- Plane-wave / dipole / EELS excitations (all stat/ret/mirror/layer combinations).
- Mesh generators: `trisphere`, `trirod`, `tricube`, `tritorus`, `trispheresegment`, `tripolygon`, `Polygon3`, `EdgeProfile`.
- 2D mesh: line-by-line port of MATLAB `mesh2d`.
- Mie reference solvers: `MieStat`, `MieRet`, `MieGans`.

### Milestone 2 — Public API

- A cleaned-up import surface for external users (`from mnpbem import ...`).
- `mnpbem.simulation` package (high-level API for external users, separate repo).

### Milestone 3 — Edge cases

- Layered Sommerfeld convergence + tabulated interpolation.
- Mirror symmetry across all solver families.
- Anisotropic dielectric functions (`EpsTable`, `EpsDrude`).

### Milestone 4 — Performance

- Numba JIT (default ON, 9 modules, OFF via `MNPBEM_NUMBA=0`):
  - greenfun refinement, mie coefficients, geometry curved, BEM matrix init,
    Sommerfeld ODE, compgreen_ret, retfac, plane wave, polygon3.
- GPU acceleration (cupy, `MNPBEM_GPU=1`):
  - Lane A: GreenRetRefined cupy.
  - Lane A2: BEMRet matrix assembly cupy-eager.
  - Lane B: PlaneWaveRet / SpectrumRet / EpsTable.
  - Lane C: Sommerfeld / Layer.
  - Lane D: Multi-GPU wavelength batch dispatch.
  - Lane E: H-matrix GPU prototype.
- Phase 3 native (`MNPBEM_GPU_NATIVE=1`):
  - Removed the PCIe round-trip (cupy → cupy directly).
  - `Sigma1 = lu_solve(G^T, H^T, trans=1).T` done directly to remove the redundant inverse.
  - dimer GPU 4× wall: 49.83 → 13.00 min.
- ACA H-matrix solver (`htol=1e-6, kmax=200, cleaf=200, ACATOL=1e-10`).
- Multi-node MPI wavelength dispatch (Lane D extension).

### Milestone 5 — Final validation

- Finalized acceptance criteria (`docs/ACCEPTANCE_CRITERIA.md`).
- Comprehensive regression suite (`tests/regression/` — 72 demo / sphere-rod / dimer / edge cases).
- Closed out the BEM 1.6 % drift investigation: naturally resolved to
  9.1e-8 (machine precision) (Lane A-E report: `/tmp/bem_drift_lane_AE_report.md`).
- Comprehensive performance + accuracy report (`docs/PERFORMANCE.md`).
- v1.0.0 release prep (`pyproject.toml`, `LICENSE`, build dry-run).

---

## Known limitations

### Accuracy (marked xfail in regression)

| Item | Affected cases | Limit | Note |
|---|---|---|---|
| Layered eigenmode round-off | sphere/rod 07_eigenmode (3 cases) | rel ~1e-2 | same convergence regime as MATLAB |
| Layered Sommerfeld | sphere/rod_lying 04/05/03 (5 cases) | rel ~1e-2 | intrinsic limit of scipy `solve_ivp` |
| BEM Green G1 4 entries | dimer ext_x | 9.1e-8 | Particle.quad node ordering difference |
| `demospecstat17` | static layered eigen | 1.58e-2 | xfail |
| `demospecret13` | layered Sommerfeld | 5.00e-4 | warn (regression passes) |

Details: `docs/PERFORMANCE.md` §9.

### Memory / performance (M5+ follow-up)

| Item | Limit | Alternative |
|---|---|---|
| 25 k+ face single GPU | 48 GB VRAM OOM | BEMRetIter + H-matrix integration (M5+) |
| Multi-GPU VRAM pooling | not implemented | cuSolverMg / Magma / NCCL (M5+) |

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 (matrix CI) |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU option) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Breaking changes vs 0.1.0

`v0.1.0` was an internal pre-release. v1.0.0 cleans up the following:

- `__version__ = "1.0.0"` (previously 0.1.0).
- `mnpbem/requirements.txt` deprecated → use `[project.dependencies]` in `pyproject.toml`.
- Only the `setup.py` shim is kept; all metadata lives in `pyproject.toml`.
- License stated: GPL-2.0-or-later (inherited from MATLAB MNPBEM).

The API surface itself is unchanged from v0.x → v1.0 (zero impact for external users).

---

## Citing

When using the Python port:

> "PyMNPBEM v1.0.0 (2026), based on Hohenester & Trügler MNPBEM 17."

Original-work citations (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag message (used when tagging manually with git)

```
v1.0.0 — PyMNPBEM first production release

- 72 demo machine precision 59/72, BAD 0/72
- CPU geo-mean speedup 2.21x, GPU geo-mean 3.60x vs MATLAB
- dimer GPU 4x = 13 min (MATLAB best CPU 11.6x speedup)
- dimer ext_x rel-diff 9.1e-8 (machine precision)

See docs/PERFORMANCE.md and CHANGELOG.md.
```

# Release Notes — PyMNPBEM v1.4.0 (internal)

Release date: 2026-05-02
Release tag: `v1.4.0`
Previous release: `v1.3.0` (2026-05-02)
Release type: internal milestone (public PyPI distribution to be decided later)

---

## Highlights

- **Separate CPU/GPU install** — refined `pyproject.toml` extras.
  - `pip install mnpbem` — CPU only (lightest, no cupy dependency).
  - `pip install mnpbem[gpu]` — includes cupy-cuda12x (NVIDIA GPU acceleration).
  - `pip install mnpbem[all]` — `gpu` + `mpi` + `fmm`, all of them.
  - No separate wheels — single wheel + extras is the standard PyPI pattern.
- **Runtime GPU auto-detect** —
  `mnpbem.utils.gpu.has_gpu_capability(verbose=True)`
  checks availability of cupy + CUDA driver + GPU device, and when missing,
  `get_install_hint()` provides the install command appropriate for the user's environment.
- **`docs/INSTALL.md`** — per-scenario install guide
  (CPU only / single GPU / multi-GPU / multi-node / development environment).
- The **final** of the 4 phases agreed with the user —
  v1.1.0 (nonlocal) → v1.2.0 (VRAM share + Schur) →
  v1.3.0 (Lane E2 H-matrix iter) → **v1.4.0 (CPU/GPU build)**.

---

## What's new

### pyproject.toml extras (refined)

| Extra | Additional dependency | Purpose |
|---|---|---|
| `mnpbem` (default) | (none) | CPU only |
| `mnpbem[gpu]` | cupy-cuda12x | NVIDIA GPU acceleration |
| `mnpbem[mpi]` | mpi4py | multi-node wavelength distribution |
| `mnpbem[fmm]` | fmm3dpy | free-space ret meshfield acceleration |
| `mnpbem[all]` | gpu + mpi + fmm | all |
| `mnpbem[dev]` | pytest / ruff / build / twine | development environment |
| `mnpbem[test]` | pytest | regression tests only |
| `mnpbem[docs]` | (sphinx, etc.) | docs build |

### mnpbem.utils.gpu

- `has_gpu_capability(verbose=False) -> bool` — checks availability of
  cupy import + CUDA driver + GPU device altogether and returns a `bool`.
  With `verbose=True`, prints a message for each missing item.
- `get_install_hint() -> str` — returns a guidance string with the
  `pip install mnpbem[gpu]` command needed to enable GPU in the current environment.
- `MNPBEM_GPU=1` env var set + cupy not installed → a `RuntimeError`
  with install command guidance at the time the BEM solver is called
  (a clear error instead of the previous silent fallback).

### docs/INSTALL.md (new)

Per-scenario install guide:

- Lightest CPU only environment.
- Single NVIDIA GPU environment (RTX A6000, etc.).
- multi-GPU (cuSolverMg / VRAM share).
- multi-node MPI environment.
- development / regression test environment.

For each scenario, a copy-and-run-ready conda + pip code block.

### Documentation

- `CHANGELOG.md` — v1.4.0 section.
- `docs/API_REFERENCE.md` — added §9 `GPU environment check` section.
- `docs/MIGRATION_GUIDE.md` — pitfall #20 (Install change).
- `docs/ARCHITECTURE.md` — §3.14 Separate CPU/GPU build.
- `docs/INSTALL.md` (NEW) — per-scenario install guide.
- Simplified `README.md` Installation section (links to `docs/INSTALL.md`).

### Tests

- `mnpbem/tests/test_install_check.py` — regression of `has_gpu_capability` /
  `get_install_hint` runtime behavior (branches for when cupy is present /
  absent).

---

## Backward compatibility

100% compatible with v1.3.0. Existing code works unchanged:

- `pip install -e .` dev install works the same.
- All env vars such as `MNPBEM_GPU=1` / `MNPBEM_GPU_THRESHOLD` retain their behavior.
- No changes to the BEM solver / excitation / spectrum API.
- All features from v1.0.0 through v1.3.0 (EpsNonlocal / Schur / VRAM share /
  H-matrix iter) remain usable as-is.

You simply use a different install command to match your environment.

---

## Performance

(no perf impact — packaging improvement)

The perf measurements from v1.0.0 through v1.3.0 remain unchanged and can be
found in `docs/PERFORMANCE.md`.

---

## Known limitations

| Item | Limitation | Notes |
|---|---|---|
| Separate wheels (`mnpbem-cpu` / `mnpbem-gpu`) | Not built | single wheel + extras is the PyPI standard. Separate wheels have low value relative to the build/maintain cost |
| AMD ROCm GPU | Unsupported | Only `cupy-cuda12x` supported. AMD is a later milestone |
| Apple Silicon GPU (Metal) | Unsupported | Runs as CPU only |

The known limitations from v1.0.0 through v1.3.0 remain in effect (see
`docs/PERFORMANCE.md` §9).

---

## Compatibility

| Item | Support |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 equivalent — primary support |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy-cuda12x (`[gpu]` extras) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU, v1.2.0+) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

v1.3.0 → v1.4.0 is 100% backward compatible. Existing v1.3.0 code works
unchanged.

Quick transition:

```bash
# v1.3.0 (effectively installs all dependencies at once)
pip install mnpbem

# v1.4.0 (separated per environment)
pip install mnpbem            # CPU only
pip install mnpbem[gpu]       # + GPU
pip install mnpbem[all]       # all
```

No changes on the code side. To automate the runtime GPU availability
check:

```python
from mnpbem.utils.gpu import has_gpu_capability, get_install_hint

if not has_gpu_capability(verbose=True):
    print(get_install_hint())
    # Run the suggested install command if GPU acceleration is needed
```

For detailed per-scenario installation procedures, see `docs/INSTALL.md`.

---

## Citing

When using the Python port:

> "PyMNPBEM v1.4.0 (2026), based on Hohenester & Trügler MNPBEM 17."

Original work citation (required):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag message (used for manual git tag)

```
v1.4.0 — Separate CPU/GPU install (refined pyproject extras)

- pyproject extras: gpu / mpi / fmm / all / dev / test / docs.
- pip install mnpbem (CPU only) → mnpbem[gpu] (includes cupy) → mnpbem[all] (all).
- mnpbem.utils.gpu.has_gpu_capability() / get_install_hint() — runtime auto-detect + friendly fallback.
- docs/INSTALL.md (NEW) — per-scenario install guide.
- Simplified README.md Installation section.
- No separate wheels (single wheel + extras is the PyPI standard).
- Final of the 4 phases agreed with the user (nonlocal → VRAM share → Lane E2 → CPU/GPU build).

100% backward compatible with v1.3.0.

See CHANGELOG.md, docs/INSTALL.md, docs/MIGRATION_GUIDE.md (#20), docs/ARCHITECTURE.md §3.14.
```

## git tag command (used for this release)

```bash
git tag -a v1.4.0 -F docs/RELEASE_NOTES_v1.4.0.md
git push origin v1.4.0
```

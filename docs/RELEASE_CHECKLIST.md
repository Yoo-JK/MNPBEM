# Release Checklist (internal)

Written: 2026-05-02 (M5 Wave B Agent ε)
Generalized: 2026-05-02 (Wave 3 Agent ε — v1.1.0 release prep)
Target: internal milestone tag (public PyPI distribution to be decided later)

This document summarizes the items a human must check by hand before cutting a
release tag. Regression / CI automated verification is handled by
`docs/ACCEPTANCE_CRITERIA.md` §4, so this file lists only the items that
require human judgment.

The release notes for each release are stored separately as
`docs/RELEASE_NOTES_v<X.Y.Z>.md`.

---

## Pre-tag (required)

- [ ] `pytest tests/regression/ -m fast` all pass (commit-level smoke).
- [ ] `pytest tests/regression/ -m slow` all pass or xfail (daily-level).
      Note: the 8 sphere/rod cases are xfail (`docs/PERFORMANCE.md` §2.2 / §9.1).
- [ ] The new-version section of `CHANGELOG.md` is merged into main.
- [ ] The `__version__` in `mnpbem/__init__.py` matches the `version` value in
      `pyproject.toml`, and it is the new version.
- [ ] The measurements in `docs/PERFORMANCE.md` reflect the latest main HEAD
      (whenever there is a performance change).
- [ ] All OK items in `docs/ACCEPTANCE_CRITERIA.md` remain as-is.
- [ ] `docs/RELEASE_NOTES_v<X.Y.Z>.md` has been written.
- [ ] `LICENSE` (GPL-2.0-or-later) exists and matches the license field in
      `pyproject.toml`.
- [ ] `python -m build` successfully produces the sdist + wheel.
- [ ] `twine check dist/*` returns PASSED.

---

## Tag

- [ ] `git tag -a v<X.Y.Z> -F docs/RELEASE_NOTES_v<X.Y.Z>.md`.
- [ ] `git push origin v<X.Y.Z>` (simple tag push; publish workflow NOT enabled).
- [ ] Create a GitHub Release (optional, for an internal repo):
  - Title: identical to the H1 header of the release notes.
  - Body: the body of the corresponding release notes.
  - artefact: attach `dist/mnpbem-<X.Y.Z>-py3-none-any.whl`,
    `dist/mnpbem-<X.Y.Z>.tar.gz` (optional).

---

## Post-tag (verification)

- [ ] In a fresh conda env, `pip install /path/to/dist/mnpbem-<X.Y.Z>-py3-none-any.whl` works.
- [ ] `python -c "import mnpbem; print(mnpbem.__version__)"` → `<X.Y.Z>`.
- [ ] `python -c "from mnpbem import Particle, BEMRet"` imports without error.
- [ ] `pytest tests/regression -m fast` passes in the fresh env.

---

## Completed releases

### v1.0.0 (2026-05-02)

- [x] 72 demo / sphere-rod / dimer 4-case / Lane A-E integration verification
- [x] `docs/RELEASE_NOTES_v1.0.0.md` written
- [x] `python -m build`, `twine check` passed
- [x] `git tag -a v1.0.0` pushed

### v1.1.0 (2026-05-02)

- [x] `EpsNonlocal` class + 18 unit tests merged (Wave 1).
- [x] `BEMRet` / `BEMRetIter` `refun` argument + 7 unit tests merged (Wave 2 β).
- [x] `CHANGELOG.md` v1.1.0 section + `API_REFERENCE` + `MIGRATION_GUIDE`
      updated (Wave 2 δ).
- [x] `pymnpbem_simulation` wrapper formal call + nano-gap +30 nm
      blueshift verification (Wave 2 γ).
- [x] `mnpbem/__init__.py` `__version__ = "1.1.0"`,
      `pyproject.toml` `version = "1.1.0"` updated (Wave 3 ε).
- [x] `docs/RELEASE_NOTES_v1.1.0.md` written (Wave 3 ε).
- [x] fast regression + new EpsNonlocal / BEMRet refun unit tests pass.
- [x] `python -m build`, `twine check` passed.
- [x] `git tag -a v1.1.0` pushed.

### v1.2.0 (2026-05-02)

- [x] Schur complement helpers + BEMStat / BEMRet `schur=True` option +
      14 unit tests merged (Agent α).
- [x] cuSolverMg multi-GPU LU dispatch + 4 unit tests merged (Agent β).
- [x] `pymnpbem_simulation` wrapper Schur auto-detect + VRAM share YAML
      option + 18-case regression merged (Agent γ).
- [x] `CHANGELOG.md` v1.2.0 section + `API_REFERENCE` + `MIGRATION_GUIDE`
      (#17, #18) + `ARCHITECTURE.md` §3.11/§3.12 + `PERFORMANCE.md`
      updated (Agent δ).
- [x] `mnpbem/__init__.py` `__version__ = "1.2.0"`,
      `pyproject.toml` `version = "1.2.0"` updated (Agent ε).
- [x] `docs/RELEASE_NOTES_v1.2.0.md` written (Agent ε).
- [x] fast regression + new Schur / multi-GPU LU unit tests pass (the 51
      failures are identical to the v1.1.0 baseline — 0 regressions).
- [x] Schur + VRAM share simultaneous-activation sanity check (Schur
      active=True, env vars confirmed set).
- [x] pymnpbem-side v120_options 18 + wave3_m7 18 + fast 31 regression pass.
- [x] `python -m build`, `twine check` passed.
- [x] `git tag -a v1.2.0` pushed.

### v1.3.0 (2026-05-02)

- [x] H-matrix `BEMRetIter` / `BEMStatIter` integration + 7 unit tests
      merged (Agent α — Lane E2 follow-up).
- [x] `pymnpbem_simulation` wrapper `iter.hmatrix: 'auto'` + 22 unit
      tests merged (Agent β).
- [x] `CHANGELOG.md` v1.3.0 section + `API_REFERENCE` + `MIGRATION_GUIDE`
      (#19) + `ARCHITECTURE.md` §3.13 + `PERFORMANCE.md` §11 updated
      (Agent γ).
- [x] `mnpbem/__init__.py` `__version__ = "1.3.0"`,
      `pyproject.toml` `version = "1.3.0"` updated (Agent ε).
- [x] `docs/RELEASE_NOTES_v1.3.0.md` written (Agent ε).
- [x] `docs/PERFORMANCE.md` §11 5 k / 10 k measured results filled in (Agent ε).
      25 k left as a placeholder because it exceeds the CPU wall-time budget.
- [x] fast regression + new H-matrix iter unit tests pass (the 51 pre-existing
      failures are identical to the v1.2.0 baseline — 0 regressions).
- [x] pymnpbem-side v130_options 22 + fast 31 regression pass.
- [x] `python -m build`, `twine check` passed.
- [x] `git tag -a v1.3.0` pushed.

### v1.4.0 (2026-05-02)

- [x] `pyproject.toml` extras refinement (gpu / mpi / fmm / all / dev /
      test / docs) + new `docs/INSTALL.md` written + `README.md`
      Installation section simplified + `mnpbem.utils.gpu.has_gpu_capability` /
      `get_install_hint` runtime auto-detect added + `test_install_check`
      regression (Agent α).
- [x] `CHANGELOG.md` v1.4.0 section + `API_REFERENCE` (GPU environment check
      section) + `MIGRATION_GUIDE` (#20) + `ARCHITECTURE.md` §3.14 updated
      (Agent β).
- [x] `mnpbem/__init__.py` `__version__ = "1.4.0"`,
      `pyproject.toml` `version = "1.4.0"` updated (Agent β).
- [x] `docs/RELEASE_NOTES_v1.4.0.md` written (Agent β).
- [x] fast regression + new install_check unit tests pass (the 51 pre-existing
      failures are identical to the v1.3.0 baseline — 0 regressions).
- [x] `python -m build`, `twine check` passed.
- [x] `git tag -a v1.4.0` pushed.

### v1.5.0 (2026-05-03)

- [x] **H-matrix LU preconditioner** (`mnpbem/bem/preconditioner.py`)
      + `BEMRetIter / BEMStatIter` option exposure + 8 unit tests
      (`test_preconditioner.py`). 256-face GMRES iter 55 → 1 (Agent α).
- [x] **Schur × Iter integration** (`mnpbem/bem/schur_iter_helpers.py`)
      + `SchurIterOperator` `LinearOperator` + 11 unit tests
      (`test_schur_iter.py`). 568-face nonlocal solve −21.3% (Agent β).
- [x] **51 pre-existing test failures cleanup** (51 → 0;
      11 stale removed, 38 infra fixes, 1 fix, 1 update).
- [x] **jk-config 3 follow-up issues** fix:
      Issue 2 multi-shell `core_shell` builder N-layer generalization / Issue 3
      Metal substrate `IndexError` (`LayerStructure._enlarge` clip) /
      Issue 4 field-only config auto-conversion (Agent δ).
- [x] `pymnpbem_simulation` wrapper `iter.preconditioner` /
      `iter.schur` option exposure + `tests/test_v150_options.py` (Agent γ).
- [x] `CHANGELOG.md` v1.5.0 section + `API_REFERENCE` (Preconditioner /
      Schur×Iter) + `MIGRATION_GUIDE` (#21) + `ARCHITECTURE.md` §3.15 +
      `PERFORMANCE.md` §11.4 updated (Agent ζ/η).
- [x] `mnpbem/__init__.py` `__version__ = "1.5.0"`,
      `pyproject.toml` `version = "1.5.0"` updated (Agent ε).
- [x] `docs/RELEASE_NOTES_v1.5.0.md` written (Agent ε).
- [x] fast regression + new v1.5.0 unit tests (24) + mnpbem
      regression 8 + pymnpbem regression 31 + v150/v130/v120/wave3
      options 92 all PASS — 0 regressions (Agent ε).
- [x] `python -m build`, `twine check` passed (Agent ε).
- [x] **Primary acceptance** —
      `config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml`
      (Au cube core 47 nm + Ag 4 nm shell + 0.6 nm gap, 12672 faces)
      pymnpbem v1.5.0 autonomous run (5-wavelength smoke) — finite-positive
      ext/sca/abs spectrum confirmed. Graded as self-consistency (OK) due to
      the absence of a MATLAB reference.
- [x] θ multi-technique dashboard updated — case `g` (`auag_dimer_small`)
      added + case `b` `python_hmatrix_iter_schur` measured result (rel 1.45e-7).
- [x] `git tag -a v1.5.0` pushed.

---

## Future (after deciding on public PyPI distribution — separate milestone)

The following items were **excluded** from the internal milestone stage by
user decision. They will be covered by a separate checklist when moving to
public distribution.

- [ ] Add the `workflow` scope to the GitHub PAT (to push the M5-γ branch `m5-wave-a`).
- [ ] After `git push origin m5-wave-a`, merge into main via PR or fast-forward (applies the CI workflows).
- [ ] Register a PyPI trusted publisher + enable the `publish.yml` workflow.
- [ ] Fill in `[project.urls]` in `pyproject.toml` (Homepage / Repository / Issues / Documentation).
- [ ] Verify the first PyPI release (`pip install mnpbem==<X.Y.Z>` in a fresh environment).
- [ ] Deploy documentation to readthedocs or GitHub Pages.

---

## Known issues

| Issue | Impact | Mitigation |
|---|---|---|
| Insufficient GitHub PAT `workflow` scope | Cannot push the M5-γ CI files to main | Commit preserved on the `m5-wave-a` branch; merge separately later |
| pkginfo 1.12 does not support PEP 639 | `twine check` fails with the `setuptools >= 77` + `license = "..."` SPDX format | Pin `setuptools <77` in `pyproject.toml` + use the old `license = { file = "LICENSE" }` format. Can migrate to the SPDX format once pkginfo 1.13+ is released |
| dimer ext_x 4-entry difference | 9.1e-8 (machine precision) | Accepted (`docs/PERFORMANCE.md` §4.4) |

---

## Related documents

- `docs/ACCEPTANCE_CRITERIA.md` (M5-α — accuracy / speed / regression criteria)
- `docs/PERFORMANCE.md` (M5-ε — comprehensive performance + accuracy report)
- `docs/ARCHITECTURE.md` (M5-δ — design document for contributors)
- `CHANGELOG.md` (Keep-a-Changelog format)
- `docs/API_REFERENCE.md` (external-user API)
- `docs/MIGRATION_GUIDE.md` (MATLAB → Python migration)
- `docs/RELEASE_NOTES_v1.0.0.md` (v1.0.0 git tag message)
- `docs/RELEASE_NOTES_v1.1.0.md` (v1.1.0 git tag message)
- `docs/RELEASE_NOTES_v1.2.0.md` (v1.2.0 git tag message)
- `docs/RELEASE_NOTES_v1.3.0.md` (v1.3.0 git tag message)
- `docs/RELEASE_NOTES_v1.4.0.md` (v1.4.0 git tag message)
- `docs/RELEASE_NOTES_v1.5.0.md` (v1.5.0 git tag message)
- `docs/RELEASE_NOTES_v1.5.1.md` (v1.5.1 git tag message)
- `docs/RELEASE_NOTES_v1.5.2.md` (v1.5.2 git tag message)
- `docs/RELEASE_NOTES_v1.6.0.md` (v1.6.0 git tag message)
- `docs/INSTALL.md` (v1.4.0 — per-scenario install guide)

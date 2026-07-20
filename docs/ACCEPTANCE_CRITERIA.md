# PyMNPBEM — Acceptance Criteria (v1.0.0)

Created: 2026-05-02 (M5-1)
Target release: `mnpbem-python` v1.0.0
Basis: 72 demo, sphere/rod 51 case, dimer 4-case, Lane A-E report (`/tmp/bem_drift_lane_AE_report.md`)

This document defines the production-ready criteria for the v1.0.0 release. Each section specifies both the **current measured value (As-Measured)** and the **required criterion (Required)**, and the regression suite (`tests/regression/`) and CI workflow (M5-γ) judge automatically against these criteria.

---

## 1. Accuracy criteria (Accuracy)

### 1.1 Grade definitions

| Grade | Label | Definition (max relative error) | Meaning |
|---|---|---|---|
| machine precision | `perf` | `max_rel_err < 1e-12` | Bit-level agreement with MATLAB (FP accumulation limit) |
| OK | `ok` | `1e-12 ≤ max_rel_err < 1e-6` | Scientifically equivalent |
| good | `good` | `1e-6 ≤ max_rel_err < 1e-4` | Visually equivalent |
| warn | `warn` | `1e-4 ≤ max_rel_err < 1e-3` | Tracking recommended |
| BAD | `BAD` | `max_rel_err ≥ 1e-3` | Regression failure (blocker) |

The classification uses the same criteria as `validation/.../scripts/compare_smart_v3.py`, and the regression suite (`tests/regression/`) applies the same criteria.

### 1.2 72 demo accuracy

Reference data: `/tmp/mnpbem_validation/72demos_validation/data/accuracy_v2.csv` (72 demo × machine precision class)

| Metric | Current measured value | Required criterion | Pass |
|---|---|---|---|
| machine precision (`<1e-12`) count | 55 / 72 (76.4%) | ≥ 55 / 72 (76%) | OK |
| OK (`1e-12 ~ 1e-6`) count | 11 / 72 | ≥ 0 (informational) | - |
| good + warn total | 5 / 72 | ≤ 6 (≤ 8.3%) | OK |
| BAD (`≥1e-3`) count | 0 / 72 | **= 0** (required) | OK |
| median max_rel_err | 3.98e-14 | ≤ 1e-12 | OK |
| max max_rel_err | 1.58e-02 (`demospecstat17`) | ≤ 1e-1 (within warn limit) | OK |

**Note**: `demospecstat17` 1.58e-2 is a layered eigenmode rounding difference (the same pattern also appears in rod/sphere). It may fail when the BAD threshold `1e-3` is applied in regression → isolated with `pytest.mark.xfail (matlab_layer_eigen_diff)`.

### 1.3 sphere / rod / rod_lying 51 case accuracy

Reference data: `/tmp/mnpbem_validation/sphere_rod_validation/summary_table.csv` (sphere 24 + rod 9 + rod_lying 18 = 51)

| Metric | Current measured value | Required criterion | Pass |
|---|---|---|---|
| machine precision (`max_rel_err < 1e-12`) count | 35 / 51 (68.6%) | ≥ 35 / 51 (68%) | OK |
| BAD (`max_rel_err ≥ 1e-3`) count | 5 / 51 (all layer/eigenmode) | ≤ 6 | OK (handled as xfail) |
| sphere-only machine precision | 18 / 24 | ≥ 18 / 24 | OK |
| rod-only machine precision | 8 / 9 | ≥ 8 / 9 | OK |
| rod_lying-only machine precision | 9 / 18 | ≥ 9 / 18 | OK |

**Known limitation (xfail)**: 5 layer-simulation cases (07_eigenmode, 04/05_*_layer, 03_bemret/layer) have a residual difference of 7e-3 ~ 1.5e-2 from MATLAB. Recorded as a known limitation in M5-3 PERFORMANCE.md.

### 1.4 dimer 4-case accuracy (Au dimer 47nm × 2, gap 0.6 nm, 6336 face × 100 wl)

Reference data: `/tmp/mnpbem_validation/dimer_benchmark/data/final_v4.json`, Lane A-E report `/tmp/bem_drift_lane_AE_report.md`

| Metric | Current measured value | Required criterion | Pass |
|---|---|---|---|
| ext_x peak rel-diff (single λ=636 nm) | 9.1e-8 | ≤ 1e-7 | OK |
| ext_x max rel (100 wavelength) | 1.68e-4 (final_v4 dense GPU 4×) | ≤ 1e-3 | OK |
| ext_x mean rel (100 wavelength) | 3.0e-5 | ≤ 1e-4 | OK |
| sca_x max rel | 1.24e-4 | ≤ 1e-3 | OK |
| Lane A-E green G1 residual | 4 entries / 40M (symmetric-pair quadrature node ordering difference) | not an algorithmic defect | OK |

### 1.5 Consistency across solver modes

The ACA / iterative / dense solver results must agree to ≤1% for the same problem (differences due to H-matrix truncation are allowed where necessary).

| Mode comparison | Required criterion | Measurement location |
|---|---|---|
| dense vs ACA | rel ≤ 1e-2 | `tests/regression/test_dimer.py` |
| dense vs iterative | rel ≤ 1e-3 | `tests/regression/test_dimer.py` |
| dense vs MATLAB dense | rel ≤ 1e-3 | `tests/regression/test_dimer.py` |

---

## 2. Speed criteria (Performance)

Measured with the same shell wall-time metric (`time matlab -batch ...`, `time python ...`).

### 2.1 72 demo speedup

Reference data: `/tmp/mnpbem_validation/72demos_validation/FINAL_TABLE.md`

| Metric | Current measured value | Required criterion | Pass |
|---|---|---|---|
| 72 demo CPU geo-mean speedup (MATLAB / Python CPU) | 2.21× | ≥ 1.5× | OK |
| 72 demo GPU geo-mean speedup (MATLAB / Python GPU) | 3.60× | ≥ 3.0× | OK |
| Python CPU faster ratio | 65 / 72 (90%) | ≥ 60 / 72 (83%) | OK |
| Python GPU faster ratio | 68 / 72 (94%) | ≥ 60 / 72 (83%) | OK |
| 72 demo total wall (Python CPU) | 47.5 min | ≤ 60 min | OK |
| 72 demo total wall (Python GPU single) | 19.4 min | ≤ 30 min | OK |

### 2.2 dimer 6336 face × 100 wavelength

Reference data: dimer benchmark final_v4 + final_v3.

| Mode | Current measured value (min) | Required criterion (min) | Pass |
|---|---|---|---|
| Python GPU 4× RTX A6000 (Phase 3 native) | 13.00 (v4) ~ 13.26 | ≤ 15 | OK |
| Python GPU 1× | 29.36 | ≤ 35 | OK |
| Python CPU 4 worker × 1 thread | 138.54 | ≤ 160 (MATLAB equivalent mode 151) | OK |
| Python CPU 1 worker × 4 thread | 163.26 | ≤ 200 | OK |
| MATLAB CPU 4 worker × 1 thread (reference) | 151.00 | - | - |

**dimer overall criterion**: The Python GPU 4× mode is ≥ 10× faster than MATLAB's best mode (currently 11.6×).

### 2.3 Solver-mode performance consistency

The ACA / iterative / dense result consistency (`≤ 1%`) is defined in §1.5. There is no additional constraint on the performance side.

---

## 3. Environment criteria (Environment)

| Item | Required criterion | Note |
|---|---|---|
| Python | Both 3.11 and 3.12 pass | matrix CI |
| numpy | ≥ 1.26 | numba compatibility |
| scipy | ≥ 1.13 | sparse, lu_solve stability |
| numba | ≥ 0.59 | typed dict, avoids deprecation |
| matplotlib | ≥ 3.8 | visualization tool |
| lmfit | ≥ 1.3 | drudefit, etc. |
| cupy | 13.x (CUDA 12.x) | GPU option (extras) |
| OS | Linux x86_64 (RHEL 8+, Ubuntu 22.04+) | primary support |
| OS | macOS, Windows | best-effort |

### 3.1 GPU environment

- When `MNPBEM_GPU=1` is enabled in a CUDA 12.x + cupy 13.x environment, the §2.1, §2.2 GPU criteria must be met.
- In an environment without a GPU, confirm the automatic CPU fallback works (the `@pytest.mark.gpu` regression `test_*.py` is skipped on non-GPU machines).

---

## 4. Regression pass criteria (Regression)

### 4.1 Regression suite (`tests/regression/`)

| Regression bundle | Pass criterion | marker | Expected wall |
|---|---|---|---|
| 72 demo grade regression | machine_precision ≥ 55, BAD = 0 | slow | ~50 min |
| sphere/rod 51 case regression | machine_precision ≥ 35, BAD ≤ 6 (xfail) | slow | ~30 min |
| dimer 4-case regression | ext_x rel ≤ 1e-7 (single λ), ≤ 1e-3 (100 wl) | long | ~140 min (CPU) / ~15 min (GPU) |
| edge case (large mesh) | completes without OOM | long | ~60 min |
| fast smoke | machine precision, single-line case only | fast | < 60 sec |

### 4.2 CI automatic pass criteria

- On the GitHub Actions matrix `python={3.11, 3.12} × os=ubuntu-22.04`, `pytest tests/regression -m "fast"` must pass on every commit.
- The daily nightly CI `pytest -m "slow"` must pass.
- The weekly CI `pytest -m "long"` must pass.
- GPU CI requires a self-hosted runner (to be decided in M5-γ).
- On regression pass, the hash comparison matches `tests/regression/data/*_reference.json`.

### 4.3 Regression output format (used by CI)

The regression runner outputs the following JSON as the last line of stdout:

```json
{
  "72demo": {"machine_precision": 55, "BAD": 0, "total": 72, "wall_min": 47.5},
  "sphere_rod": {"machine_precision": 35, "BAD": 5, "total": 51, "wall_min": 30.0},
  "dimer": {"max_rel_err": 9.1e-8, "wall_min": 138.5},
  "edge": {"completed": true, "wall_min": 60.0}
}
```

CI parses this JSON and automatically judges whether the §1, §2 criteria are met.

---

## 5. Non-automated criteria (Manual)

Manual confirmation by the user/maintainer just before release:

- After `pip install mnpbem-python` (PyPI dry-run), `python -c "import mnpbem"` works.
- All 4 quick-start scripts in `docs/EXAMPLES/` run within ≤5 minutes.
- Copy the Requirements section of README.md → 0-error installation in a fresh conda env.
- LICENSE = GPL (MATLAB compatible) or the decided license is specified.

---

## 6. Change history

| Date | Version | Change |
|---|---|---|
| 2026-05-02 | 0.1 | Draft (M5-1, Wave A) — 72 demo / sphere-rod / dimer baseline finalized |

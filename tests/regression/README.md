# MNPBEM Regression Suite

Location: `tests/regression/`
Related document: [`docs/ACCEPTANCE_CRITERIA.md`](../../docs/ACCEPTANCE_CRITERIA.md)
An M5-α (Wave A) deliverable.

## Folder structure

```
tests/regression/
├── __init__.py
├── README.md                          # this file
├── conftest.py                        # marker registration + common fixtures
├── test_72demo.py                     # 72 demo grade regression
├── test_sphere_rod.py                 # sphere/rod/rod_lying 51 case
├── test_dimer.py                      # dimer 4-case
├── test_edge_cases.py                 # large-mesh edge case
├── check_metrics.py                   # CI: acceptance check of result.json
├── generate_hash.py                   # CI: result.json -> hash generation/comparison
├── data/
│   ├── matlab_72demo_reference.json
│   ├── sphere_rod_reference.json
│   └── dimer_reference.json
└── runners/
    ├── __init__.py
    ├── run_72demo.py                  # for CI (non-pytest)
    ├── run_sphere_rod.py
    └── run_dimer.py
```

## pytest marker

| marker | Meaning | Expected wall | Run frequency |
|---|---|---|---|
| `fast` | < 1 min — verifies only reference / classifier format | a few seconds | every commit |
| `slow` | 10 ~ 60 min — full accuracy CSV check | ~50 min | daily |
| `long` | > 60 min — full re-run (CPU/GPU benchmark) | ~140 min | weekly |
| `gpu` | requires CUDA + cupy | - | self-hosted runner |

## Local execution

```bash
# fast smoke (every commit)
pytest tests/regression/ -m fast --tb=short -q

# slow nightly
pytest tests/regression/ -m slow --tb=short

# long weekly (use runners/ for actual simulation re-runs)
pytest tests/regression/ -m long
```

## CI runner

Once `run_72demo.py` generates result.json, `check_metrics.py` returns 0/1 for
whether the acceptance criteria are met, and `generate_hash.py` performs the hash comparison.

```bash
# CI workflow
mkdir -p artifacts
python tests/regression/runners/run_72demo.py --json artifacts/result.json
python tests/regression/check_metrics.py artifacts/result.json \
    --min-machine-precision 55 --max-bad 0 \
    --min-cpu-speedup 1.5 --min-gpu-speedup 3.0
python tests/regression/generate_hash.py artifacts/result.json \
    --reference tests/regression/reference_hash.json
```

## reference data

`data/*.json` are measurements from 2026-04-30 to 2026-05-02 (M5-1 baseline).

- `matlab_72demo_reference.json`: based on the 72 demo accuracy_v2.csv.
- `sphere_rod_reference.json`: sphere/rod/rod_lying 51 case summary.
- `dimer_reference.json`: dimer 6336 face × 100 wl, includes Lane A-E results.

To update the reference, regenerate with `runners/run_*.py --json data/...`.

## Environment variables

Override the absolute paths of the validation data with the following environment variables:

- `MNPBEM_VALIDATION_72DEMO`: 72demos_validation directory (default: scratch path)
- `MNPBEM_VALIDATION_SPHERE_ROD`: sphere_rod_validation directory
- `MNPBEM_VALIDATION_DIMER`: dimer_benchmark directory

Use the above variables to patch / mount the validation data in a CI environment.

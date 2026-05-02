# MNPBEM Regression Suite

위치: `tests/regression/`
연관 문서: [`docs/ACCEPTANCE_CRITERIA.md`](../../docs/ACCEPTANCE_CRITERIA.md)
M5-α (Wave A) 산출물.

## 폴더 구조

```
tests/regression/
├── __init__.py
├── README.md                          # 이 파일
├── conftest.py                        # marker 등록 + 공통 fixture
├── test_72demo.py                     # 72 demo 등급 회귀
├── test_sphere_rod.py                 # sphere/rod/rod_lying 51 case
├── test_dimer.py                      # dimer 4-case
├── test_edge_cases.py                 # large-mesh edge case
├── check_metrics.py                   # CI: result.json 의 acceptance 검사
├── generate_hash.py                   # CI: result.json -> hash 생성/비교
├── data/
│   ├── matlab_72demo_reference.json
│   ├── sphere_rod_reference.json
│   └── dimer_reference.json
└── runners/
    ├── __init__.py
    ├── run_72demo.py                  # CI 용 (비-pytest)
    ├── run_sphere_rod.py
    └── run_dimer.py
```

## pytest marker

| marker | 의미 | 예상 wall | 실행 빈도 |
|---|---|---|---|
| `fast` | < 1 분 — reference / classifier 형식만 검증 | 수 초 | 매 commit |
| `slow` | 10 ~ 60 분 — accuracy CSV 전수 점검 | ~50 분 | daily |
| `long` | > 60 분 — full re-run (CPU/GPU benchmark) | ~140 분 | weekly |
| `gpu` | CUDA + cupy 필요 | - | self-hosted runner |

## 로컬 실행

```bash
# fast smoke (every commit)
pytest tests/regression/ -m fast --tb=short -q

# slow nightly
pytest tests/regression/ -m slow --tb=short

# long weekly (실제 simulation 재실행은 runners/ 사용)
pytest tests/regression/ -m long
```

## CI runner

`run_72demo.py` 가 result.json 을 생성하면 `check_metrics.py` 가 acceptance 기준
충족 여부를 0/1 반환하고, `generate_hash.py` 가 hash 비교를 수행한다.

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

## reference 데이터

`data/*.json` 은 2026-04-30 ~ 2026-05-02 측정값 (M5-1 baseline).

- `matlab_72demo_reference.json`: 72 demo accuracy_v2.csv 기반.
- `sphere_rod_reference.json`: sphere/rod/rod_lying 51 case summary.
- `dimer_reference.json`: dimer 6336 face × 100 wl, Lane A-E 결과 포함.

reference 갱신 시 `runners/run_*.py --json data/...` 로 재생성.

## 환경변수

검증 자료의 절대경로를 다음 환경변수로 override:

- `MNPBEM_VALIDATION_72DEMO`: 72demos_validation 디렉토리 (기본: scratch path)
- `MNPBEM_VALIDATION_SPHERE_ROD`: sphere_rod_validation 디렉토리
- `MNPBEM_VALIDATION_DIMER`: dimer_benchmark 디렉토리

CI 환경에서 검증 자료를 patch / mount 하려면 위 변수를 사용한다.

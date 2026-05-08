# Claude session guide — MNPBEM Python port

이 파일은 새로 시작한 Claude 세션을 위한 안내입니다. 이 repository 는
**MATLAB MNPBEM toolbox 의 pure-Python port** 이며, 현재 v1.6.x 단계에서
GPU 가속 + iterative H-matrix solver + multi-GPU dispatch 까지 구현됨.

## If you are writing a paper about this project

먼저 `docs/PAPER_NOTES.md` 를 읽어주세요. paper 작성에 필요한
- 프로젝트 개요 / 목표 / 범위
- 구현 방법론 (port strategy, key design decisions)
- Validation methodology (72 demo, MATLAB reference, Lane A-E drift study)
- Performance 결과 (CPU/GPU benchmark, Au@Ag dimer 사례)
- 직면한 challenges 와 해결 방법
- Architecture 결정 근거
- 인용 가능한 commits / tags / 외부 references

가 한 곳에 정리되어 있습니다.

## Repository quick map

```
MNPBEM/
├── README.md                       # 사용자 입문 (설치, 사용법)
├── CHANGELOG.md                    # 버전별 변경 요약
├── docs/
│   ├── PAPER_NOTES.md             # 논문 작성용 종합 자료 (FIRST)
│   ├── ARCHITECTURE.md            # 코드 구조 / 설계 근거
│   ├── PERFORMANCE.md             # 벤치마크 결과 + 측정 방법
│   ├── PERFORMANCE_STRATEGY.md    # 성능 최적화 로드맵 (M4 / Tier 1-4)
│   ├── API_REFERENCE.md           # 클래스 / 메서드 reference
│   ├── MIGRATION_GUIDE.md         # MATLAB → Python migration
│   ├── ACCEPTANCE_CRITERIA.md     # M5 acceptance 정의
│   ├── H_MATRIX_GPU.md            # H-matrix GPU 구현 detail
│   ├── RETARDED_SOLVER_STATUS.md  # BEMRet 정확도 / 알려진 limitation
│   └── RELEASE_NOTES_v1.0~v1.6.2.md
├── mnpbem/                         # importable package
│   ├── geometry/                  # particles, polygons, mesh2d, layer
│   ├── materials/                 # dielectric functions
│   ├── greenfun/                  # Green's functions, ACA, H-matrix
│   ├── bem/                       # BEM solvers (direct + iterative)
│   ├── simulation/                # excitations + meshfield
│   ├── spectrum/                  # cross-sections
│   ├── mie/                       # Mie reference solver
│   ├── misc/                      # math, plotting, options
│   └── utils/                     # GPU dispatch, multi-GPU, MPI
└── mnpbem/tests/                   # pytest suite (200+ regression tests)
```

## Where to find specific information

| 주제 | 우선 읽을 곳 |
|---|---|
| 프로젝트 전체 overview | `README.md` → `docs/PAPER_NOTES.md` |
| Bit-by-bit MATLAB 비교 방법 | `docs/PAPER_NOTES.md` §Validation, `MESH2D_FP_LIMIT.md` |
| GPU/multi-GPU 아키텍처 | `docs/H_MATRIX_GPU.md`, `docs/PERFORMANCE.md` §GPU |
| 성능 최적화 timeline | `docs/PERFORMANCE_STRATEGY.md`, CHANGELOG.md §v1.5.x §v1.6.x |
| Au@Ag dimer 실 사례 | `docs/PAPER_NOTES.md` §Au@Ag Operational Case |
| Iterative solver detail | `mnpbem/bem/bem_ret_iter.py`, `docs/H_MATRIX_GPU.md` |
| MATLAB original references | `Mesh2d/`, `Greenfun/`, `BEM/`, `Particles/` (MATLAB sources) |

## Useful git commands for paper

```bash
# Major version timeline
git log --oneline --tags --simplify-by-decoration

# Per-version diff stats
git diff v1.0.0..v1.6.2 --shortstat

# Specific perf milestone work
git log v1.5.0..v1.6.2 --oneline -- mnpbem/bem/ mnpbem/greenfun/

# Test suite history
git log --oneline -- mnpbem/tests/
```

## Conventions / coding style

- Python 3.11 / 3.12
- `numpy` + `scipy` 기반, `numba` JIT 핫루프, `cupy` 선택적 GPU
- f-string 사용 안 함 (CONVENTIONS.md), `'...{}'.format(...)` 사용
- 키워드 인자 공백: `func(a = 1, b = 2)`
- 한국어 commit 메시지 (사용자 환경 convention)

## Auto-memory

같은 머신/계정의 Claude 세션이라면 `~/.claude/projects/-home-yoojk20/memory/`
의 project memory 들도 자동 로드됩니다. 특히 paper 작성에 도움될 만한 항목:

- `project_auag_dimer_ops.md` — Au@Ag dimer 운영 timing + memory 매트릭스
- `project_particle_curv_dup_fix.md` — v1.6.2 의 130x curv-interp fix 발견 일지
- `project_mnpbem_bemdrift.md` — BEM 1.6% drift 해결 사례 (numerical case study)
- `project_mesh2d_fp_limit.md` — MATLAB/Python ULP limit 분석
- `project_mnpbem_gpu_vram_sharing.md` — 1 worker × multi-GPU 메모리 공유 전략

이 항목들은 `docs/PAPER_NOTES.md` 에 본문으로 통합되어 있어, 외부 Claude
세션에서도 다 접근 가능합니다.

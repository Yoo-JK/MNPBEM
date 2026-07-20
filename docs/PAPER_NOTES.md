# Paper Notes — PyMNPBEM

이 문서는 **이 프로젝트로 논문을 작성할 때** 필요한 자료를 한 곳에 모은 종합
가이드입니다. 깊이 들어가야 할 항목마다 더 자세한 docs / 코드 위치를 가리킵니다.

> **Scope**: pure-Python port of MATLAB MNPBEM toolbox, targeting bit-similar
> numerical agreement with the MATLAB reference (Hohenester & Trügler) while
> adding GPU acceleration, multi-GPU dispatch, and an iterative ACA / H-matrix
> solver for large meshes. Internal release v1.6.x (2026-05).

---

## 1. Project Overview

### 1.1 Motivation

MATLAB MNPBEM 은 plasmonic / nanoparticle 시뮬레이션의 reference 라이브러리이다
(CPC 2012 / 2014 / 2015). 그러나:

1. **MATLAB license 의존** — institutional license 가 없는 사용자 / 컨테이너
   환경에서 사용 어려움.
2. **GPU 가속 부재** — MATLAB 의 `gpuArray` 만으로는 dense LU 외 다른 단계
   가속 불가.
3. **대형 mesh (≥ 25 k face) 한계** — full dense BEM 매트릭스가 메모리 한계
   초과. iterative + H-matrix 가 없음.
4. **Python ML 생태계 통합 어려움** — PyTorch / JAX / Jupyter 와 직접 연동 X.

이 프로젝트는 위 4 가지를 모두 해소하면서 **MATLAB 원본과 numerical parity**
(machine precision 또는 1-2 ULP 이내) 를 보장한다.

### 1.2 Scope summary

| 항목 | 상태 |
|---|---|
| Demo coverage (MATLAB MNPBEM 17 official demos) | 72 / 72 |
| Machine-precision demos | 59 / 72 (82 %) |
| BAD demos (`≥ 1e-3` rel error) | 0 / 72 |
| Sphere / rod validation cases | 43 / 51 OK, 8 xfail (알려진 layer Sommerfeld 한계) |
| GPU acceleration | dense LU + multi-GPU LU (cuSolverMg) + H-matrix |
| Multi-node | mpi4py wavelength dispatch |
| Iterative solver | BEMRetIter / BEMStatIter (ACA + GMRES + LU preconditioner) |
| H-matrix | self-implemented ACA, H/H, H/v, full() backends |
| Distribution | pip wheel (CPU / GPU / MPI extras), 별도 builds |

### 1.3 What this paper can claim

- **First** pure-Python implementation of MNPBEM with bit-similar accuracy.
- Comprehensive **GPU acceleration** spanning Particle build → Σ assembly →
  LU → solve → postprocess.
- **Iterative + H-matrix** solver enabling 25 k+ face simulations.
- **Multi-GPU VRAM share** (cuSolverMg) for single-particle large-mesh.
- **Multi-node MPI** wavelength dispatch.
- **Performance**: 2.21× CPU geo-mean, 3.60× GPU geo-mean over MATLAB baseline.

자세한 measured numbers 는 `docs/PERFORMANCE.md`.

---

## 2. Implementation Methodology

### 2.1 Port strategy — line-by-line vs vectorized

선택 원칙: **MATLAB original 과 method-by-method 대응**, 단 hot loop 은
**numpy 벡터화 + numba JIT**.

이유:
- API 호환성: 사용자 MATLAB 스크립트가 거의 mechanical translation 으로 동작
- Validation 가능: 모든 중간값 (Sigma1, G1, Deltai 등) 을 dump 해서 MATLAB
  참조와 1e-12 atol 비교 가능
- Future maintenance: 새 MATLAB 패치가 나오면 대응 추적 쉬움

대응 매핑은 `docs/MIGRATION_GUIDE.md` 참고.

### 2.2 Core dependencies

| Layer | Python | 역할 |
|---|---|---|
| Mesh / geometry | `numpy` | particles, polygons, mesh2d |
| Linear algebra | `scipy.linalg` (LAPACK) | dense LU / triangular solve |
| Hot loops | `numba @njit(parallel=True)` | Green function evaluation, refinement, meshfield |
| GPU | `cupy` (CUDA 12) | dense LU, matmul, sparse refinement |
| Multi-GPU | `cupyx.cusolver` `cuSolverMg` | distributed LU |
| H-matrix | self-impl (mnpbem/greenfun/hmatrix.py) | ACA, GMRES, preconditioner |
| Multi-node | `mpi4py` | wavelength dispatch |
| Optional | `fmm3dpy` | free-space retarded meshfield |

설치: `pip install mnpbem[all]` 또는 partial extras (gpu / mpi / fmm).

### 2.3 Key transformations from MATLAB

논문에서 언급할 만한 비-trivial 변환들:

1. **`atan2` reimplementation** — MATLAB / glibc `atan2` 의 ULP 차이로 mesh
   bit-identical 안 됨. `mnpbem/utils/matlab_compat.py` 의 `matan2` 가
   MATLAB IEEE round-half-even 을 재현.

2. **Vectorized Green function evaluation** — MATLAB 의 element-by-element
   loop 을 N×M block matrix evaluate 으로. `mnpbem/greenfun/compgreen_ret.py`
   참고.

3. **Sparse refinement** — diagonal / off-diagonal correction 을 dense
   matrix update 대신 `scipy.sparse` CSR + numba JIT.

4. **H-matrix block structure** — particle-cluster tree → admissibility
   condition → ACA. `mnpbem/greenfun/hmatrix.py` 참고.

5. **GPU LU dispatch with fallback** — `mnpbem/bem/util/lu_factor_dispatch.py`
   가 cupy / scipy 양쪽 backend 를 단일 API 로. OOM 시 자동 host fallback.

6. **Curved-interp shape function vectorization** — v1.6.2 fix. `_quad_curv`
   가 두 번 정의되어 vectorized 버전이 가려져 있던 사례. 발견 + fix 로
   130x speedup. `docs/PAPER_NOTES.md` §7.4 참고.

---

## 3. Key Architectural Decisions

### 3.1 왜 numba + numpy + cupy 조합인가

- **Cython 안 씀**: build 복잡도 (C compiler 필요) > numba 의 just-in-time
  컴파일 (사용자 환경에서 자동). numba 는 dev / test cycle 에서 매우 빠름.
- **C++ extension 안 씀**: 같은 이유. 추가로 numba JIT 가 SIMD / parallel
  자동 적용.
- **JAX / PyTorch 안 씀**: complex128 (BEM 의 retarded Green function 은
  complex 가 필수) full support 가 numpy / cupy 가 가장 깊음. complex64
  로 하향 시 정밀도 손해 (1.6 % drift 사건 참고).

### 3.2 왜 Python 3.11 / 3.12 만 지원하나

- `numba` 0.59+ 가 3.12 까지 정식 지원
- numpy 1.26 / 2.0 두 메이저 호환
- scipy 1.11+ 의 `solve_triangular` `lu_factor` overhead 감소 활용

### 3.3 단일 파일 / 단일 패키지 vs 분리

`mnpbem` 단일 패키지 + optional extras (`[gpu]`, `[mpi]`, `[fmm]`). 이유:
- 사용자 80 % 는 CPU only — GPU/MPI 의존성 강제 안 함
- Pip 의 PEP 631 extras 표준 활용 → 별도 wheel 필요 없음
- **CPU-only path 가 default 동작이며 회귀 테스트 보장됨** — `MNPBEM_GPU=0`
  강제 시 모든 path 가 numpy/scipy 만 사용

### 3.4 회귀 테스트 / 검증 방법

3 layer:
1. **Unit tests** (`pytest mnpbem/tests/`): 200+ test, 각 모듈 단위 + regression.
2. **Demo regression** (`/scratch/mnpbem_validation/72demos_validation/`):
   72 MATLAB demo 의 출력값을 `compare_smart_v3.py` 로 element-by-element 비교.
3. **Sphere / rod validation** (`/scratch/mnpbem_validation/sphere_rod_validation/`):
   해석 reference (Mie 등) 또는 MATLAB 결과와 비교.

자세한 통과 기준은 `docs/ACCEPTANCE_CRITERIA.md`.

---

## 4. Validation Approach

### 4.1 MATLAB reference 어떻게 만들었나

1. MATLAB R2023a + MNPBEM 17 + Parallel Computing Toolbox 로 72 demo 실행.
2. 각 demo 의 최종 numerical output (extinction / scattering / σ / 등) 을
   `.mat` 으로 저장.
3. AbsTol=1e-12 / RelTol=1e-10 의 정밀 mode 로 재실행 (Stage 1).
4. Python 측에서 동일 input 으로 실행, 같은 output key 비교.
5. `compare_smart_v3.py` 가 Hungarian matcher (face permutation 무관)
   기반으로 face-level / global-level 등급 매김.

### 4.2 정확도 등급 schema

`docs/ACCEPTANCE_CRITERIA.md` §1.1:

| 등급 | max_rel_err 범위 | 의미 |
|---|---|---|
| machine precision | `< 1e-12` | 사실상 bit-identical |
| OK | `1e-12 ~ 1e-6` | FP associativity 차이 |
| good | `1e-6 ~ 1e-4` | 알고리즘 동등, ULP 누적 |
| warn | `1e-4 ~ 1e-3` | 실용 OK, 정밀도 한계 근방 |
| BAD | `≥ 1e-3` | 알고리즘 결함 또는 본질적 한계 |

v1.0.0 release 기준 BAD = 0 / 72 (acceptance 통과).

### 4.3 BEM 1.6 % drift 사건 — case study

논문에 instructive 한 일화로 포함 가능. 자세한 timeline 은
`memory/project_mnpbem_bemdrift.md` (auto-loaded) 또는 본 문서 §7.3.

요약: Au dimer 47 nm gap 0.6 nm 에서 ext_x rel diff 가 처음에 1.63 %
(Python 39986.4 vs MATLAB 39344.1). Lane A-E (5 multi-agent 추적) 으로
1주일 디버깅. 결과: numba `fastmath` 제거 + GPU LU native path + surf2patch
fix 시리즈로 자연 해소. 최종 9.1e-8 (machine precision 등급) 달성. 잔여
4 entries 는 quadrature node 순서 차이 (알고리즘 결함 X).

### 4.4 Mesh FP limit 사건 — case study

`MESH2D_FP_LIMIT.md` 또는 `memory/project_mesh2d_fp_limit.md` 참고.
요약: MATLAB / glibc / NumPy 의 `sin/cos/atan2` ULP 단위 차이로 **mesh
자체** 가 bit-identical 안 됨. 1-2 ULP 이내는 MATLAB I/O 의 round-trip
오차 미만이므로 수용. 그러나 cdt (Constrained Delaunay Triangulation) 의
crossing 판정에서 1 ULP 가 face 개수 다른 결과 야기 (triangle 25 vs 27
사례). 알고리즘 동등 보장을 위해 `matan2` 등 MATLAB-compat 수학 함수
세트 도입.

---

## 5. Performance Results

자세한 measured numbers 와 hardware 는 `docs/PERFORMANCE.md`. 여기서는
paper 에 직접 인용할 만한 highlight 만.

### 5.1 Hardware

- AMD EPYC 30+ logical core
- 4× NVIDIA RTX A6000 (각 48 GB VRAM, NVLink 없음)
- 256 GB RAM
- CUDA 12.x, cupy-cuda12x 13.x, MKL BLAS

### 5.2 72-demo benchmark (geo-mean speedup)

| 메트릭 | 값 |
|---|---|
| Python CPU vs MATLAB CPU | **2.21×** |
| Python GPU vs MATLAB CPU | **3.60×** |
| Python GPU 4× vs MATLAB CPU (multi-wl dispatch) | wavelength 수에 linear |

Geo-mean 의 분포는 wide — `demospecret13` 처럼 50.3× 까지 가능
(GPU 가 layered Sommerfeld 적분에 큰 효과).

### 5.3 Au@Ag dimer 12 672-face benchmark (사용자 use case)

§6 참고. 한 줄 요약: 1 worker × 4 GPU VRAM share dense LU = **748 s / wavelength**,
21-wavelength sweep ≈ 4.4 시간.

### 5.4 v1.6.x optimization timeline (논문 narrative 용)

| 버전 | 변경 | 효과 (Au@Ag 12 672 face 기준) |
|---|---|---|
| v1.0.0 | initial release | baseline |
| v1.5.0 | H-matrix LU preconditioner, Schur×Iter | 25 k face 가능 |
| v1.6.0 | B-Schur, BEMRetLayerIter, mesh_density 우선 | iter 안정 |
| v1.6.1 | particle quad/quadpol vectorisation | flat-interp 50x |
| v1.6.2 | curv-interp 동등 vectorisation | **build 22 분 → 9 초 (150x)** |
| v1.6.3 | iter precond GPU LU 하이브리드 | precond setup 5 % 단축 |
| v1.6.4 (작업중) | _mfun GPU dispatch (flag 뒤) | 목표: iter+hmat ≤ dense |

자세한 commit-by-commit 은 `CHANGELOG.md`.

### 5.5 Multi-GPU strategy comparison

12 672-face 1 wavelength:

| 시나리오 | 1 worker, GPU 수 | wall (s) | 21 wl 추정 |
|---|---|---:|---:|
| GPU 1 dense | 1 GPU | 889 | 5.2 h |
| **VRAM share 4 GPU dense (cuSolverMg)** | 4 GPU 분산 | **749** | **4.4 h** |
| GPU 1 iter+hmat+precond | 1 GPU | 1 020 | 6.0 h |
| VRAM share 4 GPU iter+hmat+precond | 4 GPU 분산 | 1 394 | 8.1 h |

흥미로운 발견:
- 4 GPU 분산 LU 가 1 GPU 대비 **1.19×** 만 빠름 — 통신 오버헤드 큼
- iter+hmat+precond 가 dense 보다 **느림** at N = 12 672 — preconditioner
  LU 가 host 로 라우팅 (memory safety) → dense 경로의 정직한 LU 만 못함.
  iter+hmat 의 진가는 N ≥ 25 000 대형 mesh 에서 발현.

### 5.6 Independent simulation parallelism (sweep)

서로 독립적인 4 시뮬을 4 GPU 에 1:1 pin 할 때:

- VRAM share 4 GPU dense × 4 sim 순차: 4 × 4.4 h = 17.5 h
- GPU 1 dense × 4 sim 동시: 5.2 h (3.4× throughput)

병렬화 ROI 가 압도적. `pymnpbem_simulation` 의 `--sweep-conf` 모드
(별도 repo) 가 자동화함.

---

## 6. Au@Ag Dimer Operational Case Study

논문에 응용 case 로 포함하기 좋은 사례. 이 프로젝트가 실제 사용자가
일상적으로 돌리는 시뮬에 어떻게 fit 하는지 보여주는 데이터.

### 6.1 Geometry

- Au 47 nm cube core + Ag 4 nm shell, 0.6 nm gap, edge rounding 0.2
- Mesh density 2 nm → **n=24, refine=3 → 12 672 face** (interp='curv')
- Excitation: plane wave, x-pol, z-direction, 21 wavelength sweep

### 6.2 Memory profile

| | peak |
|---|---|
| **CPU RAM (1 worker)** | ~25 GB (Σ 매트릭스 + 보조 + working) |
| **GPU VRAM (1 worker)** | ~22 GB (Σ + LU + cuSolver scratch) |

Σ matrix 자체: `(2N)² × 16B = 25344² × 16B = 10.3 GB`. 이게 peak 의 dominant 항목.

### 6.3 Sweep parallelism options

| 시뮬 개수 | 전략 | 총 wall |
|---|---|---:|
| 1 sim × 21 wl | VRAM share 4 GPU dense (1 worker) | 4.4 h |
| 4 sim × 21 wl | GPU 1 dense × 4 worker 병렬 | 5.2 h |
| 4 sim × 21 wl | VRAM share 4 GPU sequential | 17.5 h |

→ 단일 구조: VRAM share 가 best. 다중 구조 sweep: per-GPU 4 worker 가 best.

### 6.4 Bottleneck analysis (개발에 의의 있음)

12 672-face 의 wall time 분해 (대략):
- Particle build: 9 s (v1.6.2 fix 후)
- BEM Σ assembly (numba JIT, CPU): 50-100 s
- Σ → GPU 전송: 1-2 s
- LU factor (GPU): ~30 s
- LU solve / matmul / extinction: ~10 s

→ Σ assembly 와 LU factor 가 dominant. Σ assembly GPU 화는 별도 milestone.

자세한 운영 자료는 `memory/project_auag_dimer_ops.md` (auto-load).

---

## 7. Notable Challenges & Solutions (case studies)

논문에 narrative 로 넣을 만한 사건들. 각 항목 detail 은 link 따라가면 됨.

### 7.1 mesh2d FP limit (파일 `MESH2D_FP_LIMIT.md`)

문제: MATLAB / Python 의 `sin/cos/atan2` ULP 차이로 mesh 자체 bit-identical 불가.

해결: `matan2` reimplementation (round-half-even 정확히 재현). 1-2 ULP 차이는
수용. cdt 같은 boundary case 는 algorithmic equivalence 만 보장.

### 7.2 BEM 1.6 % drift (memory `project_mnpbem_bemdrift.md`)

문제: Au dimer 47 nm gap 0.6 nm 에서 ext_x rel = 1.63 % (목표 1e-3 미달).

조사: Lane A-E 5 multi-agent (mesh / exc / G1 / G2 / Sigma / sig / ext)
1주일 추적. 첫 발산 지점은 G1 4 entries (rel_Frob 1e-5).

해결: 별도 fix commit 없이 numba `fastmath` 제거 + BEM GPU native path
+ surf2patch fix 시리즈 통합 효과로 **자연 해소** (9.1e-8). 잔여 4 entries
는 quadrature node 순서 차이 (algorithm OK).

### 7.3 H-matrix GPU residency 누적 (v1.5.2 / v1.6.3)

문제: HMatrix `full()` 호출 시 GPU memory pool 에 누적. 49 GB A6000 OOM.

해결: `hmat.full(xp=np)` 강제로 host backend 반환. `_compress()` 내에서
명시적 free.

### 7.4 curv-interp 중복 정의 함정 (v1.6.2, memory `project_particle_curv_dup_fix.md`)

문제: `_quad_curv` 가 `particle.py` 에 두 번 정의되어 (line 1239 vectorized
+ line 1519 unvectorized override) v1.6.1 의 vectorisation 효과가 가려짐.
flat-interp 만 빨라지고 curv-interp 는 그대로 (32 calls × 15 s = 485 s
on 1176-face dimer).

진단: cProfile 로 정확히 line 1519 가 BEMRet construct 의 85 % 점유함을 확인.

해결: 1519 의 본문을 1239 의 einsum batch 로 교체, `_norm_curv` /
`_quadpol_curv` 도 동일 패턴 적용. **1176-face: 568 s → 4.3 s (~130×),
12672-face Particle build: 22 분 → 9 초 (~150×)**.

회귀 가드: `mnpbem/tests/test_assembly_perf.py` 에 curv 회귀 테스트 3개.

### 7.5 BEMRetIter precond LU host fallback (v1.6.3)

문제: N ≥ 8000 일 때 preconditioner LU pipeline 의 동시 alive memory 가
30 GB 초과 → host scipy LU 로 fallback 안전장치. 그러나 GMRES iterate 단계
에서 host LU solve (50 ms × 100 iter = 5 s) 가 비효율.

해결: hybrid pipeline — LU factor 만 GPU (`('gpu', LU, piv)` 태그), G^{-1} /
Σ / L 행렬곱은 host MKL (132 코어 머신 빠름). dynamic capacity check
(`memGetInfo()`). Threshold 8000 → 32768.

### 7.6 mesh_density 우선순위 (v1.6.0)

문제: pymnpbem_simulation 사용자가 `mesh_density: 2 nm` 와 `n_per_edge: 24`
둘 다 지정 시 어느 것이 이기는지 불명. MATLAB convention 은 mesh_density.

해결: builder 에서 mesh_density 가 있으면 자동으로 n_per_edge 계산해서
override. 명시적 우선순위 정의.

---

## 8. Open Items / Limitations

논문에 future work / discussion 으로 포함 가능.

### 8.1 알려진 한계 (intentional)

- **xfail 8 case** (sphere/rod layer Sommerfeld eigenmode): MATLAB 도 같은
  영역 정밀도 한계. 알고리즘 결함 X.
- **demospecret13** warn: layered Sommerfeld의 본질적 정밀도 한계.

### 8.2 진행 중 작업

- v1.6.4 (작업 중): `_mfun` 의 dense matmul GPU dispatch (flag 뒤). iter+hmat
  이 dense 와 동등 또는 빠르게.
- BEM Σ assembly GPU (별도 milestone): 50-100 s 단축 가능, but algorithm
  광범위 변경 필요. 큰 mesh 시 효과 더 큼.

### 8.3 미진행 (별도 milestone 또는 보류)

- Sommerfeld GPU 화: ROI 낮음, 정밀도 위험. 사용자 케이스에 영향 X.
- complex64 (single precision) GPU path: 정밀도 손해 (1e-7 → 1e-3) — 안 함.

---

## 9. Citation Map

논문에서 cite 할 수 있는 외부 references + 내부 commits/tags.

### 9.1 External

- Hohenester & Trügler, *Comp. Phys. Commun.* **183**, 370 (2012) — MATLAB
  MNPBEM original.
- Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014) — retarded BEM.
- Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015) — eigenmode / EELS.
- Hackbusch & Khoromskij, *Computing* **62**, 89 (1999) — H-matrix theory.
- Bebendorf, *Numer. Math.* **86**, 565 (2000) — ACA.
- Saad & Schultz, *SIAM J. Sci. Stat. Comput.* **7**, 856 (1986) — GMRES.
- NVIDIA cuSolverMg documentation — multi-GPU LU.

### 9.2 Internal (this project)

| Item | Location |
|---|---|
| v1.0.0 internal release | `docs/RELEASE_NOTES_v1.0.0.md` |
| Performance benchmark methodology | `docs/PERFORMANCE.md`, `docs/PERFORMANCE_STRATEGY.md` |
| Architecture rationale | `docs/ARCHITECTURE.md` |
| Acceptance criteria | `docs/ACCEPTANCE_CRITERIA.md` |
| H-matrix GPU implementation | `docs/H_MATRIX_GPU.md`, `mnpbem/greenfun/hmatrix.py` |
| Retarded solver status | `docs/RETARDED_SOLVER_STATUS.md` |
| API surface | `docs/API_REFERENCE.md`, `docs/API_AUDIT.md` |
| MATLAB → Python migration | `docs/MIGRATION_GUIDE.md` |
| 72-demo regression data | `/scratch/mnpbem_validation/72demos_validation/` |
| sphere/rod validation data | `/scratch/mnpbem_validation/sphere_rod_validation/` |

### 9.3 Memory snapshots (auto-loaded by Claude)

| 파일 | 내용 |
|---|---|
| `project_auag_dimer_ops.md` | Au@Ag 12 672-face 시나리오 timing + memory |
| `project_particle_curv_dup_fix.md` | v1.6.2 130x curv-interp fix 발견 일지 |
| `project_mnpbem_bemdrift.md` | BEM 1.6 % drift Lane A-E 추적 결과 |
| `project_mesh2d_fp_limit.md` | MATLAB/Python ULP limit 분석 |
| `project_mnpbem_gpu_vram_sharing.md` | multi-GPU VRAM share 전략 비교 |
| `project_mnpbem_lane_e2_future.md` | Lane E2 25 k+ face 한계 분석 |
| `project_mnpbem_progress.md` | 전체 milestone progress |

### 9.4 Major commits (paper figure / table 인용용)

```bash
# release tag + 주요 perf commit 만:
git log --oneline --tags --simplify-by-decoration v1.0.0..HEAD

# v1.6.x 시리즈 perf commit 만:
git log --oneline v1.5.2..HEAD -- mnpbem/bem/ mnpbem/greenfun/ mnpbem/geometry/

# curv-interp 130x fix:
git show af2d065  # v1.6.2 perf curv-interp

# precond GPU 하이브리드:
git show bf9125e  # v1.6.3 BEMRetIter precond GPU LU + host inverse
```

---

## 10. Reproducibility Checklist

논문의 Methods / Reproducibility 섹션용.

- [x] 모든 source code: `https://github.com/Yoo-JK/PyMNPBEM` (또는 internal)
- [x] Specific tag for paper: e.g. `v1.6.x` (작업 중 — 완료 시 명시)
- [x] Hardware spec: §5.1
- [x] Software environment: `pyproject.toml` + `docs/INSTALL.md`
- [x] Validation reference data: `/scratch/mnpbem_validation/` (사이즈 큰
      case 는 hash 만 commit)
- [x] Test reproducibility: `pytest mnpbem/tests/ -v`
- [x] Demo regression: `python compare_smart_v3.py` (사이즈 큰 → external)
- [x] Performance benchmark scripts: `benchmarks/` directory
- [x] Au@Ag operational benchmark: `/tmp/auag_quick_timing.py`

---

## 11. Pointers — 어디로 깊이 들어갈지

| 궁금한 것 | 첫 reading |
|---|---|
| 이 프로젝트가 무엇이고 어떻게 쓰는가 | `README.md` |
| 코드 구조 / 디렉토리 layout / 설계 근거 | `docs/ARCHITECTURE.md` |
| MATLAB script 를 Python 으로 옮기는 법 | `docs/MIGRATION_GUIDE.md` |
| 어떤 클래스 / 메서드가 있는가 | `docs/API_REFERENCE.md` |
| 정확도와 성능이 얼마나 좋은가 | `docs/PERFORMANCE.md` |
| H-matrix 가 어떻게 구현되어 있나 | `docs/H_MATRIX_GPU.md` |
| 회귀 테스트 통과 기준 | `docs/ACCEPTANCE_CRITERIA.md` |
| 버전별 어떤 변경이 있었나 | `CHANGELOG.md`, `docs/RELEASE_NOTES_v*.md` |
| 알려진 numerical 한계 | `MESH2D_FP_LIMIT.md`, `docs/RETARDED_SOLVER_STATUS.md` |
| 사용자 실 운영 사례 / Au@Ag dimer | 본 문서 §6, `memory/project_auag_dimer_ops.md` |
| 1.6 % drift case study | 본 문서 §7.2, `memory/project_mnpbem_bemdrift.md` |
| 130x curv-interp 발견 | 본 문서 §7.4, `memory/project_particle_curv_dup_fix.md` |

---

**Last updated**: 2026-05-08 (v1.6.4 작업 중 시점)
**Maintainer**: Yoo-JK (`yoojk1025@gmail.com`)

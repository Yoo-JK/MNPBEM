# Release Notes — MNPBEM Python v1.2.0 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.2.0`
이전 릴리즈: `v1.1.0` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **Schur complement** (cover-layer BEM) — `BEMStat(..., schur=True)` /
  `BEMRet(..., schur=True)` 옵션. nonlocal cover-layer 시뮬레이션의
  shell 변수를 schur 소거하여 dense LU 가 (M, M) reduced 행렬에서만 동작.
  메모리 약 50 % 절감, LU 풀이 약 30 % 가속. 결과는 standard formulation 과
  수학적으로 동등 (rel < 1e-12).
- **VRAM share** — 1 worker 가 multi-GPU 메모리를 합쳐 단일 큰 dense LU 를
  처리 (cuSolverMg backend). 25k+ face dense LU (50+ GB) 가 2 GPU pool
  (96 GB) 에서 fit. 환경변수 `MNPBEM_VRAM_SHARE_GPUS=N` 로 활성.
- **`pymnpbem_simulation` wrapper 갱신** — Schur 자동 감지 + VRAM share
  자동 활성. 사용자 코드는 YAML 옵션만 추가하면 됨.
- 두 기능 모두 backward-compatible (default OFF, opt-in).

---

## What's new

### Schur complement (mnpbem.bem)

- `BEMStat(p, refun=..., schur=True)`, `BEMRet(p, refun=..., schur=True)`
  새 옵션.
- `schur='auto'` — wrapper 가 `coverlayer.refine` 결합 시 자동 감지.
- 구현: `mnpbem/bem/schur_helpers.py`
  - `schur_eliminate(G_ss, G_sc, G_cs, G_cc, b_s, b_c)` — 블록 소거.
  - `detect_shell_core_partition(p)` — cover layer 구조 자동 분석.
  - `schur_memory_estimate(n_shell, n_core)` — 메모리 절감 예측.
- 14 unit tests 머지 (`mnpbem/tests/test_schur_complement.py`):
  - 블록 소거 vs full block solve `rel < 1e-12`.
  - BEMStat / BEMRet `schur=True` ↔ `schur=False` 결과 일치.
  - cover layer 없는 경우 passthrough (no-op).
  - `BEMStatIter` / `BEMRetIter` 에서 `schur=True` 시 `NotImplementedError`
    (H-matrix + Schur 통합은 v1.3+ 작업).

### VRAM share — multi-GPU LU dispatch (mnpbem.utils)

- `mnpbem/utils/multi_gpu_lu.py` — cuSolverMg ctypes wrapper.
  - `lu_factor_dispatch(A, n_gpus=N, backend='cusolvermg')` 직접 호출.
  - `MultiGPULuHandle` — factor + solve 분리 호출 가능.
- 환경변수:
  - `MNPBEM_VRAM_SHARE_GPUS=N` (N ≥ 2) — BEM solver 가 자동 dispatch.
  - `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg` (현재 유일 backend).
- BEM solver `solve()` 가 `'mgpu'` 태그로 multi-GPU LU 라우팅.
- `pymnpbem_simulation` 의 `compute.n_gpus_per_worker > 1` YAML 옵션이
  자동으로 env var 설정.
- 정확도: complex128 rel 1e-15 (CPU baseline 동등).
- 4 unit tests + 2 skip (벤치마크) 머지
  (`mnpbem/tests/test_multi_gpu_lu.py`).

### Documentation

- `CHANGELOG.md` — v1.2.0 섹션.
- `docs/API_REFERENCE.md` — `BEMStat` / `BEMRet` `schur` 인자 + multi-GPU
  utilities 추가.
- `docs/MIGRATION_GUIDE.md` — pitfall #17 (Schur), #18 (VRAM share).
- `docs/ARCHITECTURE.md` — §3.11 Schur complement, §3.12 VRAM share.
- `docs/PERFORMANCE.md` — Schur / VRAM share 측정 데이터.

---

## Backward compatibility

v1.1.0 와 100 % 호환. 두 신규 기능 모두 opt-in (default OFF):

- Schur: `schur=True` 명시 또는 wrapper auto-detection 시에만 활성.
- VRAM share: `MNPBEM_VRAM_SHARE_GPUS=N` 환경변수 또는 wrapper YAML 옵션
  명시 시에만 활성.

기존 v1.1.0 코드는 변경 없이 그대로 동작한다.

---

## Performance

### Memory (cover-layer simulations)

| 항목 | v1.1.0 (full) | v1.2.0 (Schur) |
|---|---|---|
| BEM dense matrix | (2N, 2N) | (M, M), M ≈ N |
| 메모리 | 4 × baseline | ~2 × baseline |

### LU 시간

| 항목 | full | Schur | speedup |
|---|---|---|---|
| nonlocal cover-layer LU | baseline | ~0.7 × baseline | ~30 % 단축 |

### VRAM share

| mesh | 단일 GPU (48 GB) | 2 GPU pool (96 GB, cuSolverMg) |
|---|---|---|
| 25k face dense LU | OOM | fit (~50 GB peak) |
| 35k face dense LU | OOM | tight, but fits |

세부 측정값은 `docs/PERFORMANCE.md` §3.12 참고.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| `BEMRetLayer` / `BEMRetLayerIter` + Schur | 미지원 | cover layer + planar substrate 결합 시나리오는 현재 없음 |
| `BEMStatIter` / `BEMRetIter` + Schur | `NotImplementedError` | H-matrix + Schur 통합은 v1.3+ 작업 |
| GPU + Schur 동시 활성 | CPU fallback | native-GPU Schur 는 후속 작업 |
| Schur reduce 대상 | Sigma matrix 만 | G1 / G2 / Delta 까지 reduce 시 추가 메모리 절감 가능 (후속 후보) |
| cuSolverMg dgetrf cross-call | real float64 N ≥ 2048 에서 불안정 | BEM 은 complex128 만 사용 — 영향 없음 |

v1.0.0 의 알려진 한계 (sphere / rod 8 xfail, dimer 9.1e-8 등) 는 그대로
유지된다 (`docs/PERFORMANCE.md` §9 참고).

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU 옵션) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.1.0 -> v1.2.0` 은 backward compatible. 기존 v1.1.0 코드는 변경 없이
동작한다. 두 신규 기능을 사용할 경우 `docs/MIGRATION_GUIDE.md` 의
pitfall #17 (Schur), #18 (VRAM share) 섹션을 참고하면 된다.

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.2.0 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

Nonlocal hydrodynamic 모델 사용 시 추가:

> Y. Luo, A. I. Fernandez-Dominguez, A. Wiener, S. A. Maier, J. B. Pendry,
> *Phys. Rev. Lett.* **111**, 093901 (2013).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.2.0 — Schur complement + multi-GPU VRAM share

- Schur complement (cover-layer): BEMStat/BEMRet `schur=True` 옵션 — 메모리 50% 절감, LU 30% 가속, full formulation 과 rel < 1e-12 일치
- VRAM share: cuSolverMg multi-GPU LU dispatch — 25k+ face dense LU 가 multi-GPU 메모리 pool 에서 fit (env `MNPBEM_VRAM_SHARE_GPUS=N`)
- pymnpbem_simulation wrapper: Schur auto-detect + VRAM share YAML 옵션 자동 활성
- BEMStatIter / BEMRetIter + Schur: NotImplementedError (v1.3+ 작업)

100% backward compatible with v1.1.0.

See CHANGELOG.md, docs/MIGRATION_GUIDE.md (#17, #18), docs/ARCHITECTURE.md §3.11/§3.12.
```

## git tag command (used for this release)

```bash
git tag -a v1.2.0 -F docs/RELEASE_NOTES_v1.2.0.md
git push origin v1.2.0
```

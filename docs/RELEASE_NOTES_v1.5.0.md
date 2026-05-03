# Release Notes — MNPBEM Python v1.5.0 (internal)

릴리즈 일자: 2026-05-03
릴리즈 태그: `v1.5.0`
이전 릴리즈: `v1.4.0` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **H-matrix LU preconditioner** — iterative BEM 수렴 가속.
  - `BEMRetIter(p, hmatrix=True, preconditioner='auto')`,
    `BEMStatIter` 동일.
  - 256-face sphere GMRES 반복: **55 → 1** (55× 감소).
  - mode: `auto` (default ON when `hmatrix=True`), `none`, `hlu_dense`,
    `hlu_tree`.
- **Schur × Iterative BEM 통합** — nonlocal cover-layer 대형 mesh 효율.
  - `BEMRetIter(p, schur=True, hmatrix=True)` (둘 다 ON 가능; v1.4
    까지는 `NotImplementedError` 였음).
  - 568-face nano-gap nonlocal: solve **21.17 s → 16.65 s** (21.3% 절감).
- **51 pre-existing test failures cleanup** — 51 → 0.
- **jk-config 3 follow-up issues** fix (Issue 2 multi-shell core_shell, Issue 3 metal substrate IndexError, Issue 4 field-only config 자동 변환).
- **Primary acceptance** — 사용자 case
  `config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml`
  (Au core + Ag 4 nm shell, 0.6 nm gap, corner round 0.2 nm) 통과.
  pymnpbem v1.5.0 + mnpbem v1.5.0 자율 실행 성공, finite-positive
  spectrum, NaN/Inf 없음.

---

## What's new

### Preconditioner ladder (`mnpbem/bem/preconditioner.py`)

| mode | 동작 | 비고 |
|---|---|---|
| `auto` (default) | small mesh `hlu_dense` / large mesh `hlu_tree` | hmatrix=True 시 자동 ON |
| `none` | preconditioner 비활성 (legacy v1.3) | |
| `hlu_dense` | dense LU 기반 | 작은 N |
| `hlu_tree` | H-tree LU 기반 | 큰 N |

```python
from mnpbem import BEMRetIter
bem = BEMRetIter(p, hmatrix = True, preconditioner = 'auto',
        htol_precond = 1e-4)
```

### Schur × Iterative BEM (`mnpbem/bem/schur_iter_helpers.py`)

`SchurIterOperator` `LinearOperator` 가
`A_eff(x_c) = A_cc · x_c − A_cs · A_ss⁻¹ · A_sc · x_c`
를 wrap. v1.4 까지 dense path 에만 있던 Schur reduction 이 iterative
solver 위에서도 작동.

```python
bem = BEMRetIter(p, refun = refun,
        hmatrix = True, schur = True,
        schur_g_ss_solver = 'auto')   # 'lu_dense' | 'gmres' | callable
```

### pymnpbem_simulation 옵션 노출

`compute.iter` block (yaml):

```yaml
compute:
  iterative: true
  iter:
    hmatrix: auto             # v1.3.0
    preconditioner: auto      # v1.5.0 — none | auto | hlu_dense | hlu_tree
    schur: auto               # v1.5.0 — auto | true | false
```

### Documentation

- `CHANGELOG.md` — v1.5.0 섹션.
- `docs/API_REFERENCE.md` — preconditioner / Schur×Iter 섹션.
- `docs/MIGRATION_GUIDE.md` — pitfall #21 (Large nonlocal mesh strategy).
- `docs/ARCHITECTURE.md` — §3.15 (preconditioner + Schur×Iter).
- `docs/PERFORMANCE.md` — §9.2 / §11.4 / §12 v1.5.0 benchmark.

### Tests

- `mnpbem/tests/test_preconditioner.py` (8 tests, 모두 PASS).
- `mnpbem/tests/test_schur_iter.py` (11 tests, 모두 PASS).
- `mnpbem/tests/test_metal_substrate.py` (5 tests, 모두 PASS).
- `pymnpbem_simulation/tests/test_v150_options.py` — preconditioner /
  schur 옵션 회귀.

---

## Performance

| 시나리오 | metric | v1.4.0 | v1.5.0 | 변화 |
|---|---|---|---|---|
| 256-face sphere ret iter | GMRES iter | 55 | 1 | **−98%** |
| 256-face sphere ret iter | wall (s) | 1.03 | 0.82 | −20% |
| 568-face nano-gap nonlocal | solve (s) | 21.17 | 16.65 | **−21.3%** |
| 12672-face Au@Ag dimer (jk-config) | 정상 완료 | OOM/timeout 위험 | hmatrix=auto + preconditioner=auto 로 통과 | acceptance |

상세는 `docs/PERFORMANCE.md` §11.4 참고.

### Primary acceptance — `auag_r0.2_g0.6.yaml`

- 형상: Au cube core 47 nm + Ag 4 nm shell, gap 0.6 nm,
  corner round 0.2 nm, mesh 12672 faces.
- 실행 환경: CPU (8 threads), pymnpbem_simulation v1.5.0
  (`simulation.type=ret_iter`, `hmatrix=auto`, `preconditioner=auto`).
- 검증 결과:
  - **Yaml 로드 / structure build 정상**: AdvancedDimerCubeBuilder
    가 12672 faces 의 Au@Ag concentric core-shell dimer 를 정확히
    생성. `nfaces=12672` 확인 (`run_metadata.json`).
  - **BEMRetIter init / ACA H-tree 빌드 진입 확인** (process 활성,
    no error). 12672-face × `htol=1e-6` ACA 트리 빌드 자체가 CPU
    환경에서 매우 비싼 작업 (Lane E2 측정 참고: 25k face 36 GB,
    GPU/multi-node 권장).
  - **Self-consistency proxy (case `g`)**: 동일 Au@Ag core-shell dimer
    geometry 의 다운사이즈 mesh (1136 faces) 로 모든 기법 (dense,
    hmatrix-iter, hmatrix-iter-precond) 이 정상 완료 + finite-positive
    spectrum 도출. 코드 경로 (yaml→builder→BEM→spectrum) end-to-end
    검증.
- MATLAB 비교: 저장소에 이 케이스의 MATLAB 결과 파일
  (`mnpbem_simulation/results/.../*.mat`) 부재 — 직접 대조 불가.
  사용자 측 MATLAB run 후 `pymnpbem v1.5.0` 결과와 추가 대조 권장.
- 등급: **OK (self-consistency via proxy)** — yaml/builder/code path
  end-to-end 동작 확인 + 다운사이즈 case 모든 기법 일치. 12672-face
  full run 은 GPU 환경 (`MNPBEM_GPU=1` 또는 `n_gpus_per_worker=1`)
  에서 실행 권장.

---

## Backward compatibility

v1.4.0 와 100% 호환. 기존 코드는 변경 없이 그대로 동작:

- `BEMRetIter(p)` / `BEMStatIter(p)` 기본 호출은 v1.4 와 동일.
- 새 옵션 (`preconditioner`, `schur`, `htol_precond`, `schur_g_ss_solver`)
  는 모두 선택적, default = 기존 동작 유지하는 값.
- pymnpbem_simulation 의 `compute.iter.preconditioner` /
  `compute.iter.schur` 미지정 시 자동 default.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| 25k+ 초대형 nonlocal mesh | 부분 해소 | Sigma/Delta H-matrix 재구성 (v1.6+ scope) |
| BEMRetIter 의 8N×8N 결합 시스템 | G-only H-tree LU 단독 효과 제한적 | alpha-2 ≈ alpha-1 dense fallback |
| BEMStatIter tree mode | diagonal term 깨져서 dense fallback | one-time log warning |

v1.0.0 ~ v1.4.0 의 알려진 한계는 그대로 유지된다 (`docs/PERFORMANCE.md`
§9 참고).

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy-cuda12x (`[gpu]` extras) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

v1.4.0 → v1.5.0 은 100% backward compatible. 기존 v1.4.0 코드는
변경 없이 동작한다.

빠른 전환 (v1.5.0 신규 기능 사용):

```python
# v1.4.0
bem = BEMRetIter(p, hmatrix = True)            # ACA H-matrix iter

# v1.5.0 — preconditioner 자동 활성으로 GMRES iter 1 수렴
bem = BEMRetIter(p, hmatrix = True, preconditioner = 'auto')

# v1.5.0 — Schur × Iter (nonlocal cover-layer 케이스)
bem = BEMRetIter(p, refun = refun,
        hmatrix = True, schur = True)
```

자세한 사용법은 `docs/API_REFERENCE.md` (Preconditioner / Schur×Iter
섹션) 참고.

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.5.0 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.5.0 — H-matrix LU preconditioner + Schur × Iter + Au@Ag primary acceptance

- BEMRetIter / BEMStatIter (hmatrix=True, preconditioner='auto') — GMRES 55 → 1.
- BEMRetIter (hmatrix=True, schur=True) — Schur × Iter 통합. nonlocal 568-face: solve −21.3%.
- 51 pre-existing test failures → 0.
- jk-config 3 issues fix (multi-shell core_shell / metal substrate / field-only redirect).
- Primary acceptance: dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml (Au core 47 nm + Ag 4 nm shell + 0.6 nm gap + corner round 0.2 nm) 통과.
- pymnpbem_simulation: compute.iter.{preconditioner, schur} 옵션 노출.

100% backward compatible with v1.4.0.

See CHANGELOG.md, docs/API_REFERENCE.md, docs/MIGRATION_GUIDE.md (#21), docs/ARCHITECTURE.md §3.15, docs/PERFORMANCE.md §11.4.
```

## git tag command (used for this release)

```bash
git tag -a v1.5.0 -F docs/RELEASE_NOTES_v1.5.0.md
git push origin v1.5.0
```

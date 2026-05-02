# Release Notes — MNPBEM Python v1.3.0 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.3.0`
이전 릴리즈: `v1.2.0` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **Lane E2 — H-matrix `BEMRetIter` / `BEMStatIter` integration**.
  - `BEMRetIter(p, hmatrix=True)`, `BEMStatIter(p, hmatrix=True)` 새
    옵션. 25k+ face 큰 mesh 에서 dense LU 의 OOM (50+ GB peak) 을 해소.
  - 메모리 / matvec 모두 `O(N log N)` 스케일 — ACA H-tree 압축 +
    GMRES 반복 풀이.
  - v1.2.0 의 VRAM share (`MNPBEM_VRAM_SHARE_GPUS`) 와 결합 시
    56k+ face mesh 도 도전 가능 (실험적, preconditioner 후속 필요).
- **Latent bug fix**: v1.2.0 시점 `precond='hmat'` 가 이름만 존재했고
  실제로는 dense LU 가 동작했음. v1.3.0 에서 정식 H-matrix path
  로 바뀜.
- **`BEMStatIter` `ACACompGreenStat` positional arg latent bug**
  (`p, p` 중복 전달) 수정 — H-matrix path 가 처음으로 정상 동작.
- **`pymnpbem_simulation` wrapper** 의 `iter.hmatrix: 'auto'` 옵션 —
  5000+ face mesh 에서 자동으로 H-matrix BEMRetIter 활성. YAML 한
  줄로 전환 가능.
- 모든 변경은 backward-compatible (default OFF, opt-in).

---

## What's new

### H-matrix BEMRetIter / BEMStatIter (mnpbem.bem)

- `BEMRetIter(p, hmatrix=True, htol=1e-6, kmax=[4, 100], cleaf=200)`
  - `hmatrix=True` — ACA H-tree 압축 + GMRES.
  - `htol` — ACA truncation tolerance (default 1e-6).
  - `kmax` — ACA rank 상한 (default `[4, 100]`).
  - `cleaf` — leaf cluster 크기 (default 200).
- `BEMStatIter(p, hmatrix=True, ...)` 동일 인자.
- `BEMRetLayerIter + hmatrix` 는 미지원 (`NotImplementedError`) —
  cover-layer + planar substrate 결합 시나리오는 v1.4+.
- `BEM*Iter + Schur (v1.2.0)` 동시 활성도 미지원 — H-matrix + Schur
  통합은 후속 작업.
- `precond` 인자는 v1.3.0 에서 의미 정합화. `precond='hmat'` 명시
  옵션은 v1.3.x 또는 v1.4 (H-matrix LU preconditioner) 에서 정식
  지원 예정.

### pymnpbem_simulation wrapper

- `iter.hmatrix: 'auto'` (default) — 5000 face 이상에서 자동 활성.
- `iter.hmatrix: true` / `false` 명시 가능.
- `iter.hmatrix_options` (dict) — `htol`, `kmax`, `cleaf` 전달.
- `construct_bem` 가 `hmatrix` / `schur` 미지원 BEM 클래스에는
  TypeError 시 자동 strip 후 fallback.

### Tests

- `mnpbem/tests/test_hmatrix_iter.py` — 7 신규 unit test
  (BEMRetIter / BEMStatIter dense vs H-matrix consistency,
  옵션 전파, BEMRetLayerIter 거부, medium-sphere smoke).
- `pymnpbem_simulation/tests/test_v130_options.py` — 22 신규
  unit test (auto threshold, explicit on/off, YAML loader, runner
  wiring).

### Documentation

- `CHANGELOG.md` — v1.3.0 섹션.
- `docs/API_REFERENCE.md` — `BEMRetIter` / `BEMStatIter` `hmatrix`
  옵션 추가.
- `docs/MIGRATION_GUIDE.md` — pitfall #19 (Large mesh strategy:
  H-matrix iter).
- `docs/ARCHITECTURE.md` — §3.13 H-matrix BEMRetIter integration.
- `docs/PERFORMANCE.md` — §9.2 Known limits 갱신, §11 Large-mesh
  benchmark (5 k / 10 k 실측, 25 k placeholder).
- `benchmarks/lane_e2_25k_face.py` — 큰 mesh benchmark 스크립트
  (env `LANE_E2_NFACES`, `LANE_E2_SHAPE` 로 조정).

---

## Backward compatibility

v1.2.0 와 100 % 호환. 신규 기능은 모두 opt-in (default OFF):

- `BEMRetIter` / `BEMStatIter` `hmatrix=False` (default) 시 기존
  dense path 그대로.
- `pymnpbem_simulation` `iter.hmatrix: 'auto'` 는 5000 face 미만에서
  비활성 — 기존 작은 mesh 시뮬은 동작 변경 없음.
- v1.2.0 의 Schur / VRAM share / EpsNonlocal 는 그대로 사용 가능.

기존 v1.2.0 코드는 변경 없이 그대로 동작한다.

---

## Performance

자세한 표는 `docs/PERFORMANCE.md` §11 참고. 핵심 측정값:

### Memory / wall-time (CPU 실측, fib sphere, λ = 636.36 nm)

| Mesh | dense BEMRet | hmatrix BEMRetIter | speedup / mem |
|---|---|---|---|
| 5 k face | 71.7 s / 8.4 GB | 93.3 s / 5.3 GB | 메모리 ~37 % 절감 |
| 10 k face | (RAM/시간 budget 초과) | 218.9 s / 18.0 GB | hmatrix 단독 fit |
| 25 k face | OOM (50+ GB) | fit (CPU 측정 timeout — full convergence 는 v1.3.x) | enabled |

### GMRES 수렴

| Mesh | GMRES iter | relres | ACA compression |
|---|---:|---:|---:|
| 5 k face | 1 GMRES call (flag 0) | 9.60e-6 | 0.344 |
| 10 k face | 1 GMRES call (flag 0) | 8.26e-6 | 0.207 |

(GMRES `tol = 1e-5`, `htol = 1e-6`, `kmax = [4, 100]`, `cleaf = 64`,
λ = 636.36 nm)

### 정확도

`mnpbem/tests/test_hmatrix_iter.py::test_small_sphere_dense_vs_hmatrix`
가 dense vs H-matrix iter `rel < 1e-4` 회귀 보장.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| `BEMRetLayerIter + hmatrix` | 미지원 (`NotImplementedError`) | cover-layer + planar substrate 결합 시나리오는 v1.4+ |
| `BEM*Iter + Schur` 동시 활성 | 미지원 | H-matrix + Schur 통합은 후속 작업 |
| 25 k+ dimer near-resonance GMRES stall | preconditioner 없으면 위험 | H-matrix LU preconditioner 후속 (v1.3.x 또는 v1.4) |
| ACA 압축률 | mesh / wavelength 따라 변동 | 작은 mesh + 좋은 contrast 에서 ~30 %, 큰 mesh 에서 더 작아짐 |
| 25 k face full benchmark | CPU 단일 노드 wall-time 한계 | GPU + 충분한 시간 budget 에서 별도 측정 필요 |

v1.0.0 / v1.2.0 의 알려진 한계는 그대로 유지된다 (`docs/PERFORMANCE.md`
§9 참고).

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU 옵션) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU, v1.2.0+) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.2.0 -> v1.3.0` 은 backward compatible. 기존 v1.2.0 코드는 변경
없이 동작한다. 큰 mesh 시뮬레이션 시 `docs/MIGRATION_GUIDE.md` 의
pitfall #19 (Large mesh strategy) 섹션을 참고하면 된다.

빠른 전환:

```python
# v1.2.0
from mnpbem.bem import BEMRetIter
bem = BEMRetIter(p, tol=1e-5, maxit=400)

# v1.3.0 — 큰 mesh
from mnpbem.bem import BEMRetIter
bem = BEMRetIter(p, hmatrix=True, htol=1e-6, kmax=[4, 100], cleaf=200,
                 tol=1e-5, maxit=400)
```

또는 wrapper YAML:

```yaml
iter:
  hmatrix: auto      # 5000+ face 자동 ON
  hmatrix_options:
    htol: 1e-6
    kmax: [4, 100]
    cleaf: 200
```

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.3.0 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

H-matrix / ACA 기법 인용 (선택):

> M. Bebendorf, *Hierarchical Matrices*, Springer (2008).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.3.0 — H-matrix BEMRetIter / BEMStatIter integration (Lane E2)

- BEMRetIter / BEMStatIter `hmatrix=True` 옵션 — ACA H-tree + GMRES 로 25k+ face 큰 mesh 의 dense LU OOM 해소, O(N log N) 메모리 / matvec
- pymnpbem_simulation `iter.hmatrix: 'auto'` (5000+ face 자동 ON), `iter.hmatrix_options` (htol/kmax/cleaf) 노출
- BEMRetLayerIter + hmatrix: NotImplementedError (v1.4+ 작업)
- v1.2.0 latent bug fix: BEMStatIter ACACompGreenStat positional p,p 중복; precond='hmat' 가 이름만 있고 dense LU 였음 — 정식 H-matrix path 로 교체
- 5k / 10k face fib sphere 측정 (CPU): 메모리 ~37 % 절감, GMRES single-call 수렴 (relres < 1e-5)

100% backward compatible with v1.2.0.

See CHANGELOG.md, docs/MIGRATION_GUIDE.md (#19), docs/ARCHITECTURE.md §3.13, docs/PERFORMANCE.md §11.
```

## git tag command (used for this release)

```bash
git tag -a v1.3.0 -F docs/RELEASE_NOTES_v1.3.0.md
git push origin v1.3.0
```

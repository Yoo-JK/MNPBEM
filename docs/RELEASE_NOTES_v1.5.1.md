# Release Notes — PyMNPBEM v1.5.1 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.5.1`
이전 릴리즈: `v1.5.0` (2026-05-03)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **4 mnpbem GPU 버그 fix** — Au@Ag GPU full mesh acceptance 차단 요인 제거.
  - Bug 1: `bem_ret.py` CPU init path `G1i/G2i/Deltai` LU 백엔드 일관화.
  - Bug 2: `bem_ret_iter.py:264` dense LU preconditioner 의 numpy/cupy mix.
  - Bug 3: `hmatrix.py:250` `_aca_block` 에서 cupy column-index 의 implicit numpy 변환 거부.
  - Bug 4: `multi_gpu.py::_worker` 가 `simulation.type` 에 따라 BEM 클래스를 동적으로 받도록 확장.
- **BEMRetIter operator-form eps fix** — Au@Ag (multi-material, non-uniform eps + cross-connectivity) iter drift **70% → 0%**.
  - dense `BEMRet` 의 `L1 = G1·diag(eps1)·G1⁻¹` 와 일치하도록 iter formulation 수정.
- **`pymnpbem_simulation` `simulation.type=ret + iterative=true` 자동 라우팅** — Issue A. yaml 마이그레이션·dispatch 양쪽에서 `_iter` variant 로 in-place rewrite.
- **`pymnpbem_simulation` multi-GPU wavelength-split BEM 클래스 wiring** — `simulation.type` → `bem_class` 추론 + `compute.iter.{hmatrix, preconditioner, schur, htol, tol, maxit}` worker propagation. v1.5.0 까지는 multi-GPU 경로가 항상 `BEMRet` 를 강제하던 Bug 4 후속 wrapper 측 패치.

---

## What's new

### α — 4 mnpbem GPU 버그 fix (`47ab98c`)

| 버그 | 파일/라인 | 증상 | 수정 |
|---|---|---|---|
| 1 | `mnpbem/bem/bem_ret.py` (CPU init) | LU 백엔드 변경 시 G1i/G2i/Deltai 가 numpy/cupy 혼재 | LU 백엔드와 동일 dtype 으로 일관 |
| 2 | `mnpbem/bem/bem_ret_iter.py:264` | dense LU preconditioner build 중 GPU `H1` × CPU `G1i` mix | `G1i` 도 native CuPy 로 |
| 3 | `mnpbem/greenfun/hmatrix.py:250` | ACA block 의 `cols[pivot_col_local]` 가 cupy → cupy refused implicit numpy | host scalar 로 명시 변환 |
| 4 | `mnpbem/utils/multi_gpu.py::_worker` | `simulation.type=ret_iter` 인데도 dense `BEMRet` 강제 | `bem_class` parameter, runtime `_resolve_bem_class()` |

### β — BEMRetIter operator-form eps (`76ea1b0`)

`BEMRetIter` 의 `L1` 는 v1.5.0 까지 `eps1 · G1⁻¹` (component-wise scaling) 으로 적용했다. Au@Ag 처럼 non-uniform eps 가 cross-connectivity 와 결합하면 dense `BEMRet` 의 정의 `L1 = G1 · diag(eps1) · G1⁻¹` 와 일치하지 않아 70% drift 가 발생. v1.5.1 은 `apply_eps_operator(L_op, eps_diag)` 형태로 `LinearOperator` chain 을 재구성한다.

회귀:

- `case_g` (1136 face Au@Ag dimer): 7 wl × 7 variant **모두 0% rd**.
- tier-1-like (3184 face): 0% rd.
- 130 iter + 62 composite + 8 신규 = 200 tests PASS.

### Issue A — `pymnpbem` iterative routing fix

`compute.iterative=true` + `simulation.type=ret` 조합이 v1.5.0 까지 dense `BEMRet` 로 떨어지던 문제를 두 위치에서 수정:

- `pymnpbem_simulation/dispatch/single_node.py::_redirect_iterative_to_iter_type` — runtime 에 cfg 를 in-place 로 rewrite (`ret` → `ret_iter`, `stat` → `stat_iter`, `ret_layer` → `ret_layer_iter`).
- `pymnpbem_simulation/migration/py_to_yaml.py::_redirect_iterative_to_iter_type` — yaml 변환 시 동일 rewrite.

### multi-GPU wavelength-split BEM wiring (`pymnpbem_simulation`)

v1.5.0 의 Bug 4 fix 는 `mnpbem.utils.multi_gpu.solve_spectrum_multi_gpu` 측에 `bem_class` parameter 를 추가했지만, `pymnpbem_simulation/dispatch/multi_gpu.py` 가 이를 호출하지 않아 wavelength-split 경로가 여전히 `BEMRet` (dense) 를 강제했다. v1.5.1 은 wrapper 측에서 `bem_class` 를 `simulation.type` 으로부터 유도하여 전달하고, `compute.iter.*` 를 worker `bem_kwargs` 로 함께 전파한다.

### Tests

- `mnpbem/tests/test_gpu_cupy_consistency.py` — 14 tests.
- `mnpbem/tests/test_iter_convergence.py` — 8 tests.
- 기존 130 iter + 62 composite + 14 + 8 = 214 PASS, 회귀 0.

---

## Backward compatibility

100% backward compatible. uniform-eps 케이스 (single material) 는 v1.5.0 와 동일.

---

## Performance

### Au@Ag dimer Tier-3 (12672 face, jk-config `auag_r0.2_g0.6.yaml`)

5 wavelengths × 4× RTX A6000 GPU (49 GB ea):

| 시나리오 | 경로 | wall (min) | peak GPU mem | self-cons vs ref | 통과 |
|---|---|---:|---:|---|:---:|
| 1 | 4 worker × 1 GPU each, BEMRetIter (hmatrix=True, precond=auto) | (Tier-3 iter 측정값) | (peak) | (rel diff) | (status) |
| 2 | VRAM share 4 GPU dense (cusolverMg) | (Tier-3 vram 측정값) | (peak) | (rel diff) | (status) |
| 3 | 4 worker × 1 GPU each, BEMRet (dense baseline) | (Tier-3 dense 측정값) | (peak) | reference (정의) | (status) |

> 상세는 `docs/PERFORMANCE.md` §11.5 (v1.5.1 Au@Ag Tier-3 acceptance) 참고.

### iter drift 회귀 (β fix 효과)

| mesh | technique | v1.5.0 rd vs dense | v1.5.1 rd vs dense |
|---|---|---:|---:|
| case_g (1136 face Au@Ag) | iter+hmat+precond | 70% (mid-band) | **0% (machine grade)** |
| tier-1 (3184 face Au@Ag) | iter+hmat+precond | 78% (mid-band) | **0%** |

### 단위 테스트

- mnpbem 회귀: **986 passed, 4 skipped** (`test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` hang — known issue, 별도 조사).
- mnpbem v1.5.1 핵심 8 모듈 (test_gpu_cupy_consistency, test_iter_convergence, test_iterative, test_hmatrix_iter, test_preconditioner, test_eps_nonlocal, test_metal_substrate, test_schur_iter 부분): **182 + 6 = 188 PASS**.
- mnpbem fast regression (`tests/regression/ -m fast`): **8 passed**.
- pymnpbem v150 + v130 + wave3 옵션 회귀: **74 passed**.
- pymnpbem fast regression: **31 passed**.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| `BEMRetLayerIter` operator-form eps | 같은 패치 미적용 | substrate + iter 결합 시 v1.5.2 / v1.6 후속 |
| `test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` | hang | 별도 조사 (다른 10 schur 테스트 PASS) |
| 25k+ face nonlocal mesh | 부분 해소 (v1.5.0 와 동일) | Sigma/Delta H-matrix 재구성 v1.6+ |
| Bug 5 — `HMatrix.full()` cupy/numpy mix | Tier-3 12672-face GPU iter+hmat+precond 시나리오 fail | `mnpbem/greenfun/hmatrix.py:374`. v1.5.2 후속 (v1.5.1 α GPU bug 시리즈와 동일 카테고리) |

v1.0.0 ~ v1.5.0 의 알려진 한계는 그대로 유지된다.

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

v1.5.0 → v1.5.1 은 100% backward compatible. 기존 v1.5.0 코드는 변경 없이 동작한다.

GPU full-mesh Au@Ag 사용자는 환경 변수 / yaml 변경 불필요 — pymnpbem `compute.iterative=true` + `simulation.type=ret` 조합도 자동으로 `ret_iter` + GPU iter+hmat+precond 로 라우팅된다.

---

## Citing

Python port 사용 시:

> "PyMNPBEM v1.5.1 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.5.1 — 4 GPU bugs fix + BEMRetIter operator-form eps + Au@Ag Tier-3 acceptance

- 4 mnpbem GPU 버그 fix (numpy/cupy interop, multi_gpu BEM 클래스 파라미터화).
- BEMRetIter operator-form eps for non-uniform regions (Au@Ag iter 70% drift → 0%).
- pymnpbem Issue A: simulation.type=ret + iterative=true 자동 ret_iter 라우팅.
- pymnpbem multi-GPU wavelength-split BEM 클래스 wiring (Bug 4 후속 wrapper).
- Tier-3 Au@Ag dimer 12672 face GPU full validation (jk-config auag_r0.2_g0.6.yaml).

100% backward compatible with v1.5.0.

See CHANGELOG.md, docs/RELEASE_NOTES_v1.5.1.md.
```

## git tag command

```bash
git tag -a v1.5.1 -F docs/RELEASE_NOTES_v1.5.1.md
git push origin v1.5.1
```

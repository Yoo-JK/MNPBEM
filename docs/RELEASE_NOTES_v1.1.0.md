# Release Notes — MNPBEM Python v1.1.0 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.1.0`
이전 릴리즈: `v1.0.0` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **`EpsNonlocal` 클래스** — Hydrodynamic Drude nonlocal dielectric
  function 의 cover-layer 형식 포팅. Yu Luo et al., *Phys. Rev. Lett.*
  **111**, 093901 (2013) 의 effective-layer mapping 을 따른다.
  nano-gap (< 1 nm) / sub-5 nm particle 시뮬레이션을 위한 표준 진입점.
- **`BEMRet` `refun` 인자** — retarded 경로도 cover-layer integration 과
  결합 가능 (`BEMStat` 은 v1.0.0 부터 이미 지원). `BEMRetIter` 동시 반영.
- **`pymnpbem_simulation` wrapper 갱신** — 이전의 M7 우회 코드 제거,
  정식 `EpsNonlocal` 호출 경로 사용.
- **검증** — Au dimer 1 nm gap blueshift 재현: peak λ 520 → 490 nm
  (+30 nm shift, Ciraci 2012 / Luo 2013 일치).

---

## What's new

### `EpsNonlocal` (mnpbem.materials)

- `EpsNonlocal(metal, eps_embed, delta_d, beta=...)` 생성자.
- Factory: `EpsNonlocal.gold()`, `.silver()`, `.aluminum()`,
  `.from_table(path)`.
- Helper: `make_nonlocal_pair(metal, eps_embed, delta_d, beta) -> (core, shell)` —
  cover-layer 와 함께 사용할 core/shell `EpsConst`-호환 객체 한 쌍.
- 호출 시 `(eps_complex, k_longitudinal)` 튜플 반환 (스칼라 / array 모두 지원).
- 18 unit tests; MATLAB `demospecstat19` reference formula 와
  `rtol = 1e-12` 수준에서 bit-identical.

### `BEMRet` `refun` 인자

- `BEMRet(p, refun=...)` — `BEMStat` 와 동일한 시그니처.
- `coverlayer.refine` 등 user-defined refinement function 을 retarded
  path 에서도 동일하게 적용한다.
- `BEMRetIter` 도 같은 방식으로 갱신 (iterative + cover-layer 시나리오).
- 7 unit tests (시그니처 호환성, no-op 일치, twice-call, matrix 차이,
  cover-layer smoke, nano-dimer with nonlocal, signature).

### Documentation

- `docs/API_REFERENCE.md` — `EpsNonlocal` 섹션 추가.
- `docs/MIGRATION_GUIDE.md` — "Nonlocal hydrodynamic Drude" 섹션 추가
  (MATLAB demospecstat19 → Python EpsNonlocal 매핑).
- `CHANGELOG.md` — v1.1.0 섹션.

---

## Backward compatibility

v1.0.0 와 100 % 호환. 추가만 있고, 기존 동작/시그니처/디폴트 변경 없음.
v1.0.0 코드는 수정 없이 그대로 동작한다.

---

## Performance

`EpsNonlocal` 자체는 algorithmic feature (성능에 영향 없음).
다만 cover-layer refinement 적용 시 mesh face count 가 약 2× 늘어나며,
대응 BEM 행렬 메모리는 약 4× 가 된다. 향후 v1.2.0 에서 도입 예정인
Schur complement reduction 으로 약 50 % 절감을 목표로 한다.

회귀 fast suite (`tests/regression/ -m fast`) 와 v1.0.0 의 모든
기존 unit / 회귀 측정값에는 변화가 없다.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| `BEMRetLayer` / `BEMRetLayerIter` + cover-layer | 미지원 | planar substrate + nonlocal cover-layer 조합 시나리오는 현재 없음. v1.2.x 후속 작업 |
| ACA H-matrix + `EpsNonlocal` 결합 | 미검증 | 다음 perf milestone (v1.1.x / v1.2.0) 에서 통합 예정 |
| Multi-GPU VRAM 합산 | 미구현 | M6+ (cuSolverMg / Magma / NCCL) |

v1.0.0 의 알려진 한계 (sphere/rod 8 xfail, dimer 9.1e-8 등) 는 그대로
유지된다 (`docs/PERFORMANCE.md` §9 참고).

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU 옵션) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

`v1.0.0 -> v1.1.0` 은 backward compatible. 기존 v1.0.0 코드는 변경 없이
동작한다.

새 nonlocal 시뮬레이션을 작성할 경우 `docs/MIGRATION_GUIDE.md` 의
"Nonlocal hydrodynamic Drude" 섹션을 참고하면 된다 (MATLAB
`demospecstat19` 패턴 → Python `EpsNonlocal` + `coverlayer.refine`).

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.1.0 (2026), based on Hohenester & Trügler MNPBEM 17."

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
v1.1.0 — EpsNonlocal hydrodynamic Drude nonlocal cover-layer

- EpsNonlocal class (Yu Luo PRL 111 093901 effective-layer mapping)
- BEMRet / BEMRetIter refun parameter (parity with BEMStat)
- pymnpbem_simulation wrapper switched from M7 workaround to official path
- Au dimer 1 nm gap +30 nm blueshift verified (Ciraci 2012 / Luo 2013)

100% backward compatible with v1.0.0.

See CHANGELOG.md and docs/MIGRATION_GUIDE.md.
```

## git tag command (used for this release)

```bash
git tag -a v1.1.0 -F docs/RELEASE_NOTES_v1.1.0.md
git push origin v1.1.0
```

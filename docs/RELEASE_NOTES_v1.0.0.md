# Release Notes — MNPBEM Python v1.0.0 (internal)

릴리즈 일자: 2026-05-XX
릴리즈 태그: `v1.0.0`
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

MATLAB MNPBEM 17 의 순수 Python 포팅 첫 production 릴리즈.

| 영역 | 결과 |
|---|---|
| 72 demo 정확도 | machine precision 59 / 72 (82 %), BAD 0 / 72 |
| 72 demo 속도 | CPU geo-mean 2.21×, GPU geo-mean 3.60× (vs MATLAB) |
| sphere / rod / rod_lying 51 case | machine precision 35 / 51 (69 %), 알려진 한계 8 (xfail) |
| dimer 6336 face × 100 wl | GPU 4× = 13 min (MATLAB best CPU 대비 11.6×) |
| dimer ext_x rel-diff (single λ) | 9.1e-8 (machine precision 등급) |

---

## What's new (M1 - M5)

### Milestone 1 — Demo complete

- 50 + 22 = 72 공식 MATLAB demo 정확도 달성 (BAD 12 → 0).
- BEMStat / BEMRet / BEMRetMirror / BEMStatLayer / BEMRetLayer / iterative / eigenmode 솔버.
- 평면파 / dipole / EELS 여기 (모든 stat/ret/mirror/layer 조합).
- 메시 생성기: `trisphere`, `trirod`, `tricube`, `tritorus`, `trispheresegment`, `tripolygon`, `Polygon3`, `EdgeProfile`.
- 2D 메쉬: MATLAB `mesh2d` line-by-line 포팅.
- Mie 참조 솔버: `MieStat`, `MieRet`, `MieGans`.

### Milestone 2 — Public API

- 외부 사용자용 정리된 import surface (`from mnpbem import ...`).
- `mnpbem.simulation` 패키지 (외부 사용자용 high-level API, 별도 repo).

### Milestone 3 — Edge cases

- Layered Sommerfeld convergence + tabulated interpolation.
- Mirror symmetry across all solver families.
- 비-등방성 dielectric 함수 (`EpsTable`, `EpsDrude`).

### Milestone 4 — Performance

- Numba JIT (default ON, 9 모듈, `MNPBEM_NUMBA=0` 으로 OFF):
  - greenfun refinement, mie coefficients, geometry curved, BEM matrix init,
    Sommerfeld ODE, compgreen_ret, retfac, plane wave, polygon3.
- GPU 가속 (cupy, `MNPBEM_GPU=1`):
  - Lane A: GreenRetRefined cupy.
  - Lane A2: BEMRet matrix assembly cupy-eager.
  - Lane B: PlaneWaveRet / SpectrumRet / EpsTable.
  - Lane C: Sommerfeld / Layer.
  - Lane D: Multi-GPU wavelength batch dispatch.
  - Lane E: H-matrix GPU prototype.
- Phase 3 native (`MNPBEM_GPU_NATIVE=1`):
  - PCIe round-trip 제거 (cupy → cupy 직행).
  - `Sigma1 = lu_solve(G^T, H^T, trans=1).T` 직행으로 잉여 inverse 제거.
  - dimer GPU 4× wall: 49.83 → 13.00 min.
- ACA H-matrix solver (`htol=1e-6, kmax=200, cleaf=200, ACATOL=1e-10`).
- Multi-node MPI wavelength dispatch (Lane D 확장).

### Milestone 5 — Final validation

- Acceptance criteria 확정 (`docs/ACCEPTANCE_CRITERIA.md`).
- 종합 회귀 스위트 (`tests/regression/` — 72 demo / sphere-rod / dimer / edge cases).
- BEM 1.6 % drift 추적 종결: 9.1e-8 (machine precision) 자연 해소
  (Lane A-E 보고서: `/tmp/bem_drift_lane_AE_report.md`).
- 종합 성능 + 정확도 보고서 (`docs/PERFORMANCE.md`).
- v1.0.0 release prep (`pyproject.toml`, `LICENSE`, build dry-run).

---

## Known limitations

### 정확도 (회귀에서 xfail 처리)

| 항목 | 영향 case | 한계 | 비고 |
|---|---|---|---|
| Layered eigenmode 반올림 | sphere/rod 07_eigenmode (3 case) | rel ~1e-2 | MATLAB과 동일 수렴 영역 |
| Layered Sommerfeld | sphere/rod_lying 04/05/03 (5 case) | rel ~1e-2 | scipy `solve_ivp` 본질 한계 |
| BEM Green G1 4 entries | dimer ext_x | 9.1e-8 | Particle.quad 노드 순서 차이 |
| `demospecstat17` | static layered eigen | 1.58e-2 | xfail |
| `demospecret13` | layered Sommerfeld | 5.00e-4 | warn (회귀 통과) |

상세: `docs/PERFORMANCE.md` §9.

### 메모리 / 성능 (M5+ 후속)

| 항목 | 한계 | 대안 |
|---|---|---|
| 25 k+ face 단일 GPU | 48 GB VRAM OOM | BEMRetIter + H-matrix 통합 (M5+) |
| Multi-GPU VRAM 합산 | 미구현 | cuSolverMg / Magma / NCCL (M5+) |

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 (matrix CI) |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy 13.x (GPU 옵션) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Breaking changes vs 0.1.0

`v0.1.0` 은 internal pre-release. v1.0.0 에서 다음 정리:

- `__version__ = "1.0.0"` (이전 0.1.0).
- `mnpbem/requirements.txt` deprecated → `pyproject.toml` 의 `[project.dependencies]` 사용.
- `setup.py` shim 만 유지, 모든 메타데이터는 `pyproject.toml`.
- License 명시: GPL-2.0-or-later (MATLAB MNPBEM 상속).

API surface 자체는 v0.x → v1.0 변경 없음 (외부 사용자 zero-impact).

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.0.0 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.0.0 — MNPBEM Python port first production release

- 72 demo machine precision 59/72, BAD 0/72
- CPU geo-mean speedup 2.21x, GPU geo-mean 3.60x vs MATLAB
- dimer GPU 4x = 13 min (MATLAB best CPU 11.6x speedup)
- dimer ext_x rel-diff 9.1e-8 (machine precision)

See docs/PERFORMANCE.md and CHANGELOG.md.
```

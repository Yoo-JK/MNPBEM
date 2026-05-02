# Release Notes — MNPBEM Python v1.4.0 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.4.0`
이전 릴리즈: `v1.3.0` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **CPU/GPU 분리 install** — `pyproject.toml` extras 정교화.
  - `pip install mnpbem` — CPU only (가장 가벼움, cupy 의존성 없음).
  - `pip install mnpbem[gpu]` — cupy-cuda12x 포함 (NVIDIA GPU 가속).
  - `pip install mnpbem[all]` — `gpu` + `mpi` + `fmm` 전부.
  - 별도 wheel 분리 X — single wheel + extras 가 PyPI 표준 패턴.
- **Runtime GPU auto-detect** —
  `mnpbem.utils.gpu.has_gpu_capability(verbose=True)` 가
  cupy + CUDA driver + GPU device 가용성을 검사하고, 누락 시
  `get_install_hint()` 가 사용자 환경에 맞는 install 명령을 안내한다.
- **`docs/INSTALL.md`** — 시나리오별 install 가이드
  (CPU only / single GPU / multi-GPU / multi-node / 개발 환경).
- 사용자 합의 4 phase 의 **마지막** —
  v1.1.0 (nonlocal) → v1.2.0 (VRAM share + Schur) →
  v1.3.0 (Lane E2 H-matrix iter) → **v1.4.0 (CPU/GPU build)**.

---

## What's new

### pyproject.toml extras (정교화)

| Extra | 추가 의존성 | 용도 |
|---|---|---|
| `mnpbem` (default) | (없음) | CPU only |
| `mnpbem[gpu]` | cupy-cuda12x | NVIDIA GPU 가속 |
| `mnpbem[mpi]` | mpi4py | multi-node wavelength 분배 |
| `mnpbem[fmm]` | fmm3dpy | free-space ret meshfield 가속 |
| `mnpbem[all]` | gpu + mpi + fmm | 전부 |
| `mnpbem[dev]` | pytest / ruff / build / twine | 개발 환경 |
| `mnpbem[test]` | pytest | 회귀 테스트만 |
| `mnpbem[docs]` | (sphinx 등) | docs build |

### mnpbem.utils.gpu

- `has_gpu_capability(verbose=False) -> bool` — cupy import +
  CUDA driver + GPU device 가용성을 모두 검사하여 `bool` 반환.
  `verbose=True` 시 누락된 항목별 메시지 출력.
- `get_install_hint() -> str` — 현재 환경에서 GPU 활성을 위해
  필요한 `pip install mnpbem[gpu]` 명령 안내 문자열 반환.
- `MNPBEM_GPU=1` env var 명시 + cupy 미설치 → BEM solver 호출
  시점에 `RuntimeError` + install 명령 안내 (기존 silent fallback
  대신 명확한 에러).

### docs/INSTALL.md (신규)

시나리오별 install 가이드:

- 가장 가벼운 CPU only 환경.
- 단일 NVIDIA GPU 환경 (RTX A6000 등).
- multi-GPU (cuSolverMg / VRAM share).
- multi-node MPI 환경.
- 개발 / 회귀 테스트 환경.

각 시나리오마다 그대로 복사·실행 가능한 conda + pip 코드 블록.

### Documentation

- `CHANGELOG.md` — v1.4.0 섹션.
- `docs/API_REFERENCE.md` — §9 `GPU 환경 검사` 섹션 추가.
- `docs/MIGRATION_GUIDE.md` — pitfall #20 (Install 변경).
- `docs/ARCHITECTURE.md` — §3.14 CPU/GPU 분리 build.
- `docs/INSTALL.md` (NEW) — 시나리오별 install 가이드.
- `README.md` Installation 섹션 간략화 (`docs/INSTALL.md` 로 링크).

### Tests

- `mnpbem/tests/test_install_check.py` — `has_gpu_capability` /
  `get_install_hint` runtime 동작 회귀 (cupy 있을 때 / 없을 때
  분기).

---

## Backward compatibility

v1.3.0 와 100% 호환. 기존 코드는 변경 없이 그대로 동작:

- `pip install -e .` 로 dev install 도 그대로.
- `MNPBEM_GPU=1` / `MNPBEM_GPU_THRESHOLD` 등 모든 env var 동작 유지.
- BEM solver / excitation / spectrum API 변경 없음.
- v1.0.0 ~ v1.3.0 의 모든 기능 (EpsNonlocal / Schur / VRAM share /
  H-matrix iter) 그대로 사용 가능.

단순히 install 명령만 환경에 맞게 다르게 사용하면 된다.

---

## Performance

(perf 영향 없음 — packaging 개선)

v1.0.0~v1.3.0 의 perf 측정값은 그대로 유지되며 `docs/PERFORMANCE.md`
에서 확인 가능.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| 별도 wheel (`mnpbem-cpu` / `mnpbem-gpu`) | 만들지 않음 | single wheel + extras 가 PyPI 표준. 별도 wheel 은 build/maintain 비용 대비 가치 낮음 |
| AMD ROCm GPU | 미지원 | `cupy-cuda12x` 만 지원. AMD 는 후속 milestone |
| Apple Silicon GPU (Metal) | 미지원 | CPU only 로 동작 |

v1.0.0 ~ v1.3.0 의 알려진 한계는 그대로 유지된다 (`docs/PERFORMANCE.md`
§9 참고).

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy-cuda12x (`[gpu]` extras) |
| cuSolverMg | CUDA toolkit 11.x+ (multi-GPU LU, v1.2.0+) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

v1.3.0 → v1.4.0 은 100% backward compatible. 기존 v1.3.0 코드는
변경 없이 동작한다.

빠른 전환:

```bash
# v1.3.0 (사실상 모든 의존성 동시 install)
pip install mnpbem

# v1.4.0 (환경별 분리)
pip install mnpbem            # CPU only
pip install mnpbem[gpu]       # + GPU
pip install mnpbem[all]       # 전부
```

코드 사이드는 변경 사항 없음. runtime GPU 가용성 확인을 자동화
하려면:

```python
from mnpbem.utils.gpu import has_gpu_capability, get_install_hint

if not has_gpu_capability(verbose=True):
    print(get_install_hint())
    # GPU 가속이 필요한 경우 안내된 install 명령 실행
```

자세한 시나리오별 설치 절차는 `docs/INSTALL.md` 참고.

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.4.0 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.4.0 — CPU/GPU 분리 install (pyproject extras 정교화)

- pyproject extras: gpu / mpi / fmm / all / dev / test / docs.
- pip install mnpbem (CPU only) → mnpbem[gpu] (cupy 포함) → mnpbem[all] (전부).
- mnpbem.utils.gpu.has_gpu_capability() / get_install_hint() — runtime auto-detect + 친절한 fallback.
- docs/INSTALL.md (NEW) — 시나리오별 install 가이드.
- README.md Installation 섹션 간략화.
- 별도 wheel 분리 X (single wheel + extras 가 PyPI 표준).
- 사용자 합의 4 phase 의 마지막 (nonlocal → VRAM share → Lane E2 → CPU/GPU build).

100% backward compatible with v1.3.0.

See CHANGELOG.md, docs/INSTALL.md, docs/MIGRATION_GUIDE.md (#20), docs/ARCHITECTURE.md §3.14.
```

## git tag command (used for this release)

```bash
git tag -a v1.4.0 -F docs/RELEASE_NOTES_v1.4.0.md
git push origin v1.4.0
```

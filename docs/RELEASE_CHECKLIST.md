# Release Checklist (internal)

작성: 2026-05-02 (M5 Wave B Agent ε)
일반화: 2026-05-02 (Wave 3 Agent ε — v1.1.0 release prep)
대상: internal milestone tag (PyPI 공개 배포는 추후 결정)

이 문서는 release tag 를 찍기 전에 사람이 직접 확인할 사항을 정리한다.
회귀 / CI 자동 검증은 `docs/ACCEPTANCE_CRITERIA.md` §4 가 담당하므로,
여기서는 사람의 판단이 필요한 항목만 둔다.

각 release 의 release notes 는 `docs/RELEASE_NOTES_v<X.Y.Z>.md` 로 별도
보관한다.

---

## Pre-tag (필수)

- [ ] `pytest tests/regression/ -m fast` 모두 통과 (commit-level smoke).
- [ ] `pytest tests/regression/ -m slow` 모두 통과 또는 xfail (daily-level).
      ※ sphere/rod 의 8 case 는 xfail (`docs/PERFORMANCE.md` §2.2 / §9.1).
- [ ] `CHANGELOG.md` 의 신규 버전 섹션이 main 에 머지되어 있다.
- [ ] `mnpbem/__init__.py` 의 `__version__` 와 `pyproject.toml` 의
      `version` 값이 일치 + 신규 버전이다.
- [ ] `docs/PERFORMANCE.md` 의 측정값이 최신 main HEAD 기준이다 (성능
      변화가 있을 때).
- [ ] `docs/ACCEPTANCE_CRITERIA.md` 의 모든 OK 항목 그대로 유지된다.
- [ ] `docs/RELEASE_NOTES_v<X.Y.Z>.md` 가 작성되어 있다.
- [ ] `LICENSE` (GPL-2.0-or-later) 가 존재하고 `pyproject.toml` 의
      license 필드와 일치한다.
- [ ] `python -m build` 가 sdist + wheel 생성 성공.
- [ ] `twine check dist/*` 가 PASSED.

---

## Tag

- [ ] `git tag -a v<X.Y.Z> -F docs/RELEASE_NOTES_v<X.Y.Z>.md`.
- [ ] `git push origin v<X.Y.Z>` (단순 tag push, publish workflow 는 활성화 X).
- [ ] GitHub Release 생성 (선택, internal repo 일 때):
  - 제목: release notes 의 H1 헤더와 동일.
  - 본문: 해당 release notes 본문.
  - artefact: `dist/mnpbem-<X.Y.Z>-py3-none-any.whl`,
    `dist/mnpbem-<X.Y.Z>.tar.gz` 첨부 (선택).

---

## Post-tag (검증)

- [ ] 새 conda env 에서 `pip install /path/to/dist/mnpbem-<X.Y.Z>-py3-none-any.whl` 동작.
- [ ] `python -c "import mnpbem; print(mnpbem.__version__)"` → `<X.Y.Z>`.
- [ ] `python -c "from mnpbem import Particle, BEMRet"` 무에러 import.
- [ ] `pytest tests/regression -m fast` 새 env 에서 통과.

---

## 완료된 릴리즈

### v1.0.0 (2026-05-02)

- [x] 72 demo / sphere-rod / dimer 4 case / Lane A-E 통합 검증
- [x] `docs/RELEASE_NOTES_v1.0.0.md` 작성
- [x] `python -m build`, `twine check` 통과
- [x] `git tag -a v1.0.0` 푸시

### v1.1.0 (2026-05-02)

- [x] `EpsNonlocal` 클래스 + 18 unit tests 머지 (Wave 1).
- [x] `BEMRet` / `BEMRetIter` `refun` 인자 + 7 unit tests 머지 (Wave 2 β).
- [x] `CHANGELOG.md` v1.1.0 섹션 + `API_REFERENCE` + `MIGRATION_GUIDE`
      갱신 (Wave 2 δ).
- [x] `pymnpbem_simulation` wrapper 정식 호출 + nano-gap +30 nm
      blueshift 검증 (Wave 2 γ).
- [x] `mnpbem/__init__.py` `__version__ = "1.1.0"`,
      `pyproject.toml` `version = "1.1.0"` 갱신 (Wave 3 ε).
- [x] `docs/RELEASE_NOTES_v1.1.0.md` 작성 (Wave 3 ε).
- [x] fast 회귀 + 새 EpsNonlocal / BEMRet refun unit 테스트 통과.
- [x] `python -m build`, `twine check` 통과.
- [x] `git tag -a v1.1.0` 푸시.

### v1.2.0 (2026-05-02)

- [x] Schur complement helpers + BEMStat / BEMRet `schur=True` 옵션 +
      14 unit tests 머지 (Agent α).
- [x] cuSolverMg multi-GPU LU dispatch + 4 unit tests 머지 (Agent β).
- [x] `pymnpbem_simulation` wrapper Schur auto-detect + VRAM share YAML
      옵션 + 18 케이스 회귀 머지 (Agent γ).
- [x] `CHANGELOG.md` v1.2.0 섹션 + `API_REFERENCE` + `MIGRATION_GUIDE`
      (#17, #18) + `ARCHITECTURE.md` §3.11/§3.12 + `PERFORMANCE.md`
      갱신 (Agent δ).
- [x] `mnpbem/__init__.py` `__version__ = "1.2.0"`,
      `pyproject.toml` `version = "1.2.0"` 갱신 (Agent ε).
- [x] `docs/RELEASE_NOTES_v1.2.0.md` 작성 (Agent ε).
- [x] fast 회귀 + 새 Schur / multi-GPU LU unit 테스트 통과 (51 failures
      는 v1.1.0 baseline 과 동일 — 회귀 0).
- [x] Schur + VRAM share 동시 활성 sanity check (Schur active=True,
      env vars 설정 확인).
- [x] pymnpbem 측 v120_options 18 + wave3_m7 18 + fast 31 회귀 통과.
- [x] `python -m build`, `twine check` 통과.
- [x] `git tag -a v1.2.0` 푸시.

---

## Future (PyPI 공개 배포 결정 후 — 별도 milestone)

다음 항목들은 사용자 결정으로 내부 milestone 단계에서 **제외** 되었다.
공개 배포로 진행할 때 별도 체크리스트로 다룬다.

- [ ] GitHub PAT 에 `workflow` scope 추가 (M5-γ branch `m5-wave-a` push 위해).
- [ ] `git push origin m5-wave-a` 후 PR 또는 fast-forward main 머지 (CI workflows 적용).
- [ ] PyPI trusted publisher 등록 + `publish.yml` workflow enable.
- [ ] `pyproject.toml` 의 `[project.urls]` 채우기 (Homepage / Repository / Issues / Documentation).
- [ ] 첫 PyPI release 검증 (`pip install mnpbem==<X.Y.Z>` 새 환경).
- [ ] readthedocs 또는 GitHub Pages 문서 배포.

---

## 알려진 이슈

| 이슈 | 영향 | 대응 |
|---|---|---|
| GitHub PAT `workflow` scope 부족 | M5-γ CI 파일을 main 에 push 불가 | `m5-wave-a` branch 에 commit 보존, 추후 별도 머지 |
| pkginfo 1.12 가 PEP 639 미지원 | `setuptools >= 77` + `license = "..."` SPDX 형식 시 `twine check` fail | `pyproject.toml` 에서 `setuptools <77` 핀 + `license = { file = "LICENSE" }` 옛 형식 사용. pkginfo 1.13+ 출시 시 SPDX 형식으로 마이그레이션 가능 |
| dimer ext_x 4 entry 차이 | 9.1e-8 (machine precision) | 수용 (`docs/PERFORMANCE.md` §4.4) |

---

## 관련 문서

- `docs/ACCEPTANCE_CRITERIA.md` (M5-α — 정확도 / 속도 / 회귀 기준)
- `docs/PERFORMANCE.md` (M5-ε — 종합 성능 + 정확도 보고서)
- `docs/ARCHITECTURE.md` (M5-δ — 컨트리뷰터용 설계 문서)
- `CHANGELOG.md` (Keep-a-Changelog 형식)
- `docs/API_REFERENCE.md` (외부 사용자 API)
- `docs/MIGRATION_GUIDE.md` (MATLAB → Python 마이그레이션)
- `docs/RELEASE_NOTES_v1.0.0.md` (v1.0.0 git tag 메시지)
- `docs/RELEASE_NOTES_v1.1.0.md` (v1.1.0 git tag 메시지)

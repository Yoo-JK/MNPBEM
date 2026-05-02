# Release Checklist v1.0.0 (internal)

작성: 2026-05-02 (M5 Wave B Agent ε)
대상: internal milestone tag `v1.0.0` (PyPI 공개 배포는 추후 결정)

이 문서는 v1.0.0 tag 를 찍기 전에 사람이 직접 확인할 사항을 정리한다.
회귀 / CI 자동 검증은 `docs/ACCEPTANCE_CRITERIA.md` §4 가 담당하므로,
여기서는 사람의 판단이 필요한 항목만 둔다.

---

## Pre-tag (필수)

- [ ] `pytest tests/regression/ -m fast` 모두 통과 (commit-level smoke).
- [ ] `pytest tests/regression/ -m slow` 모두 통과 또는 xfail (daily-level).
      ※ sphere/rod 의 8 case 는 xfail (`docs/PERFORMANCE.md` §2.2 / §9.1).
- [ ] `CHANGELOG.md` v1.0.0 섹션이 main 에 머지되어 있다 (M5-δ).
- [ ] `README.md` 의 버전 정보가 `mnpbem/__init__.py` `__version__ = "1.0.0"` 와 일치 (M5-β).
- [ ] `docs/PERFORMANCE.md` 의 측정값이 최신 main HEAD 기준이다.
- [ ] `docs/ACCEPTANCE_CRITERIA.md` 의 모든 OK 항목 그대로 유지된다.
- [ ] `LICENSE` (GPL-2.0-or-later) 가 존재하고 `pyproject.toml` 의 license 필드와 일치한다.
- [ ] `python -m build` 가 sdist + wheel 생성 성공.
- [ ] `twine check dist/*` 가 PASSED.

---

## Tag

- [ ] `git tag -a v1.0.0 -F docs/RELEASE_NOTES_v1.0.0.md` (또는 -m 으로 인라인).
- [ ] `git push origin v1.0.0` (단순 tag push, publish workflow 는 활성화 X).
- [ ] GitHub Release 생성 (선택, internal repo 일 때):
  - 제목: `v1.0.0 — MNPBEM Python port first production release`
  - 본문: `docs/RELEASE_NOTES_v1.0.0.md` 복사.
  - artefact: `dist/mnpbem-1.0.0-py3-none-any.whl`, `dist/mnpbem-1.0.0.tar.gz` 첨부 (선택).

---

## Post-tag (검증)

- [ ] 새 conda env 에서 `pip install /path/to/dist/mnpbem-1.0.0-py3-none-any.whl` 동작.
- [ ] `python -c "import mnpbem; print(mnpbem.__version__)"` → `1.0.0`.
- [ ] `python -c "from mnpbem import Particle, BEMRet"` 무에러 import.
- [ ] `pytest tests/regression -m fast` 새 env 에서 통과.

---

## Future (PyPI 공개 배포 결정 후 — 별도 milestone)

다음 항목들은 사용자 결정으로 내부 milestone 단계에서 **제외** 되었다.
공개 배포로 진행할 때 별도 체크리스트로 다룬다.

- [ ] GitHub PAT 에 `workflow` scope 추가 (M5-γ branch `m5-wave-a` push 위해).
- [ ] `git push origin m5-wave-a` 후 PR 또는 fast-forward main 머지 (CI workflows 적용).
- [ ] PyPI trusted publisher 등록 + `publish.yml` workflow enable.
- [ ] `pyproject.toml` 의 `[project.urls]` 채우기 (Homepage / Repository / Issues / Documentation).
- [ ] 첫 PyPI release 검증 (`pip install mnpbem==1.0.0` 새 환경).
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
- `CHANGELOG.md` (M5-δ — Keep-a-Changelog 형식)
- `docs/API_REFERENCE.md` (M5-β — 외부 사용자 API)
- `docs/MIGRATION_GUIDE.md` (M5-β — MATLAB → Python 마이그레이션)
- `docs/RELEASE_NOTES_v1.0.0.md` (M5-ε — 본 릴리즈의 git tag 메시지)

# 06_mirror / rod — Skipped

`trirod` geometry는 `trispheresegment`처럼 대칭 영역(1/4)만 생성하는 빌더가 MNPBEM에 자연스럽게 제공되지 않는다. `comparticlemirror` + `sym = 'xy'`로 rod mirror 검증을 하려면 trirod mesh를 1/4로 자르는 추가 작업이 필요해 이 검증 케이스는 sphere에만 적용한다.

rod에 대해서도 mirror symmetry 검증을 원하면 다음 중 하나로 확장 가능:
- `trirod` 구현 변형으로 quarter-only mesh 생성 헬퍼 작성
- MATLAB demo `demospecret16.m`(nanodisk + mirror) 스타일로 `tripolygon` 기반 대체 geometry 사용

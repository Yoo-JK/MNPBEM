# Mesh2d FP Precision Limit

MATLAB → Python 포팅 과정에서 mesh 생성이 bit-identical이 **구조적으로 불가능**함을 설명한다.

## 증상 (Wave 5 δ 시점)

- `demoeelsret3` (Ag nanodisk, 단일 disk, face metric): 7.1e-03 (OK)
- `demoeelsret7` (Ag nanotriangle on membrane, plate-with-hole): 2.2e-01 (warn)
- `demoeelsret8` (same + 2D spatial map): 4.8e-01 (warn)

demoeelsret7/8의 corner 위치 peak는 MATLAB 대비 0.3% 이내로 정확.
Middle·edge 위치 peak는 5–15% 차이 — 이는 mesh 한 점 ≲0.1 nm 차이가
공진 peak에 frequency shift (수십 meV)를 유발한 결과.

## Root Cause

1. **Mesh shape 자체는 거의 동일**
   - MATLAB plate `up`: 322 verts, 536 faces
   - Python `Polygon3.plate_from_list` 결과: 325 verts, 542 faces
   - 양측 vertices KDTree 매칭 median Δ = 0.07 nm (plate 150 nm 대비 5e-4)

2. **차이의 기원은 Mesh2d 반복 수렴**
   - `quadtree` → `boundarynodes`(spring smoothing, maxit=50) →
     `meshpoly`(CVT-style Laplacian, maxit=20)의 iterative flow 내부에서
     ULP-level FP 차이(MATLAB MKL `cos/sin` vs Python glibc `np.cos/np.sin`)가
     누적되어 수렴점이 약간 다른 local minimum으로 수렴.
   - 이는 `project_mesh2d_fp_limit.md`에 이미 기록된 quadtree-level 차이가
     후속 smoothing에 증폭되어 나타난 것이며, `poly25` 25-gon에서 interior
     노드 ≲0.07 nm 드리프트와 동일한 현상.

3. **공진 민감성**
   - Silver triangle on dielectric membrane은 edge/corner plasmon mode의
     Q ≳ 20을 가지며, mesh 0.1 nm 차이 → resonance frequency 수십 meV shift
     → 고정 energy grid에서 5–15% loss 차이.

## 해결책 (= 수용)

현재 `Polygon3.plate_from_list` (mnpbem/geometry/polygon3.py) 은
MATLAB `@polygon3/plate.m` 과 step-by-step 일치:

- L20 z-uniqueness assert
- L43-48 `polymesh2d(poly_closed)` 호출
- L50-53 `interp1(obj.poly, verts)` boundary enrichment
- L58 `midpoints(p, 'flat')` 로 9-column faces2 생성
- L70-89 per-polygon boundary smoothing + `vshift` edge z 적용
- L95 `Particle(verts2, faces2)` → 9-column 분기로 unique vertex 추출

algorithm 자체는 이미 정합이며, 더 이상 수학적/구조적으로 당길 여지 없음.

## 시도 및 결과

### 2026-04-23 Wave 5 δ
- `_classify_faces` / `_detect_loops` / `meshpoly` inpoly filter — 정상 동작
- MATLAB mesh2d single-face path vs Python multi-face path — 동일 결과 확인
  (poly4+poly-triangle-46 → 양 경로 모두 346 verts / 580 faces)
- `polymesh2d` 의 `face=face_list` 전달 여부는 결과에 영향 없음 (auto-detect
  경로와 explicit face 경로가 동일 mesh 생성)
- `CubicSpline` / `interp1` / `midpoints('same')` 모두 MATLAB 의미 일치

→ Wave 5 δ 시점에서 Python plate-with-hole mesh 는 MATLAB 대비
수치적으로 구조적 한계에 도달. `demoeelsret7/8` warn 등급은 수용.

## Tolerance Guidelines

- Analytical geometry (sphere, fvgrid): bit-identical (1e-14 이하)
- `tripolygon`, `plate`, `plate_from_list`: ±0.1 nm mesh noise →
  - face-level σ: O(1e-3) (OK)
  - 1D spectrum: 공진 peak height ±5–15%, corner peak ±0.3% 수준
  - 2D spatial map: 평균 편차 ±5%, 공진 위치 spatial shift 수 nm
- MATLAB과의 완전 정합이 필요하면 mesh injection (MATLAB mesh → Python 로드)
  사용 권장

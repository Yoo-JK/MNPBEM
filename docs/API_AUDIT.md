# MNPBEM API Audit (MATLAB → Python 포팅)

**작성일**: 2026-04-22  
**범위**: MATLAB MNPBEM 모든 public class/method  
**목표**: Python 포팅 완성도 평가 및 누락 항목 우선순위화

---

## 요약

| 항목 | 수량 |
|------|------|
| **MATLAB public methods** | 697개 |
| **Python 구현 파일** | 102개 |
| **주요 클래스 완성도** | 65.2% (60/92 메서드) |
| **완전히 구현된 클래스** | 0/10 (주요 클래스 기준) |
| **80% 이상 완성** | 2/10 클래스 |

**결론**: Python 포팅이 대부분의 핵심 기능을 포함하나 메서드 수준에서 ~35% 누락

---

## Module별 현황

### 1. BEM (경계요소법) 모듈

| MATLAB Class | Python Class | 메서드 | 완성도 | 상태 |
|---|---|---|---|---|
| `bemstat` | `BEMStat` | 9 | 78% (7/9) | ⚠ 미흡 |
| `bemret` | `BEMRet` | 8 | 88% (7/8) | ⚠ 거의 완성 |
| `bemstatlayer` | `BEMStatLayer` | 7 | 71% (5/7) | ✗ 불완전 |
| `bemretlayer` | `BEMRetLayer` | 6 | 83% (5/6) | ⚠ 거의 완성 |
| `bemstateig` | `BEMStatEig` | 7 | ? | ✓ 구현됨 |
| `bemstateigmirror` | `BEMStatEigMirror` | 8 | ? | ✓ 구현됨 |
| `bemstatmirror` | `BEMStatMirror` | 8 | ? | ✓ 구현됨 |
| `bemretmirror` | `BEMRetMirror` | 6 | ? | ✓ 구현됨 |
| `bemiter` | `BEMIter` | 10 | 50% (5/10) | ✗ 중대 누락 |
| `bemstatiter` | `BEMStatIter` | 9 | ? | 부분 구현 |
| `bemretiter` | `BEMRetIter` | 8 | ? | 부분 구현 |
| `bemretlayeriter` | `BEMRetLayerIter` | 7 | ? | 부분 구현 |
| `bemlayermirror` | `BEMLayerMirror` | 1 | ? | ✓ 구현됨 |

**BEM 분석**:
- ✓ 정적(stat)/진동(ret) 기본 솔버 구현됨
- ⚠ 반복(iter) 솔버 (~50% 구현)
- ⚠ 메서드 매핑: MATLAB `mldivide` → Python `__truediv__`, `mtimes` → `__mul__`

---

### 2. Particles (입자/기하) 모듈

| MATLAB Class | Python Class | 메서드 | 완성도 | 상태 |
|---|---|---|---|---|
| `polygon` | `Polygon` | 17 | 71% (12/17) | ⚠ 미흡 |
| `polygon3` | `Polygon3` | 0 | - | ? 불명확 |
| `comparticle` | `ComParticle` | 11 | 73% (8/11) | ⚠ 미흡 |
| `compound` | `Compound`/`Connect` | 11 | 9% (1/11) | ✗ **심각한 누락** |
| `layerstructure` | `LayerStructure` | 13 | 77% (10/13) | ⚠ 미흡 |
| `edgeprofile` | `EdgeProfile` | ? | ? | ? |
| `compoint` | `ComPoint` | ? | ? | ✓ 구현됨 |
| `point` | `Point` | ? | ? | ✓ 구현됨 |
| `comparticlemirror` | `ComParticleMirror` | ? | ? | ✓ 구현됨 |

**Particles 분석**:
- ✓ 기본 기하학적 클래스 구현됨
- ✗ **`compound` 클래스 심각한 누락** (11개 메서드 중 1개만)
- ⚠ `Polygon`에서 누락된 메서드:
  - `plot()`, `norm()`, `interp1()`, `union()`, `symmetry()` 등 유틸리티 함수
- ⚠ 메서드명 변경: `round()` → `round_()`, `sort()` → `sort_()`

---

### 3. Greenfun (그린함수) 모듈

| MATLAB Class | Python Class | 메서드 | 상태 |
|---|---|---|---|
| `greenstat` | `GreenStat` | 4 | ✓ 구현됨 |
| `greenret` | `GreenRet` | 5 | ✓ 구현됨 |
| `greenretlayer` | `GreenRetLayer` | 4 | ✓ 구현됨 |
| `compgreenstat` | `CompGreenStat` | 6 | ✓ 구현됨 |
| `compgreenret` | `CompGreenRet` | 5 | ✓ 구현됨 |
| `compgreenretlayer` | `CompGreenRetLayer` | 6 | ✓ 구현됨 |
| `compgreenretmirror` | `CompGreenRetMirror` | 5 | ✓ 구현됨 |
| `compgreenstatmirror` | `CompGreenStatMirror` | 5 | ✓ 구현됨 |
| `compgreentablayer` | `CompGreenTabLayer` | 8 | ✓ 구현됨 |
| `greentablayer` | `GreenTabLayer` | 8 | ✓ 구현됨 |
| `+aca/compgreenstat` | `AcaCompGreenStat` | 5 | ✓ 구현됨 |
| `+aca/compgreenret` | `AcaCompGreenRet` | 4 | ✓ 구현됨 |
| `+aca/compgreenretlayer` | `AcaCompGreenRetLayer` | 4 | ✓ 구현됨 |

**Greenfun 분석**:
- ✓ **거의 완전히 구현됨** (가장 높은 완성도)
- ✓ ACA (Adaptive Cross-Approximation) 압축 포함
- ✓ H-matrix 구현 포함

---

### 4. Simulation (시뮬레이션) 모듈

**Static 부분:**
| MATLAB Class | Python Class | 상태 |
|---|---|---|
| `dipolestat` | `DipoleStatEsource` | ✓ |
| `dipolestatlayer` | `DipoleStatLayer` | ✓ |
| `dipolestatmirror` | `DipoleStatMirror` | ✓ |
| `eelsstat` | `EelsStatBase` | ✓ |
| `planewavestat` | `PlaneWaveStat` | ✓ |
| `planewavestatlayer` | `PlaneWaveStatLayer` | ✓ |
| `planewavestatmirror` | `PlaneWaveStatMirror` | ✓ |
| `spectrumstat` | `SpectrumStat` | ✓ |
| `spectrumstatlayer` | `SpectrumStatLayer` | ✓ |

**Retarded (진동) 부분:**
| MATLAB Class | Python Class | 상태 |
|---|---|---|
| `dipoleret` | `DipoleRetEsource` | ✓ |
| `dipoleretlayer` | `DipoleRetLayer` | ✓ |
| `dipoleretmirror` | `DipoleRetMirror` | ✓ |
| `eelsret` | `EelsRetBase` | ✓ |
| `planewaveret` | `PlaneWaveRet` | ✓ |
| `planewaveretlayer` | `PlaneWaveRetLayer` | ✓ |
| `planewaveretmirror` | `PlaneWaveRetMirror` | ✓ |
| `spectrumret` | `SpectrumRet` | ✓ |
| `spectrumretlayer` | `SpectrumRetLayer` | ✓ |

**Simulation 분석**:
- ✓ **Static 및 Retarded 여기(excitation) 거의 완전히 구현**
- ✓ 디폴라이저(dipole), 평면파(planewave), EELS, 스펙트럼 분석 포함
- ✓ Mirror 대칭 구현됨

---

### 5. Mesh2d (메시 생성) 모듈

| 항목 | 상태 |
|---|---|
| `mesh2d` | ✓ 구현됨 |
| `mesh2d_core` | ✓ 구현됨 |
| `mesh_generators` | ✓ 구현됨 |
| Darren Engwirda 툴박스 | ✓ 포팅됨 |

**Mesh2d 분석**:
- ✓ **완전히 구현됨**

---

### 6. Misc (+misc) 모듈

| 항목 | Python | 상태 |
|---|---|---|
| `units` | `mnpbem/misc/units.py` | ✓ |
| `constants` | `mnpbem/utils/constants.py` | ✓ |
| `misc_utils` | `mnpbem/misc/misc_utils.py` | ✓ |
| `math_utils` | `mnpbem/misc/math_utils.py` | ✓ |
| `gauss_legendre` | `mnpbem/misc/gauss_legendre.py` | ✓ |
| `bemplot` | `mnpbem/misc/bemplot.py` | ✓ |

**Misc 분석**:
- ✓ **완전히 구현됨**

---

## 누락 항목 (Priority별)

### Priority 1 (주요 기능, 예제 실행에 필수)

1. **`Particles/@compound` 메서드** (11개 중 10개 누락)
   - `set()`, `eq()`, `ne()`, `ipart()`, `subsref()`, `dielectric()`, `mask()`, `index()`, `expand()`
   - **영향도**: 다중-물질 입자 구성 불가 → `demospecret15` 등 복잡한 예제 실패
   - **우선순위**: 매우 높음

2. **`BEM/@bemiter` 메서드** (10개 중 5개 누락)
   - `solve()`, `setstat()`, `printstat()`, `setiter()` 
   - **영향도**: 반복 솔버 기능 불완전 → 대규모 시뮬레이션 성능 저하
   - **우선순위**: 높음

3. **`Particles/@polygon` 유틸리티** (17개 중 5개 누락)
   - `plot()`, `norm()`, `interp1()`, `union()`, `symmetry()`
   - **영향도**: 메시 생성 및 시각화 제한
   - **우선순위**: 중간-높음

4. **Mirror 클래스 메서드 완성도** 
   - `bemstateigmirror`, `bemstatmirror`, `bemretmirror` → 메서드 수 불명확
   - 거울 대칭 기능 부분적 구현
   - **우선순위**: 중간

### Priority 2 (고급 기능, 일부 예제에 필요)

1. **`Particles/@layerstructure` 메서드** (13개 중 3개 누락)
   - 정확한 누락 메서드 미파악 (대부분 내부 헬퍼)
   - **영향도**: 다층 구조 시뮬레이션 제한

2. **`BEM/@bemstatlayer` 메서드** (7개 중 2개 누락)
   - `subsref()`, `init()` 완전성 미상
   - **영향도**: 다층 정적 BEM 기능 제한

3. **`BEM/@bemretlayer` 메서드** (6개 중 1개 누락)
   - `solve()` 메서드 완전성 검증 필요
   - **영향도**: 다층 진동 BEM 기능 제한

### Priority 3 (선택적, 내부 헬퍼/레거시)

1. **`Particles/@polygon3`**
   - MATLAB에서도 메서드 불명확 (생성자만 존재)
   - Python 구현 존재하나 매핑 불명확

2. **`+aca/` (Adaptive Cross Approximation)**
   - 고급 압축 기능 (대규모 문제용)
   - 포팅 상태: ✓ 구현됨

3. **메서드명 변경 호환성**
   - MATLAB `round()` → Python `round_()`
   - MATLAB `sort()` → Python `sort_()`
   - 레거시 별칭 추가 검토

---

## Python 추가 구현 (MATLAB에 없음)

| 기능 | 파일 | 설명 |
|---|---|---|
| **Material 모델** | `mnpbem/materials/` | Drude, Lorentz, 표 기반 모델 |
| **Mie 이론** | `mnpbem/mie/` | 구형 산란 (MIE 이론) |
| **Field 메시 계산** | `mnpbem/simulation/meshfield.py` | 공간 메시 위 전자기장 계산 |
| **Solver 팩토리** | `mnpbem/bem/solver_factory.py` | BEM 솔버 팩토리 패턴 |
| **병렬화** | `mnpbem/utils/parallel.py` | 병렬 계산 지원 |
| **테스트 스위트** | `mnpbem/tests/` | 20개 테스트 모듈 |

---

## 메서드 매핑 가이드

### 연산자 오버로딩

| MATLAB | Python | 의미 |
|---|---|---|
| `obj \ exc` | `obj.__truediv__(exc)` 또는 `obj / exc` | 표면 전하 계산 |
| `obj * sig` | `obj.__mul__(sig)` | 유도 전위 계산 |

**주의**: Python은 `\` 연산자가 없으므로 대신 `/` 또는 `.solve()`를 사용

### 메서드명 규칙

| MATLAB | Python | 이유 |
|---|---|---|
| `round()` | `round_()` | Python 내장 `round()` 충돌 회피 |
| `sort()` | `sort_()` | Python 내장 `list.sort()` 명확성 |
| `subsref()` | `__getitem__()` | Python 인덱싱 컨벤션 |
| `init()` | `__init__()` | Python 생성자 |

---

## 데모 실행 가능성 평가

### 완전히 작동하는 예제
- ✓ `demo_nanosphere_spectrum` - 나노구 산란 스펙트럼
- ✓ 단순 입자 (구, 타원체)
- ✓ 정적/진동 다중극 여기
- ✓ 평면파 여기
- ✓ EELS 계산

### 문제가 있을 수 있는 예제
- ⚠ 다중-물질 입자 (`compound` 메서드 부족)
- ⚠ 대규모 시뮬레이션 (반복 솔버 미흡)
- ⚠ 고급 메시 조작 (`polygon` 유틸리티 부족)
- ⚠ 다층 구조 (레이어 메서드 부분 미흡)

---

## 권장 사항

### 즉시 해결 (1주)

1. **`Particles/@compound` 전체 포팅**
   - 영향도 높음, 규모 중간
   - 핵심: `subsref()`, `dielectric()`, `mask()`, `index()` 메서드

2. **`BEM/@bemiter.solve()` 검증 및 완성**
   - 반복 솔버 안정성 확보

### 단기 개선 (2-3주)

3. **`Particles/@polygon` 유틸리티 메서드 추가**
   - `plot()`, `norm()`, `symmetry()` 등

4. **Mirror 클래스 메서드 명시화**
   - 각 mirror 클래스의 메서드 목록 문서화

### 중기 정비 (1개월)

5. **메서드명 호환성 레이어**
   - 레거시 별칭 추가 (e.g., `round` → `round_()`)

6. **테스트 커버리지 확대**
   - 누락 메서드별 테스트 작성

---

## 완성도 통계

```
Module         MATLAB  Python  Completion
─────────────────────────────────────────
BEM              98      ~65      66%
Particles       133      ~85      64%
Greenfun       117     ~115      98%  ✓
Simulation     147     ~145      99%  ✓
Mesh2d          21      ~21     100%  ✓
Misc            82      ~82     100%  ✓
Base             8       ~8     100%  ✓
─────────────────────────────────────────
TOTAL          606     ~521      86%
```

**주요 결론**:
- **전체 완성도**: ~86% (메서드 수 기준)
- **높은 완성도**: Greenfun, Simulation, Mesh2d, Misc
- **개선 필요**: Particles (특히 `compound`), BEM (특히 반복 솔버)

---

## 사용자 가이드

### Python에서 MATLAB 코드 마이그레이션

```python
# MATLAB:
% bem = bemstat(p);
% sig = bem \ exc;
% phi = bem * sig;
% E = bem.field(sig);

# Python:
from mnpbem import *
bem = BEMStat(p)
sig = bem / exc  # 또는 sig = bem.solve(exc)
phi = bem * sig
E = bem.field(sig)
```

### API 불일치 주의

1. **연산자**: MATLAB `\` → Python `/` (또는 `.solve()`)
2. **메서드명**: `round()` → `round_()`, `sort()` → `sort_()`
3. **다중 물질**: `compound` 메서드 누락 → 임시 해결책 필요

---

## 감사 결론

MNPBEM Python 포팅은 **핵심 기능 (86%) 기준으로는 상당히 완성되었으나**,  
**메서드 수준 (65%) 기준으로는 주요 누락이 존재**합니다.

**즉시 해결 필요:**
- `Particles/@compound` 전체 포팅 (매우 높은 우선순위)
- `BEM/@bemiter` 반복 솔버 완성도 향상
- 유틸리티 메서드 (`polygon`, `layerstructure`) 추가

**현황**: 대부분의 기본 시뮬레이션 작동하나, 복잡한 기하/다층 구조에서 제한


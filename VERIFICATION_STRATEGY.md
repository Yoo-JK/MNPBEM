# MNPBEM MATLAB→Python 변환 전수조사 전략

## 목표
MATLAB 기반 MNPBEM과 Python 변환 코드가 **기능적/구성적으로 100% 동일**함을 검증

---

## 전략 개요

### Phase 1: 변환 범위 매핑 (Scope Mapping)
- [ ] MATLAB 클래스/메소드 완전 목록 생성
- [ ] Python 변환 완료 항목 매핑
- [ ] 미변환 항목 식별

### Phase 2: 계층적 검증 (Hierarchical Verification)
- [ ] Level 1: 단위 테스트 (개별 메소드)
- [ ] Level 2: 클래스 테스트 (클래스 동작)
- [ ] Level 3: 통합 테스트 (모듈 간 상호작용)
- [ ] Level 4: 시스템 테스트 (전체 시뮬레이션)

### Phase 3: 수치 정확도 검증 (Numerical Validation)
- [ ] MATLAB 기준 데이터 생성
- [ ] Python 결과와 비교 (상대오차 < 1e-10)
- [ ] Edge case 테스트

### Phase 4: 자동화 파이프라인 (Automation)
- [ ] 자동 테스트 실행 스크립트
- [ ] 회귀 테스트 프레임워크
- [ ] CI/CD 통합

---

## 상세 전략

## 1️⃣ 변환 범위 매핑

### 1.1 MATLAB 코드 완전 목록화

**자동화 스크립트로 추출:**
```bash
# 모든 MATLAB 클래스 디렉토리 찾기
find . -type d -name '@*'

# 각 클래스의 public 메소드 추출
for dir in @*; do
  echo "Class: $dir"
  grep -h "^function" $dir/*.m | sed 's/.*function /  - /'
done
```

**출력 형식:**
```
Module: BEM
  Class: bemstat
    ✅ __init__ (constructor)
    ✅ solve
    ⬜ field
    ⬜ potential
  Class: bemret
    ✅ __init__
    ⬜ solve
    ...
```

### 1.2 매핑 테이블 생성

| MATLAB Class | MATLAB Method | Python Class | Python Method | Status | Test Coverage |
|--------------|---------------|--------------|---------------|--------|---------------|
| @bemstat | bemstat.m | BEMStat | \_\_init\_\_ | ✅ | ✅ |
| @bemstat | solve.m | BEMStat | solve | ✅ | ✅ |
| @bemstat | field.m | BEMStat | field | ✅ | ⚠️ partial |
| ... | ... | ... | ... | ... | ... |

---

## 2️⃣ 계층적 검증 프레임워크

### Level 1: 단위 테스트 (Method-Level)

**각 메소드마다:**
1. **입력 동일성**: MATLAB과 동일한 입력 사용
2. **출력 비교**: 수치 결과 비교 (rtol=1e-10)
3. **예외 처리**: 동일한 에러 발생 확인

**테스트 템플릿:**
```python
# test_bemstat_solve.py
def test_bemstat_solve_vs_matlab():
    """BEMStat.solve() matches MATLAB bemstat/solve.m"""

    # 1. Load MATLAB reference data
    matlab_data = scipy.io.loadmat('tests/references/bemstat_solve_ref.mat')

    # 2. Create identical Python inputs
    p = Particle(...)  # Same geometry as MATLAB
    bem_stat = BEMStat(p, ...)

    # 3. Execute Python method
    python_result = bem_stat.solve(...)

    # 4. Compare outputs
    np.testing.assert_allclose(
        python_result.sig,
        matlab_data['sig'],
        rtol=1e-10,
        err_msg="Surface charges differ from MATLAB"
    )
```

### Level 2: 클래스 테스트 (Class-Level)

**클래스 생명주기 전체 검증:**
```python
def test_bemstat_full_workflow():
    """Complete BEMStat workflow matches MATLAB"""
    # __init__ → solve → field → potential
    # All intermediate states match MATLAB
```

### Level 3: 통합 테스트 (Module Integration)

**모듈 간 상호작용:**
```python
def test_materials_geometry_integration():
    """EpsTable + Particle integration matches MATLAB"""
    # epstable → particle → comparticle workflow
```

### Level 4: 시스템 테스트 (Full Simulation)

**실제 물리 시뮬레이션:**
```python
def test_gold_nanosphere_spectrum():
    """Full spectrum calculation matches MATLAB demospecstat01.m"""
    # Complete workflow: material → geometry → BEM → excitation → spectrum
```

---

## 3️⃣ 수치 정확도 검증

### 3.1 MATLAB 기준 데이터 생성

**자동화 스크립트 (MATLAB):**
```matlab
% generate_all_references.m
% Run all MATLAB demos and save outputs

demos = {
    'Demos/demostatic01.m',
    'Demos/demospecstat01.m',
    'Demos/demoret01.m',
    ...
};

for i = 1:length(demos)
    run(demos{i});
    save(sprintf('references/demo%02d_ref.mat', i));
end
```

### 3.2 Python 비교 테스트

```python
def compare_with_matlab(python_result, matlab_ref_file, var_name):
    """Generic comparison function"""
    matlab_data = scipy.io.loadmat(matlab_ref_file)
    matlab_result = matlab_data[var_name]

    # Numerical comparison
    np.testing.assert_allclose(python_result, matlab_result, rtol=1e-10)

    # Statistical comparison
    relative_error = np.abs((python_result - matlab_result) / matlab_result)
    print(f"Max relative error: {relative_error.max():.2e}")
    print(f"Mean relative error: {relative_error.mean():.2e}")
```

### 3.3 Edge Case 테스트

**경계 조건:**
- 극한값 (wavelength → 0, wavelength → ∞)
- 특이점 (touching particles, self-interaction)
- 수치 안정성 (ill-conditioned matrices)

---

## 4️⃣ 자동화 파이프라인

### 4.1 테스트 디렉토리 구조

```
tests/
├── unit/                     # Level 1: 단위 테스트
│   ├── test_materials.py
│   ├── test_geometry.py
│   ├── test_greenfun.py
│   ├── test_bem.py
│   ├── test_excitation.py
│   └── test_spectrum.py
├── integration/              # Level 2-3: 통합 테스트
│   ├── test_material_geometry.py
│   ├── test_bem_workflow.py
│   └── test_excitation_bem.py
├── system/                   # Level 4: 시스템 테스트
│   ├── test_demo_static.py
│   ├── test_demo_spectrum.py
│   └── test_demo_retarded.py
├── references/               # MATLAB 기준 데이터
│   ├── bemstat_solve_ref.mat
│   ├── compgreenstat_init_ref.mat
│   └── ...
└── conftest.py               # pytest 설정
```

### 4.2 자동 실행 스크립트

```bash
#!/bin/bash
# run_full_verification.sh

echo "=== MNPBEM Verification Pipeline ==="

# Step 1: Generate MATLAB references (if needed)
if [ ! -d "tests/references" ]; then
    echo "Generating MATLAB references..."
    matlab -batch "cd tests/matlab; generate_all_references"
fi

# Step 2: Run Python tests
echo "Running Python unit tests..."
pytest tests/unit/ -v --tb=short

echo "Running integration tests..."
pytest tests/integration/ -v

echo "Running system tests..."
pytest tests/system/ -v

# Step 3: Generate coverage report
echo "Generating coverage report..."
pytest --cov=mnpbem --cov-report=html

echo "=== Verification Complete ==="
```

### 4.3 GitHub Actions CI/CD

```yaml
# .github/workflows/verification.yml
name: MNPBEM Verification

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r mnpbem/requirements.txt
          pip install pytest pytest-cov
      - name: Run verification tests
        run: pytest tests/ -v --cov=mnpbem
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 5️⃣ 검증 메트릭 (Verification Metrics)

### 정량적 지표

| Metric | Target | Current |
|--------|--------|---------|
| **Code Coverage** | 100% | TBD |
| **Method Coverage** | 100% of converted methods | TBD |
| **Numerical Accuracy** | rtol < 1e-10 | TBD |
| **Test Pass Rate** | 100% | TBD |

### 체크리스트

- [ ] 모든 변환된 클래스에 단위 테스트 존재
- [ ] 모든 public 메소드에 MATLAB 비교 테스트 존재
- [ ] 모든 MATLAB demo에 대응하는 Python 테스트 존재
- [ ] Edge case 테스트 커버리지 > 90%
- [ ] 문서화된 알려진 차이점 (known differences)

---

## 6️⃣ 실행 계획

### Week 1: Setup & Mapping
1. MATLAB 클래스/메소드 완전 목록 생성
2. 변환 상태 매핑 테이블 작성
3. 테스트 디렉토리 구조 생성

### Week 2-3: Unit Tests
1. 각 모듈별 단위 테스트 작성
2. MATLAB 기준 데이터 생성 스크립트 작성
3. 비교 자동화 유틸리티 개발

### Week 4: Integration & System Tests
1. 통합 테스트 작성
2. 전체 시뮬레이션 재현 테스트
3. Edge case 테스트

### Week 5: Automation
1. CI/CD 파이프라인 구축
2. 자동 리포트 생성
3. 문서화

---

## 7️⃣ 알려진 차이점 허용 기준

### 허용 가능한 차이
1. **언어 차이**: MATLAB handle class vs Python object
2. **인덱싱**: MATLAB 1-based vs Python 0-based (내부적으로만)
3. **출력 형식**: MATLAB struct vs Python dict/object
4. **성능**: 실행 시간 차이 (기능은 동일)

### 허용 불가능한 차이
1. **수치 결과**: 상대 오차 > 1e-10
2. **알고리즘**: 다른 계산 방법 사용
3. **기본값**: 다른 default parameter
4. **물리적 결과**: 다른 스펙트럼/필드 분포

---

## 📊 진행 상황 추적

### Conversion Coverage
- Materials: ✅ 100% (3/3 classes)
- Geometry: ✅ 100% (4/4 core classes)
- Green Functions: ✅ 100% (2/2 core classes)
- BEM Solvers: ✅ 100% (2/2 core classes)
- Excitation: ✅ 100% (4/4 core classes)
- Spectrum: ✅ 100% (2/2 classes)
- **Advanced Features**: ⬜ 0% (layers, mirrors, iterative, H-matrices)

### Test Coverage
- Level 1 (Unit): ⚠️ 50% (step1-7 exist, but not comprehensive)
- Level 2 (Class): ⚠️ 30%
- Level 3 (Integration): ⚠️ 20%
- Level 4 (System): ⚠️ 10%

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
cd /home/user/MNPBEM

# 2. Create test infrastructure
mkdir -p tests/{unit,integration,system,references}

# 3. Generate MATLAB references
matlab -batch "run('tests/matlab/generate_all_references.m')"

# 4. Run verification
./run_full_verification.sh

# 5. View results
open htmlcov/index.html
```

---

## 📚 참고 문서

- 기존 테스트: `mnpbem/examples/test_step*.py`
- MATLAB 원본: `BEM/`, `Greenfun/`, `Simulation/`, etc.
- Python 변환: `mnpbem/` 모듈

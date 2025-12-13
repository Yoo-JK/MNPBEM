
# MNPBEM 검증 테스트 가이드

## 디렉토리 구조

```
tests/
├── unit/                 # 단위 테스트 (메소드별)
│   ├── test_materials.py
│   ├── test_geometry.py
│   ├── test_greenfun.py
│   ├── test_bem.py
│   ├── test_excitation.py
│   └── test_spectrum.py
├── integration/          # 통합 테스트 (워크플로우)
│   └── test_workflows.py
├── references/           # MATLAB 기준 데이터 (.mat 파일)
│   └── (생성 필요)
├── matlab_references/    # MATLAB 기준 데이터 생성 스크립트
│   └── generate_*.m
└── conftest.py           # pytest 설정
```

## 사용 방법

### 1단계: MATLAB 기준 데이터 생성

```bash
cd tests/matlab_references
matlab -batch "run('generate_all_references.m')"
```

또는 개별 생성:
```bash
matlab -batch "generate_bemstat_solve_reference"
```

### 2단계: Python 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 모듈만 테스트
pytest tests/unit/test_bem.py -v

# 통합 테스트만 실행
pytest tests/integration/ -v

# 커버리지 리포트 생성
pytest tests/ --cov=mnpbem --cov-report=html
```

### 3단계: 결과 확인

```bash
# 커버리지 리포트 보기
open htmlcov/index.html
```

## 테스트 구현 가이드

각 테스트 파일에는 템플릿이 생성되어 있습니다. 다음 순서로 구현하세요:

1. **MATLAB 기준 데이터 생성 스크립트 완성**
   - `tests/matlab_references/generate_*.m` 파일 수정
   - 적절한 입력 데이터로 MATLAB 코드 실행
   - 결과를 .mat 파일로 저장

2. **Python 테스트 구현**
   - 동일한 입력으로 Python 코드 실행
   - `compare_with_matlab()` 함수로 비교
   - 필요시 추가 검증 로직 작성

3. **pytest 실행 및 디버깅**
   - 실패한 테스트 분석
   - 수치 오차 확인 (rtol 조정 필요 여부)
   - 알고리즘 차이 확인

## 예시: bemstat solve 테스트 구현

### MATLAB 스크립트 (generate_bemstat_solve_reference.m)

```matlab
function generate_bemstat_solve_reference()
    addpath(genpath('../../'));

    % Create test geometry
    diameter = 30;  % nm
    epstab = epstable('gold.dat');
    p = comparticle({{epsconst(1), epstab}}, {{trisphere(144, diameter)}}, [2, 1], 1);

    % Create BEM solver
    bem = bemstat(p, 'waitbar', 0);

    % Create excitation
    enei = 600;  % nm
    exc = planewavestat([0, 0, 1], [1, 0, 0]);

    % Solve
    sig = exc.potential(p, enei);
    sig = bem \ sig;

    % Save
    save('../../tests/references/bemstat_solve_ref.mat', 'sig', 'p', 'enei', '-v7.3');
end
```

### Python 테스트 (test_bem.py)

```python
def test_bemstat_solve():
    # Load MATLAB reference
    ref = scipy.io.loadmat('tests/references/bemstat_solve_ref.mat')

    # Create identical setup
    diameter = 30
    eps_gold = EpsTable('gold.dat')
    p = ComParticle([EpsConst(1), eps_gold], [trisphere(144, diameter)], [2, 1])

    # BEM solver
    bem_stat = BEMStat(p)

    # Excitation
    enei = 600
    exc = PlaneWaveStat([0, 0, 1], [1, 0, 0])

    # Solve
    sig = exc.potential(p, enei)
    sig = bem_stat.solve(sig, enei)

    # Compare
    compare_with_matlab(sig, 'tests/references/bemstat_solve_ref.mat', 'sig')
```

## 팁

- MATLAB 1-based indexing → Python 0-based indexing 주의
- MATLAB struct → Python dict/object 변환 확인
- 복소수 배열 비교 시 real/imag 각각 확인
- 수치 안정성 문제 시 rtol 조정 (기본 1e-10)

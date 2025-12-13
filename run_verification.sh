#!/bin/bash
#
# MNPBEM 전수조사 실행 스크립트
#
# 용도: MATLAB → Python 변환 검증을 자동으로 실행
#

set -e  # 에러 발생 시 중단

echo "================================================================"
echo "MNPBEM MATLAB→Python 변환 전수조사"
echo "================================================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 프로젝트 루트
PROJECT_ROOT="/home/user/MNPBEM"
cd "$PROJECT_ROOT"

# Step 1: 코드베이스 분석
echo -e "${YELLOW}[1/5] MATLAB 코드베이스 분석 중...${NC}"
python3 tools/analyze_matlab_code.py
echo -e "${GREEN}✅ 분석 완료${NC}"
echo ""

# Step 2: 변환 상태 리포트
echo -e "${YELLOW}[2/5] 변환 상태 리포트 생성 중...${NC}"
if [ -f "CONVERSION_MAPPING.md" ]; then
    echo "변환 통계:"
    grep -A 5 "## 📊 변환 통계" CONVERSION_MAPPING.md
    echo -e "${GREEN}✅ 리포트 생성 완료: CONVERSION_MAPPING.md${NC}"
else
    echo -e "${RED}❌ 리포트 생성 실패${NC}"
    exit 1
fi
echo ""

# Step 3: 테스트 스위트 생성
echo -e "${YELLOW}[3/5] 테스트 스위트 생성 중...${NC}"
python3 tools/test_generator.py
echo -e "${GREEN}✅ 테스트 생성 완료${NC}"
echo ""

# Step 4: 기존 테스트 실행
echo -e "${YELLOW}[4/5] 기존 검증 테스트 실행 중...${NC}"
if [ -d "mnpbem/examples" ]; then
    cd mnpbem/examples

    # Step 1-7 테스트 실행
    for i in {1..7}; do
        if [ -f "test_step${i}_*.py" ]; then
            echo "  Running step ${i}..."
            python3 test_step${i}_*.py 2>&1 | grep -E "(✅|❌|Test|Pass|Fail)" || true
        fi
    done

    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✅ 기존 테스트 완료${NC}"
else
    echo -e "${YELLOW}⚠️  examples 디렉토리 없음${NC}"
fi
echo ""

# Step 5: 리포트 생성
echo -e "${YELLOW}[5/5] 최종 리포트 생성 중...${NC}"
cat > VERIFICATION_REPORT.md <<EOF
# MNPBEM 전수조사 리포트

**생성일**: $(date +"%Y-%m-%d %H:%M:%S")

## 🎯 목표
MATLAB 기반 MNPBEM과 Python 변환 코드의 100% 동일성 검증

## 📊 현재 상태

$(grep -A 10 "## 📊 변환 통계" CONVERSION_MAPPING.md)

## ✅ 생성된 검증 도구

### 1. 분석 도구
- \`tools/analyze_matlab_code.py\`: MATLAB 코드 자동 분석
- \`CONVERSION_MAPPING.md\`: 변환 상태 매핑 테이블 (554개 메소드)

### 2. 테스트 프레임워크
- \`tests/unit/\`: 단위 테스트 (메소드별)
- \`tests/integration/\`: 통합 테스트 (워크플로우)
- \`conftest.py\`: pytest 설정 및 유틸리티
- \`matlab_references/\`: MATLAB 기준 데이터 생성 스크립트

### 3. 문서화
- \`VERIFICATION_STRATEGY.md\`: 전체 전략 문서
- \`README_TESTS.md\`: 테스트 실행 가이드

## 📋 다음 단계

### Phase 1: MATLAB 기준 데이터 생성 (예상 시간: 1-2일)

1. MATLAB 환경 준비
\`\`\`bash
cd matlab_references
matlab
\`\`\`

2. 각 변환된 클래스에 대한 기준 데이터 생성
\`\`\`matlab
% 예시
generate_particle_particle_reference
generate_epsconst_epsconst_reference
% ... (27개 스크립트)
\`\`\`

### Phase 2: Python 테스트 구현 (예상 시간: 3-5일)

1. 각 테스트 템플릿 완성
\`\`\`bash
# tests/unit/test_geometry.py 등 수정
# TODO 부분을 실제 테스트 코드로 대체
\`\`\`

2. 테스트 실행 및 디버깅
\`\`\`bash
pytest tests/unit/ -v
\`\`\`

### Phase 3: 고급 기능 검증 (예상 시간: 1-2주)

**미변환 기능 (527개 메소드):**
- Layer structures (stratified media)
- Mirror symmetry
- Iterative solvers (BiCG, GMRES)
- H-matrices
- EELS (Electron Energy Loss Spectroscopy)

**검증 필요 시:**
1. 해당 기능 Python 변환
2. 동일한 검증 프로세스 적용

### Phase 4: 지속적 검증 (Continuous Verification)

1. GitHub Actions CI/CD 설정
\`\`\`yaml
# .github/workflows/verification.yml 활성화
# 모든 커밋마다 자동 테스트
\`\`\`

2. 정기적 회귀 테스트
\`\`\`bash
# 매주 실행
./run_verification.sh
\`\`\`

## 🔍 핵심 검증 메트릭

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| 변환 완료율 | 100% | 4.9% (27/554) |
| 테스트 커버리지 | 100% | 0% (미구현) |
| 수치 정확도 | rtol < 1e-10 | TBD |
| 통합 테스트 Pass율 | 100% | TBD |

## 📝 주요 발견사항

### 변환된 모듈 (Core Physics)
✅ Materials (EpsConst, EpsTable, EpsDrude)
✅ Geometry (Particle, ComParticle, trisphere)
✅ Green Functions (CompGreenStat, CompGreenRet)
✅ BEM Solvers (BEMStat, BEMRet)
✅ Excitations (PlaneWave, Dipole - static & retarded)
✅ Spectrum Analysis

### 미변환 모듈 (Advanced Features)
⬜ Layer structures (bemstatlayer, bemretlayer)
⬜ Mirror symmetry (bemstatmirror, bemretmirror)
⬜ Iterative solvers (bemstatiter, bemretiter)
⬜ H-matrices (hmatrix, clustertree)
⬜ EELS (eelsstat, eelsret)
⬜ Mie theory (miestat, mieret)

## 🚀 빠른 시작

\`\`\`bash
# 1. 전체 검증 실행
./run_verification.sh

# 2. 특정 테스트만 실행
pytest tests/unit/test_geometry.py -v

# 3. 상태 확인
cat CONVERSION_MAPPING.md
\`\`\`

## 📚 참고 자료

- 전략 문서: \`VERIFICATION_STRATEGY.md\`
- 테스트 가이드: \`README_TESTS.md\`
- 매핑 테이블: \`CONVERSION_MAPPING.md\`
- 기존 테스트: \`mnpbem/examples/test_step*.py\`
EOF

echo -e "${GREEN}✅ 리포트 생성 완료: VERIFICATION_REPORT.md${NC}"
echo ""

# 최종 요약
echo "================================================================"
echo -e "${GREEN}✅ 전수조사 준비 완료!${NC}"
echo "================================================================"
echo ""
echo "생성된 파일:"
echo "  📄 VERIFICATION_STRATEGY.md   - 전략 문서"
echo "  📄 VERIFICATION_REPORT.md     - 실행 리포트"
echo "  📄 CONVERSION_MAPPING.md      - 변환 상태 매핑"
echo "  📄 conversion_mapping.json    - JSON 데이터"
echo "  📁 tests/                     - 테스트 프레임워크"
echo "  📁 matlab_references/         - MATLAB 기준 생성 스크립트"
echo "  📁 tools/                     - 자동화 도구"
echo ""
echo "다음 단계:"
echo "  1. VERIFICATION_REPORT.md 확인"
echo "  2. MATLAB 기준 데이터 생성 (matlab_references/)"
echo "  3. Python 테스트 구현 (tests/unit/)"
echo "  4. pytest 실행"
echo ""
echo "상세 가이드: README_TESTS.md"
echo "================================================================"

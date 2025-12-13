#!/usr/bin/env python3
"""
MATLAB 코드베이스 자동 분석 도구

목적: MATLAB 클래스와 메소드를 자동으로 추출하여 변환 상태 추적 테이블 생성
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json


class MATLABCodeAnalyzer:
    """MATLAB 코드베이스 분석기"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.classes = defaultdict(dict)

    def find_matlab_classes(self):
        """@ 디렉토리 형태의 MATLAB 클래스 찾기"""
        class_dirs = []
        for item in self.root_dir.rglob('@*'):
            if item.is_dir():
                class_dirs.append(item)
        return sorted(class_dirs)

    def extract_methods(self, class_dir):
        """클래스의 모든 메소드 추출"""
        methods = []
        class_name = class_dir.name.lstrip('@')

        # .m 파일 찾기
        for m_file in class_dir.glob('*.m'):
            method_info = self.parse_method_file(m_file, class_name)
            if method_info:
                methods.append(method_info)

        return methods

    def parse_method_file(self, m_file, class_name):
        """개별 .m 파일에서 메소드 정보 추출"""
        try:
            with open(m_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # function 시그니처 찾기
            # 패턴: function [output] = methodname(inputs) 또는 function output = methodname(inputs)
            function_pattern = r'^\s*function\s+(?:\[([^\]]+)\]\s*=\s*|(\w+)\s*=\s*)?(\w+)\s*\(([^)]*)\)'
            match = re.search(function_pattern, content, re.MULTILINE)

            if match:
                outputs_bracket = match.group(1)  # [out1, out2]
                output_single = match.group(2)    # out
                method_name = match.group(3)
                inputs = match.group(4)

                # 출력 파라미터 정리
                if outputs_bracket:
                    outputs = [o.strip() for o in outputs_bracket.split(',')]
                elif output_single:
                    outputs = [output_single.strip()]
                else:
                    outputs = []

                # 입력 파라미터 정리
                if inputs.strip():
                    input_params = [i.strip() for i in inputs.split(',')]
                else:
                    input_params = []

                # docstring 추출 (% 주석)
                doc_pattern = r'^\s*%+\s*(.+)$'
                doc_lines = re.findall(doc_pattern, content, re.MULTILINE)
                docstring = '\n'.join(doc_lines[:5]) if doc_lines else ''  # 처음 5줄만

                return {
                    'name': method_name,
                    'file': m_file.name,
                    'inputs': input_params,
                    'outputs': outputs,
                    'docstring': docstring,
                    'is_constructor': method_name == class_name,
                    'lines': len(content.splitlines())
                }
            else:
                # function 정의가 없는 경우 (스크립트 파일)
                return {
                    'name': m_file.stem,
                    'file': m_file.name,
                    'inputs': [],
                    'outputs': [],
                    'docstring': '',
                    'is_constructor': False,
                    'lines': len(content.splitlines())
                }

        except Exception as e:
            print(f"Warning: Could not parse {m_file}: {e}")
            return None

    def analyze(self):
        """전체 코드베이스 분석"""
        print("🔍 Analyzing MATLAB codebase...")

        class_dirs = self.find_matlab_classes()
        print(f"Found {len(class_dirs)} MATLAB classes")

        for class_dir in class_dirs:
            class_name = class_dir.name.lstrip('@')
            parent_module = class_dir.parent.name

            methods = self.extract_methods(class_dir)

            self.classes[parent_module][class_name] = {
                'path': str(class_dir.relative_to(self.root_dir)),
                'methods': methods,
                'total_methods': len(methods),
                'total_lines': sum(m['lines'] for m in methods)
            }

            print(f"  📦 {parent_module}/{class_name}: {len(methods)} methods, "
                  f"{sum(m['lines'] for m in methods)} lines")

        return self.classes

    def generate_mapping_table(self, python_dir):
        """Python 변환 상태 매핑 테이블 생성"""
        python_path = Path(python_dir)
        mapping = []

        for module, classes in self.classes.items():
            for class_name, class_info in classes.items():
                for method in class_info['methods']:
                    # Python 파일 존재 여부 확인
                    python_status = self.check_python_conversion(
                        python_path, module, class_name, method['name']
                    )

                    mapping.append({
                        'matlab_module': module,
                        'matlab_class': class_name,
                        'matlab_method': method['name'],
                        'matlab_file': method['file'],
                        'matlab_lines': method['lines'],
                        'is_constructor': method['is_constructor'],
                        'python_status': python_status['status'],
                        'python_file': python_status['file'],
                        'python_class': python_status['class'],
                        'python_method': python_status['method'],
                        'test_status': python_status['test_status']
                    })

        return mapping

    def check_python_conversion(self, python_dir, module, matlab_class, matlab_method):
        """Python 변환 여부 확인"""
        # 모듈 이름 매핑 (MATLAB → Python)
        module_map = {
            'BEM': 'bem',
            'Greenfun': 'greenfun',
            'Material': 'materials',
            'Particles': 'geometry',
            'Simulation': 'excitation',
            'Solver': 'spectrum'
        }

        python_module = module_map.get(module, module.lower())
        python_class_name = self.to_python_class_name(matlab_class)
        python_method_name = self.to_python_method_name(matlab_method, matlab_class)

        # Python 파일 찾기
        possible_files = [
            python_dir / python_module / f"{matlab_class.lower()}.py",
            python_dir / python_module / f"{self.to_snake_case(matlab_class)}.py",
        ]

        for py_file in possible_files:
            if py_file.exists():
                # 파일 내에서 메소드 찾기
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 클래스와 메소드 존재 확인
                class_pattern = rf'class\s+{python_class_name}\s*[\(:]'
                method_pattern = rf'def\s+{python_method_name}\s*\('

                has_class = re.search(class_pattern, content) is not None
                has_method = re.search(method_pattern, content) is not None

                if has_class and has_method:
                    # 테스트 존재 여부 확인
                    test_status = self.check_test_exists(
                        python_dir.parent, python_module, python_class_name, python_method_name
                    )

                    return {
                        'status': '✅',
                        'file': str(py_file.relative_to(python_dir)),
                        'class': python_class_name,
                        'method': python_method_name,
                        'test_status': test_status
                    }
                elif has_class:
                    return {
                        'status': '⚠️',
                        'file': str(py_file.relative_to(python_dir)),
                        'class': python_class_name,
                        'method': f'Missing: {python_method_name}',
                        'test_status': '⬜'
                    }

        return {
            'status': '⬜',
            'file': 'Not converted',
            'class': python_class_name,
            'method': python_method_name,
            'test_status': '⬜'
        }

    def check_test_exists(self, project_root, module, class_name, method_name):
        """테스트 존재 여부 확인"""
        project_root = Path(project_root)

        # 테스트 파일 경로들
        test_files = []

        # tests/unit/ 에서 찾기
        unit_test = project_root / 'tests' / 'unit' / f'test_{module}.py'
        if unit_test.exists():
            test_files.append(unit_test)

        # mnpbem/examples/ 에서 step 테스트 찾기
        examples_dir = project_root / 'mnpbem' / 'examples'
        if examples_dir.exists():
            for test_file in examples_dir.glob('test_step*.py'):
                test_files.append(test_file)

        # 각 테스트 파일 확인
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 테스트 함수 찾기
                test_func_pattern = rf'def\s+test.*{method_name}'
                if re.search(test_func_pattern, content, re.IGNORECASE):
                    return '✅'
            except Exception:
                continue

        return '⬜'

    @staticmethod
    def to_python_class_name(matlab_class):
        """MATLAB 클래스명을 Python 스타일로 변환"""
        # bemstat → BEMStat, compgreenstat → CompGreenStat
        return ''.join(word.capitalize() for word in re.split(r'(?=[A-Z])|_', matlab_class))

    @staticmethod
    def to_python_method_name(matlab_method, matlab_class):
        """MATLAB 메소드명을 Python 스타일로 변환"""
        # Constructor: bemstat → __init__
        if matlab_method == matlab_class:
            return '__init__'
        # Others: solve → solve, getFields → get_fields
        return matlab_method

    @staticmethod
    def to_snake_case(name):
        """CamelCase를 snake_case로 변환"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def export_markdown(self, mapping):
        """마크다운 테이블로 출력"""
        md = "# MATLAB → Python 변환 상태 매핑\n\n"
        md += f"**생성일**: {self.get_timestamp()}\n\n"

        # 통계
        total = len(mapping)
        converted = sum(1 for m in mapping if m['python_status'] == '✅')
        partial = sum(1 for m in mapping if m['python_status'] == '⚠️')
        not_converted = sum(1 for m in mapping if m['python_status'] == '⬜')
        tested = sum(1 for m in mapping if m['test_status'] == '✅')

        md += "## 📊 변환 통계\n\n"
        md += f"- **전체 메소드**: {total}\n"
        md += f"- **변환 완료**: {converted} ({converted/total*100:.1f}%)\n"
        md += f"- **부분 변환**: {partial} ({partial/total*100:.1f}%)\n"
        md += f"- **미변환**: {not_converted} ({not_converted/total*100:.1f}%)\n"
        md += f"- **테스트 커버리지**: {tested} ({tested/total*100:.1f}%)\n\n"

        md += "## 📋 상세 매핑 테이블\n\n"
        md += "| MATLAB Module | MATLAB Class | MATLAB Method | Lines | Python Status | Python File | Python Method | Test |\n"
        md += "|---------------|--------------|---------------|-------|---------------|-------------|---------------|------|\n"

        for m in sorted(mapping, key=lambda x: (x['matlab_module'], x['matlab_class'], x['matlab_method'])):
            md += f"| {m['matlab_module']} | {m['matlab_class']} | {m['matlab_method']} | "
            md += f"{m['matlab_lines']} | {m['python_status']} | {m['python_file']} | "
            md += f"{m['python_method']} | {m['test_status']} |\n"

        return md

    @staticmethod
    def get_timestamp():
        """현재 시각 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def export_json(self, mapping, output_file):
        """JSON 형식으로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON exported to {output_file}")


def main():
    """메인 실행 함수"""
    import sys

    if len(sys.argv) < 2:
        matlab_root = '/home/user/MNPBEM'
        python_root = '/home/user/MNPBEM/mnpbem'
    else:
        matlab_root = sys.argv[1]
        python_root = sys.argv[2] if len(sys.argv) > 2 else f"{matlab_root}/mnpbem"

    print("=" * 60)
    print("MATLAB → Python 변환 상태 분석 도구")
    print("=" * 60)
    print(f"MATLAB Root: {matlab_root}")
    print(f"Python Root: {python_root}")
    print()

    # 분석 실행
    analyzer = MATLABCodeAnalyzer(matlab_root)
    classes = analyzer.analyze()

    print()
    print("=" * 60)
    print("📊 분석 완료 - 통계")
    print("=" * 60)
    total_classes = sum(len(c) for c in classes.values())
    total_methods = sum(c['total_methods'] for module in classes.values() for c in module.values())
    total_lines = sum(c['total_lines'] for module in classes.values() for c in module.values())

    print(f"총 모듈: {len(classes)}")
    print(f"총 클래스: {total_classes}")
    print(f"총 메소드: {total_methods}")
    print(f"총 코드 라인: {total_lines:,}")
    print()

    # 매핑 테이블 생성
    print("=" * 60)
    print("🗺️  변환 상태 매핑 생성 중...")
    print("=" * 60)
    mapping = analyzer.generate_mapping_table(python_root)

    # 마크다운 출력
    md_content = analyzer.export_markdown(mapping)
    md_file = Path(matlab_root) / 'CONVERSION_MAPPING.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✅ Markdown exported to {md_file}")

    # JSON 출력
    json_file = Path(matlab_root) / 'conversion_mapping.json'
    analyzer.export_json(mapping, json_file)

    print()
    print("=" * 60)
    print("✅ 분석 완료!")
    print("=" * 60)
    print(f"결과 파일:")
    print(f"  - {md_file}")
    print(f"  - {json_file}")


if __name__ == '__main__':
    main()

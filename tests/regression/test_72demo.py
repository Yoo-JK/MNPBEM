import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List

import pytest


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_72DEMO',
    '/home/yoojk20/scratch/mnpbem_validation/72demos_validation'))


def _load_accuracy_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mre = float(row['max_rel_err'])
            except (KeyError, ValueError):
                continue
            rows.append({
                'demo': row.get('demo', ''),
                'demo_type': row.get('demo_type', ''),
                'max_rel_err': mre,
                'classification': row.get('classification', ''),
            })
    return rows


@pytest.fixture(scope = 'module')
def accuracy_rows() -> List[Dict[str, Any]]:
    csv_path = VALIDATION_BASE / 'data' / 'accuracy_v2.csv'
    rows = _load_accuracy_csv(csv_path)
    if not rows:
        pytest.skip('[info] 72demo accuracy_v2.csv not available at {}'.format(csv_path))
    return rows


@pytest.fixture(scope = 'module')
def grade_counts(accuracy_rows, classifier) -> Dict[str, int]:
    counts = {'machine_precision': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    for r in accuracy_rows:
        counts[classifier(r['max_rel_err'])] += 1
    return counts


@pytest.mark.fast
def test_reference_loaded(reference_72demo):
    assert reference_72demo['total'] == 72
    assert reference_72demo['requirements']['BAD_count_max'] == 0


@pytest.mark.fast
def test_classifier_thresholds(classifier):
    assert classifier(1e-15) == 'machine_precision'
    assert classifier(1e-13) == 'machine_precision'
    assert classifier(1e-10) == 'OK'
    assert classifier(1e-5) == 'good'
    assert classifier(5e-4) == 'warn'
    assert classifier(1e-2) == 'BAD'


@pytest.mark.fast
def test_72demo_dataset_structure(accuracy_rows):
    assert len(accuracy_rows) == 72
    for r in accuracy_rows:
        assert r['demo'].startswith('demo')
        assert r['max_rel_err'] >= 0


@pytest.mark.slow
def test_72demo_machine_precision_count(grade_counts, reference_72demo):
    required = reference_72demo['requirements']['machine_precision_count_min']
    actual = grade_counts['machine_precision']
    assert actual >= required, (
        '[error] 72demo machine_precision={} < required {}'.format(actual, required))


@pytest.mark.slow
def test_72demo_no_BAD(grade_counts, reference_72demo):
    allowed = reference_72demo['requirements']['BAD_count_max']
    actual = grade_counts['BAD']
    assert actual <= allowed, (
        '[error] 72demo BAD={} > allowed {}'.format(actual, allowed))


@pytest.mark.slow
def test_72demo_total_count(accuracy_rows):
    assert len(accuracy_rows) == 72


@pytest.mark.slow
def test_72demo_speedup_reference_consistency(reference_72demo):
    cpu = reference_72demo['speedup']['cpu_geo_mean']
    gpu = reference_72demo['speedup']['gpu_geo_mean']
    assert cpu >= reference_72demo['speedup']['cpu_geo_mean_min']
    assert gpu >= reference_72demo['speedup']['gpu_geo_mean_min']


@pytest.mark.slow
def test_72demo_per_demo_warn_threshold(accuracy_rows, classifier):
    bad_demos = [r['demo'] for r in accuracy_rows if classifier(r['max_rel_err']) == 'BAD']
    assert bad_demos == [], '[error] BAD demos found: {}'.format(bad_demos)

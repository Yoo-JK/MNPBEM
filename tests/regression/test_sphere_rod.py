import os
import csv
from pathlib import Path
from typing import Dict, Any, List

import pytest


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_SPHERE_ROD',
    '/home/yoojk20/scratch/mnpbem_validation/sphere_rod_validation'))


KNOWN_XFAIL_KEYS = {
    ('sphere', '04_bemstat_layer/normal'),
    ('sphere', '04_bemstat_layer/oblique'),
    ('sphere', '05_bemret_layer'),
    ('sphere', '07_eigenmode'),
    ('rod', '07_eigenmode'),
    ('rod_lying', '03_bemret/layer'),
    ('rod_lying', '03_bemret/nolayer'),
    ('rod_lying', '07_eigenmode'),
}


def _load_summary_csv(path: Path) -> List[Dict[str, Any]]:
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
                'shape': row.get('shape', ''),
                'category': row.get('category', ''),
                'max_rel_err': mre,
                'status': row.get('status', ''),
            })
    return rows


@pytest.fixture(scope = 'module')
def sphere_rod_rows() -> List[Dict[str, Any]]:
    csv_path = VALIDATION_BASE / 'summary_table.csv'
    rows = _load_summary_csv(csv_path)
    if not rows:
        pytest.skip('[info] sphere_rod summary not available at {}'.format(csv_path))
    return rows


@pytest.fixture(scope = 'module')
def grade_counts(sphere_rod_rows, classifier) -> Dict[str, int]:
    counts = {'machine_precision': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    for r in sphere_rod_rows:
        counts[classifier(r['max_rel_err'])] += 1
    return counts


@pytest.mark.fast
def test_reference_loaded(reference_sphere_rod):
    assert reference_sphere_rod['total'] == 51
    assert reference_sphere_rod['requirements']['machine_precision_count_min'] == 35


@pytest.mark.fast
def test_xfail_table_consistent(reference_sphere_rod):
    expected = set(reference_sphere_rod['known_xfail_categories'])
    actual = {c for _, c in KNOWN_XFAIL_KEYS}
    assert expected.intersection(actual), (
        '[error] xfail categories should overlap with reference')


@pytest.mark.slow
def test_sphere_rod_total(sphere_rod_rows):
    assert len(sphere_rod_rows) == 51


@pytest.mark.slow
def test_sphere_rod_machine_precision_count(grade_counts, reference_sphere_rod):
    required = reference_sphere_rod['requirements']['machine_precision_count_min']
    actual = grade_counts['machine_precision']
    assert actual >= required, (
        '[error] sphere_rod machine_precision={} < required {}'.format(actual, required))


@pytest.mark.slow
def test_sphere_rod_BAD_within_known(sphere_rod_rows, classifier):
    bad = [(r['shape'], r['category']) for r in sphere_rod_rows if classifier(r['max_rel_err']) == 'BAD']
    unknown = [b for b in bad if b not in KNOWN_XFAIL_KEYS]
    assert unknown == [], (
        '[error] sphere_rod BAD outside xfail list: {}'.format(unknown))


@pytest.mark.slow
def test_sphere_rod_per_shape_minimum(grade_counts, sphere_rod_rows, classifier, reference_sphere_rod):
    per_shape = {'sphere': 0, 'rod': 0, 'rod_lying': 0}
    for r in sphere_rod_rows:
        if classifier(r['max_rel_err']) == 'machine_precision' and r['shape'] in per_shape:
            per_shape[r['shape']] += 1

    expected = reference_sphere_rod['machine_precision_per_shape']
    for shape, count in per_shape.items():
        ref_count = expected.get(shape, 0)
        assert count >= ref_count, (
            '[error] {} machine_precision={} < reference {}'.format(shape, count, ref_count))

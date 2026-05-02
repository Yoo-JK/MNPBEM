import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_SPHERE_ROD',
    '/home/yoojk20/scratch/mnpbem_validation/sphere_rod_validation'))


KNOWN_XFAIL = {
    ('sphere', '04_bemstat_layer/normal'),
    ('sphere', '04_bemstat_layer/oblique'),
    ('sphere', '05_bemret_layer'),
    ('sphere', '07_eigenmode'),
    ('rod', '07_eigenmode'),
    ('rod_lying', '03_bemret/layer'),
    ('rod_lying', '03_bemret/nolayer'),
    ('rod_lying', '07_eigenmode'),
}


def classify(rel_err: float) -> str:
    if rel_err < 1e-12:
        return 'machine_precision'
    if rel_err < 1e-6:
        return 'OK'
    if rel_err < 1e-4:
        return 'good'
    if rel_err < 1e-3:
        return 'warn'
    return 'BAD'


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            try:
                mre = float(row['max_rel_err'])
            except (KeyError, ValueError):
                continue
            rows.append({
                'shape': row.get('shape', ''),
                'category': row.get('category', ''),
                'max_rel_err': mre,
            })
    return rows


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description = 'sphere/rod 51-case regression runner')
    p.add_argument('--json', type = Path, default = Path('result_sphere_rod.json'))
    args = p.parse_args(argv)

    csv_path = VALIDATION_BASE / 'summary_table.csv'
    rows = load_rows(csv_path)

    counts = {'machine_precision': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    bad_inside = 0
    bad_outside = []
    per_shape = {'sphere': 0, 'rod': 0, 'rod_lying': 0}

    for r in rows:
        cls = classify(r['max_rel_err'])
        counts[cls] += 1
        if cls == 'machine_precision' and r['shape'] in per_shape:
            per_shape[r['shape']] += 1
        if cls == 'BAD':
            key = (r['shape'], r['category'])
            if key in KNOWN_XFAIL:
                bad_inside += 1
            else:
                bad_outside.append({'shape': r['shape'], 'category': r['category'],
                        'max_rel_err': r['max_rel_err']})

    result = {
        'machine_precision_count': counts['machine_precision'],
        'ok_count': counts['OK'],
        'good_count': counts['good'],
        'warn_count': counts['warn'],
        'bad_count': counts['BAD'],
        'bad_inside_xfail': bad_inside,
        'bad_outside_xfail': bad_outside,
        'machine_precision_per_shape': per_shape,
        'n_cases': len(rows),
    }

    args.json.parent.mkdir(parents = True, exist_ok = True)
    with open(args.json, 'w') as f:
        json.dump(result, f, indent = 2)

    print('[info] wrote {}'.format(args.json))
    print(json.dumps(result, indent = 2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

import os
import sys
import csv
import json
import argparse
import math
from pathlib import Path
from typing import Dict, Any, List


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_72DEMO',
    '/home/yoojk20/scratch/mnpbem_validation/72demos_validation'))


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


def load_accuracy(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mre = float(row['max_rel_err'])
            except (KeyError, ValueError):
                continue
            rows.append({
                'demo': row.get('demo', ''),
                'max_rel_err': mre,
                'demo_type': row.get('demo_type', ''),
            })
    return rows


def load_summary(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {}
            for k, v in row.items():
                rec[k] = v
            try:
                rec['cpu_speedup'] = float(rec.get('cpu_speedup', 'nan'))
            except ValueError:
                rec['cpu_speedup'] = float('nan')
            try:
                rec['gpu_speedup'] = float(rec.get('gpu_speedup', 'nan'))
            except ValueError:
                rec['gpu_speedup'] = float('nan')
            try:
                rec['cpu_sec'] = float(rec.get('cpu_sec', 'nan'))
            except ValueError:
                rec['cpu_sec'] = float('nan')
            try:
                rec['gpu_sec'] = float(rec.get('gpu_sec', 'nan'))
            except ValueError:
                rec['gpu_sec'] = float('nan')
            try:
                rec['matlab_sec'] = float(rec.get('matlab_sec', 'nan'))
            except ValueError:
                rec['matlab_sec'] = float('nan')
            rows.append(rec)
    return rows


def geo_mean(vals: List[float]) -> float:
    cleaned = [v for v in vals if v is not None and not math.isnan(v) and v > 0]
    if not cleaned:
        return float('nan')
    s = sum(math.log(v) for v in cleaned)
    return math.exp(s / len(cleaned))


def build_result() -> Dict[str, Any]:
    acc_path = VALIDATION_BASE / 'data' / 'accuracy_v2.csv'
    sum_path = VALIDATION_BASE / 'summary_table.csv'

    acc_rows = load_accuracy(acc_path)
    sum_rows = load_summary(sum_path) if sum_path.exists() else []

    counts = {'machine_precision': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    for r in acc_rows:
        counts[classify(r['max_rel_err'])] += 1

    cpu_geo = geo_mean([r['cpu_speedup'] for r in sum_rows])
    gpu_geo = geo_mean([r['gpu_speedup'] for r in sum_rows])

    total_matlab_s = sum(r['matlab_sec'] for r in sum_rows if not math.isnan(r['matlab_sec']))
    total_cpu_s = sum(r['cpu_sec'] for r in sum_rows if not math.isnan(r['cpu_sec']))
    total_gpu_s = sum(r['gpu_sec'] for r in sum_rows if not math.isnan(r['gpu_sec']))

    return {
        'machine_precision_count': counts['machine_precision'],
        'ok_count': counts['OK'],
        'good_count': counts['good'],
        'warn_count': counts['warn'],
        'bad_count': counts['BAD'],
        'n_demos': len(acc_rows),
        'cpu_speedup': round(cpu_geo, 3) if not math.isnan(cpu_geo) else None,
        'gpu_speedup': round(gpu_geo, 3) if not math.isnan(gpu_geo) else None,
        'matlab_total_min': round(total_matlab_s / 60.0, 2),
        'cpu_total_min': round(total_cpu_s / 60.0, 2),
        'gpu_total_min': round(total_gpu_s / 60.0, 2),
    }


def build_sphere_rod() -> Dict[str, Any]:
    base = Path(os.environ.get(
        'MNPBEM_VALIDATION_SPHERE_ROD',
        '/home/yoojk20/scratch/mnpbem_validation/sphere_rod_validation'))
    summ = base / 'summary_table.csv'
    if not summ.exists():
        return {}
    counts = {'machine_precision': 0, 'OK': 0, 'good': 0, 'warn': 0, 'BAD': 0}
    n = 0
    with open(summ, 'r') as f:
        for row in csv.DictReader(f):
            try:
                mre = float(row['max_rel_err'])
            except (KeyError, ValueError):
                continue
            counts[classify(mre)] += 1
            n += 1
    return {
        'machine_precision_count': counts['machine_precision'],
        'ok_count': counts['OK'],
        'good_count': counts['good'],
        'warn_count': counts['warn'],
        'bad_count': counts['BAD'],
        'n_cases': n,
    }


def build_dimer() -> Dict[str, Any]:
    base = Path(os.environ.get(
        'MNPBEM_VALIDATION_DIMER',
        '/home/yoojk20/scratch/mnpbem_validation/dimer_benchmark'))
    final_v4 = base / 'data' / 'final_v4.json'
    if not final_v4.exists():
        return {}
    with open(final_v4, 'r') as f:
        d = json.load(f)
    max_rel = d['accuracy_vs_matlab_ref']['ext_x']['max_rel']
    return {
        'max_drift_pct': round(d['peak_resonance']['diff_pct_ext_x'], 6),
        'spectrum_max_rel': max_rel,
        'wall_min': round(d['wall_min'], 2),
        'bad_count': 1 if max_rel >= 1e-3 else 0,
    }


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description = '72-demo regression runner (CI artefact)')
    p.add_argument('--json', type = Path, default = Path('result.json'),
            help = 'path to write the result JSON')
    args = p.parse_args(argv)

    result = {
        '72demo': build_result(),
        'sphere_rod': build_sphere_rod(),
        'dimer': build_dimer(),
    }

    args.json.parent.mkdir(parents = True, exist_ok = True)
    with open(args.json, 'w') as f:
        json.dump(result, f, indent = 2)

    print('[info] wrote {}'.format(args.json))
    print(json.dumps(result, indent = 2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

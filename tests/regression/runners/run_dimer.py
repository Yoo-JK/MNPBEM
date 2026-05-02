import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_DIMER',
    '/home/yoojk20/scratch/mnpbem_validation/dimer_benchmark'))


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description = 'dimer 4-case regression runner')
    p.add_argument('--json', type = Path, default = Path('result_dimer.json'))
    args = p.parse_args(argv)

    final_v4 = VALIDATION_BASE / 'data' / 'final_v4.json'
    with open(final_v4, 'r') as f:
        v4 = json.load(f)

    spec = v4['accuracy_vs_matlab_ref']
    peak = v4['peak_resonance']

    cases = []
    for variant in ['baseline_cpu.json', 'baseline_gpu.json', 'final_v3.json', 'final_v4.json']:
        p2 = VALIDATION_BASE / 'data' / variant
        if not p2.exists():
            continue
        with open(p2, 'r') as f:
            d = json.load(f)
        cases.append({
            'name': variant.replace('.json', ''),
            'wall_min': d.get('wall_min', None),
            'multi_gpu': d.get('multi_gpu', None),
            'use_gpu': d.get('use_gpu', None),
            'hmode': d.get('hmode', None),
        })

    max_rel = spec['ext_x']['max_rel']
    result = {
        'max_drift_pct': round(peak['diff_pct_ext_x'], 6),
        'spectrum_max_rel_ext_x': max_rel,
        'spectrum_mean_rel_ext_x': spec['ext_x']['mean_rel'],
        'spectrum_max_rel_sca_x': spec['sca_x']['max_rel'],
        'wall_min_GPU4x_dense': round(v4['wall_min'], 2),
        'bad_count': 1 if max_rel >= 1e-3 else 0,
        'n_face': v4.get('nfaces', 6336),
        'cases': cases,
    }

    args.json.parent.mkdir(parents = True, exist_ok = True)
    with open(args.json, 'w') as f:
        json.dump(result, f, indent = 2)

    print('[info] wrote {}'.format(args.json))
    print(json.dumps(result, indent = 2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

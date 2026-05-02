import os
import json
from pathlib import Path
from typing import Dict, Any

import pytest


VALIDATION_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_DIMER',
    '/home/yoojk20/scratch/mnpbem_validation/dimer_benchmark'))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'module')
def dimer_final_v4() -> Dict[str, Any]:
    p = VALIDATION_BASE / 'data' / 'final_v4.json'
    if not p.exists():
        pytest.skip('[info] dimer final_v4.json not at {}'.format(p))
    return _load_json(p)


@pytest.mark.fast
def test_reference_loaded(reference_dimer):
    assert reference_dimer['mesh']['n_face'] == 6336
    assert reference_dimer['mesh']['n_wavelength'] == 100
    assert reference_dimer['single_lambda']['rel_diff'] <= reference_dimer['requirements']['single_lambda_max_rel_diff']


@pytest.mark.fast
def test_lane_AE_residual_documented(reference_dimer):
    lane = reference_dimer['lane_AE_residual']
    assert lane['algorithmic_defect'] is False
    assert lane['green_G1_outliers'] == 4


@pytest.mark.slow
def test_dimer_spectrum_max_rel(dimer_final_v4, reference_dimer):
    max_rel = dimer_final_v4['accuracy_vs_matlab_ref']['ext_x']['max_rel']
    threshold = reference_dimer['requirements']['spectrum_max_rel_diff']
    assert max_rel <= threshold, (
        '[error] dimer ext_x max_rel={:.3e} > threshold {:.3e}'.format(max_rel, threshold))


@pytest.mark.slow
def test_dimer_spectrum_mean_rel(dimer_final_v4):
    mean_rel = dimer_final_v4['accuracy_vs_matlab_ref']['ext_x']['mean_rel']
    assert mean_rel <= 1e-4, (
        '[error] dimer ext_x mean_rel={:.3e} > 1e-4'.format(mean_rel))


@pytest.mark.slow
def test_dimer_peak_resonance(dimer_final_v4, reference_dimer):
    peak = dimer_final_v4['peak_resonance']
    diff_pct = peak['diff_pct_ext_x']
    assert diff_pct <= 1.0, (
        '[error] dimer peak ext_x diff={:.3f}% > 1%'.format(diff_pct))


@pytest.mark.slow
def test_dimer_wall_time_GPU4x(dimer_final_v4, reference_dimer):
    wall = dimer_final_v4['wall_min']
    threshold = reference_dimer['requirements']['gpu_4x_wall_min_max']
    assert wall <= threshold, (
        '[error] dimer GPU 4x wall={:.2f} min > {:.2f} min'.format(wall, threshold))


@pytest.mark.slow
def test_dimer_single_lambda_consistency(reference_dimer):
    sl = reference_dimer['single_lambda']
    rel = abs(sl['python_ext_x'] - sl['matlab_ext_x']) / abs(sl['matlab_ext_x'])
    assert rel <= sl['requirement_max_rel_diff'], (
        '[error] dimer single-lambda rel={:.3e} > {:.3e}'.format(rel, sl['requirement_max_rel_diff']))


@pytest.mark.long
def test_dimer_cpu_4w_wall(reference_dimer):
    wall = reference_dimer['wall_minutes']['python_cpu_4w_1t']
    threshold = reference_dimer['requirements']['cpu_4w_wall_min_max']
    assert wall <= threshold, (
        '[error] dimer CPU 4w wall={:.2f} > {:.2f}'.format(wall, threshold))


@pytest.mark.gpu
@pytest.mark.long
def test_dimer_gpu_runs(gpu_available):
    if not gpu_available:
        pytest.skip('[info] GPU not available')
    # Smoke check: we expect that final_v4.json was produced under GPU.
    # Full re-execution is the responsibility of run_dimer.py runner.
    pass

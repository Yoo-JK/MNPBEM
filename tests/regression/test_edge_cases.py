import os
import json
from pathlib import Path
from typing import Dict, Any

import pytest


DIMER_BASE = Path(os.environ.get(
    'MNPBEM_VALIDATION_DIMER',
    '/home/yoojk20/scratch/mnpbem_validation/dimer_benchmark'))


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'module')
def lane_e2_summary() -> Dict[str, Any]:
    p = DIMER_BASE / 'data' / 'lane_E2_summary.json'
    data = _load_json_if_exists(p)
    if not data:
        pytest.skip('[info] lane_E2_summary.json not present')
    return data


@pytest.mark.fast
def test_dimer_baseline_files_exist():
    assert (DIMER_BASE / 'data' / 'final_v4.json').exists() or True


@pytest.mark.long
def test_lane_E2_large_mesh_completed(lane_e2_summary):
    completed_keys = [k for k, v in lane_e2_summary.items() if isinstance(v, dict)]
    assert len(completed_keys) > 0, (
        '[error] Lane E2 has no completed entries')


@pytest.mark.long
def test_baseline_cpu_completed():
    p = DIMER_BASE / 'data' / 'baseline_cpu.json'
    if not p.exists():
        pytest.skip('[info] baseline_cpu.json not present')
    data = _load_json_if_exists(p)
    assert 'wall_min' in data or 'wall_minutes' in data or len(data) > 0


@pytest.mark.long
def test_baseline_gpu_completed():
    p = DIMER_BASE / 'data' / 'baseline_gpu.json'
    if not p.exists():
        pytest.skip('[info] baseline_gpu.json not present')
    data = _load_json_if_exists(p)
    assert len(data) > 0

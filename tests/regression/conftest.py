import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

import pytest


REGRESSION_DIR = Path(__file__).resolve().parent
DATA_DIR = REGRESSION_DIR / 'data'


def pytest_configure(config):
    config.addinivalue_line('markers', 'fast: < 1 minute, runs every commit')
    config.addinivalue_line('markers', 'slow: 10-60 minutes, daily nightly')
    config.addinivalue_line('markers', 'long: > 60 minutes, weekly')
    config.addinivalue_line('markers', 'gpu: requires CUDA + cupy')

    # honour optional MNPBEM_REGRESSION_FAST env var to lock to fast
    if os.environ.get('MNPBEM_REGRESSION_FAST', '0') == '1':
        config.option.markexpr = (config.option.markexpr or '') + ' and fast'


@pytest.fixture(scope = 'session')
def regression_data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope = 'session')
def reference_72demo() -> Dict[str, Any]:
    path = DATA_DIR / 'matlab_72demo_reference.json'
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'session')
def reference_sphere_rod() -> Dict[str, Any]:
    path = DATA_DIR / 'sphere_rod_reference.json'
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'session')
def reference_dimer() -> Dict[str, Any]:
    path = DATA_DIR / 'dimer_reference.json'
    with open(path, 'r') as f:
        return json.load(f)


@pytest.fixture(scope = 'session')
def gpu_available() -> bool:
    try:
        import cupy
        try:
            cupy.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            return False
    except ImportError:
        return False


def classify_rel_err(rel_err: float) -> str:
    if rel_err < 1e-12:
        return 'machine_precision'
    if rel_err < 1e-6:
        return 'OK'
    if rel_err < 1e-4:
        return 'good'
    if rel_err < 1e-3:
        return 'warn'
    return 'BAD'


@pytest.fixture(scope = 'session')
def classifier():
    return classify_rel_err

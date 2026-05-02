# Installation Guide

This document covers every supported install path for the MNPBEM Python
port — from a minimal CPU-only setup to a full multi-node, multi-GPU
deployment.

## TL;DR

| Scenario | Command |
|---|---|
| CPU only (simplest, most portable) | `pip install mnpbem` |
| Single GPU acceleration | `pip install "mnpbem[gpu]"` |
| Multi-GPU + multi-node MPI | `pip install "mnpbem[gpu,mpi]"` |
| FMM acceleration | `pip install "mnpbem[fmm]"` |
| Everything | `pip install "mnpbem[all]"` |
| Developer / contributor | `pip install -e ".[all,dev]"` |
| Documentation build | `pip install "mnpbem[docs]"` |

A single wheel ships every Python module. Optional extras only pull
additional Python dependencies (cupy / mpi4py / fmm3dpy / sphinx). The
runtime auto-detects what is available and falls back to a pure NumPy
path when GPU dependencies are missing — see Section "Runtime detection".

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11 | 3.12 supported. 3.13/3.14 will be added once cupy releases wheels. |
| OS | Linux x86_64 | Tested on Ubuntu 22.04. macOS / Windows: CPU-only path is expected to work; not currently in CI. |
| CUDA | 12.x | Required only for `[gpu]`. Driver: 525+ for CUDA 12. |
| OpenMPI | 4.0+ | Required only for `[mpi]`. MPICH 3.4+ also works. |
| C/C++ build tools | gcc 9+ | Only needed if `mpi4py` builds from source. |

The `[gpu]` extra installs `cupy-cuda12x`, which expects a working CUDA
12 runtime on the host. CUDA 11.x environments need to install a
matching `cupy-cuda11x` wheel manually — `pyproject.toml` does not pin
that variant by default.

## 2. Recommended: conda environment

```bash
conda create -n mnpbem python=3.11 -y
conda activate mnpbem

# CPU only (no GPU dependencies pulled at all)
pip install mnpbem

# OR: GPU acceleration
pip install "mnpbem[gpu]"

# OR: everything
pip install "mnpbem[all]"
```

For multi-node MPI, OS-level OpenMPI is required before installing
`mpi4py`. The cleanest option on a workstation is conda-forge:

```bash
conda install -c conda-forge openmpi mpi4py
pip install mnpbem  # mpi4py already provided by conda
```

On HPC clusters, prefer the site-provided OpenMPI module and let pip
build mpi4py against it:

```bash
module load openmpi/4.1.5
pip install "mnpbem[mpi]"
```

## 3. Editable / developer install

```bash
git clone <repo-url> MNPBEM
cd MNPBEM
pip install -e ".[all,dev]"
pytest -m fast tests/   # smoke check
```

The `dev` extra adds pytest, pytest-timeout, pytest-xdist, build, and
twine. Combine with `all` to also pull the GPU / MPI / FMM stacks.

## 4. Runtime detection

The package never errors at import time when an optional accelerator is
missing. Use the helper to verify what is active:

```python
from mnpbem.utils.gpu import has_gpu_capability, get_install_hint

if has_gpu_capability(verbose=True):
    print('GPU detected — set MNPBEM_GPU=1 to enable cupy dispatch')
else:
    print('Running CPU-only')
    print(get_install_hint())
```

`has_gpu_capability(verbose=True)` emits a `RuntimeWarning` describing
the exact reason GPU is unavailable (cupy not importable / no CUDA
device / runtime check failed).

If `MNPBEM_GPU=1` is set but cupy is missing, the helper
`require_gpu_or_raise()` raises a `RuntimeError` whose message includes
`get_install_hint()` so the user knows exactly which extra to install.

## 5. Environment variables

GPU and multi-GPU dispatch is opt-in via environment variables. Defaults
are conservative (CPU + single process) so a fresh install behaves
identically across hardware tiers.

| Variable | Default | Effect |
|---|---|---|
| `MNPBEM_GPU` | `0` | `1` enables cupy LU / GEMM dispatch (single GPU). |
| `MNPBEM_GPU_THRESHOLD` | `1500` | Matrix size below which CPU is used even when `MNPBEM_GPU=1`. |
| `MNPBEM_GPU_NATIVE` | `0` | `1` keeps tensors on device, removes host round-trip. |
| `MNPBEM_GPU_LAYER` | `1` | `0` disables GPU on Sommerfeld / layer-Green kernels. |
| `MNPBEM_VRAM_SHARE_GPUS` | `1` | `>=2` distributes a single LU across multiple GPUs (cuSolverMg). |
| `MNPBEM_VRAM_SHARE_BACKEND` | `cusolvermg` | `cusolvermg` / `magma` / `nccl`. |
| `MNPBEM_NUMBA` | `1` | `0` bypasses njit (debugging only). |

Multi-node MPI is detected automatically when the program is launched
under `mpirun` / `srun` (mpi4py is imported from inside
`mnpbem.utils.mpi_dispatch`). No env var toggle is required — single
process simply behaves as `COMM_WORLD.size == 1`.

## 6. Verifying the install

```bash
python -c "import mnpbem; print('mnpbem', mnpbem.__version__)"
python -c "from mnpbem.utils.gpu import has_gpu_capability; has_gpu_capability(verbose=True)"
```

A 5-line smoke calculation:

```python
from mnpbem.geometry import trisphere
from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import ComParticle
print('trisphere n =', trisphere(144, 20).n)  # 144
```

## 7. Multi-GPU VRAM share

`MNPBEM_VRAM_SHARE_GPUS=N` (with `N>=2`) lets a single BEM solve span
the combined VRAM of `N` GPUs through cuSolverMg. Useful when the dense
LU exceeds a single device. Backend is selectable via
`MNPBEM_VRAM_SHARE_BACKEND` (`cusolvermg` is the default and most
robust; `magma` and `nccl` are experimental).

```bash
MNPBEM_GPU=1 MNPBEM_VRAM_SHARE_GPUS=4 \
    python my_simulation.py
```

If cuSolverMg is not available (e.g. older driver, no
`libcusolverMg.so`), the dispatcher emits a warning and falls back to
single-GPU LU automatically. See `docs/PERFORMANCE.md` Section 6 for
benchmarks.

## 8. Multi-node MPI

Wavelength sweeps parallelize across MPI ranks: each rank computes a
disjoint subset of the wavelength grid; the root reduces. Launch the
exact same script under `mpirun`:

```bash
mpirun -np 8 python my_spectrum.py
# or under SLURM
srun -n 8 python my_spectrum.py
```

The script does not need to import `mpi4py` directly — the package's
internal dispatcher does so lazily. `pip install "mnpbem[mpi]"` is the
only requirement.

## 9. Troubleshooting

### "ImportError: cupy is not installed"

Either:
1. `pip install "mnpbem[gpu]"` to add cupy to the active environment.
2. Unset `MNPBEM_GPU` (or set to `0`) — the package will run CPU-only.

### "libcudart.so.12: cannot open shared object file"

The CUDA toolkit runtime is missing. Install via the system package
manager (Ubuntu: `apt install cuda-toolkit-12-x`) or activate a conda
env that pulls in `cuda-runtime`:

```bash
conda install -c nvidia cuda-runtime=12
```

### "mpi4py and OpenMPI version mismatch"

Reinstall mpi4py against the active OpenMPI:

```bash
pip uninstall -y mpi4py
pip install --no-binary mpi4py "mpi4py>=3.1"
```

On HPC, ensure the `openmpi` module is loaded *before* invoking pip.

### "cupy installed but no CUDA devices found"

`nvidia-smi` should list at least one GPU. If it does not, the host has
no usable GPU (driver not loaded, container without `--gpus`, etc.). The
package falls back to CPU automatically when `has_gpu_capability()`
returns `False`.

### Older pip does not resolve nested extras

`mnpbem[all]` expands to `mnpbem[gpu,mpi,fmm]`, which pip 21.2+ resolves
natively. On older pip:

```bash
pip install --upgrade pip
# or expand manually
pip install "mnpbem[gpu,mpi,fmm]"
```

## 10. Why a single wheel?

v1.4.0 ships a single wheel rather than separate `mnpbem-cpu` and
`mnpbem-gpu` distributions. Rationale:

- All Python modules import cleanly without cupy — there is nothing to
  strip out for a CPU-only build.
- Extras are a PyPI / pip standard and require no special tooling.
- A separate `mnpbem-gpu` package would force the user to choose
  *before* knowing whether their environment has CUDA, and would
  fragment the namespace (`import mnpbem` vs `import mnpbem_gpu`).

The runtime path is gated by `MNPBEM_GPU=1` and `has_gpu_capability()`,
not by the wheel filename. This keeps the install matrix simple and the
fallback behaviour uniform across deployments.

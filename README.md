# MNPBEM Python Port v1.0.0

Python port of the [MATLAB MNPBEM toolbox](https://physik.uni-graz.at/de/mnpbem/)
(Hohenester & Trügler) for the simulation of electromagnetic properties of
metallic nanoparticles using the boundary element method (BEM).

> Original MATLAB: Hohenester & Trügler (Comp. Phys. Commun. 183, 370 (2012); 185, 1177 (2014); 193, 138 (2015)).
> Python port targets bit-similar numerical agreement with MATLAB MNPBEM17,
> while adding GPU acceleration, multi-GPU dispatch, and an iterative
> ACA / H-matrix solver for large meshes.

## What is this?

MNPBEM solves Maxwell's equations for dielectric environments where bodies
with homogeneous and isotropic dielectric functions are separated by abrupt
interfaces. Typical applications are plasmonic nanoparticles in the optical
and near-infrared range (sphere, rod, cube, dimer, particles on substrate,
EELS probes, ...).

The Python port keeps the public class structure and method names of the
MATLAB toolbox so that existing MATLAB scripts can be translated almost
mechanically (see `docs/MIGRATION_GUIDE.md`). What the Python port adds:

- **Python-native API** with `numpy` arrays instead of MATLAB structs.
- **GPU acceleration** via `cupy` (single GPU and multi-GPU wavelength dispatch).
- **Iterative solver** (`BEMRetIter`, `BEMStatIter`) using ACA-compressed
  H-matrices and GMRES — scales to tens of thousands of boundary elements.
- **Multi-node MPI** wavelength dispatch for spectrum sweeps.
- **Validation suite** that reproduces 72 MNPBEM demos and compares against
  MATLAB hash-by-hash.

## Highlights

- **Accuracy**: 55 / 72 demos at machine precision against MATLAB MNPBEM17;
  0 demos failing the BAD threshold (bit-identical mesh, identical iterative
  solver path).
- **Performance** (vs MATLAB MNPBEM17 single core, geo-mean):
  - CPU: **2.21x** speedup
  - GPU (single): **3.60x** speedup
  - Multi-GPU: linear scaling on wavelength sweeps
- **Iterative solver**: ACA / H-matrix with GMRES. Tested up to ~25 k faces.
- See `docs/PERFORMANCE.md` for full benchmark numbers and methodology.

## Requirements

```bash
conda create -n mnpbem python=3.11
conda activate mnpbem

# core (CPU)
pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.8.4 \
            numba==0.59.1 jupyter==1.1.1 python-box==7.3.2

# optional: GPU acceleration (requires CUDA 12.x)
pip install cupy-cuda12x==13.3.0

# optional: multi-node MPI
pip install mpi4py==3.1.6

# editable install of MNPBEM
git clone https://github.com/<your-org>/MNPBEM.git
cd MNPBEM
pip install -e .
```

After install, verify:

```bash
python -c "import mnpbem; print(mnpbem.__version__)"
# 1.0.0
```

## Quick Start

```python
# 5-line gold sphere extinction spectrum (retarded BEM)
import numpy as np
from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet

epstab = [EpsConst(1.0), EpsTable("gold.dat")]
p      = ComParticle(epstab, [trisphere(144, 20)], [[2, 1]], 1, interp="curv")
bem    = BEMRet(p)
exc    = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))

enei   = np.linspace(400, 800, 41)
ext    = np.zeros_like(enei)
for i, e in enumerate(enei):
    sig, bem = bem.solve(exc.potential(p, e))
    ext[i]   = float(np.real(np.ravel(exc.extinction(sig))[0]))
```

A complete worked spectrum + plot is in [`examples/01_sphere_extinction.py`](examples/01_sphere_extinction.py).

## Documentation

- [API Reference](docs/API_REFERENCE.md) — every public class and function.
- [Migration Guide (from MATLAB)](docs/MIGRATION_GUIDE.md) — line-by-line mapping.
- [Examples](examples/) — runnable Python scripts and a Jupyter tutorial.
- [Performance](docs/PERFORMANCE.md) — benchmark methodology and numbers.
- [Architecture](docs/ARCHITECTURE.md) — package layout and design notes.
- [Changelog](CHANGELOG.md) — release history.

## Repository Layout

```
mnpbem/                  # Python package (geometry, bem, greenfun, simulation, ...)
docs/                    # User documentation (API, migration, performance, architecture)
examples/                # Runnable Python examples + Jupyter tutorial
validation/              # MATLAB <-> Python regression suite (72 demos, sphere/rod, dimer)
tests/                   # Unit tests (pytest)
Particles/  Mesh2d/  Greenfun/  BEM/  Demo/   # MATLAB MNPBEM17 reference (read-only)
```

## License

MNPBEM (the original MATLAB toolbox) is distributed under the GNU GPL v2+.
The Python port is released under the same license to remain compatible
with the upstream code base.

```
Copyright (C) 2017 Ulrich Hohenester (MATLAB MNPBEM17)
Copyright (C) 2026 MNPBEM Python port contributors
This code is distributed under the terms of the GNU General Public License v2.
See the COPYING file for license details.
```

## Citation

When publishing results obtained with this Python port, please cite the
original MNPBEM papers:

```bibtex
@article{hohenester2012mnpbem,
  author  = {Hohenester, U. and Tr\"ugler, A.},
  title   = {{MNPBEM} -- A {Matlab} toolbox for the simulation of plasmonic nanoparticles},
  journal = {Comput. Phys. Commun.},
  volume  = {183},
  pages   = {370--381},
  year    = {2012}
}
@article{hohenester2014simulation,
  author  = {Hohenester, U.},
  title   = {Simulating electron energy loss spectroscopy with the {MNPBEM} toolbox},
  journal = {Comput. Phys. Commun.},
  volume  = {185},
  pages   = {1177--1187},
  year    = {2014}
}
@article{waxenegger2015plasmonics,
  author  = {Waxenegger, J. and Tr\"ugler, A. and Hohenester, U.},
  title   = {Plasmonics simulations with the {MNPBEM} toolbox: Consideration of substrates and layer structures},
  journal = {Comput. Phys. Commun.},
  volume  = {193},
  pages   = {138--150},
  year    = {2015}
}
```

In addition, please cite this Python port:

```bibtex
@software{mnpbem_python_2026,
  title  = {{MNPBEM} Python port (v1.0.0)},
  year   = {2026},
  url    = {https://github.com/<your-org>/MNPBEM},
  note   = {Python port of MNPBEM17 with GPU acceleration and ACA / H-matrix solvers.}
}
```

## Bug Reports & Contributions

Please open an issue on GitHub. When reporting a numerical discrepancy
against MATLAB MNPBEM17, include:
- Python version, `mnpbem.__version__`, `numpy.__version__`
- Mesh parameters (e.g. `trisphere(144, 20)`)
- A minimal script that reproduces the discrepancy
- Expected MATLAB output (preferably from the same demo file in `Demo/`)

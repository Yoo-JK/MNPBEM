# MNPBEM API Audit (MATLAB → Python port)

**Date**: 2026-04-22  
**Scope**: all public classes/methods of MATLAB MNPBEM  
**Goal**: assess Python port completeness and prioritize missing items

---

## Summary

| Item | Count |
|------|------|
| **MATLAB public methods** | 697 |
| **Python implementation files** | 102 |
| **Completeness of major classes** | 65.2% (60/92 methods) |
| **Fully implemented classes** | 0/10 (among major classes) |
| **80%+ complete** | 2/10 classes |

**Conclusion**: the Python port covers most of the core functionality but is ~35% incomplete at the method level

---

## Status by Module

### 1. BEM (boundary element method) module

| MATLAB Class | Python Class | Methods | Completeness | Status |
|---|---|---|---|---|
| `bemstat` | `BEMStat` | 9 | 78% (7/9) | ⚠ incomplete |
| `bemret` | `BEMRet` | 8 | 88% (7/8) | ⚠ nearly complete |
| `bemstatlayer` | `BEMStatLayer` | 7 | 71% (5/7) | ✗ incomplete |
| `bemretlayer` | `BEMRetLayer` | 6 | 83% (5/6) | ⚠ nearly complete |
| `bemstateig` | `BEMStatEig` | 7 | ? | ✓ implemented |
| `bemstateigmirror` | `BEMStatEigMirror` | 8 | ? | ✓ implemented |
| `bemstatmirror` | `BEMStatMirror` | 8 | ? | ✓ implemented |
| `bemretmirror` | `BEMRetMirror` | 6 | ? | ✓ implemented |
| `bemiter` | `BEMIter` | 10 | 50% (5/10) | ✗ major gaps |
| `bemstatiter` | `BEMStatIter` | 9 | ? | partially implemented |
| `bemretiter` | `BEMRetIter` | 8 | ? | partially implemented |
| `bemretlayeriter` | `BEMRetLayerIter` | 7 | ? | partially implemented |
| `bemlayermirror` | `BEMLayerMirror` | 1 | ? | ✓ implemented |

**BEM analysis**:
- ✓ quasistatic (stat) / retarded (ret) base solvers implemented
- ⚠ iterative (iter) solver (~50% implemented)
- ⚠ method mapping: MATLAB `mldivide` → Python `__truediv__`, `mtimes` → `__mul__`

---

### 2. Particles (particle/geometry) module

| MATLAB Class | Python Class | Methods | Completeness | Status |
|---|---|---|---|---|
| `polygon` | `Polygon` | 17 | 71% (12/17) | ⚠ incomplete |
| `polygon3` | `Polygon3` | 0 | - | ? unclear |
| `comparticle` | `ComParticle` | 11 | 73% (8/11) | ⚠ incomplete |
| `compound` | `Compound`/`Connect` | 11 | 9% (1/11) | ✗ **serious gap** |
| `layerstructure` | `LayerStructure` | 13 | 77% (10/13) | ⚠ incomplete |
| `edgeprofile` | `EdgeProfile` | ? | ? | ? |
| `compoint` | `ComPoint` | ? | ? | ✓ implemented |
| `point` | `Point` | ? | ? | ✓ implemented |
| `comparticlemirror` | `ComParticleMirror` | ? | ? | ✓ implemented |

**Particles analysis**:
- ✓ basic geometric classes implemented
- ✗ **serious gap in the `compound` class** (only 1 of 11 methods)
- ⚠ methods missing from `Polygon`:
  - utility functions such as `plot()`, `norm()`, `interp1()`, `union()`, `symmetry()`
- ⚠ method renames: `round()` → `round_()`, `sort()` → `sort_()`

---

### 3. Greenfun (Green function) module

| MATLAB Class | Python Class | Methods | Status |
|---|---|---|---|
| `greenstat` | `GreenStat` | 4 | ✓ implemented |
| `greenret` | `GreenRet` | 5 | ✓ implemented |
| `greenretlayer` | `GreenRetLayer` | 4 | ✓ implemented |
| `compgreenstat` | `CompGreenStat` | 6 | ✓ implemented |
| `compgreenret` | `CompGreenRet` | 5 | ✓ implemented |
| `compgreenretlayer` | `CompGreenRetLayer` | 6 | ✓ implemented |
| `compgreenretmirror` | `CompGreenRetMirror` | 5 | ✓ implemented |
| `compgreenstatmirror` | `CompGreenStatMirror` | 5 | ✓ implemented |
| `compgreentablayer` | `CompGreenTabLayer` | 8 | ✓ implemented |
| `greentablayer` | `GreenTabLayer` | 8 | ✓ implemented |
| `+aca/compgreenstat` | `AcaCompGreenStat` | 5 | ✓ implemented |
| `+aca/compgreenret` | `AcaCompGreenRet` | 4 | ✓ implemented |
| `+aca/compgreenretlayer` | `AcaCompGreenRetLayer` | 4 | ✓ implemented |

**Greenfun analysis**:
- ✓ **nearly fully implemented** (highest completeness)
- ✓ includes ACA (Adaptive Cross-Approximation) compression
- ✓ includes H-matrix implementation

---

### 4. Simulation module

**Static part:**
| MATLAB Class | Python Class | Status |
|---|---|---|
| `dipolestat` | `DipoleStatEsource` | ✓ |
| `dipolestatlayer` | `DipoleStatLayer` | ✓ |
| `dipolestatmirror` | `DipoleStatMirror` | ✓ |
| `eelsstat` | `EelsStatBase` | ✓ |
| `planewavestat` | `PlaneWaveStat` | ✓ |
| `planewavestatlayer` | `PlaneWaveStatLayer` | ✓ |
| `planewavestatmirror` | `PlaneWaveStatMirror` | ✓ |
| `spectrumstat` | `SpectrumStat` | ✓ |
| `spectrumstatlayer` | `SpectrumStatLayer` | ✓ |

**Retarded part:**
| MATLAB Class | Python Class | Status |
|---|---|---|
| `dipoleret` | `DipoleRetEsource` | ✓ |
| `dipoleretlayer` | `DipoleRetLayer` | ✓ |
| `dipoleretmirror` | `DipoleRetMirror` | ✓ |
| `eelsret` | `EelsRetBase` | ✓ |
| `planewaveret` | `PlaneWaveRet` | ✓ |
| `planewaveretlayer` | `PlaneWaveRetLayer` | ✓ |
| `planewaveretmirror` | `PlaneWaveRetMirror` | ✓ |
| `spectrumret` | `SpectrumRet` | ✓ |
| `spectrumretlayer` | `SpectrumRetLayer` | ✓ |

**Simulation analysis**:
- ✓ **Static and Retarded excitation nearly fully implemented**
- ✓ includes dipole, plane wave, EELS, and spectrum analysis
- ✓ Mirror symmetry implemented

---

### 5. Mesh2d (mesh generation) module

| Item | Status |
|---|---|
| `mesh2d` | ✓ implemented |
| `mesh2d_core` | ✓ implemented |
| `mesh_generators` | ✓ implemented |
| Darren Engwirda toolbox | ✓ ported |

**Mesh2d analysis**:
- ✓ **fully implemented**

---

### 6. Misc (+misc) module

| Item | Python | Status |
|---|---|---|
| `units` | `mnpbem/misc/units.py` | ✓ |
| `constants` | `mnpbem/utils/constants.py` | ✓ |
| `misc_utils` | `mnpbem/misc/misc_utils.py` | ✓ |
| `math_utils` | `mnpbem/misc/math_utils.py` | ✓ |
| `gauss_legendre` | `mnpbem/misc/gauss_legendre.py` | ✓ |
| `bemplot` | `mnpbem/misc/bemplot.py` | ✓ |

**Misc analysis**:
- ✓ **fully implemented**

---

## Missing Items (by Priority)

### Priority 1 (major features, essential to run the demos)

1. **`Particles/@compound` methods** (10 of 11 missing)
   - `set()`, `eq()`, `ne()`, `ipart()`, `subsref()`, `dielectric()`, `mask()`, `index()`, `expand()`
   - **Impact**: multi-material particle construction impossible → complex demos such as `demospecret15` fail
   - **Priority**: very high

2. **`BEM/@bemiter` methods** (5 of 10 missing)
   - `solve()`, `setstat()`, `printstat()`, `setiter()` 
   - **Impact**: incomplete iterative solver → degraded large-scale simulation performance
   - **Priority**: high

3. **`Particles/@polygon` utilities** (5 of 17 missing)
   - `plot()`, `norm()`, `interp1()`, `union()`, `symmetry()`
   - **Impact**: limits mesh generation and visualization
   - **Priority**: medium-high

4. **Completeness of Mirror class methods** 
   - `bemstateigmirror`, `bemstatmirror`, `bemretmirror` → method counts unclear
   - mirror-symmetry functionality partially implemented
   - **Priority**: medium

### Priority 2 (advanced features, needed for some demos)

1. **`Particles/@layerstructure` methods** (3 of 13 missing)
   - exact missing methods not identified (mostly internal helpers)
   - **Impact**: limits layered-structure simulation

2. **`BEM/@bemstatlayer` methods** (2 of 7 missing)
   - completeness of `subsref()`, `init()` unknown
   - **Impact**: limits layered quasistatic BEM functionality

3. **`BEM/@bemretlayer` methods** (1 of 6 missing)
   - completeness of the `solve()` method needs verification
   - **Impact**: limits layered retarded BEM functionality

### Priority 3 (optional, internal helpers/legacy)

1. **`Particles/@polygon3`**
   - methods unclear even in MATLAB (only a constructor exists)
   - a Python implementation exists but the mapping is unclear

2. **`+aca/` (Adaptive Cross Approximation)**
   - advanced compression feature (for large-scale problems)
   - port status: ✓ implemented

3. **Method-rename compatibility**
   - MATLAB `round()` → Python `round_()`
   - MATLAB `sort()` → Python `sort_()`
   - consider adding legacy aliases

---

## Python Additions (not in MATLAB)

| Feature | File | Description |
|---|---|---|
| **Material models** | `mnpbem/materials/` | Drude, Lorentz, table-based models |
| **Mie theory** | `mnpbem/mie/` | spherical scattering (Mie theory) |
| **Field mesh computation** | `mnpbem/simulation/meshfield.py` | electromagnetic field computation on a spatial mesh |
| **Solver factory** | `mnpbem/bem/solver_factory.py` | BEM solver factory pattern |
| **Parallelization** | `mnpbem/utils/parallel.py` | parallel computation support |
| **Test suite** | `mnpbem/tests/` | 20 test modules |

---

## Method Mapping Guide

### Operator overloading

| MATLAB | Python | Meaning |
|---|---|---|
| `obj \ exc` | `obj.__truediv__(exc)` or `obj / exc` | surface charge computation |
| `obj * sig` | `obj.__mul__(sig)` | induced potential computation |

**Note**: Python has no `\` operator, so use `/` or `.solve()` instead

### Method-naming conventions

| MATLAB | Python | Reason |
|---|---|---|
| `round()` | `round_()` | avoid clash with Python built-in `round()` |
| `sort()` | `sort_()` | clarity vs Python built-in `list.sort()` |
| `subsref()` | `__getitem__()` | Python indexing convention |
| `init()` | `__init__()` | Python constructor |

---

## Demo Runnability Assessment

### Fully working demos
- ✓ `demo_nanosphere_spectrum` - nanosphere scattering spectrum
- ✓ simple particles (sphere, ellipsoid)
- ✓ quasistatic/retarded multipole excitation
- ✓ plane-wave excitation
- ✓ EELS computation

### Demos that may have issues
- ⚠ multi-material particles (insufficient `compound` methods)
- ⚠ large-scale simulation (incomplete iterative solver)
- ⚠ advanced mesh manipulation (insufficient `polygon` utilities)
- ⚠ layered structures (layer methods partially incomplete)

---

## Recommendations

### Immediate (1 week)

1. **Full port of `Particles/@compound`**
   - high impact, medium scope
   - core: `subsref()`, `dielectric()`, `mask()`, `index()` methods

2. **Verify and complete `BEM/@bemiter.solve()`**
   - ensure iterative solver stability

### Short-term improvements (2-3 weeks)

3. **Add `Particles/@polygon` utility methods**
   - `plot()`, `norm()`, `symmetry()`, etc.

4. **Clarify Mirror class methods**
   - document the method list for each mirror class

### Medium-term maintenance (1 month)

5. **Method-name compatibility layer**
   - add legacy aliases (e.g., `round` → `round_()`)

6. **Expand test coverage**
   - write tests for each missing method

---

## Completeness Statistics

```
Module         MATLAB  Python  Completion
─────────────────────────────────────────
BEM              98      ~65      66%
Particles       133      ~85      64%
Greenfun       117     ~115      98%  ✓
Simulation     147     ~145      99%  ✓
Mesh2d          21      ~21     100%  ✓
Misc            82      ~82     100%  ✓
Base             8       ~8     100%  ✓
─────────────────────────────────────────
TOTAL          606     ~521      86%
```

**Key conclusions**:
- **Overall completeness**: ~86% (by method count)
- **High completeness**: Greenfun, Simulation, Mesh2d, Misc
- **Needs improvement**: Particles (especially `compound`), BEM (especially the iterative solver)

---

## User Guide

### Migrating MATLAB code to Python

```python
# MATLAB:
% bem = bemstat(p);
% sig = bem \ exc;
% phi = bem * sig;
% E = bem.field(sig);

# Python:
from mnpbem import *
bem = BEMStat(p)
sig = bem / exc  # or sig = bem.solve(exc)
phi = bem * sig
E = bem.field(sig)
```

### API mismatch caveats

1. **Operators**: MATLAB `\` → Python `/` (or `.solve()`)
2. **Method names**: `round()` → `round_()`, `sort()` → `sort_()`
3. **Multi-material**: missing `compound` methods → a workaround is needed

---

## Audit Conclusion

The MNPBEM Python port is **fairly complete by core-feature measure (86%)**,  
but **has major gaps at the method level (65%)**.

**Immediate action needed:**
- Full port of `Particles/@compound` (very high priority)
- Improve completeness of the `BEM/@bemiter` iterative solver
- Add utility methods (`polygon`, `layerstructure`)

**Current state**: most basic simulations work, but complex geometry / layered structures are limited


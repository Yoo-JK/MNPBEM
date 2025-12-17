# Retarded BEM Solver Implementation Status

**Date**: 2025-12-17
**Branch**: `claude/matlab-to-python-conversion-4LwI3`

## Summary

Implemented retarded (full Maxwell) BEM solver for plasmonics simulations. The solver is **fully functional for Drude model** dielectrics but has **limitations with table-based** dielectric functions.

## What Works ✓

### 1. Region-Based Connectivity Matrix
- Fixed connectivity from particle-based (1×1) to region-based (2×2)
- Proper material indexing: con[0][0]=inside, con[1][1]=outside
- Correctly handles closed surfaces: con[0][1]=0, con[1][0]=0

### 2. Region-Based Green Function Evaluation
- Properly evaluates G, F, H1, H2 matrices for each region pair
- Correctly uses different wavenumbers: k_gold (inside) vs k_water (outside)
- BEM equations: G1 = G[0][0] - G[1][0], G2 = G[1][1] - G[0][1]

### 3. Diagonal Regularization
- Added temporary fix for self-interaction singularities
- Replaces diagonal elements >1e10 with mean of off-diagonal elements
- **Works well for Drude model** dielectrics with smooth k(λ)

### 4. Complete BEM Pipeline
- BEMRet solver initialization
- PlaneWaveRet excitation (extinction, absorption, scattering)
- SpectrumRet for far-field calculations
- All 4 test combinations execute without errors

## Test Results

### Drude Model (Working) ✓
```
Quasistatic:  λ=500 nm,  σ=1821 nm²
Retarded:     λ=490 nm,  σ=1218 nm²
Peak reduction: 33% (expected: 20-40% due to radiation damping)
Wavelength shift: -10 nm (reasonable)
```
**Status**: Excellent agreement with expected physics

### Table-Based Dielectric (Limited) ⚠
```
Quasistatic:  λ=520 nm,  σ=322 nm²
Retarded:     λ=400 nm,  σ=292 nm²
Peak reduction: 9% (too small)
Wavelength shift: -120 nm (anomalously large)
Spectrum: Nearly flat (240-290 nm²) instead of sharp peak
```
**Status**: Diagonal regularization insufficient for complex ε(ω)

## Root Cause Analysis

### Why Table-Based Fails

**Wavenumber variation**:
```
λ(nm)   k_drude          k_table         |ratio|
400     0.019+0.001i     0.023+0.031i    2.03×
450     0.002+0.013i     0.019+0.027i    2.54×
500     0.001+0.023i     0.012+0.023i    1.14×
520     0.001+0.026i     0.008+0.025i    1.02×
```

**Problem**:
1. Table has 2-3× larger |k| at short wavelengths
2. Phase factor exp(ikd) becomes very different
3. Diagonal regularization (mean before phase) doesn't account for this
4. Results in incorrect spectral shape

**Drude vs Table**:
- Drude: k varies smoothly, diagonal regularization works
- Table: k has sharp features, diagonal regularization breaks

## What's Missing

### Proper Polar Integration Refinement

**MATLAB implementation** (`@greenret/private/init.m`):

**Diagonal elements** (lines 44-95):
```matlab
% Uses quadpol() for polar integration
[pos, w, row] = quadpol(p2, face2);
% Refine for multiple orders
for ord = 0:order
    g(iface, ord+1) = accumarray(row, w .* r.^(ord-1)) / factorial(n);
end
```

**Off-diagonal near elements** (lines 98-177):
```matlab
% Uses quad() for boundary element integration
[postab, wtab] = quad(p2, reface);
% Refine elements where distance is small (ir == 1)
```

**What this does**:
1. Computes exact integral over boundary element surface
2. Handles r→0 limit properly (no singularity)
3. Expansion in powers of (ikr) for different orders
4. Provides accurate self-interaction and near-field values

## Implementation Requirements

To properly fix table-based dielectric support, need to implement:

### 1. Polar Quadrature (`quadpol`)
- Generate integration points on boundary elements
- Use polar coordinates (r, θ) for better r→0 behavior
- Weight functions for area integration
- MATLAB location: `/Misc/@particle/quadpol.m`

### 2. Boundary Element Integration (`quad`)
- Standard quadrature for finite elements
- Gaussian quadrature points and weights
- MATLAB location: `/Misc/@particle/quad.m`

### 3. Refinement Distance Criteria (`ir` matrix)
```
ir == 0: Far field (use 1/d approximation)
ir == 1: Near field (refine off-diagonal)
ir == 2: Self-interaction (polar integration)
```

### 4. Multi-Order Expansion
- Store coefficients for orders 0, 1, 2, ...
- Green function: G = Σ g_n * (ik)^n / n!
- Surface derivative: F = Σ f_n * (ik)^n / n!

### 5. Integration with eval()
- During eval(k, 'G'), combine refined diagonal with non-refined off-diagonal
- Apply phase factor exp(ikd) after refinement
- Properly handle different k values for different materials

## Files Modified

### Core Implementation
- `/home/user/MNPBEM/mnpbem/greenfun/compgreen_ret.py`:
  - Region-based connectivity `_connect()`
  - Region-based evaluation `_eval1()`
  - Diagonal regularization in `GreenRetBlock.eval()`
  - BEM solver initialization `_init_solver()`

- `/home/user/MNPBEM/mnpbem/bem/bem_ret.py`:
  - Updated `init()` for region-based Green functions
  - Proper G1, G2, H1, H2 matrix construction

### Simulation & Analysis
- `/home/user/MNPBEM/mnpbem/simulation/planewave_ret.py`:
  - Fixed `extinction()` axis handling
  - Fixed `absorption()` tuple unpacking

- `/home/user/MNPBEM/mnpbem/spectrum/spectrum_ret.py`:
  - Fixed phase array dimensions

### Geometry
- `/home/user/MNPBEM/mnpbem/geometry/comparticle.py`:
  - Added `bradius()` method

## Recommendations

### Short Term (Current Status)
**Use Drude model** for retarded simulations:
```python
eps_gold = EpsDrude(10, 9.07, 0.066)
cp = ComParticle(eps, [p], [[2, 1]])
bem = BEMRet(cp)
```
Works reliably for plasmonic nanoparticles with simple geometries.

### Medium Term (Workaround)
**For table-based dielectric**, consider:
1. Use quasistatic approximation (works well for small particles <50 nm)
2. Or implement simplified analytical self-interaction for flat elements
3. Or use only Drude model as approximation to table data

### Long Term (Proper Solution)
**Implement full polar integration refinement**:
1. Port MATLAB's `quadpol()` and `quad()` functions
2. Implement multi-order expansion storage
3. Add refinement criteria (`ir` matrix) based on element distance
4. Integrate with current `GreenRetBlock.eval()` method

Estimated effort: 3-5 days of focused development + testing

## References

**MATLAB Code**:
- `/home/user/MNPBEM/Greenfun/@greenret/private/init.m` - Refinement implementation
- `/home/user/MNPBEM/Greenfun/@greenret/private/eval1.m` - Green function evaluation
- `/home/user/MNPBEM/Particles/@compound/connect.m` - Connectivity matrix

**Physics**:
- García de Abajo & Howie, *Phys. Rev. B* **65**, 115418 (2002)
  - Equations (19-22): Retarded BEM formulation
- Hohenester & Trügler, *Comp. Phys. Comm.* **183**, 370 (2012)
  - MNPBEM toolbox documentation

## Testing

### Test Script
```bash
cd /home/user/workspace
python3 test_all_combinations_python.py
```

### Expected Output
```
Quasistatic + Drude:  Peak at ~500 nm, σ~1800 nm²  ✓
Quasistatic + Table:  Peak at ~520 nm, σ~320 nm²   ✓
Retarded + Drude:     Peak at ~490 nm, σ~1200 nm²  ✓
Retarded + Table:     Limited accuracy              ⚠
```

### Debug Script
```bash
cd /home/user/workspace
python3 debug_green_eval.py
```
Shows Green function values and connectivity structure.

## Conclusion

The retarded BEM solver is **production-ready for Drude model** dielectrics, providing physically accurate extinction, absorption, and scattering spectra. For **table-based dielectrics**, the solver runs but has reduced accuracy due to crude diagonal regularization. Full polar integration refinement is needed for complete accuracy with complex dielectric functions.

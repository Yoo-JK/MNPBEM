# Mesh2d FP Precision Limit

This document explains why bit-identical mesh generation is **structurally impossible** during the MATLAB → Python port.

## Symptoms (as of Wave 5 δ)

- `demoeelsret3` (Ag nanodisk, single disk, face metric): 7.1e-03 (OK)
- `demoeelsret7` (Ag nanotriangle on membrane, plate-with-hole): 2.2e-01 (warn)
- `demoeelsret8` (same + 2D spatial map): 4.8e-01 (warn)

The corner-location peaks in demoeelsret7/8 are accurate to within 0.3% of MATLAB.
The middle/edge-location peaks differ by 5–15% — this results from a ≲0.1 nm difference in a single mesh point
inducing a frequency shift (tens of meV) in the resonance peak.

## Root Cause

1. **The mesh shape itself is nearly identical**
   - MATLAB plate `up`: 322 verts, 536 faces
   - Python `Polygon3.plate_from_list` result: 325 verts, 542 faces
   - KDTree matching of both vertex sets gives median Δ = 0.07 nm (5e-4 relative to the 150 nm plate)

2. **The origin of the difference is Mesh2d iterative convergence**
   - Within the iterative flow of `quadtree` → `boundarynodes` (spring smoothing, maxit=50) →
     `meshpoly` (CVT-style Laplacian, maxit=20), ULP-level FP differences
     (MATLAB MKL `cos/sin` vs Python glibc `np.cos/np.sin`)
     accumulate, causing convergence to a slightly different local minimum.
   - This is the quadtree-level difference already recorded in `project_mesh2d_fp_limit.md`
     amplified through the subsequent smoothing, and it is the same phenomenon as the ≲0.07 nm interior-node
     drift observed in the `poly25` 25-gon.

3. **Resonance sensitivity**
   - A silver triangle on a dielectric membrane has edge/corner plasmon modes with
     Q ≳ 20, so a 0.1 nm mesh difference → a resonance frequency shift of tens of meV
     → a 5–15% loss difference on a fixed energy grid.

## Solution (= acceptance)

The current `Polygon3.plate_from_list` (mnpbem/geometry/polygon3.py) matches
MATLAB `@polygon3/plate.m` step-by-step:

- L20 z-uniqueness assert
- L43-48 `polymesh2d(poly_closed)` call
- L50-53 `interp1(obj.poly, verts)` boundary enrichment
- L58 9-column faces2 generation via `midpoints(p, 'flat')`
- L70-89 per-polygon boundary smoothing + `vshift` edge z application
- L95 `Particle(verts2, faces2)` → unique vertex extraction via the 9-column branch

The algorithm itself is already consistent, and there is no further mathematical or structural room to close the gap.

## Attempts and Results

### 2026-04-23 Wave 5 δ
- `_classify_faces` / `_detect_loops` / `meshpoly` inpoly filter — working correctly
- MATLAB mesh2d single-face path vs Python multi-face path — confirmed identical results
  (poly4+poly-triangle-46 → both paths give 346 verts / 580 faces)
- Whether `polymesh2d` is passed `face=face_list` has no effect on the result (the auto-detect
  path and the explicit-face path generate identical meshes)
- `CubicSpline` / `interp1` / `midpoints('same')` all match MATLAB semantics

→ As of Wave 5 δ, the Python plate-with-hole mesh has reached a
structural numerical limit relative to MATLAB. The `demoeelsret7/8` warn grade is accepted.

## Tolerance Guidelines

- Analytical geometry (sphere, fvgrid): bit-identical (below 1e-14)
- `tripolygon`, `plate`, `plate_from_list`: ±0.1 nm mesh noise →
  - face-level σ: O(1e-3) (OK)
  - 1D spectrum: resonance peak height ±5–15%, corner peak on the order of ±0.3%
  - 2D spatial map: mean deviation ±5%, resonance-location spatial shift of a few nm
- If exact agreement with MATLAB is required, using mesh injection (MATLAB mesh → load into Python)
  is recommended

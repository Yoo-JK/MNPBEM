# Au@Ag Iter Drift — Diagnostic

## Reproduction (case_g 1136 face)

case_auag_dimer_small (Au-core 5nm + Ag-shell 1.5nm, dimer gap 0.6nm),
medium=1.77 (water).  Excitation: PlaneWaveRet pol=[1,0,0], dir=[0,0,1].

| variant            | hmatrix | htol  | tol   | maxit | restart | precond | max rel diff vs dense |
|--------------------|---------|-------|-------|-------|---------|---------|-----------------------|
| iter_baseline      | T       | 1e-6  | 1e-4  | 200   | 20      | hmat    | 0.700 |
| iter_tol1e-6       | T       | 1e-6  | 1e-6  | 200   | 20      | hmat    | 0.700 |
| iter_tol1e-8       | T       | 1e-6  | 1e-8  | 400   | 20      | hmat    | 0.700 |
| iter_htol1e-8      | T       | 1e-8  | 1e-6  | 400   | 20      | hmat    | 0.700 |
| iter_htol1e-10     | T       | 1e-10 | 1e-6  | 400   | 20      | hmat    | 0.700 |
| iter_no_hmat       | F       | -     | 1e-8  | 400   | 20      | hmat    | 0.700 |
| iter_restart200    | T       | 1e-8  | 1e-8  | 400   | 200     | hmat    | 0.700 |

**Per-wl drift (uniform across all variants)**: 70%, 39%, 5%, 5%, 0%, 2%, 2%
at enei = 380, 433, 487, 540, 593, 647, 700 nm.

**GMRES residuals** all reach ~1e-14 (machine epsilon) on first restart of
20 iterations (so iter count is essentially "converged immediately" at 20).
The residual norm `‖A x - b‖ / ‖b‖ ≈ 1e-14` confirms GMRES converges to
the true solution of the iter system — the 70% drift is **NOT** a
preconditioner / tolerance / hmatrix-htol issue.

## Root cause

`mnpbem/bem/bem_ret.py:360-366` (BEMRet, dense):
```python
if np.all(self.g.con[0][1] == 0) or np.isscalar(self.eps1):
    self.L1 = self.eps1
    self.L2 = self.eps2
else:
    self.L1 = G1 @ self.eps1 @ G1i  # operator form
    self.L2 = G2 @ self.eps2 @ G2i
```

`mnpbem/bem/bem_ret_iter.py:534` (BEMRetIter, iter):
```python
alpha = Hh1 - Hh2 - 1j * k * self._outer(nvec,
            _matmul_diag(eps1, Gsig1) - _matmul_diag(eps2, Gsig2))
```

For Au@Ag dimer:
- `g.con[0][1]` = `[[0,3,0,3],[0,0,0,0],[0,3,0,3],[0,0,0,0]]` -> non-zero
  (cross-connectivity between core/shell faces of two particles)
- `eps1` non-uniform: Au-Ag interface (-12.3+0.4i) vs Ag-core interface
  (-5.26+2.19i)

So BEMRet uses the **operator form** `L1 = G1 @ eps1 @ G1i`, while
BEMRetIter applies eps1 as **point-wise scalar** -> the two are
mathematically distinct, and the iter is solving a **different system**.

GMRES correctly converges to that wrong system's solution (residual = 1e-14).

This matches the original MATLAB MNPBEM iter implementation
(`BEM/@bemretiter/private/afun.m`), which has the same limitation. The
v1.5.0 AUAG_REPORT note "GMRES + dense-LU preconditioner not converging
tightly on the Au/Ag dielectric jump" was wrong about the cause — the
iter is converging tightly, but to the **wrong** linearised operator.

## Fix

Apply the operator form in `_afun` when `g.con[0][1] != 0` AND `eps1`
non-uniform: replace `eps1 * Gsig1` with `G1 @ (eps1 * (G1^{-1} Gsig1))`
i.e. `G1 @ diag(eps1) @ G1^{-1} @ Gsig1`.  Need cached LU of `G1, G2`
(shared with `_init_precond`).  Fallback to scalar form when scalar /
no cross-connectivity (preserves bit-identity for single-particle and
homogeneous-eps cases).

import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..greenfun import CompStruct
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array
from ..utils.matlab_compat import msqrt
from .bem_iter import BEMIter

# v1.7.2 GPU memory-pool cleanup: mirror BEMRet's wavelength-end immediate free
# pattern so cupy returns blocks to the driver at every wavelength rather than
# accumulating across the sweep (MATLAB-parity).  When cupy is not importable
# the helper becomes a no-op so the CPU path stays untouched.
try:
    import cupy as _cp_stat  # type: ignore
    _CUPY_OK_STAT = True
except Exception:
    _cp_stat = None  # type: ignore
    _CUPY_OK_STAT = False


def _gpu_pool_cleanup_stat(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned memory pools.

    Mirrors the v1.7.2 BEMRet cleanup (bem_ret.py:485-503, 734-737): the
    deviceSynchronize() before free_all_blocks() is load-bearing — without
    it, blocks that are still in flight on the CUDA stream are NOT actually
    idle yet and the pool refuses to return them to the driver, so the
    high-water mark keeps creeping up across wavelengths.

    Honours MNPBEM_GPU_POOL_LIMIT_GB (legitimate peaks past this cap will
    OOM; default 0 = uncapped).
    """
    if not _CUPY_OK_STAT:
        return
    try:
        mempool = _cp_stat.get_default_memory_pool()
        pinned = _cp_stat.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_stat.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


class BEMStatIter(BEMIter):

    # MATLAB: @bemstatiter properties (Constant)
    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        # Schur option (v1.5.0): cover-layer (EpsNonlocal) shell-face
        # elimination on the iterative path. Combines with hmatrix=True
        # via SchurIterOperator -- no explicit inv(G_ss) is built, only
        # full matvecs and a small shell-block solve are required.
        self._schur_opt = options.pop('schur', False)
        self._schur_g_ss_solver = options.pop('schur_g_ss_solver', 'auto')
        self._schur_inner_tol = options.pop('schur_inner_tol', 1e-8)
        self._schur_inner_maxit = options.pop('schur_inner_maxit', 200)
        self._schur_active = False
        self._shell_face_idx = None
        self._core_face_idx = None
        self._schur_op = None

        # H-matrix (v1.3.0): opt-in ACA acceleration of F. The matvec used
        # by GMRES then uses HMatrix @ x rather than dense matmul.
        self._hmatrix = bool(options.pop('hmatrix', False))
        self._htol = options.pop('htol', 1e-6)
        self._kmax = options.pop('kmax', [4, 100])
        self._cleaf = options.pop('cleaf', 200)
        self._fadmiss = options.pop('fadmiss', None)

        # H-matrix LU preconditioner (v1.5.0, agent alpha) for the
        # quasistatic iterative solver. See BEMRetIter for the full mode
        # list. Active only when self._hmatrix is True.
        self._hlu_mode = options.pop('preconditioner', 'auto')
        self._htol_precond = options.pop('htol_precond', 1e-4)
        self._hlu_object = None

        # Same default-precond logic as BEMRetIter: don't densify F just to
        # build an LU when the user opted into compression.
        if self._hmatrix and 'precond' not in options:
            options['precond'] = None

        # Initialize BEMIter base class
        super(BEMStatIter, self).__init__(**options)

        # MATLAB: @bemstatiter properties
        self.p = p
        self.enei = None
        self.F = None

        # MATLAB: @bemstatiter properties (Access = private)
        self._op = options
        self._g = None
        self._lambda = None
        self._mat_lu = None

        # Green function
        # MATLAB: obj.g = aca.compgreenstat(p, varargin{:}, 'htol', ...)
        # For iterative solver, Green function is computed as H-matrix
        self._init_green(p, **options)

        # Initialize for given wavelength
        if enei is not None:
            self._init_matrices(enei)

    def _init_green(self,
            p: Any,
            **options: Any) -> None:

        # MATLAB: bemstatiter/private/init.m
        # H-matrix path uses ACACompGreenStat with cluster-tree ACA on F.
        # Dense path uses CompGreenStat (legacy / small mesh / tests).
        # ``hmode`` legacy alias maps onto hmatrix=True.
        hmode = options.pop('hmode', None)
        if self._hmatrix or hmode is not None:
            from ..greenfun import ACACompGreenStat
            kmax_scalar = (max(self._kmax) if hasattr(self._kmax, '__iter__')
                    else self._kmax)
            htol_scalar = (max(self._htol) if hasattr(self._htol, '__iter__')
                    else self._htol)
            aca_kwargs = {
                'htol': htol_scalar,
                'kmax': kmax_scalar,
                'cleaf': self._cleaf,
            }
            if self._fadmiss is not None:
                aca_kwargs['fadmiss'] = self._fadmiss
            self._g = ACACompGreenStat(p, **aca_kwargs, **options)
        else:
            from ..greenfun import CompGreenStat
            self._g = CompGreenStat(p, p, **options)

        # Surface derivative of Green function
        # MATLAB: obj.F = eval(obj.g, 'F')
        self.F = self._g.F

    def _init_matrices(self,
            enei: float) -> 'BEMStatIter':

        # MATLAB: bemstatiter/private/initmat.m
        if self.enei is not None and self.enei == enei:
            return self

        # v1.7.2 wavelength-entry GPU cleanup: drain any stale residents from
        # the previous wavelength BEFORE we re-factor (-Lambda - F) so the
        # new dense LU sees the maximum amount of free device memory.
        # MATLAB-parity (MATLAB releases its workspace at the end of every
        # wavelength loop iteration; cupy needs an explicit drain).
        #
        # Pattern mirrors bem_ret.py:485-503: deviceSynchronize() then both
        # default + pinned pool free_all_blocks().  The full cleanup (sync +
        # both pools + optional limit) is the helper; the per-step inline
        # drains below are cheaper single-pool free_all_blocks() calls used
        # after individual GEMM / LU stages where the prior op has already
        # had its result captured.
        if self._mat_lu is not None:
            self._mat_lu = None
        _gpu_pool_cleanup_stat(apply_limit = True)
        # Belt-and-braces inline drain in case the helper short-circuited
        # (cupy not importable on this build).  Guarded so CPU path is
        # untouched.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        self.enei = enei

        # Dielectric functions
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda function [Garcia de Abajo, Eq. (23)]
        # MATLAB: obj.lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
        self._lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        # Initialize preconditioner
        if self.precond is not None:
            F = self.F
            # Densify if HMatrix — preconditioner LU is dense.
            if hasattr(F, 'full') and not isinstance(F, np.ndarray):
                F_dense = F.full()
            else:
                F_dense = F
            n = F_dense.shape[0]
            # v1.7.2: densification of F (when F is an HMatrix) can leave
            # the per-block GPU staging buffers in the pool.  Drain so the
            # subsequent Lambda allocation does not push the pool past the
            # 49 GB cap on a 49 GB A6000.
            if _CUPY_OK_STAT:
                _cp_stat.get_default_memory_pool().free_all_blocks()

            # Build diagonal Lambda matrix from lambda values
            # MATLAB: spdiag(obj.lambda) handles both scalar and array
            if np.isscalar(self._lambda) or (isinstance(self._lambda, np.ndarray) and self._lambda.ndim == 0):
                Lambda = self._lambda * np.eye(n)
            else:
                Lambda = np.diag(self._lambda)

            if self.precond == 'hmat':
                # MATLAB: obj.mat = lu(-lambda - F)
                self._mat_lu = lu_factor_dispatch(-Lambda - F_dense)

            elif self.precond == 'full':
                # MATLAB: obj.mat = inv(-lambda - full(F))
                self._mat_lu = lu_factor_dispatch(-Lambda - F_dense)

            else:
                raise ValueError('[error] preconditioner not known: <{}>'.format(self.precond))

            # v1.7.2: ``F_dense`` / ``Lambda`` / their sum are transient
            # buffers consumed by ``lu_factor_dispatch`` (overwrite_a paths)
            # but the Python frame still references them.  Drop the names
            # explicitly so cupy can reclaim before the next wavelength's
            # LU runs.
            del F_dense, Lambda
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()

        # Schur (v1.5.0): detect cover-layer partition and prepare the
        # SchurIterOperator that wraps _afun. Done lazily here so that
        # the partition is recomputed if the user constructs the solver
        # without enei and queries it later. When no EpsNonlocal cover
        # layer is present, schur silently falls back to the full path.
        self._schur_active = False
        self._schur_op = None
        if self._schur_opt:
            from .schur_iter_helpers import SchurIterOperator, detect_iter_partition
            partition = detect_iter_partition(self.p)
            if partition is not None:
                shell_idx, core_idx = partition
                nfaces = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
                self._shell_face_idx = shell_idx
                self._core_face_idx = core_idx
                self._schur_op = SchurIterOperator(
                        self._afun,
                        shell_idx,
                        core_idx,
                        nfaces = nfaces,
                        components = 1,
                        dtype = complex,
                        g_ss_solver = self._schur_g_ss_solver,
                        inner_tol = self._schur_inner_tol,
                        inner_maxit = self._schur_inner_maxit)
                self._schur_active = True

        return self

    def _afun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/afun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        # -(lambda + F) * vec
        # Handle both scalar and array lambda
        if np.isscalar(self._lambda) or (isinstance(self._lambda, np.ndarray) and self._lambda.ndim == 0):
            result = -(self.F @ vec_2d + vec_2d * self._lambda)
        else:
            result = -(self.F @ vec_2d + vec_2d * self._lambda[:, np.newaxis])
        return result.reshape(-1)

    def _mfun(self,
            vec: np.ndarray) -> np.ndarray:

        # MATLAB: bemstatiter/private/mfun.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        vec_2d = vec.reshape(n, -1)

        if self.precond == 'hmat' or self.precond == 'full':
            # MATLAB: vec = solve(obj.mat, vec) or obj.mat * vec
            result = lu_solve_dispatch(self._mat_lu, vec_2d)
        else:
            result = vec_2d

        return result.reshape(-1)

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/solve.m
        # Initialize BEM solver (if needed)
        self._init_matrices(exc.enei)

        # Excitation and size of excitation array
        b = exc.phip.ravel().astype(complex)
        siz = exc.phip.shape

        # v1.7.2: drain the cupy pool right before GMRES enters its Krylov
        # build-up.  The init pipeline above leaves up to 2 N^2 * 16 B of
        # transient asarray buffers in the pool; releasing them now keeps
        # peak usage during the iter loop bounded to LU + Krylov subspace.
        if _CUPY_OK_STAT:
            _cp_stat.cuda.runtime.deviceSynchronize()
            _cp_stat.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()

        if self._schur_active:
            # Schur path: GMRES is run on the reduced (core-only) operator.
            # The preconditioner is bypassed because _mfun was built for the
            # full (N, N) (-Lambda - F) factor and would need re-factoring on
            # the core block. The reduced system is well-conditioned for
            # cover-layer geometries so this is acceptable for v1.5.0.
            op = self._schur_op
            b_eff = op.reduce_rhs(b)
            x_core, _ = self._iter_solve(None, b_eff, op._matvec, None)
            x = op.recover_full(x_core, b)
            # v1.7.2: drop the Schur-reduced Krylov subspace handle and
            # the matvec closure references before the next wavelength
            # entry drains the pool.
            del x_core
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()
        else:
            # Function for matrix multiplication
            fa = self._afun
            fm = None
            if self.precond is not None:
                fm = self._mfun

            # v1.5.0 H-matrix LU preconditioner (agent alpha). Replaces fm
            # when active on the H-matrix path.
            if self._hmatrix and self._hlu_mode != 'none':
                fm = self._build_hlu_preconditioner(b.shape[0])

            # Iterative solution
            x, self_updated = self._iter_solve(None, b, fa, fm)
            # v1.7.2: GMRES holds up to ``restart`` Krylov vectors of length
            # n in its scipy internal buffer; ``fa`` / ``fm`` close over
            # ``self.F`` and ``self._mat_lu`` which may be cupy resident.
            # Once x is computed the matvec closures are dead refs;
            # drop them so the pool can compact before the next solve.
            del fa, fm
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()
            _gpu_pool_cleanup_stat()

        # Host-materialize cupy result so the returned sig is always
        # CPU-resident (mirrors BEMStat.solve defensive guard).
        sig_arr = x.reshape(siz)
        if is_cupy_array(sig_arr):
            sig_arr = to_host(sig_arr)

        # Save everything in single structure
        sig = CompStruct(self.p, exc.enei, sig = sig_arr)

        # v1.7.2 solve-exit cleanup: free any residual transient buffers
        # (RHS reshape staging, matvec scratch) before returning so the
        # caller's wavelength loop sees a fully drained pool when it
        # advances to the next enei.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()

        return sig, self

    def _build_hlu_preconditioner(self,
            n_vec: int) -> Callable:

        # v1.5.0 agent alpha — H-matrix LU preconditioner for BEMStatIter.
        # The quasistatic operator A = -(lambda*I + F) has an exact dense
        # LU built inside _init_matrices when self.precond is set. We turn
        # precond='hmat' on for this solve so that the existing _mfun acts
        # as the GMRES preconditioner. This is equivalent to v1.3
        # ``precond='hmat'`` but now triggered on the H-matrix code path
        # (where v1.3 left it disabled by default).
        if self._hlu_object is not None and self._hlu_object == (n_vec, self.enei):
            return self._mfun

        if self._mat_lu is None:
            # Build the dense LU once (lambda + F densified to ndarray).
            self.precond = 'hmat'
            cached_enei = self.enei
            self.enei = None
            self._init_matrices(cached_enei)
            # v1.7.2: the densify-and-LU pipeline above can spike GPU
            # usage by ~3 N^2 transient; drain so GMRES sees max headroom.
            if _CUPY_OK_STAT:
                _cp_stat.cuda.runtime.deviceSynchronize()
                _cp_stat.get_default_memory_pool().free_all_blocks()

        self._hlu_object = (n_vec, self.enei)
        return self._mfun

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatIter']:

        # MATLAB: bemstatiter/mldivide.m
        return self.solve(exc)

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        # MATLAB: bemstatiter/mtimes.m
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/field.m
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces
        nvec = self.p.nvec

        # Electric field in normal direction
        if inout == 1:
            H = self._g.H1
        else:
            H = self._g.H2

        # MATLAB: e = -outer(obj.p.nvec, matmul(obj.g.H, sig.sig))
        H_sig = H @ sig.sig.reshape(n, -1)
        if H_sig.ndim == 1:
            e = -nvec * H_sig[:, np.newaxis]
        else:
            e = -nvec[:, :, np.newaxis] * H_sig[:, np.newaxis, :]
        # v1.7.2: drain after the field-side matvec to keep the pool from
        # accumulating across repeated field() calls in a wavelength loop.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        # Tangential directions via interpolation
        G_sig = self._g.G @ sig.sig.reshape(n, -1)
        phi = self.p.interp(G_sig)
        phi1, phi2, t1, t2 = self.p.deriv(phi)

        # Normal vector
        nvec_c = np.cross(t1, t2)
        h = msqrt(np.sum(nvec_c * nvec_c, axis = 1, keepdims = True))
        nvec_c = nvec_c / h

        # Tangential derivative of PHI
        tvec1 = np.cross(t2, nvec_c) / h
        tvec2 = np.cross(t1, nvec_c) / h

        if phi1.ndim == 1:
            phip = tvec1 * phi1[:, np.newaxis] - tvec2 * phi2[:, np.newaxis]
        else:
            phip = tvec1[:, :, np.newaxis] * phi1[:, np.newaxis, :] - \
                   tvec2[:, :, np.newaxis] * phi2[:, np.newaxis, :]

        e = e - phip
        # v1.7.2: drain residual transient buffers from the interp/deriv
        # pipeline before returning so a follow-up potential()/field() at
        # a different wavelength does not see leftover blocks.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()

        return CompStruct(self.p, sig.enei, e = e)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        # MATLAB: bemstatiter/potential.m
        pot = self._g.potential(sig, inout)
        # v1.7.2: drain after the Green-function potential evaluation so
        # repeated potential() calls in a wavelength loop don't leak
        # transient asarray buffers into the cupy pool.
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        return pot

    def clear(self) -> 'BEMStatIter':

        # MATLAB: bemstatiter/clear.m
        # v1.7 A3 fix: also reset the cache gate (enei) and the wavelength-
        # dependent auxiliaries (_lambda, Schur state, _hlu_object).
        # Otherwise a follow-up solve at the same wavelength hits the
        # cache, finds _mat_lu=None, and crashes inside _mfun.
        self._mat_lu = None
        self.enei = None
        self._lambda = None
        self._schur_active = False
        self._schur_op = None
        self._hlu_object = None
        # v1.7.2: explicit clear() means the user wants the device drained.
        # Without this the LU buffer just released above stays in the cupy
        # pool until the next solve triggers a free_all_blocks.
        if _CUPY_OK_STAT:
            _cp_stat.cuda.runtime.deviceSynchronize()
            _cp_stat.get_default_memory_pool().free_all_blocks()
            _cp_stat.get_default_pinned_memory_pool().free_all_blocks()
        _gpu_pool_cleanup_stat()
        return self

    def __call__(self,
            enei: float) -> 'BEMStatIter':

        # v1.7.2: explicit __call__(enei) is a user-driven wavelength step.
        # ``_init_matrices`` already drains on cache miss; we make the
        # drain unconditional here so consecutive __call__(enei) /
        # __call__(enei') sequences keep the pool tight even when only
        # the H-matrix evaluator side allocates transient buffers.
        out = self._init_matrices(enei)
        if _CUPY_OK_STAT:
            _cp_stat.get_default_memory_pool().free_all_blocks()
        return out

    def __repr__(self) -> str:
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatIter(p: {} faces, solver={}, {})'.format(n, self.solver, status)

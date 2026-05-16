import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..greenfun import CompGreenStatLayer, CompStruct
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


# v1.7.3 (Phase 2): mirror BEMStat / BEMStatIter's wavelength-end cupy pool
# cleanup so the BEMStatLayer dense LU path also keeps the high-water mark
# bounded across long sweeps.  The deviceSynchronize() before free_all_blocks
# is load-bearing — without it, blocks that are still in flight on the CUDA
# stream are NOT actually idle and the pool refuses to return them to the
# driver.  Honors MNPBEM_GPU_POOL_LIMIT_GB the same way BEMStat does.
try:
    import cupy as _cp_layer  # type: ignore
    _CUPY_OK_LAYER = True
except Exception:
    _cp_layer = None  # type: ignore
    _CUPY_OK_LAYER = False


def _vram_share_lu_kwargs() -> dict:
    """Read MNPBEM_VRAM_SHARE_* env vars and return kwargs for lu_factor_dispatch.

    Returns ``{}`` when VRAM-share is not enabled (n_gpus<=1).  Mirrors the
    helper in ``bem_ret.py`` so all dense BEM solvers honour the same
    multi-GPU VRAM-share env vars.
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


def _gpu_pool_cleanup_layer(apply_limit: bool = False) -> None:
    """Synchronise CUDA stream then drain cupy default + pinned pools."""
    if not _CUPY_OK_LAYER:
        return
    try:
        mempool = _cp_layer.get_default_memory_pool()
        pinned = _cp_layer.get_default_pinned_memory_pool()
        if apply_limit:
            try:
                pool_limit_gb = float(os.environ.get(
                        'MNPBEM_GPU_POOL_LIMIT_GB', '0'))
            except (TypeError, ValueError):
                pool_limit_gb = 0.0
            if pool_limit_gb > 0:
                mempool.set_limit(size = int(pool_limit_gb * (1024 ** 3)))
        _cp_layer.cuda.runtime.deviceSynchronize()
        mempool.free_all_blocks()
        pinned.free_all_blocks()
    except Exception:
        pass


class BEMStatLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer

        self.enei = None
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None

        # Green function with layer
        # MATLAB: obj.g = compgreenstatlayer(p, p, layer, varargin{:})
        self.g = CompGreenStatLayer(p, p, layer, **options)

        # Surface derivative of Green function.
        # v1.7.3 Phase 2: F may live on cupy when the upstream Green-function
        # assembly route is GPU-resident.  Host-promote here so the
        # downstream eps1*H1 - eps2*H2 mixing op stays on a single backend
        # (cupy/numpy mix triggers TypeError in older numpy versions).  The
        # F tensor itself is reused only by downstream callers (g.field /
        # g.potential) that already accept either flavour, so dropping the
        # GPU view here also frees its device-side buffer.
        F_obj = self.g.F
        if is_cupy_array(F_obj):
            F_obj = to_host(F_obj)
        self.F = F_obj
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        if enei is not None:
            self(enei)

    def _init_matrices(self,
            enei: float) -> 'BEMStatLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # v1.7.3 Phase 2: free previous wavelength's LU before allocating
        # new buffers.  Mirrors the v1.7.2 BEMStat / BEMStatIter pattern.
        self._A_lu = None
        self._rhs_scale = None
        _gpu_pool_cleanup_layer(apply_limit = True)

        # MATLAB @bemstatlayer/subsref.m "()" branch:
        #   [H1, H2] = eval(obj.g, enei, 'H1', 'H2')
        #   mat = -inv(eps1 * H1 - eps2 * H2) * (eps1 - eps2)
        # The eps1/eps2 are inside/outside dielectric functions of the
        # particle (per-face). They are scalars for homogeneous setups.
        H1 = self.g.eval(enei, 'H1')
        H2 = self.g.eval(enei, 'H2')

        # v1.7.3 Phase 2: H1/H2 from the layer Green function may be cupy
        # arrays.  Host-promote them so the eps* per-face scaling below stays
        # numpy-only (the downstream LU factor lives behind dispatch which
        # re-uploads as needed).  Free the GPU-side intermediates immediately
        # so the pool can recycle their N^2 buffers before the GEMM below.
        if is_cupy_array(H1):
            H1 = to_host(H1)
        if is_cupy_array(H2):
            H2 = to_host(H2)
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        eps1 = np.atleast_1d(self.p.eps1(enei)).astype(complex)
        eps2 = np.atleast_1d(self.p.eps2(enei)).astype(complex)
        n = H1.shape[0]
        if eps1.size == 1:
            eps1 = np.full(n, eps1[0], dtype = complex)
        if eps2.size == 1:
            eps2 = np.full(n, eps2[0], dtype = complex)

        # Use diagonal multiplication to avoid forming dense diag matrices.
        A = eps1[:, np.newaxis] * H1 - eps2[:, np.newaxis] * H2
        rhs_scale = eps1 - eps2  # per-face
        # v1.7.3 Phase 2: H1/H2 are no longer needed after A is formed.
        # Drop them so the cupy pool can reclaim ~2 N^2 buffers before the
        # LU factor.
        del H1, H2

        # Honour MNPBEM_VRAM_SHARE_* for multi-GPU dispatch on large meshes.
        _lu_opts = _vram_share_lu_kwargs()
        self._A_lu = lu_factor_dispatch(A, **_lu_opts)
        # ``A`` is consumed by the LU factor (overwrite_a paths); drop the
        # local handle so the cupy pool reclaims its N^2 buffer.
        del A
        self._rhs_scale = rhs_scale
        self.enei = enei
        _gpu_pool_cleanup_layer()

        return self

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        return self.__truediv__(exc)

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        self._init_matrices(exc.enei)

        phip = exc.phip
        orig_shape = phip.shape
        if phip.ndim == 1:
            phip_2d = phip.reshape(-1, 1)
        elif phip.ndim > 2:
            phip_2d = phip.reshape(phip.shape[0], -1)
        else:
            phip_2d = phip

        # MATLAB mat * phip = -inv(A) * diag(eps1 - eps2) * phip
        rhs = self._rhs_scale[:, np.newaxis] * phip_2d
        sig_result = -lu_solve_dispatch(self._A_lu, rhs)

        if sig_result.shape != orig_shape:
            sig_result = sig_result.reshape(orig_shape)

        # v1.7 Phase 1.4: host-materialize before returning to user.
        if is_cupy_array(sig_result):
            sig_result = to_host(sig_result)

        # v1.7.3 Phase 2: post-solve pool drain mirrors BEMStat.__truediv__.
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()

        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMStatLayer':

        # v1.7 A3 fix: drop the real LU factor / rhs scale held in
        # _A_lu and _rhs_scale.  Previous versions only reset the
        # unused mat_lu attribute, leaving GPU LU memory pinned until
        # the next wavelength rebuild.
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None
        self.enei = None
        # v1.7.3 Phase 2: explicit clear() signals the user wants the
        # device drained.  Mirrors BEMStat.clear pattern.
        if _CUPY_OK_LAYER:
            _gpu_pool_cleanup_layer()
        return self

    def __call__(self,
            enei: float) -> 'BEMStatLayer':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        return 'BEMStatLayer(p: {} faces, {})'.format(n, status)

"""Multi-GPU dense LU factorization via cuSolverMg.

Provides a 1-worker multi-GPU LU path: a single big matrix (e.g. dense
BEM Sigma1 = lu_solve(G^T, H^T)) is distributed across N GPUs in a
1-D block-cyclic column layout. Each GPU holds N/k of the matrix
columns, allowing pooled VRAM use for problems too large to fit on a
single device.

Backend
-------
cuSolverMg (NVIDIA official multi-GPU dense LAPACK). cupy 14 does not
expose cuSolverMg, so the wrapper uses ctypes against the system
``libcusolverMg.so``. Falls back to single-GPU (cupy) or CPU (scipy)
LU when cuSolverMg is unavailable, with a warning.

Activation
----------
Controlled by ``MNPBEM_VRAM_SHARE_GPUS`` (>=2) and
``MNPBEM_VRAM_SHARE_BACKEND`` ('cusolvermg'|'magma'|'nccl'). Default
backend is cusolvermg. Magma/nccl backends raise NotImplementedError.

Reference
---------
- https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cusolverMG-api
- /usr/local/cuda-*/include/cusolverMg.h
"""

from __future__ import annotations

import os
import sys
import ctypes
import warnings
from ctypes import c_int, c_int32, c_int64, c_void_p, c_char_p, POINTER, byref
from typing import Any, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# cuSolverMg / cudart constants
# ---------------------------------------------------------------------------

# cudaDataType (library_types.h)
CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_C_32F = 4
CUDA_C_64F = 5

# cusolverStatus_t = 0 ⇒ CUSOLVER_STATUS_SUCCESS
CUSOLVER_STATUS_SUCCESS = 0

# cusolverMgGridMapping_t
GRID_MAPPING_COL_MAJOR = 0
GRID_MAPPING_ROW_MAJOR = 1

# cublasOperation_t
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


# ---------------------------------------------------------------------------
# Library loaders
# ---------------------------------------------------------------------------

def _candidate_libs(name_root: str) -> List[str]:
    cands = [
        '{}.so'.format(name_root),
        '{}.so.11'.format(name_root),
        '{}.so.12'.format(name_root),
        '/usr/local/cuda/lib64/{}.so'.format(name_root),
        '/usr/local/cuda-12.4/targets/x86_64-linux/lib/{}.so'.format(name_root),
        '/usr/local/cuda-12.4/targets/x86_64-linux/lib/{}.so.11'.format(name_root),
    ]
    return cands


def _try_load(name_root: str) -> Optional[ctypes.CDLL]:
    for cand in _candidate_libs(name_root):
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    return None


_libcusolverMg: Optional[ctypes.CDLL] = None
_libcudart: Optional[ctypes.CDLL] = None


def _load_libs() -> bool:
    global _libcusolverMg, _libcudart
    if _libcusolverMg is not None:
        return True
    mg = _try_load('libcusolverMg')
    rt = _try_load('libcudart')
    if mg is None or rt is None:
        return False
    _libcusolverMg = mg
    _libcudart = rt
    return True


def cusolvermg_available() -> bool:
    return _load_libs()


# ---------------------------------------------------------------------------
# Helpers — cudaMalloc / cudaMemcpy / device set
# ---------------------------------------------------------------------------

def _check_status(ret: int, fn: str) -> None:
    if ret != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError('[error] {} returned {}'.format(fn, ret))


def _cuda_set_device(dev: int) -> None:
    ret = _libcudart.cudaSetDevice(c_int(dev))
    if ret != 0:
        raise RuntimeError('[error] cudaSetDevice({}) failed: {}'.format(dev, ret))


def _cuda_malloc(nbytes: int) -> int:
    ptr = c_void_p(0)
    ret = _libcudart.cudaMalloc(byref(ptr), c_int64(nbytes))
    if ret != 0:
        raise RuntimeError('[error] cudaMalloc({}) failed: {}'.format(nbytes, ret))
    return ptr.value or 0


def _cuda_free(ptr: int) -> None:
    if ptr:
        _libcudart.cudaFree(c_void_p(ptr))


def _cuda_memcpy_h2d(dst_ptr: int, src_arr: np.ndarray) -> None:
    nbytes = src_arr.nbytes
    src_void = src_arr.ctypes.data_as(c_void_p)
    # cudaMemcpyHostToDevice = 1
    ret = _libcudart.cudaMemcpy(c_void_p(dst_ptr), src_void, c_int64(nbytes), c_int(1))
    if ret != 0:
        raise RuntimeError('[error] cudaMemcpy H2D failed: {}'.format(ret))


def _cuda_memcpy_d2h(dst_arr: np.ndarray, src_ptr: int, nbytes: int) -> None:
    dst_void = dst_arr.ctypes.data_as(c_void_p)
    # cudaMemcpyDeviceToHost = 2
    ret = _libcudart.cudaMemcpy(dst_void, c_void_p(src_ptr), c_int64(nbytes), c_int(2))
    if ret != 0:
        raise RuntimeError('[error] cudaMemcpy D2H failed: {}'.format(ret))


def _cuda_device_sync() -> None:
    _libcudart.cudaDeviceSynchronize()


# ---------------------------------------------------------------------------
# cuSolverMg signature setup
# ---------------------------------------------------------------------------

_CUSOLVERMG_BOUND = False


def _bind_cusolvermg() -> None:
    global _CUSOLVERMG_BOUND
    if _CUSOLVERMG_BOUND:
        return
    lib = _libcusolverMg

    lib.cusolverMgCreate.argtypes = [POINTER(c_void_p)]
    lib.cusolverMgCreate.restype = c_int
    lib.cusolverMgDestroy.argtypes = [c_void_p]
    lib.cusolverMgDestroy.restype = c_int

    lib.cusolverMgDeviceSelect.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.cusolverMgDeviceSelect.restype = c_int

    lib.cusolverMgCreateDeviceGrid.argtypes = [
        POINTER(c_void_p), c_int32, c_int32, POINTER(c_int32), c_int]
    lib.cusolverMgCreateDeviceGrid.restype = c_int
    lib.cusolverMgDestroyGrid.argtypes = [c_void_p]
    lib.cusolverMgDestroyGrid.restype = c_int

    lib.cusolverMgCreateMatrixDesc.argtypes = [
        POINTER(c_void_p), c_int64, c_int64, c_int64, c_int64, c_int, c_void_p]
    lib.cusolverMgCreateMatrixDesc.restype = c_int
    lib.cusolverMgDestroyMatrixDesc.argtypes = [c_void_p]
    lib.cusolverMgDestroyMatrixDesc.restype = c_int

    lib.cusolverMgGetrf_bufferSize.argtypes = [
        c_void_p, c_int, c_int, POINTER(c_void_p), c_int, c_int, c_void_p,
        POINTER(c_void_p), c_int, POINTER(c_int64)]
    lib.cusolverMgGetrf_bufferSize.restype = c_int

    lib.cusolverMgGetrf.argtypes = [
        c_void_p, c_int, c_int, POINTER(c_void_p), c_int, c_int, c_void_p,
        POINTER(c_void_p), c_int, POINTER(c_void_p), c_int64, POINTER(c_int)]
    lib.cusolverMgGetrf.restype = c_int

    lib.cusolverMgGetrs_bufferSize.argtypes = [
        c_void_p, c_int, c_int, c_int, POINTER(c_void_p), c_int, c_int, c_void_p,
        POINTER(c_void_p), POINTER(c_void_p), c_int, c_int, c_void_p,
        c_int, POINTER(c_int64)]
    lib.cusolverMgGetrs_bufferSize.restype = c_int

    lib.cusolverMgGetrs.argtypes = [
        c_void_p, c_int, c_int, c_int, POINTER(c_void_p), c_int, c_int, c_void_p,
        POINTER(c_void_p), POINTER(c_void_p), c_int, c_int, c_void_p,
        c_int, POINTER(c_void_p), c_int64, POINTER(c_int)]
    lib.cusolverMgGetrs.restype = c_int

    _libcudart.cudaSetDevice.argtypes = [c_int]
    _libcudart.cudaSetDevice.restype = c_int
    _libcudart.cudaMalloc.argtypes = [POINTER(c_void_p), c_int64]
    _libcudart.cudaMalloc.restype = c_int
    _libcudart.cudaFree.argtypes = [c_void_p]
    _libcudart.cudaFree.restype = c_int
    _libcudart.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_int64, c_int]
    _libcudart.cudaMemcpy.restype = c_int
    _libcudart.cudaDeviceSynchronize.argtypes = []
    _libcudart.cudaDeviceSynchronize.restype = c_int

    _CUSOLVERMG_BOUND = True


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _cuda_dtype_for(arr: np.ndarray) -> int:
    if arr.dtype == np.complex128:
        return CUDA_C_64F
    if arr.dtype == np.complex64:
        return CUDA_C_32F
    if arr.dtype == np.float64:
        return CUDA_R_64F
    if arr.dtype == np.float32:
        return CUDA_R_32F
    raise ValueError('[error] unsupported dtype <{}> for cuSolverMg'.format(arr.dtype))


def _itemsize_for(dtype: np.dtype) -> int:
    return int(np.dtype(dtype).itemsize)


# ---------------------------------------------------------------------------
# Multi-GPU LU class
# ---------------------------------------------------------------------------

class MultiGPULU(object):
    """Block-cyclic distributed LU factorization across multiple GPUs.

    Currently supports cuSolverMg backend only. Magma and NCCL options
    are reserved (NotImplementedError on construction).
    """

    def __init__(self,
            n_gpus: int,
            backend: str = 'cusolvermg',
            device_ids: Optional[List[int]] = None) -> None:

        self.n_gpus = int(n_gpus)
        self.backend = backend
        self.device_ids = list(range(self.n_gpus)) if device_ids is None else list(device_ids)
        assert len(self.device_ids) == self.n_gpus, \
            '[error] <device_ids> length must equal <n_gpus>'

        # State filled by factor()
        self.handle: Optional[c_void_p] = None
        self.grid: Optional[c_void_p] = None
        self.descr: Optional[c_void_p] = None
        self.array_d_A: Optional[Any] = None
        self.array_d_IPIV: Optional[Any] = None
        self.array_d_work: Optional[Any] = None
        self.work_lwork: int = 0
        self.N: int = 0
        self.dtype: Optional[np.dtype] = None
        self.cuda_dtype: int = -1
        self.col_blk_size: int = 0
        self.tile_cols_per_gpu: List[int] = []

        if backend == 'cusolvermg':
            if not cusolvermg_available():
                raise RuntimeError(
                    '[error] cuSolverMg unavailable — '
                    '<libcusolverMg.so> or <libcudart.so> not found')
            _bind_cusolvermg()
        elif backend == 'magma':
            raise NotImplementedError('[error] magma backend not implemented')
        elif backend == 'nccl':
            raise NotImplementedError('[error] nccl backend not implemented')
        else:
            raise ValueError('[error] unknown <backend>: {}'.format(backend))

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _local_cols(self,
            N: int,
            blk: int,
            g: int) -> int:
        # Block-cyclic 1-D: blocks of size <blk> assigned round-robin to GPUs.
        # Returns total columns owned by GPU <g>.
        nblocks = (N + blk - 1) // blk
        cnt = 0
        for ib in range(nblocks):
            owner = ib % self.n_gpus
            if owner != g:
                continue
            start = ib * blk
            stop = min(N, start + blk)
            cnt += (stop - start)
        return cnt

    def _alloc_distributed(self,
            N: int,
            dtype: np.dtype) -> None:
        """Block-cyclic per-GPU tile allocation. Block size capped at 256."""

        itemsz = _itemsize_for(dtype)
        # cuSolverMg internal panel size: 256 is the value used in NVIDIA
        # samples. Smaller blk gives finer cyclic distribution but more
        # round-trips; larger blk hits internal limits.
        blk = int(os.environ.get('MNPBEM_VRAM_SHARE_BLK', '256'))
        # Don't use a block size larger than N/n_gpus rounded up to a multiple
        # of 32 (cuSolverMg requires multiple of warp).
        max_blk = max(32, ((N // self.n_gpus + 31) // 32) * 32)
        blk = min(blk, max_blk)
        if blk < 32:
            blk = 32
        self.col_blk_size = blk

        # Per-GPU local column count (varies because cyclic).
        local_cols = [self._local_cols(N, blk, g) for g in range(self.n_gpus)]
        # cuSolverMg requires every GPU to allocate enough to host its share
        # of blocks; round up to ceil(N / n_gpus / blk) * blk for safety.
        nblocks = (N + blk - 1) // blk
        max_blocks_per_gpu = (nblocks + self.n_gpus - 1) // self.n_gpus
        local_cols_alloc = max_blocks_per_gpu * blk
        self.tile_cols_per_gpu = local_cols
        self.local_cols_alloc = local_cols_alloc

        ptrs_A = (c_void_p * self.n_gpus)()
        ptrs_IPIV = (c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            nbytes_A = N * local_cols_alloc * itemsz
            ptrs_A[g] = c_void_p(_cuda_malloc(nbytes_A))
            nbytes_p = local_cols_alloc * 4
            ptrs_IPIV[g] = c_void_p(_cuda_malloc(nbytes_p))
        self.array_d_A = ptrs_A
        self.array_d_IPIV = ptrs_IPIV

    def _scatter_columns(self,
            A: np.ndarray) -> None:
        """Block-cyclic scatter of A columns to per-GPU tiles."""

        N = A.shape[0]
        blk = self.col_blk_size
        A_f = np.asfortranarray(A)
        # For each GPU, build a (N, local_cols_alloc) F-order tile by copying
        # the assigned blocks in cyclic order.
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.zeros((N, self.local_cols_alloc), dtype=A_f.dtype, order='F')
            local_offset = 0
            nblocks = (N + blk - 1) // blk
            for ib in range(nblocks):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk
                stop = min(N, start + blk)
                ncols = stop - start
                tile_full[:, local_offset:local_offset + ncols] = A_f[:, start:stop]
                local_offset += blk  # advance by full block (zero-pad trailing partial)
            ptr = int(self.array_d_A[g] or 0)
            _cuda_memcpy_h2d(ptr, tile_full)

    def _gather_columns(self,
            N: int,
            out: np.ndarray) -> None:
        """Block-cyclic gather of factored A back to host (Fortran-ordered)."""

        blk = self.col_blk_size
        itemsz = _itemsize_for(out.dtype)
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.empty((N, self.local_cols_alloc), dtype=out.dtype, order='F')
            ptr = int(self.array_d_A[g] or 0)
            nbytes = N * self.local_cols_alloc * itemsz
            _cuda_memcpy_d2h(tile_full, ptr, nbytes)
            local_offset = 0
            nblocks = (N + blk - 1) // blk
            for ib in range(nblocks):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk
                stop = min(N, start + blk)
                ncols = stop - start
                out[:, start:stop] = tile_full[:, local_offset:local_offset + ncols]
                local_offset += blk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def factor(self,
            A: np.ndarray) -> 'MultiGPULU':

        assert A.ndim == 2 and A.shape[0] == A.shape[1], \
            '[error] cuSolverMg LU requires square matrix'
        N = A.shape[0]
        self.N = N
        self.dtype = A.dtype
        self.cuda_dtype = _cuda_dtype_for(A)
        lib = _libcusolverMg

        # Create handle / grid / descriptor
        h = c_void_p(0)
        _check_status(lib.cusolverMgCreate(byref(h)), 'cusolverMgCreate')
        self.handle = h

        dev_arr_c = (c_int * self.n_gpus)(*self.device_ids)
        _check_status(
            lib.cusolverMgDeviceSelect(h, c_int(self.n_gpus), dev_arr_c),
            'cusolverMgDeviceSelect')

        grid = c_void_p(0)
        dev_arr32 = (c_int32 * self.n_gpus)(*self.device_ids)
        _check_status(
            lib.cusolverMgCreateDeviceGrid(
                byref(grid), c_int32(1), c_int32(self.n_gpus),
                dev_arr32, c_int(GRID_MAPPING_COL_MAJOR)),
            'cusolverMgCreateDeviceGrid')
        self.grid = grid

        # Allocate per-GPU buffers BEFORE building descriptor (descriptor needs
        # to know the block size). Block size = ceil(N/n_gpus).
        self._alloc_distributed(N, A.dtype)

        descr = c_void_p(0)
        _check_status(
            lib.cusolverMgCreateMatrixDesc(
                byref(descr), c_int64(N), c_int64(N),
                c_int64(N), c_int64(self.col_blk_size),
                c_int(self.cuda_dtype), grid),
            'cusolverMgCreateMatrixDesc')
        self.descr = descr

        # Scatter A to GPUs
        self._scatter_columns(A)

        # Query workspace size
        lwork = c_int64(0)
        _check_status(
            lib.cusolverMgGetrf_bufferSize(
                h, c_int(N), c_int(N), self.array_d_A,
                c_int(1), c_int(1), descr,
                self.array_d_IPIV, c_int(self.cuda_dtype), byref(lwork)),
            'cusolverMgGetrf_bufferSize')
        self.work_lwork = int(lwork.value)

        # Allocate workspace per GPU
        ptrs_w = (c_void_p * self.n_gpus)()
        itemsz = _itemsize_for(A.dtype)
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_w[g] = c_void_p(_cuda_malloc(self.work_lwork * itemsz))
        self.array_d_work = ptrs_w

        # Run getrf
        info = c_int(0)
        _check_status(
            lib.cusolverMgGetrf(
                h, c_int(N), c_int(N), self.array_d_A,
                c_int(1), c_int(1), descr,
                self.array_d_IPIV, c_int(self.cuda_dtype),
                self.array_d_work, c_int64(self.work_lwork), byref(info)),
            'cusolverMgGetrf')
        if info.value != 0:
            raise RuntimeError(
                '[error] cusolverMgGetrf reports info={} (singular or invalid)'.format(info.value))
        _cuda_device_sync()
        return self

    def solve(self,
            B: np.ndarray,
            trans: str = 'N') -> np.ndarray:

        assert self.handle is not None, '[error] solve() called before factor()'
        if B.ndim == 1:
            B = B.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False
        N = self.N
        nrhs = B.shape[1]
        assert B.shape[0] == N, '[error] B rows must equal N'
        assert B.dtype == self.dtype, \
            '[error] B dtype <{}> != A dtype <{}>'.format(B.dtype, self.dtype)
        lib = _libcusolverMg

        # B distribution: same block-cyclic partition as A but only over the
        # nrhs columns. Use min(blk, nrhs) so for small RHS the descriptor
        # block size stays valid.
        blk_rhs = min(self.col_blk_size, max(1, nrhs))
        nblocks_rhs = (nrhs + blk_rhs - 1) // blk_rhs
        max_blocks_per_gpu_rhs = (nblocks_rhs + self.n_gpus - 1) // self.n_gpus
        local_cols_alloc_rhs = max_blocks_per_gpu_rhs * blk_rhs
        if local_cols_alloc_rhs == 0:
            local_cols_alloc_rhs = blk_rhs

        itemsz = _itemsize_for(B.dtype)
        ptrs_B = (c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_B[g] = c_void_p(_cuda_malloc(N * local_cols_alloc_rhs * itemsz))

        B_f = np.asfortranarray(B)
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.zeros((N, local_cols_alloc_rhs), dtype=B_f.dtype, order='F')
            local_offset = 0
            for ib in range(nblocks_rhs):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk_rhs
                stop = min(nrhs, start + blk_rhs)
                ncols = stop - start
                tile_full[:, local_offset:local_offset + ncols] = B_f[:, start:stop]
                local_offset += blk_rhs
            _cuda_memcpy_h2d(int(ptrs_B[g] or 0), tile_full)

        # B descriptor
        descrB = c_void_p(0)
        _check_status(
            lib.cusolverMgCreateMatrixDesc(
                byref(descrB), c_int64(N), c_int64(nrhs),
                c_int64(N), c_int64(blk_rhs),
                c_int(self.cuda_dtype), self.grid),
            'cusolverMgCreateMatrixDesc(B)')

        op = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}[trans.upper()]

        # Query workspace
        lwork = c_int64(0)
        _check_status(
            lib.cusolverMgGetrs_bufferSize(
                self.handle, c_int(op), c_int(N), c_int(nrhs),
                self.array_d_A, c_int(1), c_int(1), self.descr,
                self.array_d_IPIV, ptrs_B, c_int(1), c_int(1), descrB,
                c_int(self.cuda_dtype), byref(lwork)),
            'cusolverMgGetrs_bufferSize')
        lwork_solve = int(lwork.value)

        ptrs_w_solve = (c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_w_solve[g] = c_void_p(_cuda_malloc(max(1, lwork_solve) * itemsz))

        info = c_int(0)
        _check_status(
            lib.cusolverMgGetrs(
                self.handle, c_int(op), c_int(N), c_int(nrhs),
                self.array_d_A, c_int(1), c_int(1), self.descr,
                self.array_d_IPIV, ptrs_B, c_int(1), c_int(1), descrB,
                c_int(self.cuda_dtype),
                ptrs_w_solve, c_int64(lwork_solve), byref(info)),
            'cusolverMgGetrs')
        if info.value != 0:
            raise RuntimeError(
                '[error] cusolverMgGetrs reports info={}'.format(info.value))
        _cuda_device_sync()

        # Gather B (now solution X) back, block-cyclic in reverse.
        X_f = np.empty((N, nrhs), dtype=B.dtype, order='F')
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.empty((N, local_cols_alloc_rhs), dtype=B.dtype, order='F')
            _cuda_memcpy_d2h(
                tile_full, int(ptrs_B[g] or 0), N * local_cols_alloc_rhs * itemsz)
            local_offset = 0
            for ib in range(nblocks_rhs):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk_rhs
                stop = min(nrhs, start + blk_rhs)
                ncols = stop - start
                X_f[:, start:stop] = tile_full[:, local_offset:local_offset + ncols]
                local_offset += blk_rhs

        # Free B tiles + workspace
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            _cuda_free(int(ptrs_B[g] or 0))
            _cuda_free(int(ptrs_w_solve[g] or 0))
        lib.cusolverMgDestroyMatrixDesc(descrB)

        X = np.ascontiguousarray(X_f)
        if squeeze:
            X = X[:, 0]
        return X

    def close(self) -> None:
        lib = _libcusolverMg
        # Synchronize all devices before tearing down so any in-flight kernels
        # don't trip subsequent cusolverMg calls.
        for dev in self.device_ids:
            try:
                _cuda_set_device(dev)
                _cuda_device_sync()
            except Exception:
                pass
        # Free per-GPU buffers
        if self.array_d_work is not None:
            for g, dev in enumerate(self.device_ids):
                try:
                    _cuda_set_device(dev)
                    _cuda_free(int(self.array_d_work[g] or 0))
                except Exception:
                    pass
            self.array_d_work = None
        if self.array_d_A is not None:
            for g, dev in enumerate(self.device_ids):
                try:
                    _cuda_set_device(dev)
                    _cuda_free(int(self.array_d_A[g] or 0))
                    _cuda_free(int(self.array_d_IPIV[g] or 0))
                except Exception:
                    pass
            self.array_d_A = None
            self.array_d_IPIV = None
        if self.descr is not None:
            lib.cusolverMgDestroyMatrixDesc(self.descr)
            self.descr = None
        if self.grid is not None:
            lib.cusolverMgDestroyGrid(self.grid)
            self.grid = None
        if self.handle is not None:
            lib.cusolverMgDestroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Detection / fallback helpers
# ---------------------------------------------------------------------------

def _detect_n_gpus() -> int:
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def factor_multi_gpu(A: np.ndarray,
        n_gpus: Optional[int] = None,
        backend: str = 'cusolvermg',
        device_ids: Optional[List[int]] = None) -> MultiGPULU:

    if n_gpus is None:
        n_gpus = _detect_n_gpus()
    if n_gpus < 2:
        raise ValueError(
            '[error] factor_multi_gpu requires <n_gpus>>=2, got {}'.format(n_gpus))
    lu = MultiGPULU(n_gpus, backend=backend, device_ids=device_ids)
    return lu.factor(A)


def solve_multi_gpu(A: np.ndarray,
        b: np.ndarray,
        n_gpus: Optional[int] = None,
        backend: str = 'cusolvermg',
        device_ids: Optional[List[int]] = None) -> np.ndarray:

    lu = factor_multi_gpu(A, n_gpus=n_gpus, backend=backend, device_ids=device_ids)
    try:
        return lu.solve(b)
    finally:
        lu.close()


def warn_fallback(reason: str) -> None:
    msg = '[info] VRAM-share multi-GPU LU unavailable ({}). Falling back.'.format(reason)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)

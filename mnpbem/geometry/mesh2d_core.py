import os
import sys
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse import csr_matrix

from ..utils.matlab_compat import msqrt, mround
from .mesh2d_utils import (
    mydelaunayn, inpoly, dist2poly, triarea, quality, circumcircle,
    fixmesh, findedge, mytsearch, tinterp, checkgeometry,
)
from .mesh2d_quadtree import quadtree


# ---------------------------------------------------------------------------
# Internal helpers (meshpoly)
# ---------------------------------------------------------------------------

def _tricentre(t: np.ndarray,
        f: np.ndarray) -> np.ndarray:

    # Interpolate nodal F to the centroid of the triangles T.
    # MATLAB Mesh2d/meshpoly.m > tricentre
    return (f[t[:, 0]] + f[t[:, 1]] + f[t[:, 2]]) / 3.0


def _longest(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # Return the length of the longest edge in each triangle.
    # MATLAB Mesh2d/meshpoly.m > longest
    d1 = np.sum((p[t[:, 1]] - p[t[:, 0]]) ** 2, axis = 1)
    d2 = np.sum((p[t[:, 2]] - p[t[:, 1]]) ** 2, axis = 1)
    d3 = np.sum((p[t[:, 0]] - p[t[:, 2]]) ** 2, axis = 1)
    return msqrt(np.maximum(np.maximum(d1, d2), d3))


def _getedges(t: np.ndarray,
        n: int) -> np.ndarray:

    # Get the unique edges and boundary edges in a triangulation.
    # MATLAB Mesh2d/meshpoly.m > getedges
    #
    # Algorithm:
    #   1. Form all triangle edges, sort each row
    #   2. Sort rows lexicographically (sortrows in MATLAB)
    #   3. Shared edges appear as consecutive identical pairs
    #   4. Boundary edges = non-shared, internal = shared (take one of each pair)
    #   5. Return [boundary_edges; unique_internal_edges]

    numt = t.shape[0]

    e_all = np.empty((3 * numt, 2), dtype = int)
    e_all[:numt] = np.sort(t[:, [0, 1]], axis = 1)
    e_all[numt:2 * numt] = np.sort(t[:, [0, 2]], axis = 1)
    e_all[2 * numt:] = np.sort(t[:, [1, 2]], axis = 1)

    # sortrows equivalent: lexsort by (col1, col0)
    sorted_idx = np.lexsort((e_all[:, 1], e_all[:, 0]))
    e_sorted = e_all[sorted_idx]

    # find shared edges (consecutive identical rows)
    is_shared = np.zeros(len(e_sorted), dtype = bool)
    same = np.all(e_sorted[:-1] == e_sorted[1:], axis = 1)
    is_shared[:-1] |= same
    is_shared[1:] |= same

    bnd = e_sorted[~is_shared]       # boundary edges
    internal = e_sorted[is_shared]    # internal edges (each appears twice)

    # take every other internal edge (they come in consecutive pairs)
    internal_unique = internal[::2]

    total_len = bnd.shape[0] + internal_unique.shape[0]
    e = np.empty((total_len, 2), dtype = int)
    e[:bnd.shape[0]] = bnd
    e[bnd.shape[0]:] = internal_unique

    return e


def _cdt(p: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # Approximate geometry-constrained Delaunay triangulation.
    # MATLAB Mesh2d/meshpoly.m > cdt
    t = mydelaunayn(p)

    # impose geometry constraints: keep triangles whose centroids are inside
    centroids = _tricentre(t, p)
    inside, _ = inpoly(centroids, node, edge)
    t = t[inside]

    return p, t


def _initmesh(p: np.ndarray,
        ph: np.ndarray,
        th: np.ndarray,
        hh: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Initialise the mesh nodes.
    # MATLAB Mesh2d/meshpoly.m > initmesh
    #
    # Boundary nodes for all geometry edges have been passed in.
    # Only take those in the current face.

    enum = findedge(p, node, edge, 1.0e-08)
    p = p[enum > 0]
    fix = np.arange(p.shape[0])

    # internal nodes from quadtree
    inside, on_bnd = inpoly(ph, node, edge)
    internal = inside & ~on_bnd

    n_fix = p.shape[0]
    n_internal = np.sum(internal)
    total_p = n_fix + n_internal
    p_combined = np.empty((total_p, 2))
    p_combined[:n_fix] = p
    p_combined[n_fix:] = ph[internal]
    p = p_combined

    tndx = np.zeros(p.shape[0], dtype = int)

    return p, fix, tndx


# ---------------------------------------------------------------------------
# Internal helpers (meshfaces)
# ---------------------------------------------------------------------------

def _boundarynodes(ph: np.ndarray,
        th: np.ndarray,
        hh: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray,
        output: bool = False) -> np.ndarray:

    # Discretise the geometry based on the edge size requirements
    # interpolated from the background mesh.
    # MATLAB Mesh2d/meshfaces.m > boundarynodes

    p = node.copy()
    e = edge.copy()

    # size function at geometry nodes
    i = mytsearch(ph[:, 0], ph[:, 1], th, p[:, 0], p[:, 1])
    h = tinterp(ph, th, hh, p, i)

    if output:
        print('[info] Placing Boundary Nodes')

    # iterative edge splitting
    for _iter in range(100):
        dxy = p[e[:, 1]] - p[e[:, 0]]
        L = msqrt(np.sum(dxy ** 2, axis = 1))
        he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

        ratio = L / np.maximum(he, np.finfo(float).eps)
        split = ratio >= 1.5

        if not np.any(split):
            break

        n1 = e[split, 0]
        n2 = e[split, 1]
        pm = 0.5 * (p[n1] + p[n2])
        n3 = np.arange(pm.shape[0]) + p.shape[0]

        # update edge list: split edge -> [n1, n3], add [n3, n2]
        e_new = e.copy()
        e_new[split, 1] = n3
        extra_edges = np.column_stack([n3, n2])

        total_e = e_new.shape[0] + extra_edges.shape[0]
        e_combined = np.empty((total_e, 2), dtype = int)
        e_combined[:e_new.shape[0]] = e_new
        e_combined[e_new.shape[0]:] = extra_edges
        e = e_combined

        # update node list
        total_p = p.shape[0] + pm.shape[0]
        p_combined = np.empty((total_p, 2))
        p_combined[:p.shape[0]] = p
        p_combined[p.shape[0]:] = pm
        p = p_combined

        # size function at new nodes
        i_new = mytsearch(ph[:, 0], ph[:, 1], th, pm[:, 0], pm[:, 1])
        h_new = tinterp(ph, th, hh, pm, i_new)
        total_h = h.shape[0] + h_new.shape[0]
        h_combined = np.empty(total_h)
        h_combined[:h.shape[0]] = h
        h_combined[h.shape[0]:] = h_new
        h = h_combined

    # build sparse node-to-edge connectivity matrix
    ne = e.shape[0]
    rows = np.empty(2 * ne, dtype = int)
    cols = np.empty(2 * ne, dtype = int)
    vals = np.empty(2 * ne)
    rows[:ne] = e[:, 0]
    cols[:ne] = np.arange(ne)
    vals[:ne] = -1.0
    rows[ne:] = e[:, 1]
    cols[ne:] = np.arange(ne)
    vals[ne:] = 1.0
    S = csr_matrix((vals, (rows, cols)), shape = (p.shape[0], ne))

    # smooth boundary nodes
    if output:
        print('[info] Smoothing Boundaries')

    nnode_orig = node.shape[0]
    dxy = p[e[:, 1]] - p[e[:, 0]]
    L = msqrt(np.sum(dxy ** 2, axis = 1))
    he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

    tol = 0.02
    maxit_bnd = 50
    delta = 0.0
    i_search = np.zeros(p.shape[0], dtype = int)

    for _iter in range(maxit_bnd):
        delta_old = delta

        # spring based smoothing
        F_ratio = he / np.maximum(L, np.finfo(float).eps) - 1.0
        Fxy = dxy * F_ratio[:, np.newaxis]
        Fp = S.dot(Fxy)

        # don't move original geometry nodes
        Fp[:nnode_orig] = 0.0
        p = p + 0.2 * Fp

        # convergence
        dxy = p[e[:, 1]] - p[e[:, 0]]
        Lnew = msqrt(np.sum(dxy ** 2, axis = 1))
        delta = np.max(np.abs((Lnew - L) / np.maximum(Lnew, np.finfo(float).eps)))

        if delta < tol:
            break
        else:
            if _iter == maxit_bnd - 1:
                print('[info] WARNING: Boundary smoothing did not converge.')

        L = Lnew

        if delta > delta_old:
            # re-interpolate size function at moved nodes
            i_search = mytsearch(ph[:, 0], ph[:, 1], th, p[:, 0], p[:, 1], i_search)
            h = tinterp(ph, th, hh, p, i_search)
            he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

    return p


def _getoptions(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    # Extract the user defined options.
    # MATLAB Mesh2d/meshfaces.m > getoptions
    defaults = {
        'mlim': 0.02,
        'maxit': 20,
        'dhmax': 0.3,
        'output': False,
        'debug': False,
    }

    if options is None:
        return defaults

    result = dict(defaults)
    for key in options:
        assert key in defaults, '[error] Invalid field <{}> in OPTIONS'.format(key)
        result[key] = options[key]

    return result


# ---------------------------------------------------------------------------
# meshpoly -- core iterative meshing
# ---------------------------------------------------------------------------

def meshpoly(node: np.ndarray,
        edge: np.ndarray,
        qtree_data: Dict[str, np.ndarray],
        p: np.ndarray,
        options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/meshpoly.m - core meshing routine
    # Do not call directly, use mesh2d or meshfaces instead.

    # --- constants (exact from MATLAB) ---
    shortedge = 0.75
    longedge = 1.5
    smalltri = 0.25
    largetri = 4.0
    qlimit = 0.5
    dt = 0.2  # smoothing step size

    qp = qtree_data['p']
    qt = qtree_data['t']
    qh = qtree_data['h']

    maxit = options.get('maxit', 20)
    mlim = options.get('mlim', 0.02)
    do_output = options.get('output', False)
    do_debug = options.get('debug', False)

    # initialise mesh
    if do_output:
        print('[info] Initialising Mesh')

    p, fix, tndx = _initmesh(p, qp, qt, qh, node, edge)

    t = np.empty((0, 3), dtype = int)

    # main loop
    if do_output:
        print('[info] Iteration   Convergence (%)')

    for iteration in range(maxit):

        # ensure unique node list (MATLAB meshpoly.m L78: unique(p,'rows') — no rounding)
        _, unique_idx, inverse_idx = np.unique(
            p, axis = 0,
            return_index = True, return_inverse = True)
        p = p[unique_idx]
        fix = inverse_idx[fix]
        tndx = tndx[unique_idx] if len(tndx) > len(unique_idx) - 1 else np.zeros(p.shape[0], dtype = int)

        # constrained Delaunay triangulation
        p, t = _cdt(p, node, edge)
        if t.shape[0] == 0:
            break

        # unique edges
        e = _getedges(t, p.shape[0])
        nume = e.shape[0]

        # sparse node-to-edge connectivity matrix
        # S has shape (n_nodes, n_edges), with +1 at e[:,0] and -1 at e[:,1]
        rows = np.empty(2 * nume, dtype = int)
        cols = np.empty(2 * nume, dtype = int)
        vals = np.empty(2 * nume)
        rows[:nume] = e[:, 0]
        cols[:nume] = np.arange(nume)
        vals[:nume] = 1.0
        rows[nume:] = e[:, 1]
        cols[nume:] = np.arange(nume)
        vals[nume:] = -1.0
        S = csr_matrix((vals, (rows, cols)), shape = (p.shape[0], nume))

        # size function at nodes via interpolation from background mesh
        tndx = mytsearch(qp[:, 0], qp[:, 1], qt, p[:, 0], p[:, 1],
                         tndx if len(tndx) == p.shape[0] else None)
        hn = tinterp(qp, qt, qh, p, tndx)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])

        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        # inner smoothing sub-iterations
        # MATLAB: for subiter = 1:(iter-1)
        # iter is 1-based in MATLAB, so when iter=1 -> 0 sub-iterations
        # iteration is 0-based here, so range(iteration) gives the same count
        done = False
        move = 1.0
        for _subiter in range(iteration):

            # spring based smoothing
            L0 = h * msqrt(np.sum(L ** 2) / np.sum(h ** 2))
            F = np.maximum(L0 / L - 1.0, -0.1)
            Fxy = edgev * F[:, np.newaxis]
            Fp = S.dot(Fxy)
            Fp[fix] = 0.0
            p = p + dt * Fp

            # measure convergence
            edgev = p[e[:, 0]] - p[e[:, 1]]
            L0_new = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)
            move = np.max(np.abs((L0_new - L) / L))
            L = L0_new

            if move < mlim:
                done = True
                break

        if do_output:
            convergence_pct = 100.0 * min(1.0, mlim / max(move, np.finfo(float).eps))
            print('[info] {:2d}           {:2.1f}'.format(iteration + 1, convergence_pct))

        # constrained Delaunay triangulation (after smoothing)
        p, t = _cdt(p, node, edge)
        if t.shape[0] == 0:
            break

        # unique edges
        e = _getedges(t, p.shape[0])

        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        # size function at nodes
        tndx = mytsearch(qp[:, 0], qp[:, 1], qt, p[:, 0], p[:, 1],
                         tndx if len(tndx) == p.shape[0] else None)
        hn = tinterp(qp, qt, qh, p, tndx)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])

        r = L / np.maximum(h, np.finfo(float).eps)

        # main loop convergence check
        if done and np.max(r) < 3.0:
            break
        else:
            if iteration == maxit - 1:
                print('[info] WARNING: Maximum number of iterations reached. '
                      'Solution did not converge!')

        # --- nodal density control ---
        if iteration < maxit - 1:

            # estimate required triangle area from size function
            Ah = 0.5 * _tricentre(t, hn) ** 2

            t_area = np.abs(triarea(p, t))

            # build sparse connectivity for edge-count check
            nume2 = e.shape[0]
            rows2 = np.empty(2 * nume2, dtype = int)
            cols2 = np.empty(2 * nume2, dtype = int)
            vals2 = np.empty(2 * nume2)
            rows2[:nume2] = e[:, 0]
            cols2[:nume2] = np.arange(nume2)
            vals2[:nume2] = 1.0
            rows2[nume2:] = e[:, 1]
            cols2[nume2:] = np.arange(nume2)
            vals2[nume2:] = 1.0
            S_abs = csr_matrix((np.abs(vals2), (rows2, cols2)),
                               shape = (p.shape[0], nume2))
            edge_count = np.array(S_abs.sum(axis = 1)).ravel()

            # --- remove nodes ---
            # triangles with small area
            small_idx = np.where(t_area < smalltri * Ah)[0]
            # nodes with less than 2 edge connections
            low_conn = np.where(edge_count < 2)[0]
            # short edges
            short_idx = np.where(r < shortedge)[0]

            prob = np.zeros(p.shape[0], dtype = bool)
            if len(short_idx) > 0:
                prob[e[short_idx].ravel()] = True
            if len(small_idx) > 0:
                prob[t[small_idx].ravel()] = True
            if len(low_conn) > 0:
                prob[low_conn] = True
            prob[fix] = False

            if np.any(prob):
                pnew = p[~prob]
                tndx_new = tndx[~prob] if len(tndx) == p.shape[0] else np.zeros(pnew.shape[0], dtype = int)

                # re-index fix to remain consistent
                j_remap = np.zeros(p.shape[0], dtype = int)
                j_remap[~prob] = 1
                j_remap = np.cumsum(j_remap) - 1
                fix = j_remap[fix]
                fix = fix[fix >= 0]
            else:
                pnew = p.copy()
                tndx_new = tndx.copy() if len(tndx) == p.shape[0] else np.zeros(pnew.shape[0], dtype = int)

            # --- add new nodes ---
            large_mask = t_area > largetri * Ah
            r_tri = _longest(p, t) / np.maximum(_tricentre(t, hn), np.finfo(float).eps)
            q_tri = quality(p, t)
            low_q_mask = (r_tri > longedge) & (q_tri < qlimit)

            if np.any(large_mask | low_q_mask):
                # separate large and low-quality (but not large)
                k_mask = low_q_mask & ~large_mask
                i_idx = np.where(large_mask)[0]
                k_idx = np.where(k_mask)[0]

                # circumcentres: large triangles first, then low-quality
                tri_indices_len = len(i_idx) + len(k_idx)
                tri_indices = np.empty(tri_indices_len, dtype = int)
                tri_indices[:len(i_idx)] = i_idx
                tri_indices[len(i_idx):] = k_idx

                cc = circumcircle(p, t[tri_indices])

                # don't add multiple points in one circumcircle
                # MATLAB logic: accept all large-tri circumcentres,
                # then for low-quality ones, check overlap with accepted
                n_large = len(i_idx)
                ok = np.zeros(cc.shape[0], dtype = bool)
                ok[:n_large] = True  # accept all large-triangle circumcentres

                for ii in range(n_large, cc.shape[0]):
                    x_cc = cc[ii, 0]
                    y_cc = cc[ii, 1]
                    is_inside = False
                    accepted = np.where(ok)[0]
                    for jj in range(len(accepted)):
                        kk = accepted[jj]
                        dx_sq = (x_cc - cc[kk, 0]) ** 2
                        if dx_sq < cc[kk, 2] and (dx_sq + (y_cc - cc[kk, 1]) ** 2) < cc[kk, 2]:
                            is_inside = True
                            break
                    if not is_inside:
                        ok[ii] = True

                cc = cc[ok]

                # only take internal points
                if cc.shape[0] > 0:
                    inside_cc, _ = inpoly(cc[:, :2], node, edge)
                    cc = cc[inside_cc]

                if cc.shape[0] > 0:
                    cc_points = cc[:, :2]
                    total_new = pnew.shape[0] + cc_points.shape[0]
                    p_combined = np.empty((total_new, 2))
                    p_combined[:pnew.shape[0]] = pnew
                    p_combined[pnew.shape[0]:] = cc_points
                    pnew = p_combined

                    total_tndx = tndx_new.shape[0] + cc_points.shape[0]
                    tndx_combined = np.empty(total_tndx, dtype = int)
                    tndx_combined[:tndx_new.shape[0]] = tndx_new
                    tndx_combined[tndx_new.shape[0]:] = 0
                    tndx_new = tndx_combined

            p = pnew
            tndx = tndx_new

    if do_debug:
        print('[info] meshpoly completed after {} iterations'.format(iteration + 1))

    return p, t


# ---------------------------------------------------------------------------
# meshfaces -- multi-face orchestrator
# ---------------------------------------------------------------------------

def meshfaces(node: np.ndarray,
        edge: np.ndarray,
        face: Optional[List[np.ndarray]],
        hdata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/meshfaces.m
    # 2D unstructured mesh generation for multi-face polygonal geometry.

    node = np.asarray(node, dtype = float)
    if edge is not None:
        edge = np.asarray(edge, dtype = int)

    opts = _getoptions(options)

    # check geometry
    if opts.get('output', False):
        print('[info] Checking Geometry')

    node, edge, face, hdata = checkgeometry(node, edge, face, hdata)

    # quadtree decomposition
    qt_p, qt_t, qt_h = quadtree(node, edge, hdata, opts['dhmax'],
                                 opts.get('output', False))
    qt = {'p': qt_p, 't': qt_t, 'h': qt_h}

    # discretise edges
    pbnd = _boundarynodes(qt_p, qt_t, qt_h, node, edge,
                          opts.get('output', False))

    # mesh each face separately
    p_all = np.empty((0, 2))
    t_all = np.empty((0, 3), dtype = int)
    fnum_all = np.empty(0, dtype = int)

    for k in range(len(face)):
        face_edges = edge[face[k]]
        pnew, tnew = meshpoly(node, face_edges, qt, pbnd, opts)

        if tnew.shape[0] > 0:
            tnew_shifted = tnew + p_all.shape[0]

            # append triangles
            total_t = t_all.shape[0] + tnew_shifted.shape[0]
            t_combined = np.empty((total_t, 3), dtype = int)
            t_combined[:t_all.shape[0]] = t_all
            t_combined[t_all.shape[0]:] = tnew_shifted
            t_all = t_combined

            # append nodes
            total_p = p_all.shape[0] + pnew.shape[0]
            p_combined = np.empty((total_p, 2))
            p_combined[:p_all.shape[0]] = p_all
            p_combined[p_all.shape[0]:] = pnew
            p_all = p_combined

            # face number (1-based, matching MATLAB convention)
            fnum_new = np.full(tnew.shape[0], k + 1, dtype = int)
            total_fnum = fnum_all.shape[0] + fnum_new.shape[0]
            fnum_combined = np.empty(total_fnum, dtype = int)
            fnum_combined[:fnum_all.shape[0]] = fnum_all
            fnum_combined[fnum_all.shape[0]:] = fnum_new
            fnum_all = fnum_combined

    # ensure consistent, CCW ordered triangulation
    if p_all.shape[0] > 0 and t_all.shape[0] > 0:
        p_all, t_all, _, fnum_all = fixmesh(p_all, t_all, tfun = fnum_all)

    # element quality
    if p_all.shape[0] > 0 and t_all.shape[0] > 0:
        q = quality(p_all, t_all)
        if opts.get('output', False):
            print('[info] Triangles: {}, Nodes: {}, '
                  'Mean quality: {:.4f}, Min quality: {:.4f}'.format(
                      t_all.shape[0], p_all.shape[0],
                      np.mean(q), np.min(q)))

    return p_all, t_all, fnum_all


# ---------------------------------------------------------------------------
# mesh2d -- main entry point
# ---------------------------------------------------------------------------

def mesh2d(node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        hdata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/mesh2d.m
    # 2D unstructured mesh generation for a polygon.
    # Assumes 1 face containing all edges.
    #
    # Returns (p, t):
    #   p : Nx2 array of nodal XY coordinates
    #   t : Mx3 array of triangles as indices into p (CCW ordered)

    p, t, _ = meshfaces(node, edge, None, hdata, options)
    return p, t


# ---------------------------------------------------------------------------
# smoothmesh -- Laplacian smoothing
# ---------------------------------------------------------------------------

def smoothmesh(p: np.ndarray,
        t: np.ndarray,
        maxit: int = 20,
        tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/smoothmesh.m
    # Laplacian smoothing of a triangular mesh.

    p = np.asarray(p, dtype = float).copy()
    t = np.asarray(t, dtype = int).copy()

    p, t, _, _ = fixmesh(p, t)

    n = p.shape[0]
    numt = t.shape[0]

    # sparse connectivity matrix S (n x n)
    # S[i,j] = 1 if nodes i and j are connected
    rows = np.empty(6 * numt, dtype = int)
    cols = np.empty(6 * numt, dtype = int)
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for idx, (a, b) in enumerate(pairs):
        rows[idx * numt:(idx + 1) * numt] = t[:, a]
        cols[idx * numt:(idx + 1) * numt] = t[:, b]

    S = csr_matrix((np.ones(6 * numt), (rows, cols)), shape = (n, n))
    W = np.array(S.sum(axis = 1)).ravel()

    assert not np.any(W == 0), '[error] Invalid mesh. Hanging nodes found.'

    # find boundary nodes via non-unique edge analysis
    edge_all = np.empty((3 * numt, 2), dtype = int)
    edge_all[:numt] = t[:, [0, 1]]
    edge_all[numt:2 * numt] = t[:, [0, 2]]
    edge_all[2 * numt:] = t[:, [1, 2]]
    edge_sorted = np.sort(edge_all, axis = 1)

    # sort and find shared/boundary edges
    sorted_idx = np.lexsort((edge_sorted[:, 1], edge_sorted[:, 0]))
    edge_sorted_2 = edge_sorted[sorted_idx]
    is_dup = np.zeros(len(edge_sorted_2), dtype = bool)
    same = np.all(edge_sorted_2[:-1] == edge_sorted_2[1:], axis = 1)
    is_dup[:-1] |= same
    is_dup[1:] |= same

    bnd_edges = edge_sorted_2[~is_dup]
    bnd_nodes = np.unique(bnd_edges.ravel())

    # unique edges for length computation
    internal_edges = edge_sorted_2[is_dup]
    internal_unique = internal_edges[::2]
    total_unique = bnd_edges.shape[0] + internal_unique.shape[0]
    unique_edges = np.empty((total_unique, 2), dtype = int)
    unique_edges[:bnd_edges.shape[0]] = bnd_edges
    unique_edges[bnd_edges.shape[0]:] = internal_unique

    L = np.maximum(
        msqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)),
        np.finfo(float).eps)

    for _iter in range(maxit):
        # Laplacian smoothing: pnew = (S * p) / W
        sp = S.dot(p)
        pnew = np.empty_like(p)
        pnew[:, 0] = sp[:, 0] / W
        pnew[:, 1] = sp[:, 1] / W
        pnew[bnd_nodes] = p[bnd_nodes]  # don't move boundary nodes
        p = pnew

        Lnew = np.maximum(
            msqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)),
            np.finfo(float).eps)
        move = np.max(np.abs((Lnew - L) / Lnew))

        if move < tol:
            break
        L = Lnew

    if _iter == maxit - 1 and move >= tol:
        print('[info] WARNING: Maximum number of iterations reached, '
              'solution did not converge!')

    return p, t


# ---------------------------------------------------------------------------
# refine_mesh -- quadtree refinement
# ---------------------------------------------------------------------------

def refine_mesh(p: np.ndarray,
        t: np.ndarray,
        ti: Optional[np.ndarray] = None,
        f: Optional[np.ndarray] = None) -> Union[
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    # MATLAB Mesh2d/refine.m
    # Quadtree triangle refinement: each selected triangle is split into
    # four sub-triangles by joining nodes at edge midpoints.
    # Neighbouring triangles are refined via bisection for compatibility.

    p = np.asarray(p, dtype = float).copy()
    t = np.asarray(t, dtype = int).copy()

    got_f = f is not None
    if got_f:
        f = np.asarray(f, dtype = float).copy()
        p, t, _, f = fixmesh(p, t, tfun = f)
    else:
        p, t, _, _ = fixmesh(p, t)

    if ti is None:
        ti = np.ones(t.shape[0], dtype = bool)
    else:
        ti = np.asarray(ti, dtype = bool)

    assert ti.shape[0] == t.shape[0], '[error] ti must have length == number of triangles'
    if got_f:
        assert f.shape[0] == p.shape[0], '[error] f must have length == number of nodes'

    numt = t.shape[0]
    vect = np.arange(numt)

    # edge connectivity
    e_all = np.empty((3 * numt, 2), dtype = int)
    e_all[:numt] = t[:, [0, 1]]
    e_all[numt:2 * numt] = t[:, [1, 2]]
    e_all[2 * numt:] = t[:, [2, 0]]

    e_sorted = np.sort(e_all, axis = 1)
    e_unique, j_map = np.unique(e_sorted, axis = 0, return_inverse = True)

    te = np.empty((numt, 3), dtype = int)
    te[:, 0] = j_map[vect]
    te[:, 1] = j_map[vect + numt]
    te[:, 2] = j_map[vect + 2 * numt]

    split = np.zeros(e_unique.shape[0], dtype = bool)
    split[te[ti].ravel()] = True

    # propagate splits to maintain mesh compatibility
    nsplit = np.sum(split)
    while True:
        split3 = np.sum(split[te].astype(int), axis = 1) >= 2
        split[te[split3].ravel()] = True
        new_count = np.sum(split)
        if new_count == nsplit:
            break
        nsplit = new_count

    split1 = np.sum(split[te].astype(int), axis = 1) == 1

    # new nodes at split edge midpoints
    n_nodes = p.shape[0]
    nsplit_count = np.sum(split)
    pm = 0.5 * (p[e_unique[split, 0]] + p[e_unique[split, 1]])

    total_p = n_nodes + nsplit_count
    p_new = np.empty((total_p, 2))
    p_new[:n_nodes] = p
    p_new[n_nodes:] = pm

    # map split edges to new node indices
    i_map = np.full(e_unique.shape[0], -1, dtype = int)
    i_map[split] = np.arange(nsplit_count) + n_nodes

    # --- build new triangles ---
    split3_mask = np.sum(split[te].astype(int), axis = 1) >= 2
    keep = ~(split1 | split3_mask)
    tnew_list = [t[keep]]

    # split3 case: split into 4 sub-triangles
    if np.any(split3_mask):
        n1 = t[split3_mask, 0]
        n2 = t[split3_mask, 1]
        n3 = t[split3_mask, 2]
        n4 = i_map[te[split3_mask, 0]]
        n5 = i_map[te[split3_mask, 1]]
        n6 = i_map[te[split3_mask, 2]]

        tnew_list.append(np.column_stack([n1, n4, n6]))
        tnew_list.append(np.column_stack([n4, n2, n5]))
        tnew_list.append(np.column_stack([n5, n3, n6]))
        tnew_list.append(np.column_stack([n4, n5, n6]))

    # split1 case: bisect into 2 sub-triangles
    if np.any(split1):
        split1_indices = np.where(split1)[0]
        for idx_k in split1_indices:
            col = -1
            for c in range(3):
                if split[te[idx_k, c]]:
                    col = c
                    break

            N1 = col
            N2 = (col + 1) % 3
            N3 = (col + 2) % 3

            nn1 = t[idx_k, N1]
            nn2 = t[idx_k, N2]
            nn3 = t[idx_k, N3]
            nn4 = i_map[te[idx_k, col]]

            tnew_list.append(np.array([[nn1, nn4, nn3], [nn4, nn2, nn3]]))

    t_new = np.vstack(tnew_list) if len(tnew_list) > 0 else np.empty((0, 3), dtype = int)

    # linear interpolation of nodal function to new nodes
    if got_f:
        n_new_nodes = nsplit_count
        f_new_total = f.shape[0] + n_new_nodes
        if f.ndim == 1:
            f_new = np.empty(f_new_total)
            f_new[:f.shape[0]] = f
            f_new[f.shape[0]:] = 0.5 * (f[e_unique[split, 0]] + f[e_unique[split, 1]])
        else:
            f_new = np.empty((f_new_total, f.shape[1]))
            f_new[:f.shape[0]] = f
            f_new[f.shape[0]:] = 0.5 * (f[e_unique[split, 0]] + f[e_unique[split, 1]])
        return p_new, t_new, f_new

    return p_new, t_new

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List
from scipy.spatial import Delaunay, ConvexHull


def inpoly(p: np.ndarray,
        node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        reltol: float = 1.0e-12) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/inpoly.m - point-in-polygon test using crossing number
    p = np.asarray(p, dtype = float)
    node = np.asarray(node, dtype = float)

    if p.ndim == 1:
        p = p.reshape(1, -1)

    assert p.shape[1] == 2, '[error] P must be an Nx2 array.'
    assert node.shape[1] == 2, '[error] NODE must be an Mx2 array.'

    nnode = node.shape[0]
    if edge is None:
        idx = np.arange(nnode)
        edge = np.empty((nnode, 2), dtype = int)
        edge[:nnode - 1, 0] = idx[:nnode - 1]
        edge[:nnode - 1, 1] = idx[1:nnode]
        edge[nnode - 1, 0] = nnode - 1
        edge[nnode - 1, 1] = 0
    else:
        edge = np.asarray(edge, dtype = int)

    assert edge.shape[1] == 2, '[error] EDGE must be an Mx2 array.'

    n = p.shape[0]
    nc = edge.shape[0]

    # choose direction with biggest range as y-coordinate
    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]]
        node = node[:, [1, 0]]

    # polygon bounding-box tolerance
    dxy_node = np.max(node, axis = 0) - np.min(node, axis = 0)
    tol = reltol * min(dxy_node)
    if tol == 0.0:
        tol = reltol

    # sort test points by y-value
    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]

    cn = np.zeros(n, dtype = bool)
    on = np.zeros(n, dtype = bool)

    for k in range(nc):
        n1 = edge[k, 0]
        n2 = edge[k, 1]

        y1 = node[n1, 1]
        y2 = node[n2, 1]
        if y1 < y2:
            x1 = node[n1, 0]
            x2 = node[n2, 0]
        else:
            yt = y1
            y1 = y2
            y2 = yt
            x1 = node[n2, 0]
            x2 = node[n1, 0]

        if x1 > x2:
            xmin = x2
            xmax = x1
        else:
            xmin = x1
            xmax = x2

        # binary search for first point with y >= y1
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            for _bs in range(n):
                start = (lower + upper) // 2
                if y[start] < y1:
                    lower = start + 1
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start - 1
            else:
                start = lower

        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                if X >= xmin:
                    if X <= xmax:
                        on[j] = on[j] or (abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) <= tol)
                        if (Y < y2) and ((y2 - y1) * (X - x1) < (Y - y1) * (x2 - x1)):
                            cn[j] = not cn[j]
                elif Y < y2:
                    cn[j] = not cn[j]
            else:
                break

    # re-index to undo the sorting
    result_cn = np.zeros(n, dtype = bool)
    result_on = np.zeros(n, dtype = bool)
    result_cn[sort_idx] = cn | on
    result_on[sort_idx] = on

    return result_cn, result_on


def triarea(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/triarea.m - signed triangle area (CCW positive)
    d12 = p[t[:, 1], :] - p[t[:, 0], :]
    d13 = p[t[:, 2], :] - p[t[:, 0], :]
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]
    return A


def quality(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/quality.m - triangle quality 0 <= q <= 1
    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]

    d12 = p2 - p1
    d13 = p3 - p1
    d23 = p3 - p2

    # 3.4641 = 4 * sqrt(3)
    q = 3.4641 * np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]) / np.sum(d12 ** 2 + d13 ** 2 + d23 ** 2, axis = 1)
    return q


def circumcircle(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/circumcircle.m - circumcircle center and radius^2
    cc = np.zeros((t.shape[0], 3))

    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]

    a1 = p2 - p1
    a2 = p3 - p1
    b1 = np.sum(a1 * (p2 + p1), axis = 1)
    b2 = np.sum(a2 * (p3 + p1), axis = 1)

    idet = 0.5 / (a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1] + np.finfo(float).eps)

    cc[:, 0] = (a2[:, 1] * b1 - a1[:, 1] * b2) * idet
    cc[:, 1] = (-a2[:, 0] * b1 + a1[:, 0] * b2) * idet
    cc[:, 2] = np.sum((p1 - cc[:, :2]) ** 2, axis = 1)

    return cc


def fixmesh(p: np.ndarray,
        t: np.ndarray,
        pfun: Optional[np.ndarray] = None,
        tfun: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    # MATLAB Mesh2d/fixmesh.m - clean up mesh
    TOL = 1.0e-10

    # remove duplicate nodes
    p_rounded = np.round(p / (TOL * 0.01)) * (TOL * 0.01)
    _, i_unique, j_map = np.unique(p_rounded, axis = 0, return_index = True, return_inverse = True)
    if pfun is not None:
        pfun = pfun[i_unique]
    p = p[i_unique]
    t = j_map[t]

    # triangle area
    A = triarea(p, t)
    Ai = A < 0.0
    Aj = np.abs(A) > TOL * np.max(np.abs(A)) if len(A) > 0 else np.ones(len(A), dtype = bool)

    # flip node numbering to give CCW order
    t_flip = t[Ai].copy()
    t_flip[:, [0, 1]] = t_flip[:, [1, 0]]
    t[Ai] = t_flip

    # remove zero area triangles
    t = t[Aj]
    if tfun is not None:
        tfun = tfun[Aj]

    # remove unused nodes
    used = np.unique(t.ravel())
    if len(used) < p.shape[0]:
        remap = np.full(p.shape[0], -1, dtype = int)
        remap[used] = np.arange(len(used))
        p = p[used]
        if pfun is not None:
            pfun = pfun[used]
        t = remap[t]

    return p, t, pfun, tfun


def smoothmesh(p: np.ndarray,
        t: np.ndarray,
        maxit: int = 20,
        tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/smoothmesh.m - Laplacian smoothing
    p, t, _, _ = fixmesh(p, t)

    n = p.shape[0]
    numt = t.shape[0]

    # sparse connectivity matrix
    rows = np.empty(6 * numt, dtype = int)
    cols = np.empty(6 * numt, dtype = int)
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for idx, (a, b) in enumerate(pairs):
        rows[idx * numt:(idx + 1) * numt] = t[:, a]
        cols[idx * numt:(idx + 1) * numt] = t[:, b]

    from scipy.sparse import csr_matrix
    S = csr_matrix((np.ones(6 * numt), (rows, cols)), shape = (n, n))
    W = np.array(S.sum(axis = 1)).ravel()

    # find boundary nodes
    edge_all = np.empty((3 * numt, 2), dtype = int)
    edge_all[:numt] = t[:, [0, 1]]
    edge_all[numt:2 * numt] = t[:, [0, 2]]
    edge_all[2 * numt:] = t[:, [1, 2]]
    edge_sorted = np.sort(edge_all, axis = 1)

    # find unique and boundary edges
    _, counts = np.unique(edge_sorted, axis = 0, return_counts = True)
    # use lexsort for proper identification
    sorted_idx = np.lexsort((edge_sorted[:, 1], edge_sorted[:, 0]))
    edge_sorted_2 = edge_sorted[sorted_idx]
    is_dup = np.zeros(len(edge_sorted_2), dtype = bool)
    is_dup[:-1] |= np.all(edge_sorted_2[:-1] == edge_sorted_2[1:], axis = 1)
    is_dup[1:] |= np.all(edge_sorted_2[:-1] == edge_sorted_2[1:], axis = 1)
    bnd_edges = edge_sorted_2[~is_dup]
    bnd_nodes = np.unique(bnd_edges.ravel())

    # unique edges for length computation
    unique_edges = np.unique(edge_sorted, axis = 0)

    L = np.maximum(np.sqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)), np.finfo(float).eps)

    for it in range(maxit):
        pnew = np.zeros_like(p)
        sp = S.dot(p)
        pnew[:, 0] = sp[:, 0] / W
        pnew[:, 1] = sp[:, 1] / W
        pnew[bnd_nodes] = p[bnd_nodes]
        p = pnew

        Lnew = np.maximum(np.sqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)), np.finfo(float).eps)
        move = np.max(np.abs((Lnew - L) / Lnew))
        if move < tol:
            break
        L = Lnew

    return p, t


def refine(p: np.ndarray,
        t: np.ndarray,
        ti: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/refine.m - refine triangulation (uniform or selective)
    p, t, _, _ = fixmesh(p, t)

    if ti is None:
        ti = np.ones(t.shape[0], dtype = bool)
    else:
        ti = np.asarray(ti, dtype = bool)

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

    # propagate splits to maintain compatibility
    while True:
        split3 = np.sum(split[te].astype(int), axis = 1) >= 2
        old_count = np.sum(split)
        split[te[split3].ravel()] = True
        if np.sum(split) == old_count:
            break

    split1 = np.sum(split[te].astype(int), axis = 1) == 1

    np_count = p.shape[0]
    nsplit = np.sum(split)
    pm = 0.5 * (p[e_unique[split, 0]] + p[e_unique[split, 1]])

    total_p = np_count + nsplit
    p_new = np.empty((total_p, 2))
    p_new[:np_count] = p
    p_new[np_count:] = pm

    # map split edges to new node indices
    i_map = np.full(e_unique.shape[0], -1, dtype = int)
    i_map[split] = np.arange(nsplit) + np_count

    # new triangles in split3 case
    tnew_list = []
    keep = ~(split1 | (np.sum(split[te].astype(int), axis = 1) >= 2))
    tnew_list.append(t[keep])

    split3_mask = np.sum(split[te].astype(int), axis = 1) >= 2
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

    # new triangles in split1 case
    if np.any(split1):
        split1_idx = np.where(split1)[0]
        for k in split1_idx:
            col = -1
            for c in range(3):
                if split[te[k, c]]:
                    col = c
                    break
            N1 = col
            N2 = (col + 1) % 3
            N3 = (col + 2) % 3
            nn1 = t[k, N1]
            nn2 = t[k, N2]
            nn3 = t[k, N3]
            nn4 = i_map[te[k, col]]
            tnew_list.append(np.array([[nn1, nn4, nn3], [nn4, nn2, nn3]]))

    t_new = np.vstack(tnew_list) if len(tnew_list) > 0 else np.empty((0, 3), dtype = int)

    return p_new, t_new


def connectivity(p: np.ndarray,
        t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/connectivity.m
    numt = t.shape[0]
    vect = np.arange(numt)

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

    # edge-to-triangle connectivity
    nume = e_unique.shape[0]
    e2t = np.zeros((nume, 2), dtype = int)
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == 0:
                e2t[ce, 0] = k + 1  # 1-indexed for MATLAB compatibility
            else:
                e2t[ce, 1] = k + 1

    # boundary nodes
    bnd = np.zeros(p.shape[0], dtype = bool)
    bnd_edges = e_unique[e2t[:, 1] == 0]
    bnd[bnd_edges.ravel()] = True

    return e_unique, te, e2t, bnd


def findedge(p: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray,
        tol: float = 1.0e-08) -> np.ndarray:

    # MATLAB Mesh2d/findedge.m - locate points on edges
    n = p.shape[0]
    nc = edge.shape[0]

    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]].copy()
        node = node[:, [1, 0]].copy()
    tol_abs = tol * min(dxy)

    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]

    enum = np.zeros(n, dtype = int)
    for k in range(nc):
        n1 = edge[k, 0]
        n2 = edge[k, 1]

        y1 = node[n1, 1]
        y2 = node[n2, 1]
        if y1 < y2:
            x1 = node[n1, 0]
            x2 = node[n2, 0]
        else:
            yt = y1
            y1 = y2
            y2 = yt
            x1 = node[n2, 0]
            x2 = node[n1, 0]

        # binary search
        if n == 0:
            continue
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            start = 0
            for _bs in range(n):
                start = (lower + upper) // 2
                if y[start] < y1:
                    lower = start + 1
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start - 1

        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                if abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) < tol_abs:
                    enum[j] = k + 1  # 1-indexed
            else:
                break

    result = np.zeros(n, dtype = int)
    result[sort_idx] = enum
    return result


def dist2poly(p: np.ndarray,
        edgexy: np.ndarray,
        lim: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/dist2poly.m - distance from points to polygon boundary
    np_count = p.shape[0]
    ne = edgexy.shape[0]

    if lim is None:
        lim = np.full(np_count, np.inf)
    else:
        lim = np.asarray(lim, dtype = float).copy()

    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]].copy()
        edgexy = edgexy[:, [1, 0, 3, 2]].copy()

    # ensure edgexy[:, [0,1]] has the lower y value
    swap = edgexy[:, 3] < edgexy[:, 1]
    edgexy_swap = edgexy[swap].copy()
    edgexy[swap] = edgexy_swap[:, [2, 3, 0, 1]]

    tol = 1000.0 * np.finfo(float).eps * max(dxy)
    L = np.zeros(np_count)

    for k in range(np_count):
        x_pt = p[k, 0]
        y_pt = p[k, 1]
        d = lim[k]

        for j in range(ne):
            y1 = edgexy[j, 1]
            y2 = edgexy[j, 3]
            if y2 < y_pt - d:
                continue
            if y1 > y_pt + d:
                continue

            x1 = edgexy[j, 0]
            x2 = edgexy[j, 2]
            xmin = min(x1, x2)
            xmax = max(x1, x2)

            if xmin > x_pt + d or xmax < x_pt - d:
                continue

            x2mx1 = x2 - x1
            y2my1 = y2 - y1
            denom = x2mx1 ** 2 + y2my1 ** 2
            if denom < np.finfo(float).eps:
                continue

            r = ((x_pt - x1) * x2mx1 + (y_pt - y1) * y2my1) / denom
            r = max(min(r, 1.0), 0.0)

            dj = (x1 + r * x2mx1 - x_pt) ** 2 + (y1 + r * y2my1 - y_pt) ** 2
            if dj < d ** 2 and dj > tol:
                d = np.sqrt(dj)

        L[k] = d

    return L


def _mydelaunayn(p: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/mydelaunayn.m - Delaunay triangulation with scaling
    maxxy = np.max(p, axis = 0)
    minxy = np.min(p, axis = 0)
    center = 0.5 * (minxy + maxxy)
    scale = 0.5 * min(maxxy - minxy)
    if scale < np.finfo(float).eps:
        scale = 1.0

    ps = (p - center) / scale

    try:
        tri = Delaunay(ps)
        t = tri.simplices
    except Exception:
        # add small jitter and retry
        jitter = np.random.randn(*ps.shape) * 1e-10
        tri = Delaunay(ps + jitter)
        t = tri.simplices

    return t


def _tricentre(t: np.ndarray,
        f: np.ndarray) -> np.ndarray:

    return (f[t[:, 0]] + f[t[:, 1]] + f[t[:, 2]]) / 3.0


def _longest(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    d1 = np.sum((p[t[:, 1]] - p[t[:, 0]]) ** 2, axis = 1)
    d2 = np.sum((p[t[:, 2]] - p[t[:, 1]]) ** 2, axis = 1)
    d3 = np.sum((p[t[:, 0]] - p[t[:, 2]]) ** 2, axis = 1)
    return np.sqrt(np.maximum(np.maximum(d1, d2), d3))


def _getedges(t: np.ndarray,
        n: int) -> np.ndarray:

    # unique edges and boundary edges
    e_all = np.empty((3 * t.shape[0], 2), dtype = int)
    e_all[:t.shape[0]] = np.sort(t[:, [0, 1]], axis = 1)
    e_all[t.shape[0]:2 * t.shape[0]] = np.sort(t[:, [0, 2]], axis = 1)
    e_all[2 * t.shape[0]:] = np.sort(t[:, [1, 2]], axis = 1)

    e_sorted_idx = np.lexsort((e_all[:, 1], e_all[:, 0]))
    e_sorted = e_all[e_sorted_idx]

    is_shared = np.zeros(len(e_sorted), dtype = bool)
    is_shared[:-1] |= np.all(e_sorted[:-1] == e_sorted[1:], axis = 1)
    is_shared[1:] |= np.all(e_sorted[:-1] == e_sorted[1:], axis = 1)

    bnd = e_sorted[~is_shared]
    internal = e_sorted[is_shared]

    # take every other internal edge (they come in pairs)
    internal_unique = internal[::2]

    total_len = bnd.shape[0] + internal_unique.shape[0]
    e = np.empty((total_len, 2), dtype = int)
    e[:bnd.shape[0]] = bnd
    e[bnd.shape[0]:] = internal_unique

    return e


def _rotate(p: np.ndarray,
        theta: float) -> np.ndarray:

    s = np.sin(theta)
    c = np.cos(theta)
    rot = np.array([[c, s], [-s, c]])
    return p @ rot


def _minrectangle(p: np.ndarray) -> float:

    n = p.shape[0]
    if n <= 2:
        return 0.0

    try:
        hull = ConvexHull(p)
        vertices = hull.vertices
        p_hull = p[vertices]
    except Exception:
        return 0.0

    # edges of convex hull
    n_hull = len(p_hull)
    best_theta = 0.0
    best_area = np.inf

    for k in range(n_hull):
        i1 = k
        i2 = (k + 1) % n_hull
        dxy = p_hull[i2] - p_hull[i1]
        ang = np.arctan2(dxy[1], dxy[0])
        theta = -ang

        pr = _rotate(p_hull, theta)
        dxy_r = np.max(pr, axis = 0) - np.min(pr, axis = 0)
        area = dxy_r[0] * dxy_r[1]
        if area < best_area:
            best_area = area
            best_theta = theta

    # ensure long axis aligned with Y
    pr = _rotate(p_hull, best_theta)
    dxy_r = np.max(pr, axis = 0) - np.min(pr, axis = 0)
    if dxy_r[0] > dxy_r[1]:
        best_theta += 0.5 * np.pi

    return best_theta


def _tinterp(p: np.ndarray,
        t: np.ndarray,
        f: np.ndarray,
        pi: np.ndarray,
        i: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/tinterp.m - triangle-based linear interpolation
    fi = np.zeros(pi.shape[0])

    out = (i < 0) | np.isnan(i.astype(float))
    if np.any(out):
        # nearest neighbour extrapolation
        from scipy.spatial import cKDTree
        tree = cKDTree(p)
        _, nn_idx = tree.query(pi[out])
        fi[out] = f[nn_idx]

    valid = ~out
    if np.any(valid):
        pin = pi[valid]
        tin = t[i[valid].astype(int)]

        t1 = tin[:, 0]
        t2 = tin[:, 1]
        t3 = tin[:, 2]

        dp1 = pin - p[t1]
        dp2 = pin - p[t2]
        dp3 = pin - p[t3]

        A3 = np.abs(dp1[:, 0] * dp2[:, 1] - dp1[:, 1] * dp2[:, 0])
        A2 = np.abs(dp1[:, 0] * dp3[:, 1] - dp1[:, 1] * dp3[:, 0])
        A1 = np.abs(dp3[:, 0] * dp2[:, 1] - dp3[:, 1] * dp2[:, 0])

        denom = A1 + A2 + A3
        denom[denom < np.finfo(float).eps] = np.finfo(float).eps
        fi[valid] = (A1 * f[t1] + A2 * f[t2] + A3 * f[t3]) / denom

    return fi


def _mytsearch(x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        i_guess: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/mytsearch.m - find enclosing triangle
    p = np.column_stack([x, y])
    pi = np.column_stack([xi, yi])

    # scale to avoid precision issues
    maxxy = np.max(p, axis = 0)
    minxy = np.min(p, axis = 0)
    den = 0.5 * min(maxxy - minxy) if min(maxxy - minxy) > 0 else 1.0

    ps = (p - 0.5 * (minxy + maxxy)) / den
    pis = (pi - 0.5 * (minxy + maxxy)) / den

    ni = len(xi)
    result = np.full(ni, -1, dtype = int)

    # check initial guess if provided
    if i_guess is not None and len(i_guess) == ni:
        valid_guess = (i_guess >= 0) & (i_guess < t.shape[0])
        if np.any(valid_guess):
            k_idx = np.where(valid_guess)[0]
            tri_idx = i_guess[k_idx]

            n1 = t[tri_idx, 0]
            n2 = t[tri_idx, 1]
            n3 = t[tri_idx, 2]

            # check if point is inside triangle using cross products
            def _sameside(xa, ya, xb, yb, x1, y1, x2, y2):
                dx = xb - xa
                dy = yb - ya
                a1 = (x1 - xa) * dy - (y1 - ya) * dx
                a2 = (x2 - xa) * dy - (y2 - ya) * dx
                return a1 * a2 >= 0.0

            ok = (_sameside(ps[n1, 0], ps[n1, 1], ps[n2, 0], ps[n2, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n3, 0], ps[n3, 1]) &
                  _sameside(ps[n2, 0], ps[n2, 1], ps[n3, 0], ps[n3, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n1, 0], ps[n1, 1]) &
                  _sameside(ps[n3, 0], ps[n3, 1], ps[n1, 0], ps[n1, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n2, 0], ps[n2, 1]))

            result[k_idx[ok]] = tri_idx[ok]

    # full search for points that failed
    need_search = result < 0
    if np.any(need_search):
        try:
            tri_obj = Delaunay(ps)
            found = tri_obj.find_simplex(pis[need_search])
            # map from Delaunay simplex to our triangulation
            # since we use a different triangulation, do direct search
        except Exception:
            pass

        # direct search using point-in-triangle test
        search_idx = np.where(need_search)[0]
        for k in range(t.shape[0]):
            if len(search_idx) == 0:
                break
            v0 = ps[t[k, 0]]
            v1 = ps[t[k, 1]]
            v2 = ps[t[k, 2]]

            # barycentric coordinates
            d00 = (v1[0] - v0[0]) * (v1[0] - v0[0]) + (v1[1] - v0[1]) * (v1[1] - v0[1])
            d01 = (v1[0] - v0[0]) * (v2[0] - v0[0]) + (v1[1] - v0[1]) * (v2[1] - v0[1])
            d11 = (v2[0] - v0[0]) * (v2[0] - v0[0]) + (v2[1] - v0[1]) * (v2[1] - v0[1])
            inv_denom = d00 * d11 - d01 * d01
            if abs(inv_denom) < np.finfo(float).eps:
                continue
            inv_denom = 1.0 / inv_denom

            pts = pis[search_idx]
            d20 = (pts[:, 0] - v0[0]) * (v1[0] - v0[0]) + (pts[:, 1] - v0[1]) * (v1[1] - v0[1])
            d21 = (pts[:, 0] - v0[0]) * (v2[0] - v0[0]) + (pts[:, 1] - v0[1]) * (v2[1] - v0[1])

            u = (d11 * d20 - d01 * d21) * inv_denom
            v = (d00 * d21 - d01 * d20) * inv_denom

            inside = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1.0 + 1e-10)
            result[search_idx[inside]] = k
            search_idx = search_idx[~inside]

    return result


def _userhfun(x: np.ndarray,
        y: np.ndarray,
        fun: Optional[Callable],
        args: List,
        hmax: float,
        xymin: np.ndarray,
        xymax: np.ndarray) -> np.ndarray:

    if fun is not None:
        h = fun(x, y, *args)
    else:
        h = np.full_like(x, np.inf)

    h = np.minimum(h, hmax)
    out = (x > xymax[0]) | (x < xymin[0]) | (y > xymax[1]) | (y < xymin[1])
    h[out] = np.inf
    return h


def _gethdata(hdata: Optional[Dict[str, Any]]) -> Tuple[float, Optional[np.ndarray], Optional[Callable], List]:

    d_hmax = np.inf
    d_edgeh = None
    d_fun = None
    d_args = []

    if hdata is None:
        return d_hmax, d_edgeh, d_fun, d_args

    hmax = hdata.get('hmax', d_hmax)
    edgeh = hdata.get('edgeh', d_edgeh)
    fun = hdata.get('fun', d_fun)
    args = hdata.get('args', d_args)

    return hmax, edgeh, fun, args


def quadtree(node: np.ndarray,
        edge: np.ndarray,
        hdata: Optional[Dict[str, Any]],
        dhmax: float,
        output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/quadtree.m - quadtree decomposition
    XYmax = np.max(node, axis = 0)
    XYmin = np.min(node, axis = 0)

    theta = _minrectangle(node)
    node_r = _rotate(node, theta)

    edgexy = np.column_stack([node_r[edge[:, 0]], node_r[edge[:, 1]]])

    hmax, edgeh, fun, args = _gethdata(hdata)

    # test points along edges
    wm = 0.5 * (edgexy[:, :2] + edgexy[:, 2:])
    edge_len = np.sqrt(np.sum((edgexy[:, 2:] - edgexy[:, :2]) ** 2, axis = 1))
    L = 2.0 * dist2poly(wm, edgexy, 2.0 * edge_len)

    # add more points where edges are close
    r = 2.0 * edge_len / np.maximum(L, np.finfo(float).eps)
    r = np.round((r - 2.0) / 2.0).astype(int)
    r = np.maximum(r, 0)
    add = np.where(r > 0)[0]

    if len(add) > 0:
        new_points = []
        new_L = []
        for j in add:
            ce = j
            num = r[ce]
            tmp = (np.arange(1, num + 1)) / (num + 1)

            x1, y1 = edgexy[ce, 0], edgexy[ce, 1]
            x2, y2 = edgexy[ce, 2], edgexy[ce, 3]
            xm, ym = wm[ce, 0], wm[ce, 1]

            pts1 = np.column_stack([x1 + tmp * (xm - x1), y1 + tmp * (ym - y1)])
            pts2 = np.column_stack([xm + tmp * (x2 - xm), ym + tmp * (y2 - ym)])

            total = pts1.shape[0] + pts2.shape[0]
            combined = np.empty((total, 2))
            combined[:pts1.shape[0]] = pts1
            combined[pts1.shape[0]:] = pts2

            new_points.append(combined)
            new_L.append(np.full(total, L[ce]))

        if new_points:
            new_pts = np.vstack(new_points)
            new_ls = np.hstack(new_L)

            total_wm = wm.shape[0] + new_pts.shape[0]
            wm_new = np.empty((total_wm, 2))
            wm_new[:wm.shape[0]] = wm
            wm_new[wm.shape[0]:] = new_pts
            wm = wm_new

            new_L_computed = dist2poly(new_pts, edgexy, new_ls)
            total_L = L.shape[0] + new_L_computed.shape[0]
            L_new = np.empty(total_L)
            L_new[:L.shape[0]] = L
            L_new[L.shape[0]:] = new_L_computed
            L = L_new

    # sort by y-value
    sort_idx = np.argsort(wm[:, 1])
    wm = wm[sort_idx]
    L = L[sort_idx]
    nw = wm.shape[0]

    # quadtree decomposition
    xymin = np.min(edgexy.reshape(-1, 2), axis = 0)
    xymax = np.max(edgexy.reshape(-1, 2), axis = 0)

    dim = 2.0 * max(xymax - xymin)
    xm = 0.5 * (xymin[0] + xymax[0])
    ym = 0.5 * (xymin[1] + xymax[1])

    # initial bounding box
    p_list = [
        np.array([xm - 0.5 * dim, ym - 0.5 * dim]),
        np.array([xm + 0.5 * dim, ym - 0.5 * dim]),
        np.array([xm + 0.5 * dim, ym + 0.5 * dim]),
        np.array([xm - 0.5 * dim, ym + 0.5 * dim])
    ]
    p_arr = np.array(p_list)
    b_list = [[0, 1, 2, 3]]

    # user defined size function at initial nodes
    pr = _rotate(p_arr, -theta)
    h_arr = _userhfun(pr[:, 0], pr[:, 1], fun, args, hmax, XYmin, XYmax).tolist()

    # iterative subdivision
    max_iter = 100
    for _iter in range(max_iter):
        new_boxes = []
        changed = False

        for m in range(len(b_list)):
            n1, n2, n3, n4 = b_list[m]
            x1 = p_arr[n1, 0]
            y1 = p_arr[n1, 1]
            x2 = p_arr[n2, 0]
            y4 = p_arr[n4, 1]

            # binary search for first wm with y >= y1
            if nw == 0 or wm[0, 1] >= y1:
                start = 0
            elif wm[nw - 1, 1] < y1:
                start = nw
            else:
                lower_b = 0
                upper_b = nw - 1
                start = 0
                for _bs in range(nw):
                    start = (lower_b + upper_b) // 2
                    if wm[start, 1] < y1:
                        lower_b = start + 1
                    elif start > 0 and wm[start - 1, 1] < y1:
                        break
                    else:
                        upper_b = max(start - 1, lower_b)
                        if lower_b > upper_b:
                            start = lower_b
                            break

            # min LFS in box
            LFS = 1.5 * min(h_arr[n1], h_arr[n2], h_arr[n3], h_arr[n4])

            for i in range(start, nw):
                if wm[i, 1] <= y4:
                    if wm[i, 0] >= x1 and wm[i, 0] <= x2 and L[i] < LFS:
                        LFS = L[i]
                else:
                    break

            # split box
            if (x2 - x1) >= LFS:
                changed = True
                xm_box = x1 + 0.5 * (x2 - x1)
                ym_box = y1 + 0.5 * (y4 - y1)

                np_start = len(p_arr)
                new_nodes = np.array([
                    [xm_box, ym_box],
                    [xm_box, y1],
                    [x2, ym_box],
                    [xm_box, y4],
                    [x1, ym_box]
                ])
                p_arr = np.vstack([p_arr, new_nodes])

                # user size function at new nodes
                pr_new = _rotate(new_nodes, -theta)
                h_new = _userhfun(pr_new[:, 0], pr_new[:, 1], fun, args, hmax, XYmin, XYmax)
                h_arr.extend(h_new.tolist())

                c = np_start  # center
                s = np_start + 1  # south
                e = np_start + 2  # east
                nn = np_start + 3  # north
                w = np_start + 4  # west

                b_list[m] = [n1, s, c, w]  # box 1
                new_boxes.append([s, n2, e, c])  # box 2
                new_boxes.append([c, e, n3, nn])  # box 3
                new_boxes.append([w, c, nn, n4])  # box 4

        b_list.extend(new_boxes)

        if not changed:
            break

    # remove duplicate nodes
    p_rounded = np.round(p_arr * 1e10) / 1e10
    _, unique_idx, remap = np.unique(p_rounded, axis = 0, return_index = True, return_inverse = True)
    p_arr = p_arr[unique_idx]
    h_arr_np = np.array(h_arr)[unique_idx]

    b_arr = np.array(b_list)
    b_arr = remap[b_arr]

    # form size function based on edge lengths
    e_set = set()
    for box in b_arr:
        for i in range(4):
            j = (i + 1) % 4
            e_pair = (min(box[i], box[j]), max(box[i], box[j]))
            e_set.add(e_pair)

    edges = np.array(list(e_set))
    if len(edges) == 0:
        # degenerate case
        p_out = _rotate(p_arr, -theta)
        t_out = np.array([[0, 1, 2]])
        h_out = h_arr_np
        return p_out, t_out, h_out

    L_edges = np.sqrt(np.sum((p_arr[edges[:, 0]] - p_arr[edges[:, 1]]) ** 2, axis = 1))

    for k in range(len(edges)):
        lk = L_edges[k]
        if lk < h_arr_np[edges[k, 0]]:
            h_arr_np[edges[k, 0]] = lk
        if lk < h_arr_np[edges[k, 1]]:
            h_arr_np[edges[k, 1]] = lk

    h_arr_np = np.minimum(h_arr_np, hmax)

    # gradient limiting
    tol_grad = 1.0e-06
    for _git in range(1000):
        h_old = h_arr_np.copy()
        for k in range(len(edges)):
            n1e = edges[k, 0]
            n2e = edges[k, 1]
            lk = L_edges[k]
            if h_arr_np[n1e] > h_arr_np[n2e]:
                dh = (h_arr_np[n1e] - h_arr_np[n2e]) / lk
                if dh > dhmax:
                    h_arr_np[n1e] = h_arr_np[n2e] + dhmax * lk
            else:
                dh = (h_arr_np[n2e] - h_arr_np[n1e]) / lk
                if dh > dhmax:
                    h_arr_np[n2e] = h_arr_np[n1e] + dhmax * lk

        max_change = np.max(np.abs((h_arr_np - h_old) / np.maximum(h_arr_np, np.finfo(float).eps)))
        if max_change < tol_grad:
            break

    # triangulate quadtree
    if len(b_arr) == 1:
        t_arr = np.array([[b_arr[0, 0], b_arr[0, 1], b_arr[0, 2]],
                           [b_arr[0, 0], b_arr[0, 2], b_arr[0, 3]]])
    else:
        # check for regular boxes (all corners have <= 4 connections)
        n2n_count = np.zeros(len(p_arr), dtype = int)
        for k in range(len(edges)):
            n2n_count[edges[k, 0]] += 1
            n2n_count[edges[k, 1]] += 1

        t_list = []
        for box in b_arr:
            # simple triangulation: split box into 2 triangles
            t_list.append([box[0], box[1], box[2]])
            t_list.append([box[0], box[2], box[3]])

        t_arr = np.array(t_list, dtype = int)

    # remove bad nodes
    good = h_arr_np > 0
    if not np.all(good):
        good_idx = np.where(good)[0]
        remap2 = np.full(len(p_arr), -1, dtype = int)
        remap2[good_idx] = np.arange(len(good_idx))
        p_arr = p_arr[good_idx]
        h_arr_np = h_arr_np[good_idx]

        valid_tri = np.all(np.isin(t_arr, good_idx), axis = 1)
        t_arr = remap2[t_arr[valid_tri]]

    # undo rotation
    p_arr = _rotate(p_arr, -theta)

    return p_arr, t_arr, h_arr_np


def _boundarynodes(ph: np.ndarray,
        th: np.ndarray,
        hh: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/meshfaces.m > boundarynodes
    p = node.copy()
    e = edge.copy()

    # get size function at geometry nodes
    i = _mytsearch(ph[:, 0], ph[:, 1], th, p[:, 0], p[:, 1])
    h = _tinterp(ph, th, hh, p, i)

    for _iter in range(100):
        dxy = p[e[:, 1]] - p[e[:, 0]]
        L = np.sqrt(np.sum(dxy ** 2, axis = 1))
        he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

        ratio = L / np.maximum(he, np.finfo(float).eps)
        split = ratio >= 1.5

        if not np.any(split):
            break

        n1 = e[split, 0]
        n2 = e[split, 1]
        pm = 0.5 * (p[n1] + p[n2])
        n3 = np.arange(pm.shape[0]) + p.shape[0]

        e_new = e.copy()
        e_new[split, 1] = n3
        n_split = np.sum(split)
        extra_edges = np.column_stack([n3, n2])
        total_e = e_new.shape[0] + extra_edges.shape[0]
        e_combined = np.empty((total_e, 2), dtype = int)
        e_combined[:e_new.shape[0]] = e_new
        e_combined[e_new.shape[0]:] = extra_edges
        e = e_combined

        total_p = p.shape[0] + pm.shape[0]
        p_combined = np.empty((total_p, 2))
        p_combined[:p.shape[0]] = p
        p_combined[p.shape[0]:] = pm
        p = p_combined

        i_new = _mytsearch(ph[:, 0], ph[:, 1], th, pm[:, 0], pm[:, 1])
        h_new = _tinterp(ph, th, hh, pm, i_new)
        total_h = h.shape[0] + h_new.shape[0]
        h_combined = np.empty(total_h)
        h_combined[:h.shape[0]] = h
        h_combined[h.shape[0]:] = h_new
        h = h_combined

    # spring-based boundary smoothing
    ne = e.shape[0]
    nnode_orig = node.shape[0]
    dxy = p[e[:, 1]] - p[e[:, 0]]
    L = np.sqrt(np.sum(dxy ** 2, axis = 1))
    he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

    for _iter in range(50):
        F_x = np.zeros(p.shape[0])
        F_y = np.zeros(p.shape[0])
        factor = he / np.maximum(L, np.finfo(float).eps) - 1.0

        fx = dxy[:, 0] * factor
        fy = dxy[:, 1] * factor

        np.add.at(F_x, e[:, 0], -fx)
        np.add.at(F_x, e[:, 1], fx)
        np.add.at(F_y, e[:, 0], -fy)
        np.add.at(F_y, e[:, 1], fy)

        F_x[:nnode_orig] = 0.0
        F_y[:nnode_orig] = 0.0

        p[:, 0] += 0.2 * F_x
        p[:, 1] += 0.2 * F_y

        dxy = p[e[:, 1]] - p[e[:, 0]]
        Lnew = np.sqrt(np.sum(dxy ** 2, axis = 1))
        delta = np.max(np.abs((Lnew - L) / np.maximum(Lnew, np.finfo(float).eps)))
        if delta < 0.02:
            break
        L = Lnew

    return p


def _cdt(p: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # constrained Delaunay triangulation (approximate)
    t = _mydelaunayn(p)

    # only keep triangles with internal centroids
    centroids = _tricentre(t, p)
    inside, _ = inpoly(centroids, node, edge)
    t = t[inside]

    return p, t


def meshpoly(node: np.ndarray,
        edge: np.ndarray,
        qtree: Dict[str, np.ndarray],
        p: np.ndarray,
        options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/meshpoly.m - core meshing routine
    shortedge = 0.75
    longedge = 1.5
    smalltri = 0.25
    largetri = 4.0
    qlimit = 0.5
    dt = 0.2

    # initialize mesh
    enum = findedge(p, node, edge, 1.0e-08)
    p = p[enum > 0]
    fix = np.arange(p.shape[0])

    # add internal nodes from quadtree
    inside, on_bnd = inpoly(qtree['p'], node, edge)
    internal = inside & ~on_bnd

    total_p = p.shape[0] + np.sum(internal)
    p_combined = np.empty((total_p, 2))
    p_combined[:p.shape[0]] = p
    p_combined[p.shape[0]:] = qtree['p'][internal]
    p = p_combined

    tndx = np.zeros(p.shape[0], dtype = int)

    for iteration in range(options.get('maxit', 20)):
        # ensure unique node list
        _, unique_idx, inverse_idx = np.unique(np.round(p * 1e10) / 1e10, axis = 0, return_index = True, return_inverse = True)
        p = p[unique_idx]
        fix_new = np.unique(inverse_idx[fix])
        fix_new = fix_new[fix_new < p.shape[0]]
        fix = fix_new
        tndx = tndx[unique_idx] if len(tndx) >= len(unique_idx) else np.zeros(p.shape[0], dtype = int)

        # constrained Delaunay
        p, t = _cdt(p, node, edge)

        if t.shape[0] == 0:
            break

        e = _getedges(t, p.shape[0])
        nume = e.shape[0]

        # sparse connectivity
        from scipy.sparse import csr_matrix as csr
        rows = np.empty(2 * nume, dtype = int)
        cols = np.empty(2 * nume, dtype = int)
        vals = np.empty(2 * nume)
        rows[:nume] = e[:, 0]
        cols[:nume] = np.arange(nume)
        vals[:nume] = 1.0
        rows[nume:] = e[:, 1]
        cols[nume:] = np.arange(nume)
        vals[nume:] = -1.0
        S = csr((vals, (rows, cols)), shape = (p.shape[0], nume))

        # size function interpolation
        tndx_full = _mytsearch(qtree['p'][:, 0], qtree['p'][:, 1], qtree['t'], p[:, 0], p[:, 1], tndx if len(tndx) == p.shape[0] else None)
        hn = _tinterp(qtree['p'], qtree['t'], qtree['h'], p, tndx_full)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])

        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(np.sqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        # inner smoothing
        done = False
        for subiter in range(max(iteration, 1)):
            L0_target = h * np.sqrt(np.sum(L ** 2) / np.sum(h ** 2))
            F = np.maximum(L0_target / L - 1.0, -0.1)
            Fxy = edgev * F[:, np.newaxis]
            Fp = S.dot(Fxy)

            Fp[fix] = 0.0
            p = p + dt * Fp

            edgev = p[e[:, 0]] - p[e[:, 1]]
            L0_new = np.maximum(np.sqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)
            move = np.max(np.abs((L0_new - L) / L))
            L = L0_new

            mlim = options.get('mlim', 0.02)
            if move < mlim:
                done = True
                break

        # re-triangulate after smoothing
        p, t = _cdt(p, node, edge)
        if t.shape[0] == 0:
            break

        e = _getedges(t, p.shape[0])
        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(np.sqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        tndx_full = _mytsearch(qtree['p'][:, 0], qtree['p'][:, 1], qtree['t'], p[:, 0], p[:, 1])
        hn = _tinterp(qtree['p'], qtree['t'], qtree['h'], p, tndx_full)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])
        tndx = tndx_full

        r = L / np.maximum(h, np.finfo(float).eps)
        if done and np.max(r) < 3.0:
            break

        # nodal density control
        if iteration < options.get('maxit', 20) - 1:
            Ah = 0.5 * _tricentre(t, hn) ** 2
            t_area = np.abs(triarea(p, t))

            # remove nodes
            small_tri = np.where(t_area < smalltri * Ah)[0]
            short_edges = np.where(r < shortedge)[0]

            prob = np.zeros(p.shape[0], dtype = bool)
            if len(short_edges) > 0:
                prob[e[short_edges].ravel()] = True
            if len(small_tri) > 0:
                prob[t[small_tri].ravel()] = True
            prob[fix] = False

            pnew = p[~prob]
            tndx_new = tndx[~prob] if len(tndx) == p.shape[0] else np.zeros(pnew.shape[0], dtype = int)

            # re-index fix
            remap_arr = np.zeros(p.shape[0], dtype = int)
            remap_arr[~prob] = 1
            remap_arr = np.cumsum(remap_arr) - 1
            fix = remap_arr[fix]
            fix = fix[fix >= 0]

            # add new nodes at circumcentres of large/low-quality triangles
            large_tri = t_area > largetri * Ah
            r_tri = _longest(p, t) / np.maximum(_tricentre(t, hn), np.finfo(float).eps)
            q = quality(p, t)
            low_quality = (r_tri > longedge) & (q < qlimit)

            add_mask = large_tri | low_quality
            if np.any(add_mask):
                cc = circumcircle(p, t[add_mask])
                cc_points = cc[:, :2]

                # only keep internal points
                inside_cc, _ = inpoly(cc_points, node, edge)
                cc_points = cc_points[inside_cc]

                if len(cc_points) > 0:
                    total_new = pnew.shape[0] + cc_points.shape[0]
                    p_combined2 = np.empty((total_new, 2))
                    p_combined2[:pnew.shape[0]] = pnew
                    p_combined2[pnew.shape[0]:] = cc_points
                    pnew = p_combined2

                    total_tndx = tndx_new.shape[0] + cc_points.shape[0]
                    tndx_combined = np.empty(total_tndx, dtype = int)
                    tndx_combined[:tndx_new.shape[0]] = tndx_new
                    tndx_combined[tndx_new.shape[0]:] = 0
                    tndx_new = tndx_combined

            p = pnew
            tndx = tndx_new

    return p, t


def _checkgeometry(node: np.ndarray,
        edge: Optional[np.ndarray],
        face: Optional[List[np.ndarray]],
        hdata: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Optional[Dict[str, Any]]]:

    nnode = node.shape[0]
    if edge is None:
        idx = np.arange(nnode)
        edge = np.empty((nnode, 2), dtype = int)
        edge[:nnode - 1, 0] = idx[:nnode - 1]
        edge[:nnode - 1, 1] = idx[1:nnode]
        edge[nnode - 1] = [nnode - 1, 0]

    if face is None:
        face = [np.arange(edge.shape[0])]

    # remove duplicate nodes
    _, unique_idx, remap = np.unique(np.round(node * 1e10) / 1e10, axis = 0, return_index = True, return_inverse = True)
    if len(unique_idx) < nnode:
        node = node[unique_idx]
        edge = remap[edge]

    # remove duplicate edges
    e_sorted = np.sort(edge, axis = 1)
    _, unique_e_idx = np.unique(e_sorted, axis = 0, return_index = True)
    edge = edge[unique_e_idx]

    return node, edge, face, hdata


def _getoptions(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    defaults = {
        'mlim': 0.02,
        'maxit': 20,
        'dhmax': 0.3,
        'output': False,
        'debug': False,
    }

    if options is None:
        return defaults

    for key in defaults:
        if key not in options:
            options[key] = defaults[key]

    if 'debug' not in options:
        options['debug'] = False

    return options


def mesh2d(node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        hdata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/mesh2d.m -> meshfaces -> meshpoly
    node = np.asarray(node, dtype = float)
    if edge is not None:
        edge = np.asarray(edge, dtype = int)

    opts = _getoptions(options)
    node, edge, face, hdata = _checkgeometry(node, edge, None, hdata)

    # quadtree decomposition
    qt_p, qt_t, qt_h = quadtree(node, edge, hdata, opts['dhmax'], opts.get('output', False))
    qt = {'p': qt_p, 't': qt_t, 'h': qt_h}

    # boundary nodes
    pbnd = _boundarynodes(qt_p, qt_t, qt_h, node, edge)

    # mesh each face
    p_all = np.empty((0, 2))
    t_all = np.empty((0, 3), dtype = int)

    for k in range(len(face)):
        face_edges = edge[face[k]]
        pnew, tnew = meshpoly(node, face_edges, qt, pbnd, opts)

        if tnew.shape[0] > 0:
            tnew_shifted = tnew + p_all.shape[0]
            total_t = t_all.shape[0] + tnew_shifted.shape[0]
            t_combined = np.empty((total_t, 3), dtype = int)
            t_combined[:t_all.shape[0]] = t_all
            t_combined[t_all.shape[0]:] = tnew_shifted
            t_all = t_combined

            total_p = p_all.shape[0] + pnew.shape[0]
            p_combined = np.empty((total_p, 2))
            p_combined[:p_all.shape[0]] = p_all
            p_combined[p_all.shape[0]:] = pnew
            p_all = p_combined

    # fix mesh
    if p_all.shape[0] > 0 and t_all.shape[0] > 0:
        p_all, t_all, _, _ = fixmesh(p_all, t_all)

    return p_all, t_all

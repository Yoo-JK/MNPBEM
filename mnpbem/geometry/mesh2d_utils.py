import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from scipy.spatial import Delaunay


def mydelaunayn(p: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/mydelaunayn.m - Delaunay triangulation with coordinate normalization
    # Translate to origin and scale min xy range onto [-1, 1]
    # This is absolutely critical to avoid precision issues for large problems
    p = np.asarray(p, dtype = float).copy()

    maxxy = np.max(p, axis = 0)
    minxy = np.min(p, axis = 0)

    p[:, 0] = p[:, 0] - 0.5 * (minxy[0] + maxxy[0])
    p[:, 1] = p[:, 1] - 0.5 * (minxy[1] + maxxy[1])
    scale = 0.5 * min(maxxy - minxy)
    if scale < np.finfo(float).eps:
        scale = 1.0
    p = p / scale

    try:
        tri = Delaunay(p)
        t = tri.simplices.copy()
    except Exception:
        # if default fails, add small jitter and retry (analogous to MATLAB QJ option)
        jitter = np.random.randn(*p.shape) * 1e-10
        tri = Delaunay(p + jitter)
        t = tri.simplices.copy()

    return t


def triarea(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/triarea.m - signed triangle area assuming CCW ordering
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)

    d12 = p[t[:, 1]] - p[t[:, 0]]
    d13 = p[t[:, 2]] - p[t[:, 0]]
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]

    return A


def quality(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/quality.m - approximate triangle quality, 0 <= q <= 1
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)

    p1 = p[t[:, 0]]
    p2 = p[t[:, 1]]
    p3 = p[t[:, 2]]

    d12 = p2 - p1
    d13 = p3 - p1
    d23 = p3 - p2

    # 3.4641 = 4 * sqrt(3), quality factor for equilateral triangle normalization
    q = 3.4641 * np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]) / np.sum(d12 ** 2 + d13 ** 2 + d23 ** 2, axis = 1)

    return q


def circumcircle(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/circumcircle.m - circumcircle center and radius^2
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)

    cc = np.zeros((t.shape[0], 3))

    p1 = p[t[:, 0]]
    p2 = p[t[:, 1]]
    p3 = p[t[:, 2]]

    # set equation for center: [a1; a2] * [x; y] = [b1; b2] * 0.5
    a1 = p2 - p1
    a2 = p3 - p1
    b1 = np.sum(a1 * (p2 + p1), axis = 1)
    b2 = np.sum(a2 * (p3 + p1), axis = 1)

    # explicit inversion
    idet = 0.5 / (a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1])

    # circumcentre XY
    cc[:, 0] = (a2[:, 1] * b1 - a1[:, 1] * b2) * idet
    cc[:, 1] = (-a2[:, 0] * b1 + a1[:, 0] * b2) * idet

    # radius^2
    cc[:, 2] = np.sum((p1 - cc[:, :2]) ** 2, axis = 1)

    return cc


def fixmesh(p: np.ndarray,
        t: np.ndarray,
        pfun: Optional[np.ndarray] = None,
        tfun: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    # MATLAB Mesh2d/fixmesh.m - ensure triangular mesh data is consistent
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)

    TOL = 1.0e-10

    assert p.ndim == 2 and p.shape[1] == 2, '[error] P must be an Nx2 array'
    assert t.ndim == 2 and t.shape[1] == 3, '[error] T must be an Mx3 array'
    assert np.min(t) >= 0 and np.max(t) < p.shape[0], '[error] Invalid T'

    if pfun is not None:
        pfun = np.asarray(pfun, dtype = float)
        assert pfun.shape[0] == p.shape[0], '[error] PFUN must be an NxK array'

    if tfun is not None:
        tfun = np.asarray(tfun, dtype = float)
        assert tfun.shape[0] == t.shape[0], '[error] TFUN must be an MxK array'

    # remove duplicate nodes
    _, unique_idx, remap = np.unique(p, axis = 0, return_index = True, return_inverse = True)
    if pfun is not None:
        pfun = pfun[unique_idx]
    p = p[unique_idx]
    t = remap[t].reshape(t.shape)

    # triangle area
    A = triarea(p, t)
    Ai = A < 0.0
    Aj = np.abs(A) > TOL * np.max(np.abs(A))

    # flip node numbering to give CCW order
    t[Ai, 0], t[Ai, 1] = t[Ai, 1].copy(), t[Ai, 0].copy()

    # remove zero area triangles
    t = t[Aj]
    if tfun is not None:
        tfun = tfun[Aj]

    # remove un-used nodes
    used_nodes = np.unique(t.ravel())
    if pfun is not None:
        pfun = pfun[used_nodes]
    p = p[used_nodes]

    # re-index triangles
    remap2 = np.zeros(used_nodes.max() + 1 if len(used_nodes) > 0 else 1, dtype = int)
    remap2[used_nodes] = np.arange(len(used_nodes))
    t = remap2[t].reshape(t.shape)

    return p, t, pfun, tfun


def inpoly(p: np.ndarray,
        node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        reltol: float = 1.0e-12) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/inpoly.m - point-in-polygon using crossing number with binary search
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
    assert np.max(edge) < nnode and np.min(edge) >= 0, '[error] Invalid EDGE.'

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

        # endpoints sorted so that y1 <= y2
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

        # binary search to find first point with y >= y1
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            for _bs in range(n):
                start = int(round(0.5 * (lower + upper)))
                if y[start] < y1:
                    lower = start + 1
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start - 1

        # loop through points
        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                if X >= xmin:
                    if X <= xmax:
                        # check if on edge
                        on[j] = on[j] or (abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) <= tol)
                        # intersection test
                        if (Y < y2) and ((y2 - y1) * (X - x1) < (Y - y1) * (x2 - x1)):
                            cn[j] = not cn[j]
                    # intentionally no else here for xmax case
                elif Y < y2:
                    # point is left of edge bbox, must cross
                    cn[j] = not cn[j]
            else:
                break

    # re-index to undo sorting
    result_cn = np.zeros(n, dtype = bool)
    result_on = np.zeros(n, dtype = bool)
    result_cn[sort_idx] = cn | on
    result_on[sort_idx] = on

    return result_cn, result_on


def dist2poly(p: np.ndarray,
        edgexy: np.ndarray,
        lim: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/dist2poly.m - distance from points to polygon edges using sweep-line
    p = np.asarray(p, dtype = float)
    edgexy = np.asarray(edgexy, dtype = float)

    np_ = p.shape[0]
    ne = edgexy.shape[0]

    if lim is None:
        lim = np.full(np_, np.inf)
    else:
        lim = np.asarray(lim, dtype = float).copy()
        if lim.ndim == 0:
            lim = np.full(np_, float(lim))

    # choose direction with biggest range as y-coordinate
    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]]
        edgexy = edgexy[:, [1, 0, 3, 2]]

    # ensure edgexy[:, [0,1]] contains the lower y value
    swap = edgexy[:, 3] < edgexy[:, 1]
    edgexy[swap] = edgexy[swap][:, [2, 3, 0, 1]]

    # sort edges by lower y value
    idx_lower = np.argsort(edgexy[:, 1])
    edgexy_lower = edgexy[idx_lower]

    # sort edges by upper y value
    idx_upper = np.argsort(edgexy[:, 3])
    edgexy_upper = edgexy[idx_upper]

    # mean edge y value
    ymean = 0.5 * np.sum(edgexy[:, 1] + edgexy[:, 3]) / ne

    # tolerance
    tol = 1000.0 * np.finfo(float).eps * max(dxy)

    L = np.zeros(np_)

    for k in range(np_):
        x_pt = p[k, 0]
        y_pt = p[k, 1]
        d = lim[k]

        if y_pt < ymean:
            # loop through edges bottom up
            for j in range(ne):
                y2 = edgexy_lower[j, 3]
                if y2 >= (y_pt - d):
                    y1 = edgexy_lower[j, 1]
                    if y1 <= (y_pt + d):
                        x1 = edgexy_lower[j, 0]
                        x2 = edgexy_lower[j, 2]

                        if x1 < x2:
                            xmin = x1
                            xmax = x2
                        else:
                            xmin = x2
                            xmax = x1

                        if xmin <= (x_pt + d) and xmax >= (x_pt - d):
                            x2mx1 = x2 - x1
                            y2my1 = y2 - y1

                            r = ((x_pt - x1) * x2mx1 + (y_pt - y1) * y2my1) / (x2mx1 ** 2 + y2my1 ** 2)
                            if r > 1.0:
                                r = 1.0
                            elif r < 0.0:
                                r = 0.0

                            dj = (x1 + r * x2mx1 - x_pt) ** 2 + (y1 + r * y2my1 - y_pt) ** 2
                            if (dj < d ** 2) and (dj > tol):
                                d = np.sqrt(dj)
                    else:
                        break
        else:
            # loop through edges top down
            for j in range(ne - 1, -1, -1):
                y1 = edgexy_upper[j, 1]
                if y1 <= (y_pt + d):
                    y2 = edgexy_upper[j, 3]
                    if y2 >= (y_pt - d):
                        x1 = edgexy_upper[j, 0]
                        x2 = edgexy_upper[j, 2]

                        if x1 < x2:
                            xmin = x1
                            xmax = x2
                        else:
                            xmin = x2
                            xmax = x1

                        if xmin <= (x_pt + d) and xmax >= (x_pt - d):
                            x2mx1 = x2 - x1
                            y2my1 = y2 - y1

                            r = ((x_pt - x1) * x2mx1 + (y_pt - y1) * y2my1) / (x2mx1 ** 2 + y2my1 ** 2)
                            if r > 1.0:
                                r = 1.0
                            elif r < 0.0:
                                r = 0.0

                            dj = (x1 + r * x2mx1 - x_pt) ** 2 + (y1 + r * y2my1 - y_pt) ** 2
                            if (dj < d ** 2) and (dj > tol):
                                d = np.sqrt(dj)
                    else:
                        break

        L[k] = d

    return L


def findedge(p: np.ndarray,
        node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        TOL: float = 1.0e-08) -> np.ndarray:

    # MATLAB Mesh2d/findedge.m - find which edge a point lies on
    p = np.asarray(p, dtype = float)
    node = np.asarray(node, dtype = float)

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
    assert np.max(edge) < nnode and np.min(edge) >= 0, '[error] Invalid EDGE.'

    n = p.shape[0]
    nc = edge.shape[0]

    # choose direction with biggest range as y-coordinate
    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]]
        node = node[:, [1, 0]]

    tol = TOL * min(dxy)

    # sort test points by y-value
    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]

    # 0 means not on any edge; edge indices are 1-based like MATLAB
    enum = np.zeros(n, dtype = int)

    for k in range(nc):
        n1 = edge[k, 0]
        n2 = edge[k, 1]

        # endpoints sorted so that y1 <= y2
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

        # binary search to find first point with y >= y1
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            for _bs in range(n):
                start = int(round(0.5 * (lower + upper)))
                if y[start] < y1:
                    lower = start
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start

        # loop through points
        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                # check if on edge (cross product test)
                if abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) < tol:
                    enum[j] = k + 1  # 1-based edge index
            else:
                break

    # re-index to undo sorting
    result = np.zeros(n, dtype = int)
    result[sort_idx] = enum

    return result


def _sameside(xa: np.ndarray,
        ya: np.ndarray,
        xb: np.ndarray,
        yb: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/mytsearch.m helper - test if points lie on same side of line AB
    dx = xb - xa
    dy = yb - ya
    a1 = (x1 - xa) * dy - (y1 - ya) * dx
    a2 = (x2 - xa) * dy - (y2 - ya) * dx
    return a1 * a2 >= 0.0


def mytsearch(x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        iguess: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/mytsearch.m - find enclosing triangle with normalization
    x = np.asarray(x, dtype = float).ravel()
    y = np.asarray(y, dtype = float).ravel()
    t = np.asarray(t, dtype = int)
    xi = np.asarray(xi, dtype = float).ravel()
    yi = np.asarray(yi, dtype = float).ravel()

    ni = len(xi)
    assert len(x) == len(y), '[error] Wrong input dimensions'
    assert len(xi) == len(yi), '[error] Wrong input dimensions'

    if iguess is not None:
        iguess = np.asarray(iguess, dtype = float).ravel()
        assert len(iguess) == ni, '[error] Wrong input dimensions'

    # translate to origin and scale min xy range onto [-1, 1]
    xy = np.column_stack([x, y])
    maxxy = np.max(xy, axis = 0)
    minxy = np.min(xy, axis = 0)
    den = 0.5 * min(maxxy - minxy)
    if den < np.finfo(float).eps:
        den = 1.0

    center = 0.5 * (minxy + maxxy)
    x = (x - center[0]) / den
    y = (y - center[1]) / den
    xi = (xi - center[0]) / den
    yi = (yi - center[1]) / den

    result = np.full(ni, np.nan)

    # check initial guess
    if iguess is not None:
        k = np.where((iguess > 0) & ~np.isnan(iguess))[0]
        if len(k) > 0:
            tri = iguess[k].astype(int) - 1  # convert from 1-based to 0-based

            valid = (tri >= 0) & (tri < t.shape[0])
            k = k[valid]
            tri = tri[valid]

            if len(k) > 0:
                n1 = t[tri, 0]
                n2 = t[tri, 1]
                n3 = t[tri, 2]

                ok = (_sameside(x[n1], y[n1], x[n2], y[n2], xi[k], yi[k], x[n3], y[n3]) &
                      _sameside(x[n2], y[n2], x[n3], y[n3], xi[k], yi[k], x[n1], y[n1]) &
                      _sameside(x[n3], y[n3], x[n1], y[n1], xi[k], yi[k], x[n2], y[n2]))

                j = np.ones(ni, dtype = bool)
                j[k[ok]] = False
    else:
        j = np.ones(ni, dtype = bool)

    # full search for points that failed - using inpolygon approach like MATLAB
    if np.any(j):
        from matplotlib.path import Path
        for k_tri in range(t.shape[0]):
            verts_x = x[t[k_tri]]
            verts_y = y[t[k_tri]]
            poly_verts = np.column_stack([verts_x, verts_y])
            poly_path = Path(np.vstack([poly_verts, poly_verts[0]]))

            pts = np.column_stack([xi, yi])
            temp = poly_path.contains_points(pts, radius = 1e-10)
            result[temp] = k_tri + 1  # 1-based index like MATLAB

    return result


def tinterp(p: np.ndarray,
        t: np.ndarray,
        f: np.ndarray,
        pi_pts: np.ndarray,
        i: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/tinterp.m - barycentric interpolation within triangles
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)
    f = np.asarray(f, dtype = float).ravel()
    pi_pts = np.asarray(pi_pts, dtype = float)
    i = np.asarray(i, dtype = float).ravel()

    fi = np.zeros(pi_pts.shape[0])

    # deal with points outside convex hull (NaN index)
    out = np.isnan(i)
    if np.any(out):
        # nearest neighbour extrapolation
        from scipy.spatial import cKDTree
        tree = cKDTree(p)
        _, nn_idx = tree.query(pi_pts[out])
        fi[out] = f[nn_idx]

    # keep internal points
    valid = ~out
    if np.any(valid):
        pin = pi_pts[valid]
        # convert 1-based to 0-based index
        tin = t[i[valid].astype(int) - 1]

        t1 = tin[:, 0]
        t2 = tin[:, 1]
        t3 = tin[:, 2]

        # sub-triangle areas for barycentric interpolation
        dp1 = pin - p[t1]
        dp2 = pin - p[t2]
        dp3 = pin - p[t3]

        A3 = np.abs(dp1[:, 0] * dp2[:, 1] - dp1[:, 1] * dp2[:, 0])
        A2 = np.abs(dp1[:, 0] * dp3[:, 1] - dp1[:, 1] * dp3[:, 0])
        A1 = np.abs(dp3[:, 0] * dp2[:, 1] - dp3[:, 1] * dp2[:, 0])

        fi[valid] = (A1 * f[t1] + A2 * f[t2] + A3 * f[t3]) / (A1 + A2 + A3)

    return fi


def checkgeometry(node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        face: Optional[List[np.ndarray]] = None,
        hdata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Optional[Dict[str, Any]]]:

    # MATLAB Mesh2d/checkgeometry.m - validate/repair geometry
    node = np.asarray(node, dtype = float)
    assert node.ndim == 2 and node.shape[1] == 2, '[error] NODE must be an Nx2 array'

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

    assert edge.ndim == 2 and edge.shape[1] == 2, '[error] EDGE must be an Mx2 array'
    assert np.max(edge) < nnode and np.min(edge) >= 0, '[error] Invalid EDGE'

    if face is None:
        face = [np.arange(edge.shape[0])]

    for k_face in range(len(face)):
        face_k = np.asarray(face[k_face])
        assert len(face_k) > 0 and np.min(face_k) >= 0 and np.max(face_k) < edge.shape[0], \
            '[error] Invalid FACE[{}]'.format(k_face)

    edgeh = False
    if hdata is not None and 'edgeh' in hdata and hdata['edgeh'] is not None and len(hdata['edgeh']) > 0:
        edgeh = True

    # remove un-used nodes and re-index
    used = np.unique(edge.ravel())
    del_count = nnode - (used[-1] + 1) if len(used) > 0 else 0
    if del_count > 0:
        node = node[used]
        j = np.zeros(nnode, dtype = int)
        j[used] = 1
        j = np.cumsum(j) - 1
        edge = j[edge]

        # remove self-loop edges
        valid_e = edge[:, 0] != edge[:, 1]
        j_e = np.zeros(edge.shape[0], dtype = int)
        j_e[valid_e] = 1
        j_e = np.cumsum(j_e) - 1
        edge = edge[valid_e]

        for k_face in range(len(face)):
            face[k_face] = np.unique(j_e[face[k_face]])

        print('[info] WARNING: {} un-used node(s) removed'.format(del_count))
        nnode = node.shape[0]

    # remove duplicate nodes and re-index
    _, unique_idx, remap = np.unique(node, axis = 0, return_index = True, return_inverse = True)
    del_count = nnode - len(unique_idx)
    if del_count > 0:
        node = node[unique_idx]
        edge = remap[edge]

        valid_e = edge[:, 0] != edge[:, 1]
        j_e = np.zeros(edge.shape[0], dtype = int)
        j_e[valid_e] = 1
        j_e = np.cumsum(j_e) - 1
        edge = edge[valid_e]

        for k_face in range(len(face)):
            face[k_face] = np.unique(j_e[face[k_face]])

        print('[info] WARNING: {} duplicate node(s) removed'.format(del_count))
        nnode = node.shape[0]

    # remove duplicate edges
    nedge = edge.shape[0]
    e_sorted = np.sort(edge, axis = 1)
    _, unique_e_idx, e_remap = np.unique(e_sorted, axis = 0, return_index = True, return_inverse = True)

    if edgeh:
        hdata['edgeh'][:, 0] = e_remap[hdata['edgeh'][:, 0].astype(int)]
        j_e = np.zeros(nedge, dtype = int)
        j_e[unique_e_idx] = 1
        j_e = np.cumsum(j_e) - 1
        edge = edge[unique_e_idx]

        for k_face in range(len(face)):
            face[k_face] = np.unique(j_e[face[k_face]])

    del_count = nedge - edge.shape[0] if edgeh else nedge - len(unique_e_idx)
    if not edgeh:
        edge = edge[unique_e_idx]
    if del_count > 0:
        print('[info] WARNING: {} duplicate edge(s) removed'.format(del_count))

    # check for closed geometry loops using sparse node-to-edge connectivity
    nedge = edge.shape[0]
    from scipy.sparse import csr_matrix
    rows = np.empty(2 * nedge, dtype = int)
    cols = np.empty(2 * nedge, dtype = int)
    rows[:nedge] = edge[:, 0]
    rows[nedge:] = edge[:, 1]
    cols[:nedge] = np.arange(nedge)
    cols[nedge:] = np.arange(nedge)
    vals = np.ones(2 * nedge, dtype = int)
    S = csr_matrix((vals, (rows, cols)), shape = (nnode, nedge))

    node_edge_count = np.array(S.sum(axis = 1)).ravel()
    open_nodes = np.where(node_edge_count < 2)[0]
    if len(open_nodes) > 0:
        raise ValueError('[error] Open geometry contours detected at node(s): {}'.format(open_nodes.tolist()))

    return node, edge, face, hdata


def connectivity(p: np.ndarray,
        t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/connectivity.m - build edge connectivity
    p = np.asarray(p, dtype = float)
    t = np.asarray(t, dtype = int)

    assert p.ndim == 2 and p.shape[1] == 2, '[error] P must be an Nx2 array'
    assert t.ndim == 2 and t.shape[1] == 3, '[error] T must be an Mx3 array'
    assert np.min(t) >= 0 and np.max(t) < p.shape[0], '[error] Invalid T'

    numt = t.shape[0]
    vect = np.arange(numt)

    # all edges (not unique)
    e_all = np.empty((3 * numt, 2), dtype = int)
    e_all[:numt] = t[:, [0, 1]]
    e_all[numt:2 * numt] = t[:, [1, 2]]
    e_all[2 * numt:] = t[:, [2, 0]]

    # unique edges
    e_sorted = np.sort(e_all, axis = 1)
    e, j_map = np.unique(e_sorted, axis = 0, return_inverse = True)

    # unique edges in each triangle
    te = np.empty((numt, 3), dtype = int)
    te[:, 0] = j_map[vect]
    te[:, 1] = j_map[vect + numt]
    te[:, 2] = j_map[vect + 2 * numt]

    # edge-to-triangle connectivity (boundary edges have e2t[i, 1] == 0)
    nume = e.shape[0]
    e2t = np.zeros((nume, 2), dtype = int)

    for k in range(numt):
        j = 0
        while j <= 2:
            ce = te[k, j]
            if e2t[ce, 0] == 0:
                e2t[ce, 0] = k + 1  # 1-based like MATLAB
            else:
                e2t[ce, 1] = k + 1
            j += 1

    # flag boundary nodes
    bnd = np.zeros(p.shape[0], dtype = bool)
    bnd_edges = e[e2t[:, 1] == 0]
    bnd[bnd_edges.ravel()] = True

    return e, te, e2t, bnd

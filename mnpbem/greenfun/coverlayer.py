import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np


def refine(p: Any,
        layer: Any,
        **options: Any) -> Dict[str, Any]:

    pos = p.pos
    z = pos[:, 2]

    # Find faces close to layer interface
    zmin, ind = layer.mindist(z)

    # Mask for faces close to the layer
    close_mask = zmin < layer.ztol * 3

    return {
        'close_mask': close_mask,
        'zmin': zmin,
        'ind': ind
    }


def refineret(p1: Any,
        p2: Any,
        layer: Any,
        **options: Any) -> Dict[str, Any]:

    pos1 = p1.pos
    pos2 = p2.pos
    z1 = pos1[:, 2]
    z2 = pos2[:, 2]

    # Find faces close to layer interface
    zmin1, ind1 = layer.mindist(z1)
    zmin2, ind2 = layer.mindist(z2)

    n1 = pos1.shape[0]
    n2 = pos2.shape[0]

    # Radial distance
    dx = pos1[:, 0:1] - pos2[:, 0:1].T
    dy = pos1[:, 1:2] - pos2[:, 1:2].T
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Distance to layer for face pairs
    zmin_pair = np.minimum(
        np.tile(zmin1[:, np.newaxis], (1, n2)),
        np.tile(zmin2[np.newaxis, :], (n1, 1)))

    # Mask: face pairs that need refinement (close to layer and small r)
    refine_mask = (zmin_pair < layer.ztol * 3) & (r < layer.ztol * 5)

    return {
        'refine_mask': refine_mask,
        'r': r,
        'zmin1': zmin1,
        'zmin2': zmin2,
        'ind1': ind1,
        'ind2': ind2
    }


def refinestat(p1: Any,
        p2: Any,
        layer: Any,
        **options: Any) -> Dict[str, Any]:

    return refineret(p1, p2, layer, **options)


def shift(pos: np.ndarray,
        layer: Any,
        direction: str = 'up') -> np.ndarray:

    pos = pos.copy()
    z = pos[:, 2]

    zmin, ind = layer.mindist(z)

    # Shift faces that are too close to the boundary
    close_mask = zmin < layer.zmin

    if direction == 'up':
        z_layer = layer.z[ind[close_mask] - 1]
        pos[close_mask, 2] = z_layer + layer.zmin
    elif direction == 'down':
        z_layer = layer.z[ind[close_mask] - 1]
        pos[close_mask, 2] = z_layer - layer.zmin
    else:
        raise ValueError('[error] Invalid <direction>: {}'.format(direction))

    return pos

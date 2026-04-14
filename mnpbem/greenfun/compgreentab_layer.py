import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_ret_layer import CompGreenRetLayer
from .greentab_layer import GreenTabLayer


class CompGreenTabLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer

        # Tabulated Green function
        if tab is not None:
            self.tab = GreenTabLayer(layer, tab = tab)
        else:
            self.tab = GreenTabLayer(layer)

        # CompGreenRetLayer with tabulated Green functions
        options_with_tab = dict(options)
        if tab is not None:
            options_with_tab['tab'] = tab

        self.g = CompGreenRetLayer(p1, p2, layer, **options_with_tab)

    def set(self, enei_arr, **options):
        """Pre-compute Green function table at multiple wavelengths.

        MATLAB: greentab = set(greentab, enei, op)
        """
        self.tab.set(enei_arr, **options)
        return self

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            ind: Optional[np.ndarray] = None) -> Any:

        return self.g.eval(i, j, key, enei, ind = ind)

    def potential(self,
            sig: Any,
            inout: int = 1) -> Any:

        return self.g.potential(sig, inout)

    def field(self,
            sig: Any,
            inout: int = 1) -> Any:

        return self.g.field(sig, inout)

    def tabulate(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> None:

        self.tab.r = r
        self.tab.z1 = z1
        self.tab.z2 = z2
        self.tab._compute_tab(enei)

    def inside(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: @compgreentablayer/inside.m
        # Delegates to GreenTabLayer.inside() and returns index array.
        r = np.asarray(r, dtype = float).ravel()
        z1 = np.asarray(z1, dtype = float).ravel()
        if z2 is not None:
            z2 = np.asarray(z2, dtype = float).ravel()

        in_tab = self.tab.inside(r, z1, z2)

        # Index: 1 if inside, 0 otherwise (single table -> index is 0 or 1)
        ind = np.zeros(len(r), dtype = int)
        ind[in_tab] = 1

        return ind

    def ismember(self,
            layer: Any,
            enei: Optional[np.ndarray] = None,
            *args: Any) -> bool:
        # MATLAB: @compgreentablayer/ismember.m
        # Delegates to GreenTabLayer.ismember() and optionally checks positions.
        is_compat = self.tab.ismember(layer, enei)
        if not is_compat:
            return False

        # Handle additional particle/point arguments for position checking
        if len(args) > 0:
            from ..misc.distance_utils import pdist2

            pos_list = []
            for p in args:
                if not isinstance(p, (list, tuple)):
                    p = [p]
                pos_parts = []
                for pj in p:
                    if hasattr(pj, 'verts'):
                        pos_parts.append(pj.verts)
                    elif hasattr(pj, 'pos'):
                        pos_parts.append(pj.pos)
                total_len = sum(pp.shape[0] for pp in pos_parts)
                combined = np.empty((total_len, pos_parts[0].shape[1]), dtype = pos_parts[0].dtype)
                offset = 0
                for pp in pos_parts:
                    combined[offset:offset + pp.shape[0]] = pp
                    offset += pp.shape[0]
                pos_list.append(combined)

            pos1 = pos_list[0].copy()
            pos2 = pos_list[0].copy()
            if len(pos_list) == 2:
                total = pos1.shape[0] + pos_list[1].shape[0]
                pos1_ext = np.empty((total, pos1.shape[1]), dtype = pos1.dtype)
                pos1_ext[:pos1.shape[0]] = pos1
                pos1_ext[pos1.shape[0]:] = pos_list[1]
                pos1 = pos1_ext

            # Compute distances
            r = pdist2(pos1[:, :2], pos2[:, :2])
            z1_exp = np.repeat(pos1[:, 2:3], pos2.shape[0], axis = 1)
            z2_exp = np.repeat(pos2[:, 2:3].T, pos1.shape[0], axis = 0)

            ind = self.inside(r.ravel(), z1_exp.ravel(), z2_exp.ravel())
            return not np.any(ind == 0)

        return True

    def parset(self,
            enei_arr: np.ndarray,
            **options: Any) -> 'CompGreenTabLayer':
        # MATLAB: @compgreentablayer/parset.m
        # Delegates to GreenTabLayer.parset().
        self.tab.parset(enei_arr, **options)
        return self

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenTabLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)

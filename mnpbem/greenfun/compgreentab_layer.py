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

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenTabLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)

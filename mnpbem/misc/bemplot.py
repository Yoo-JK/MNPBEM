"""
BEM plotting class for MNPBEM.

MATLAB: @bemplot/
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .valarray import ValArray
from .vecarray import VecArray
from .options import getbemoptions


class BemPlot(object):
    """
    Plotting value arrays and vector functions within MNPBEM.

    MATLAB: @bemplot

    Parameters
    ----------
    fun : callable, optional
        Plot function (default: np.real)
    scale : float, optional
        Scale factor for vector array (default: 1)
    sfun : callable, optional
        Scale function for vector array (default: identity)

    Methods
    -------
    plotval(p, val, **kwargs) -> None
    plotarrow(pos, vec, **kwargs) -> None
    plotcone(pos, vec, **kwargs) -> None
    plottrue(p, val=None, **kwargs) -> None
    refresh(*keys) -> None
    """

    def __init__(self, **kwargs: Any) -> None:
        self.var = []
        self.siz = None
        self.opt = {
            'ind': None,
            'fun': lambda x: np.real(x),
            'scale': 1.0,
            'sfun': lambda x: x}

        op = getbemoptions(kwargs)
        if 'fun' in op:
            self.opt['fun'] = op['fun']
        if 'scale' in op:
            self.opt['scale'] = op['scale']
        if 'sfun' in op:
            self.opt['sfun'] = op['sfun']

    def plotval(self, p: object, val: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotval.m

        Plot value array on surface.
        """
        # initialization functions
        def inifun(p_arg: object) -> ValArray:
            return ValArray(p_arg, val)

        def inifun2(var: ValArray) -> ValArray:
            var.init2(val)
            return var

        self._plot(p, inifun, inifun2, **kwargs)

    def plottrue(self, p: object,
            val: Optional[np.ndarray] = None,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plottrue.m

        Plot with true colors on surface.
        """
        def inifun(p_arg: object) -> ValArray:
            return ValArray(p_arg, val, truecolor = True)

        def inifun2(var: ValArray) -> ValArray:
            var.init2(val, truecolor = True)
            return var

        self._plot(p, inifun, inifun2, **kwargs)

    def plotarrow(self, pos: np.ndarray, vec: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotarrow.m

        Plot vector array with arrows.
        """
        def inifun(pos_arg: np.ndarray) -> VecArray:
            return VecArray(pos_arg, vec, 'arrow')

        def inifun2(var: VecArray) -> VecArray:
            var.init2(vec, 'arrow')
            return var

        self._plot(pos, inifun, inifun2, **kwargs)

    def plotcone(self, pos: np.ndarray, vec: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotcone.m

        Plot vector array with cones.
        """
        def inifun(pos_arg: np.ndarray) -> VecArray:
            return VecArray(pos_arg, vec, 'cone')

        def inifun2(var: VecArray) -> VecArray:
            var.init2(vec, 'cone')
            return var

        self._plot(pos, inifun, inifun2, **kwargs)

    def _plot(self, p: object,
            inifun: Callable, inifun2: Callable,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plot.m

        Core plot function.
        """
        # initialize value array
        var = inifun(p)

        # handle size argument
        if hasattr(var, 'ispage') and var.ispage() and self.siz is not None:
            assert var.pagesize() == self.siz

        # has object been plotted before?
        ind = None
        for i, v in enumerate(self.var):
            if v.isbase(p):
                ind = i
                break

        if ind is None:
            ind = len(self.var)
            self.var.append(var)
        else:
            self.var[ind] = inifun2(self.var[ind])

        # handle paging
        if hasattr(self.var[ind], 'ispage') and self.var[ind].ispage() and self.siz is None:
            self.siz = self.var[ind].pagesize()
            self.opt['ind'] = 0

        # plot
        self.var[ind].plot(self.opt, **kwargs)

    def refresh(self, *keys: str) -> None:
        """
        MATLAB: @bemplot/refresh.m

        Refresh value and vector plots.
        """
        for i, var in enumerate(self.var):
            if var.depends(*keys):
                var.plot(self.opt)
                self.var[i] = var

    def set_opt(self, **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/set.m

        Set plot options and refresh.
        """
        keys = []
        if 'ind' in kwargs:
            ind = kwargs['ind']
            if isinstance(ind, (list, tuple)) and self.siz is not None:
                ind = np.ravel_multi_index(ind, self.siz)
            self.opt['ind'] = ind
            keys.append('ind')

        if 'fun' in kwargs:
            self.opt['fun'] = kwargs['fun']
            keys.append('fun')
        if 'scale' in kwargs:
            self.opt['scale'] = kwargs['scale']
            keys.append('scale')
        if 'sfun' in kwargs:
            self.opt['sfun'] = kwargs['sfun']
            keys.append('sfun')

        if keys:
            self.refresh(*keys)

    def __repr__(self) -> str:
        return 'BemPlot(nvars={}, siz={})'.format(
            len(self.var), self.siz)

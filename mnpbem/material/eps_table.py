"""
Tabulated dielectric function from data files.
"""

import os
import numpy as np
from typing import Tuple, Union
from scipy.interpolate import CubicSpline

from .eps_base import EpsBase
from ..misc.units import eV2nm


class EpsTable(EpsBase):
    """
    Interpolated tabulated dielectric function.

    Loads dielectric function data from ASCII files and interpolates
    for arbitrary wavelengths.

    Parameters
    ----------
    filename : str
        Path to ASCII file with "energy n k" in each line.
        - energy: photon energy (eV)
        - n: real part of refractive index
        - k: imaginary part of refractive index

    Available data files:
        - gold.dat, silver.dat (Johnson & Christy)
        - goldpalik.dat, silverpalik.dat, copperpalik.dat (Palik)

    Examples
    --------
    >>> eps_gold = EpsTable('gold.dat')
    >>> eps, k = eps_gold(500)  # At 500 nm
    """

    # Search paths for data files
    DATA_PATHS = [
        os.path.dirname(__file__),  # Same directory as this module
        os.path.join(os.path.dirname(__file__), '..', 'data'),  # ../data
        os.path.join(os.path.dirname(__file__), '..', '..', 'Material'),  # MATLAB data
        '.',  # Current directory
    ]

    def __init__(self, filename: str):
        """
        Initialize tabulated dielectric function.

        Parameters
        ----------
        filename : str
            Path to ASCII file with tabulated data.
        """
        self.filename = filename
        self._load_data(filename)

    def _find_file(self, filename: str) -> str:
        """Find the data file in search paths."""
        # If absolute path or relative path exists, use it
        if os.path.isfile(filename):
            return filename

        # Search in predefined paths
        for path in self.DATA_PATHS:
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path):
                return full_path

        raise FileNotFoundError(
            f"Could not find data file '{filename}'. "
            f"Searched in: {self.DATA_PATHS}"
        )

    def _load_data(self, filename: str) -> None:
        """Load and process tabulated data."""
        filepath = self._find_file(filename)

        # Load data (skip comment lines starting with %)
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('%') and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            ene = float(parts[0])
                            n = float(parts[1])
                            k = float(parts[2])
                            data.append([ene, n, k])
                        except ValueError:
                            continue

        data = np.array(data)
        if len(data) == 0:
            raise ValueError(f"No valid data found in '{filename}'")

        # Convert energy (eV) to wavelength (nm)
        self.enei = eV2nm / data[:, 0]

        # Sort by wavelength (ascending) for interpolation
        sort_idx = np.argsort(self.enei)
        self.enei = self.enei[sort_idx]
        n_vals = data[sort_idx, 1]
        k_vals = data[sort_idx, 2]

        # Create spline interpolators
        self._n_spline = CubicSpline(self.enei, n_vals)
        self._k_spline = CubicSpline(self.enei, k_vals)

        # Store range
        self.enei_min = self.enei.min()
        self.enei_max = self.enei.max()

    def __call__(self, enei: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate interpolated dielectric function and wavenumber.

        Parameters
        ----------
        enei : array_like
            Light wavelength in vacuum (nm).

        Returns
        -------
        eps : ndarray
            Interpolated dielectric function (complex).
        k : ndarray
            Wavenumber in medium.

        Raises
        ------
        ValueError
            If wavelength is outside the tabulated range.
        """
        enei = np.asarray(enei, dtype=float)
        scalar_input = enei.ndim == 0
        enei = np.atleast_1d(enei)

        # Check range
        if np.any(enei < self.enei_min) or np.any(enei > self.enei_max):
            raise ValueError(
                f"Wavelength must be in range [{self.enei_min:.1f}, {self.enei_max:.1f}] nm. "
                f"Got range [{enei.min():.1f}, {enei.max():.1f}] nm."
            )

        # Interpolate refractive index
        n = self._n_spline(enei)
        k_imag = self._k_spline(enei)

        # Complex refractive index -> dielectric function
        # n_complex = n + i*k
        # eps = n_complex^2 = (n + i*k)^2
        n_complex = n + 1j * k_imag
        eps = n_complex ** 2

        # Wavenumber
        k = 2 * np.pi / enei * n_complex

        if scalar_input:
            eps = eps[0]
            k = k[0]

        return eps, k

    def __repr__(self) -> str:
        return f"EpsTable('{self.filename}', range=[{self.enei_min:.1f}, {self.enei_max:.1f}] nm)"

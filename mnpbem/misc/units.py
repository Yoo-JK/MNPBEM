"""
Physical units and conversion factors.

This module provides constants for unit conversions between eV and nm.
"""

import numpy as np

# Conversion factor between eV and nm
# eV2nm = hc/e where h is Planck's constant, c is speed of light
eV2nm = 1 / 8.0655477e-4  # approximately 1239.84 nm*eV

# Inverse conversion
nm2eV = 1 / eV2nm

# Atomic units
HARTREE = 27.2116  # 2 * Rydberg in eV
TUNIT = 0.66 / HARTREE  # time unit in fs

# Speed of light in nm/fs
SPEED_OF_LIGHT = 299792.458  # nm/fs


def wavelength_to_energy(wavelength_nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength in nm to energy in eV.

    Parameters
    ----------
    wavelength_nm : array_like
        Wavelength in nanometers.

    Returns
    -------
    energy_eV : ndarray
        Energy in electron volts.
    """
    return eV2nm / np.asarray(wavelength_nm)


def energy_to_wavelength(energy_eV: np.ndarray) -> np.ndarray:
    """
    Convert energy in eV to wavelength in nm.

    Parameters
    ----------
    energy_eV : array_like
        Energy in electron volts.

    Returns
    -------
    wavelength_nm : ndarray
        Wavelength in nanometers.
    """
    return eV2nm / np.asarray(energy_eV)

"""
Particle and geometry classes for MNPBEM.

This module provides classes for defining particle geometries:
- Particle: Basic discretized particle surface
- Point: Collection of evaluation points
- Compound: Base class for compound particles
- ComParticle: Compound of particles with dielectric functions
- ComPoint: Compound of points with dielectric functions
"""

from .particle import Particle
from .point import Point
from .compound import Compound
from .comparticle import ComParticle
from .compoint import ComPoint
from .compstruct import CompStruct
from .layer_structure import LayerStructure

__all__ = [
    "Particle",
    "Point",
    "Compound",
    "ComParticle",
    "ComPoint",
    "CompStruct",
    "LayerStructure",
]

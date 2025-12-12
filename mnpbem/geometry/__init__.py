"""
Geometry and mesh generation module.

Classes:
- Particle: Basic particle with triangular mesh
- ComParticle: Compound particle with multiple materials

Functions:
- trisphere: Generate triangulated sphere
"""

from .particle import Particle
from .comparticle import ComParticle
from .mesh_generators import trisphere

__all__ = [
    "Particle",
    "ComParticle",
    "trisphere",
]

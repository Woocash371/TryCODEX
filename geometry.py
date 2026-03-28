"""TPMS geometry generation utilities.

This module defines standard implicit TPMS fields on normalized periodic
coordinates and extracts one-unit-cell isosurfaces using marching cubes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class GridSpec:
    """Structured sampling grid for one unit cell."""

    resolution: int = 56
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0


def _periodic_coordinates(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic coordinates in [0, 2π] for one unit cell.

    Scaling is applied by stretching physical axes while preserving one periodic
    cycle over each axis.
    """

    x = np.linspace(0.0, spec.lx, spec.resolution, endpoint=True)
    y = np.linspace(0.0, spec.ly, spec.resolution, endpoint=True)
    z = np.linspace(0.0, spec.lz, spec.resolution, endpoint=True)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")

    xp = 2.0 * np.pi * (xg / spec.lx)
    yp = 2.0 * np.pi * (yg / spec.ly)
    zp = 2.0 * np.pi * (zg / spec.lz)
    return xp, yp, zp


def implicit_field(tpms_type: str, spec: GridSpec) -> np.ndarray:
    """Evaluate TPMS implicit field F(x,y,z).

    The isosurface is defined by F(x, y, z) = iso_level.

    Implemented families (standard trigonometric forms):
    - gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)
    - diamond: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z)
               + cos(x)sin(y)cos(z) + cos(x)cos(y)sin(z)
    - primitive: cos(x) + cos(y) + cos(z)
    - i-wp (optional)
    - neovius (optional)
    - fischer_koch_s (optional, simplified trigonometric proxy)
    """

    x, y, z = _periodic_coordinates(spec)
    name = tpms_type.lower().strip()

    if name == "gyroid":
        f = np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
    elif name == "diamond":
        f = (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        )
    elif name == "primitive":
        f = np.cos(x) + np.cos(y) + np.cos(z)
    elif name in {"i-wp", "iwp"}:
        f = (
            2.0 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x))
            - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))
        )
    elif name == "neovius":
        f = 3.0 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4.0 * np.cos(x) * np.cos(y) * np.cos(z)
    elif name in {"fischer-koch s", "fischer_koch_s", "fischer koch s"}:
        # Compact approximate harmonic representation used as a visualization proxy.
        f = np.cos(2 * x) * np.sin(y) * np.cos(z) + np.cos(2 * y) * np.sin(z) * np.cos(x) + np.cos(2 * z) * np.sin(x) * np.cos(y)
    else:
        raise ValueError(f"Unsupported TPMS type: {tpms_type}")

    return f.astype(np.float32)


def voxel_masks(field: np.ndarray, iso_level: float) -> Dict[str, np.ndarray]:
    """Return boolean masks for solid and pore regions.

    Convention:
    - solid: F >= iso
    - pore/void: F < iso
    """

    solid = field >= iso_level
    pore = ~solid
    return {"solid": solid, "pore": pore}


def extract_isosurface(field: np.ndarray, iso_level: float, spec: GridSpec):
    """Extract mesh vertices/faces for F=iso_level over one unit cell."""

    spacing = (
        spec.lx / (spec.resolution - 1),
        spec.ly / (spec.resolution - 1),
        spec.lz / (spec.resolution - 1),
    )
    verts, faces, normals, values = marching_cubes(field, level=iso_level, spacing=spacing)
    return verts, faces, normals, values


def bounds(spec: GridSpec):
    return (0.0, spec.lx), (0.0, spec.ly), (0.0, spec.lz)

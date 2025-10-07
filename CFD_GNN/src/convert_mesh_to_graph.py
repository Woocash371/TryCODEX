"""Utilities for converting CFD meshes exported from OpenFOAM into graph data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import meshio
import numpy as np
import torch
from torch_geometric.data import Data


LOGGER = logging.getLogger(__name__)


def _collect_edges(cells: Iterable[np.ndarray]) -> np.ndarray:
    """Build an undirected edge list from cell connectivity.

    Each cell contributes fully connected edges between its vertices so that
    neighbouring nodes share an edge in the final graph. Duplicate edges are
    removed and the resulting array follows the shape ``(2, num_edges)`` as
    required by PyTorch Geometric.
    """

    edge_set: set[Tuple[int, int]] = set()
    for cell in cells:
        if len(cell) < 2:
            continue
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                # Store undirected edges with sorted indices to avoid duplicates.
                edge = tuple(sorted((int(cell[i]), int(cell[j]))))
                edge_set.add(edge)

    if not edge_set:
        raise ValueError("Unable to construct any edges from the provided cells.")

    edges = np.array(list(edge_set), dtype=np.int64).T
    return edges


def _ensure_point_data(mesh: meshio.Mesh) -> meshio.Mesh:
    """Ensure velocity and pressure are available, otherwise synthesise them.

    When the mesh does not contain ``U`` or ``p`` fields (for example when the
    provided VTU file is only a placeholder) this helper fabricates a smooth
    synthetic velocity and pressure field so the remainder of the pipeline still
    works. The synthetic fields mimic a vortex around the origin which is
    sufficient for demonstration purposes.
    """

    if "U" in mesh.point_data and "p" in mesh.point_data:
        return mesh

    LOGGER.warning("Mesh is missing CFD fields; generating synthetic example data.")
    points = mesh.points
    x = points[:, 0]
    y = points[:, 1]
    r2 = x**2 + y**2 + 1e-6

    # Synthetic rotational flow around the origin with decaying magnitude.
    Ux = -y / r2
    Uy = x / r2
    Uz = np.zeros_like(Ux)
    mesh.point_data["U"] = np.stack([Ux, Uy, Uz], axis=1)

    # Pressure decays logarithmically with radius from the origin.
    mesh.point_data["p"] = 1.0 - np.log(r2)
    return mesh


def _load_mesh(path: Path) -> meshio.Mesh:
    """Load a mesh from ``path``; if loading fails fabricate a synthetic case."""

    if path.exists():
        LOGGER.info("Reading mesh from %s", path)
        mesh = meshio.read(path)
        return _ensure_point_data(mesh)

    LOGGER.warning("Mesh file %s not found; creating a synthetic example mesh.", path)
    # Build a simple unit disc mesh using polar coordinates.
    theta = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
    radius = np.linspace(0.2, 1.0, 5)
    nodes = [(0.0, 0.0, 0.0)]
    for r in radius:
        for t in theta:
            nodes.append((r * np.cos(t), r * np.sin(t), 0.0))
    points = np.array(nodes, dtype=np.float64)

    # Triangulate by connecting centre to ring points (fan triangulation).
    num_rings = len(radius)
    cells = []
    prev_ring_start = 1
    ring_size = len(theta)
    for ring_idx in range(num_rings):
        ring_start = prev_ring_start
        next_ring_start = ring_start + ring_size
        if ring_idx == 0:
            centre = 0
            for i in range(ring_size):
                a = ring_start + i
                b = ring_start + (i + 1) % ring_size
                cells.append([centre, a, b])
        else:
            for i in range(ring_size):
                a_inner = prev_ring_start + i
                b_inner = prev_ring_start + (i + 1) % ring_size
                a_outer = ring_start + i
                b_outer = ring_start + (i + 1) % ring_size
                cells.append([a_inner, a_outer, b_outer])
                cells.append([a_inner, b_outer, b_inner])
        prev_ring_start = next_ring_start

    mesh = meshio.Mesh(points=points, cells=[("triangle", np.array(cells, dtype=np.int32))])
    return _ensure_point_data(mesh)


def mesh_to_graph(mesh: meshio.Mesh, include_z: bool = False) -> Data:
    """Convert a ``meshio.Mesh`` instance to ``torch_geometric.data.Data``."""

    mesh = _ensure_point_data(mesh)
    points = mesh.points

    # Retrieve any cell block that contains connectivity information.
    if hasattr(mesh, "cells_dict") and mesh.cells_dict:
        cells_iterable = mesh.cells_dict.values()
    else:
        cells_iterable = (block.data for block in mesh.cells)

    edges = _collect_edges(cells_iterable)

    velocity = np.asarray(mesh.point_data["U"], dtype=np.float64)
    pressure = np.asarray(mesh.point_data["p"], dtype=np.float64).reshape(-1)

    x = points[:, 0]
    y = points[:, 1]
    if include_z and points.shape[1] > 2:
        z = points[:, 2]
        features = np.stack([x, y, z, velocity[:, 0], velocity[:, 1], velocity[:, 2]], axis=1)
    else:
        features = np.stack([x, y, velocity[:, 0], velocity[:, 1]], axis=1)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
        y=torch.tensor(pressure, dtype=torch.float32),
        pos=torch.tensor(points[:, :3], dtype=torch.float32),
    )
    return data


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a VTU mesh into a PyG graph.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("CFD_GNN/data/case_0001.vtu"),
        help="Path to the VTU mesh exported from OpenFOAM (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("CFD_GNN/data/data_graph.pt"),
        help="Path where the PyTorch Geometric Data object will be saved.",
    )
    parser.add_argument(
        "--include-z",
        action="store_true",
        help="Include the z-coordinate and Uz component as node features.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of console logging.",
    )

    parsed = parser.parse_args(args=args)
    logging.basicConfig(level=getattr(logging, parsed.log_level), format="%(levelname)s: %(message)s")

    mesh = _load_mesh(parsed.mesh)
    data = mesh_to_graph(mesh, include_z=parsed.include_z)

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, parsed.output)
    LOGGER.info("Saved graph with %d nodes and %d edges to %s", data.num_nodes, data.num_edges, parsed.output)


if __name__ == "__main__":
    main()

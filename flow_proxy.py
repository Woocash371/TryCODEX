"""Qualitative flow interpretation for TPMS pore space.

This module intentionally provides heuristic (non-CFD) descriptors.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


DIR_TO_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _face_indices(shape: Tuple[int, int, int], axis: int, side: int):
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = 0 if side == 0 else shape[axis] - 1
    return tuple(sl)


def _connected_component_for_flow(pore: np.ndarray, axis: int) -> np.ndarray:
    """Return pore component that spans inlet/outlet for given axis if present."""

    labels = label(pore, connectivity=3)
    if labels.max() == 0:
        return np.zeros_like(pore, dtype=bool)

    in_face = labels[_face_indices(labels.shape, axis, 0)]
    out_face = labels[_face_indices(labels.shape, axis, 1)]
    in_ids = set(np.unique(in_face[in_face > 0]))
    out_ids = set(np.unique(out_face[out_face > 0]))
    spanning = in_ids.intersection(out_ids)
    if not spanning:
        return np.zeros_like(pore, dtype=bool)

    # Use largest spanning connected component.
    chosen = max(spanning, key=lambda cid: np.count_nonzero(labels == cid))
    return labels == chosen


def _shortest_path_steps(mask: np.ndarray, axis: int) -> float | None:
    """Approximate shortest pore path using 6-neighborhood BFS."""

    shape = mask.shape
    dist = -np.ones(shape, dtype=np.int32)
    q = deque()

    outlet = _face_indices(shape, axis, 1)

    for idx in np.argwhere(mask):
        if idx[axis] == 0:
            t = tuple(idx.tolist())
            dist[t] = 0
            q.append(t)

    if not q:
        return None

    neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    while q:
        i, j, k = q.popleft()
        d = dist[i, j, k]
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                if mask[ni, nj, nk] and dist[ni, nj, nk] < 0:
                    dist[ni, nj, nk] = d + 1
                    q.append((ni, nj, nk))

    outlet_vals = dist[outlet]
    valid = outlet_vals[outlet_vals >= 0]
    if valid.size == 0:
        return None
    return float(valid.min())


def directional_flow_metrics(pore: np.ndarray, direction: str) -> Dict[str, float | bool]:
    axis = DIR_TO_AXIS[direction.upper()]
    span_mask = _connected_component_for_flow(pore, axis)
    connected = bool(np.any(span_mask))

    if not connected:
        return {
            "connected": False,
            "tortuosity_proxy": np.nan,
            "openness_proxy": 0.0,
            "constriction_proxy": 1.0,
            "flow_score": 0.0,
        }

    shortest = _shortest_path_steps(span_mask, axis)
    direct = pore.shape[axis] - 1
    tort = float(shortest / max(direct, 1)) if shortest is not None else np.nan

    # Openness proxy: spanning void fraction.
    openness = float(np.count_nonzero(span_mask) / span_mask.size)

    # Constriction proxy from local pore-throat radius statistics.
    edt = distance_transform_edt(span_mask)
    throat = np.percentile(edt[span_mask], 10)
    body = np.percentile(edt[span_mask], 90)
    constriction = float(1.0 - (throat / (body + 1e-6)))
    constriction = float(np.clip(constriction, 0.0, 1.0))

    # Higher score means easier flow (heuristic).
    flow_score = float(np.clip((openness * (1.0 - constriction)) / (tort + 1e-6), 0.0, 1.0))

    return {
        "connected": connected,
        "tortuosity_proxy": tort,
        "openness_proxy": openness,
        "constriction_proxy": constriction,
        "flow_score": flow_score,
    }


def anisotropy_indicators(pore: np.ndarray) -> Dict[str, float]:
    results = {d: directional_flow_metrics(pore, d)["flow_score"] for d in ("X", "Y", "Z")}
    arr = np.array([results["X"], results["Y"], results["Z"]], dtype=float)
    arr = np.nan_to_num(arr)
    max_v = arr.max() if arr.size else 0.0
    min_v = arr.min() if arr.size else 0.0
    anis = float((max_v - min_v) / (max_v + 1e-6)) if max_v > 0 else 0.0
    return {
        "flow_x": float(results["X"]),
        "flow_y": float(results["Y"]),
        "flow_z": float(results["Z"]),
        "anisotropy_index": anis,
    }

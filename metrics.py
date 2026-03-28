"""Scientific proxy metrics for TPMS unit-cell visualization."""

from __future__ import annotations

from typing import Dict

import numpy as np

from flow_proxy import anisotropy_indicators, directional_flow_metrics


def basic_metrics(tpms_type: str, iso_level: float, solid: np.ndarray, pore: np.ndarray, lx: float, ly: float, lz: float) -> Dict[str, float | str | bool]:
    total = solid.size
    porosity = float(np.count_nonzero(pore) / total)
    solid_fraction = 1.0 - porosity

    # Surface-area proxy: interfacial voxel face count normalized by volume.
    interface_count = 0
    for axis in range(3):
        interface_count += np.count_nonzero(np.diff(solid.astype(np.int8), axis=axis) != 0)

    voxel_volume = (lx * ly * lz) / total
    area_proxy = float(interface_count * (voxel_volume ** (2.0 / 3.0)))
    ssa_proxy = float(area_proxy / (lx * ly * lz + 1e-9))

    flow_x = directional_flow_metrics(pore, "X")
    flow_y = directional_flow_metrics(pore, "Y")
    flow_z = directional_flow_metrics(pore, "Z")
    ani = anisotropy_indicators(pore)

    return {
        "tpms_type": tpms_type,
        "iso_level": float(iso_level),
        "porosity_est": porosity,
        "solid_fraction_est": solid_fraction,
        "lx": float(lx),
        "ly": float(ly),
        "lz": float(lz),
        "specific_surface_area_proxy": ssa_proxy,
        "connectivity_x": bool(flow_x["connected"]),
        "connectivity_y": bool(flow_y["connected"]),
        "connectivity_z": bool(flow_z["connected"]),
        "flow_score_x": float(flow_x["flow_score"]),
        "flow_score_y": float(flow_y["flow_score"]),
        "flow_score_z": float(flow_z["flow_score"]),
        "tortuosity_x": float(flow_x["tortuosity_proxy"]),
        "tortuosity_y": float(flow_y["tortuosity_proxy"]),
        "tortuosity_z": float(flow_z["tortuosity_proxy"]),
        "constriction_x": float(flow_x["constriction_proxy"]),
        "constriction_y": float(flow_y["constriction_proxy"]),
        "constriction_z": float(flow_z["constriction_proxy"]),
        "anisotropy_index": float(ani["anisotropy_index"]),
    }

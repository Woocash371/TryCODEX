# TPMS Unit Cell Explorer (Streamlit MVP)

A publication-friendly prototype for interactive visualization of TPMS unit cells with qualitative flow interpretation (heuristic proxies, not CFD).

## Features

- TPMS families: **Gyroid**, **Diamond**, **Primitive** (+ optional I-WP, Neovius, Fischer-Koch S)
- Single elementary unit-cell rendering with anisotropic scaling (`Lx`, `Ly`, `Lz`)
- Porosity control via implicit-surface iso-level (porosity **proxy**)
- Multi-view layout:
  - Perspective
  - Front
  - Side
  - Top
- Comparison strip:
  - baseline
  - porosity-modified
  - X/Y/Z-scale modifications
- Qualitative flow interpretation panel:
  - connectivity
  - tortuosity proxy
  - constriction proxy
  - openness proxy
  - directional flow scores X/Y/Z
- Scientific metrics panel:
  - voxel-estimated porosity/solid fraction
  - specific surface area proxy
  - anisotropy indicator
- Publication-oriented styling:
  - monochrome/scientific color modes
  - white/transparent/dark background
- Export:
  - PNG (high-res via Kaleido)
  - SVG (when available)
  - 4-panel / current / comparison exports
- Reproducibility panel:
  - current JSON config
  - one-click JSON download
  - defaults reset + presets

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).

## Project structure

- `app.py` – Streamlit UI, plotting, export, reproducibility
- `geometry.py` – implicit TPMS fields + marching-cubes extraction
- `flow_proxy.py` – qualitative directional connectivity/tortuosity/constriction proxies
- `metrics.py` – porosity and scientific proxy metrics

## Scientific notes

- The iso-level slider is a **porosity proxy** control for implicit level-set geometry.
- Porosity is estimated from voxel occupancy, not exact CAD integration.
- Flow metrics are **qualitative heuristic proxies** and explicitly **not CFD**.

## Example screenshot

Use the in-app export buttons to generate publication-ready figures after launching locally.

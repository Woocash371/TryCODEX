from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from flow_proxy import directional_flow_metrics
from geometry import GridSpec, extract_isosurface, implicit_field, voxel_masks
from metrics import basic_metrics

st.set_page_config(page_title="TPMS Unit Cell Explorer", layout="wide")

PRESETS = {
    "Gyroid isotropic": {"tpms": "Gyroid", "iso": 0.0, "lx": 1.0, "ly": 1.0, "lz": 1.0},
    "Gyroid high porosity": {"tpms": "Gyroid", "iso": 0.55, "lx": 1.0, "ly": 1.0, "lz": 1.0},
    "Diamond anisotropic X": {"tpms": "Diamond", "iso": 0.0, "lx": 1.4, "ly": 1.0, "lz": 1.0},
    "Diamond anisotropic Z": {"tpms": "Diamond", "iso": 0.0, "lx": 1.0, "ly": 1.0, "lz": 1.5},
    "Primitive compact": {"tpms": "Primitive", "iso": -0.35, "lx": 1.0, "ly": 1.0, "lz": 1.0},
    "Primitive open pore": {"tpms": "Primitive", "iso": 0.45, "lx": 1.0, "ly": 1.0, "lz": 1.0},
}

CAMERAS = {
    "perspective": dict(eye=dict(x=1.75, y=1.75, z=1.35)),
    "front": dict(eye=dict(x=0.001, y=0.0, z=2.2)),
    "side": dict(eye=dict(x=2.2, y=0.001, z=0.0)),
    "top": dict(eye=dict(x=0.001, y=2.2, z=0.001)),
}


def build_mesh_trace(verts, faces, color, opacity=1.0, name="TPMS"):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting=dict(ambient=0.42, diffuse=0.65, specular=0.18, roughness=0.55, fresnel=0.05),
        lightposition=dict(x=1, y=2, z=3),
        showscale=False,
    )


def add_flow_arrows(fig, lx, ly, lz, direction="X", scene="scene"):
    centers = np.linspace(0.2, 0.8, 5)
    if direction == "X":
        x0, x1 = 0.05 * lx, 0.95 * lx
        for y in centers * ly:
            for z in centers * lz:
                fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y, y], z=[z, z], mode="lines", line=dict(color="royalblue", width=3), showlegend=False), row=1, col=1)
    elif direction == "Y":
        y0, y1 = 0.05 * ly, 0.95 * ly
        for x in centers * lx:
            for z in centers * lz:
                fig.add_trace(go.Scatter3d(x=[x, x], y=[y0, y1], z=[z, z], mode="lines", line=dict(color="royalblue", width=3), showlegend=False), row=1, col=1)
    else:
        z0, z1 = 0.05 * lz, 0.95 * lz
        for x in centers * lx:
            for y in centers * ly:
                fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[z0, z1], mode="lines", line=dict(color="royalblue", width=3), showlegend=False), row=1, col=1)


def styled_scene(lx, ly, lz, bg_color, show_axes=True, camera=None):
    axis_style = dict(showbackground=False, showgrid=False, showline=show_axes, linewidth=1, linecolor="black", ticks="")
    return dict(
        xaxis=dict(range=[0, lx], **axis_style),
        yaxis=dict(range=[0, ly], **axis_style),
        zaxis=dict(range=[0, lz], **axis_style),
        aspectmode="manual",
        aspectratio=dict(x=lx, y=ly, z=lz),
        bgcolor=bg_color,
        camera=camera or CAMERAS["perspective"],
    )


def export_figure_button(fig, label, fmt="png", filename="tpms_export", scale=3):
    try:
        image_bytes = fig.to_image(format=fmt, scale=scale)
        st.download_button(
            label=label,
            data=image_bytes,
            file_name=f"{filename}.{fmt}",
            mime=f"image/{fmt}",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"{label} unavailable (install kaleido for static export). Details: {exc}")


with st.sidebar:
    st.title("TPMS Controls")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    if st.button("Apply preset", use_container_width=True):
        st.session_state["preset_values"] = PRESETS[preset_name]

    p = st.session_state.get("preset_values", PRESETS["Gyroid isotropic"])

    tpms_type = st.selectbox("TPMS type", ["Gyroid", "Diamond", "Primitive", "I-WP", "Neovius", "Fischer-Koch S"], index=["Gyroid", "Diamond", "Primitive", "I-WP", "Neovius", "Fischer-Koch S"].index(p["tpms"]))
    iso_level = st.slider("Porosity proxy (iso-level)", -1.4, 1.4, float(p["iso"]), 0.02)

    lx = st.slider("Lx", 0.5, 2.0, float(p["lx"]), 0.05)
    ly = st.slider("Ly", 0.5, 2.0, float(p["ly"]), 0.05)
    lz = st.slider("Lz", 0.5, 2.0, float(p["lz"]), 0.05)

    resolution = st.slider("Surface resolution", 32, 92, 56, 4)
    style_mode = st.radio("Render style", ["Monochrome", "Scientific color"], horizontal=True)
    bg_mode = st.radio("Background", ["White", "Transparent", "Dark"], horizontal=True)

    flow_direction = st.radio("Nominal inlet direction", ["X", "Y", "Z"], horizontal=True)
    pore_emphasis = st.checkbox("Pore-space emphasis (higher transparency)", value=False)
    show_axes = st.checkbox("Show axes", value=True)
    show_comparison = st.checkbox("Show comparison strip", value=True)

    if st.button("Reset to defaults", use_container_width=True):
        st.session_state["preset_values"] = PRESETS["Gyroid isotropic"]
        st.rerun()

spec = GridSpec(resolution=resolution, lx=lx, ly=ly, lz=lz)
field = implicit_field(tpms_type, spec)
masks = voxel_masks(field, iso_level)
solid, pore = masks["solid"], masks["pore"]
verts, faces, _, _ = extract_isosurface(field, iso_level, spec)
metrics = basic_metrics(tpms_type, iso_level, solid, pore, lx, ly, lz)
flow_dir_metrics = directional_flow_metrics(pore, flow_direction)

base_color = "#7f7f7f" if style_mode == "Monochrome" else "#4063D8"
opacity = 0.55 if pore_emphasis else 0.95
bg_color = "rgba(0,0,0,0)" if bg_mode == "Transparent" else ("#ffffff" if bg_mode == "White" else "#101010")
font_color = "black" if bg_mode != "Dark" else "white"

st.title("TPMS Unit Cell Explorer")
st.caption("Single-cell TPMS visualization with qualitative flow interpretation (heuristic proxy, not CFD).")

col_main, col_metrics = st.columns([2.7, 1.3])

with col_main:
    fig_main = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]])
    for r in (1, 2):
        for c in (1, 2):
            fig_main.add_trace(build_mesh_trace(verts, faces, color=base_color, opacity=opacity), row=r, col=c)

    # Add directional arrows only in perspective panel.
    add_flow_arrows(fig_main, lx, ly, lz, direction=flow_direction, scene="scene")

    fig_main.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=900,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        title="Main + Orthographic Views",
    )
    fig_main.update_scenes(**styled_scene(lx, ly, lz, bg_color=bg_color, show_axes=show_axes, camera=CAMERAS["perspective"]), row=1, col=1)
    fig_main.update_scenes(**styled_scene(lx, ly, lz, bg_color=bg_color, show_axes=show_axes, camera=CAMERAS["front"]), row=1, col=2)
    fig_main.update_scenes(**styled_scene(lx, ly, lz, bg_color=bg_color, show_axes=show_axes, camera=CAMERAS["side"]), row=2, col=1)
    fig_main.update_scenes(**styled_scene(lx, ly, lz, bg_color=bg_color, show_axes=show_axes, camera=CAMERAS["top"]), row=2, col=2)

    st.plotly_chart(fig_main, use_container_width=True)

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        export_figure_button(fig_main, "Export 4-panel PNG", fmt="png", filename="tpms_4panel", scale=3)
    with exp_c2:
        export_figure_button(fig_main, "Export 4-panel SVG", fmt="svg", filename="tpms_4panel", scale=1)
    with exp_c3:
        export_figure_button(fig_main, "Export current view PNG (high-res)", fmt="png", filename="tpms_current", scale=4)

with col_metrics:
    st.subheader("Scientific Metrics (proxies)")
    st.metric("Estimated porosity", f"{metrics['porosity_est']:.3f}")
    st.metric("Estimated solid fraction", f"{metrics['solid_fraction_est']:.3f}")
    st.metric("Specific surface area proxy", f"{metrics['specific_surface_area_proxy']:.3f}")

    st.markdown("**Connectivity**")
    st.write(
        f"X: {'Connected' if metrics['connectivity_x'] else 'Blocked'} | "
        f"Y: {'Connected' if metrics['connectivity_y'] else 'Blocked'} | "
        f"Z: {'Connected' if metrics['connectivity_z'] else 'Blocked'}"
    )

    st.markdown("**Flow scores (qualitative)**")
    st.write(f"X: {metrics['flow_score_x']:.3f}  ")
    st.write(f"Y: {metrics['flow_score_y']:.3f}  ")
    st.write(f"Z: {metrics['flow_score_z']:.3f}  ")
    st.write(f"Anisotropy index: {metrics['anisotropy_index']:.3f}")

    st.subheader("Qualitative Flow Interpretation")
    st.info(
        "This panel reports qualitative proxy descriptors based on voxelized pore-space connectivity and geometry. "
        "It is not a CFD or experimentally validated permeability model."
    )
    st.write(f"Inlet direction: **{flow_direction}**")
    st.write(f"Connected path: **{flow_dir_metrics['connected']}**")
    st.write(f"Tortuosity proxy: **{flow_dir_metrics['tortuosity_proxy']:.3f}**")
    st.write(f"Constriction severity proxy: **{flow_dir_metrics['constriction_proxy']:.3f}**")
    st.write(f"Openness proxy: **{flow_dir_metrics['openness_proxy']:.3f}**")
    st.write(f"Overall qualitative flow score: **{flow_dir_metrics['flow_score']:.3f}**")

if show_comparison:
    st.subheader("Comparison Strip")
    variants = [
        ("Baseline", dict(tpms=tpms_type, iso=iso_level, lx=lx, ly=ly, lz=lz)),
        ("Porosity Δ", dict(tpms=tpms_type, iso=iso_level + 0.25, lx=lx, ly=ly, lz=lz)),
        ("X scale Δ", dict(tpms=tpms_type, iso=iso_level, lx=lx * 1.25, ly=ly, lz=lz)),
        ("Y scale Δ", dict(tpms=tpms_type, iso=iso_level, lx=lx, ly=ly * 1.25, lz=lz)),
        ("Z scale Δ", dict(tpms=tpms_type, iso=iso_level, lx=lx, ly=ly, lz=lz * 1.25)),
    ]

    fig_cmp = make_subplots(rows=1, cols=5, specs=[[{"type": "scene"}] * 5], subplot_titles=[v[0] for v in variants])
    for idx, (_, cfg) in enumerate(variants, start=1):
        sp = GridSpec(resolution=max(32, resolution // 2), lx=cfg["lx"], ly=cfg["ly"], lz=cfg["lz"])
        fld = implicit_field(cfg["tpms"], sp)
        v, f, _, _ = extract_isosurface(fld, cfg["iso"], sp)
        fig_cmp.add_trace(build_mesh_trace(v, f, color=base_color, opacity=0.92), row=1, col=idx)
        fig_cmp.update_scenes(**styled_scene(cfg["lx"], cfg["ly"], cfg["lz"], bg_color=bg_color, show_axes=False, camera=CAMERAS["perspective"]), row=1, col=idx)

    fig_cmp.update_layout(height=330, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=font_color))
    st.plotly_chart(fig_cmp, use_container_width=True)
    export_figure_button(fig_cmp, "Export comparison PNG", fmt="png", filename="tpms_comparison", scale=3)

st.subheader("Reproducibility")
config = {
    "tpms_type": tpms_type,
    "iso_level": iso_level,
    "lx": lx,
    "ly": ly,
    "lz": lz,
    "resolution": resolution,
    "style_mode": style_mode,
    "background": bg_mode,
    "flow_direction": flow_direction,
    "pore_emphasis": pore_emphasis,
}
st.code(json.dumps(config, indent=2), language="json")
st.download_button("Copy config JSON", json.dumps(config, indent=2), file_name="tpms_config.json", mime="application/json")

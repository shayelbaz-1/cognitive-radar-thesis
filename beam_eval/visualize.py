"""Interactive Dash app for beam-filtered radar evaluation visualization.

Usage::

    pip install dash dash-bootstrap-components
    python beam_eval/visualize.py \\
        --data_root data/nuScenes \\
        --ensemble_ckpt /path/to/ensemble_lss.ckpt

Controls:
    Scene dropdown   – pick a validation scene
    Sample slider    – navigate keyframes within the scene
    Budget slider    – adjust beam selection percentage
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import dash
    from dash import dcc, html, Input, Output, State, no_update
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
except ImportError:
    sys.exit(
        "Visualization requires Dash.\n"
        "Install with:  pip install dash dash-bootstrap-components"
    )

from PIL import Image as PILImage

# ── Path setup ────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_CRN_DIR = os.path.join(_REPO_ROOT, "crn")

for _p in (_REPO_ROOT, _CRN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beam_eval.config import BeamEvalConfig  # noqa: E402
from beam_eval.beam_selector.ensemble_lss import (  # noqa: E402
    EnsembleBeamSelector,
)
from beam_eval.radar_filter import filter_bev_points_by_beams  # noqa: E402

# ── Colour palette & shared Plotly styling ────────────────────────────
_C = {
    "accent": "#00d4ff",
    "accent_fill": "rgba(0, 212, 255, 0.22)",
    "accent_line": "rgba(0, 212, 255, 0.55)",
    "pts_all": "rgba(160, 160, 180, 0.50)",
    "pts_filt": "#00d4ff",
    "ego": "#ff6b6b",
    "ring": "rgba(100, 100, 110, 0.25)",
    "card_bg": "#2b2b3b",
}

_PLOT_KW = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 15, 30, 0.85)",
    font=dict(size=11, color="#ccc"),
    margin=dict(l=45, r=15, t=40, b=40),
)

_CAM_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
_CAM_BOT = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# ── Application state (filled once by main()) ────────────────────────


class _VizState:
    def __init__(self) -> None:
        self.config: Optional[BeamEvalConfig] = None
        self.data_root: str = ""
        self.val_infos: list = []
        self.scene_samples: Dict[str, List[int]] = {}
        self.scene_names: Dict[str, str] = {}
        self.selector: Optional[EnsembleBeamSelector] = None
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def init(
        self,
        config: BeamEvalConfig,
        data_root: str,
        ensemble_ckpt: str,
    ) -> None:
        self.config = config
        self.data_root = data_root

        pkl = os.path.join(data_root, "nuscenes_infos_val.pkl")
        if not os.path.isfile(pkl):
            sys.exit(f"Val info pickle not found: {pkl}")
        with open(pkl, "rb") as f:
            self.val_infos = pickle.load(f)

        for i, info in enumerate(self.val_infos):
            self.scene_samples.setdefault(info["scene_token"], []).append(i)

        scene_json = os.path.join(data_root, "v1.0-trainval", "scene.json")
        if os.path.isfile(scene_json):
            with open(scene_json) as f:
                for s in json.load(f):
                    if s["token"] in self.scene_samples:
                        self.scene_names[s["token"]] = s["name"]
        for st in self.scene_samples:
            self.scene_names.setdefault(st, st[:12])

        if ensemble_ckpt:
            self.selector = EnsembleBeamSelector(
                config.ensemble_grid_conf,
                config.ensemble_data_aug_conf,
                model_path=ensemble_ckpt,
                beam_width=config.beam_width,
                min_range=config.min_range,
                max_range=config.max_range,
            )

    # ── data helpers ──────────────────────────────────────────────────

    def get_entropy(
        self, token: str, info: dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.selector is None:
            return None, None
        if token not in self._cache:
            batch = self.selector.prepare_batch(info["cam_infos"], self.data_root)
            self._cache[token] = self.selector.compute_entropy_map(batch)
        return self._cache[token]

    def load_bev_points(self, info: dict, max_sweeps: int = 7) -> np.ndarray:
        lidar_file = info["lidar_infos"]["LIDAR_TOP"]["filename"]
        bev_path = os.path.join(
            self.data_root, "radar_bev_filter", os.path.basename(lidar_file)
        )
        if not os.path.isfile(bev_path):
            return np.zeros((0, 7), dtype=np.float32)
        pts = np.fromfile(bev_path, dtype=np.float32).reshape(-1, 7)
        return pts[pts[:, 6] < max_sweeps]


_viz = _VizState()

# ── Helpers ───────────────────────────────────────────────────────────


def _encode_image(filepath: str, max_w: int = 420) -> str:
    img = PILImage.open(filepath)
    ratio = max_w / img.width
    img = img.resize((max_w, int(img.height * ratio)), PILImage.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _beam_wedge_xy(
    az_deg: float, bw_deg: float, max_r: float, n: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Polygon vertices: origin → arc → origin (BEV: x=right, y=forward)."""
    angles_rad = np.radians(np.linspace(az_deg - bw_deg / 2, az_deg + bw_deg / 2, n))
    x = np.concatenate([[0], max_r * np.sin(angles_rad), [0]])
    y = np.concatenate([[0], max_r * np.cos(angles_rad), [0]])
    return x, y


def _add_beam_wedges(
    fig: go.Figure,
    azimuths: list,
    bw: float,
    max_r: float,
    fill: str,
    line: str,
) -> None:
    for az in azimuths:
        x, y = _beam_wedge_xy(az, bw, max_r)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor=fill,
                line=dict(color=line, width=0.7),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _add_range_rings(fig: go.Figure, rings: tuple = (25, 50, 75, 100)) -> None:
    t = np.linspace(0, 2 * np.pi, 120)
    for r in rings:
        fig.add_trace(
            go.Scatter(
                x=(r * np.cos(t)).tolist(),
                y=(r * np.sin(t)).tolist(),
                mode="lines",
                line=dict(color=_C["ring"], dash="dot", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def _add_ego_marker(fig: go.Figure) -> None:
    fig.add_trace(
        go.Scatter(
            x=[0, -1.2, 1.2, 0],
            y=[2.2, -0.8, -0.8, 2.2],
            fill="toself",
            fillcolor=_C["ego"],
            line=dict(color=_C["ego"], width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ── Plot factories ────────────────────────────────────────────────────


def _empty_fig(msg: str = "", h: int = 480) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_PLOT_KW, height=h)
    if msg:
        fig.add_annotation(
            text=msg,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#666"),
        )
    return fig


def _bev_axes(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="X → right (m)", scaleanchor="y", range=[-55, 55]),
        yaxis=dict(title="Y → forward (m)", range=[-55, 55]),
    )


def create_entropy_bev(
    entropy: Optional[np.ndarray],
    selected: list,
    cfg: BeamEvalConfig,
    h: int = 500,
) -> go.Figure:
    if entropy is None:
        return _empty_fig("No ensemble model – entropy unavailable", h)

    gc = cfg.ensemble_grid_conf
    xs = np.linspace(gc["xbound"][0], gc["xbound"][1], entropy.shape[1])
    ys = np.linspace(gc["ybound"][0], gc["ybound"][1], entropy.shape[0])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=entropy,
            x=xs.tolist(),
            y=ys.tolist(),
            colorscale="Inferno",
            colorbar=dict(title="H", len=0.75, thickness=12),
            hovertemplate="x=%{x:.1f}m  y=%{y:.1f}m<br>entropy=%{z:.3f}<extra></extra>",
        )
    )
    _add_beam_wedges(
        fig, selected, cfg.beam_width, cfg.max_range, _C["accent_fill"], _C["accent_line"]
    )
    _add_ego_marker(fig)
    fig.update_layout(**_PLOT_KW, height=h, **_bev_axes("Entropy BEV + Selected Beams"))
    return fig


def create_radar_bev(
    pts_all: np.ndarray,
    pts_filt: np.ndarray,
    selected: list,
    cfg: BeamEvalConfig,
    h: int = 500,
) -> go.Figure:
    fig = go.Figure()
    _add_range_rings(fig)

    if len(pts_all) > 0:
        fig.add_trace(
            go.Scatter(
                x=pts_all[:, 0].tolist(),
                y=pts_all[:, 1].tolist(),
                mode="markers",
                marker=dict(size=3, color=_C["pts_all"]),
                name=f"All  ({len(pts_all):,})",
                hovertemplate="x=%{x:.1f}  y=%{y:.1f}<extra>all</extra>",
            )
        )
    if len(pts_filt) > 0:
        fig.add_trace(
            go.Scatter(
                x=pts_filt[:, 0].tolist(),
                y=pts_filt[:, 1].tolist(),
                mode="markers",
                marker=dict(
                    size=4,
                    color=pts_filt[:, 3].tolist(),
                    colorscale="Viridis",
                    colorbar=dict(title="RCS", len=0.6, thickness=12, x=1.02),
                ),
                name=f"Filtered  ({len(pts_filt):,})",
                hovertemplate="x=%{x:.1f}  y=%{y:.1f}  rcs=%{marker.color:.1f}<extra>beam</extra>",
            )
        )

    _add_beam_wedges(
        fig, selected, cfg.beam_width, cfg.max_range, _C["accent_fill"], _C["accent_line"]
    )
    _add_ego_marker(fig)
    fig.update_layout(
        **_PLOT_KW,
        height=h,
        **_bev_axes("Radar BEV  (all vs. filtered)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.45)"),
    )
    return fig


def create_3d_scatter(
    pts: np.ndarray,
    title: str,
    h: int = 500,
) -> go.Figure:
    if len(pts) == 0:
        return _empty_fig("No points", h)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0].tolist(),
                y=pts[:, 1].tolist(),
                z=pts[:, 2].tolist(),
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=pts[:, 3].tolist(),
                    colorscale="Viridis",
                    colorbar=dict(title="RCS", len=0.7, thickness=12),
                    opacity=0.85,
                ),
                hovertemplate=(
                    "x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<br>"
                    "rcs=%{marker.color:.1f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        **_PLOT_KW,
        height=h,
        title=dict(text=title, font=dict(size=13)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=0.5, y=-1.8, z=1.2)),
        ),
    )
    return fig


# ── Camera-image card helper ─────────────────────────────────────────


def _cam_card(label: str, src: str) -> dbc.Card:
    body = (
        html.Img(
            src=src,
            style={"width": "100%", "borderRadius": "4px", "border": "1px solid #333"},
        )
        if src
        else html.Div(
            "N/A",
            className="text-muted text-center",
            style={"height": "90px", "lineHeight": "90px"},
        )
    )
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    label,
                    className="text-center text-muted mb-0",
                    style={"fontSize": "0.7rem"},
                ),
                body,
            ],
            className="p-1",
        ),
        className="bg-dark border-secondary",
    )


# ── Dash application ─────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Beam Radar Viz",
)


def _build_layout() -> dbc.Container:
    scenes = sorted(_viz.scene_samples, key=lambda t: _viz.scene_names.get(t, ""))
    opts = [{"label": _viz.scene_names[t], "value": t} for t in scenes]
    default = scenes[0] if scenes else None
    n_default = max(0, len(_viz.scene_samples.get(default, [])) - 1) if default else 0
    budget = _viz.config.beam_budget_pct if _viz.config else 20

    return dbc.Container(
        [
            # header
            dbc.Row(
                dbc.Col(
                    html.H4(
                        "Beam-Filtered Radar Visualization",
                        className="text-center my-3",
                        style={"color": _C["accent"], "letterSpacing": "1px"},
                    )
                )
            ),
            # controls
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Scene", className="small text-muted"),
                                    dcc.Dropdown(
                                        id="scene-dd",
                                        options=opts,
                                        value=default,
                                        clearable=False,
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Sample", className="small text-muted"),
                                    dcc.Slider(
                                        id="sample-sl",
                                        min=0,
                                        max=n_default,
                                        step=1,
                                        value=0,
                                        marks=None,
                                        tooltip=dict(
                                            placement="bottom", always_visible=True
                                        ),
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "Beam Budget (%)", className="small text-muted"
                                    ),
                                    dcc.Slider(
                                        id="budget-sl",
                                        min=5,
                                        max=100,
                                        step=5,
                                        value=budget,
                                        marks={v: str(v) for v in range(0, 101, 25)},
                                        tooltip=dict(
                                            placement="bottom", always_visible=True
                                        ),
                                    ),
                                ],
                                md=4,
                            ),
                        ]
                    )
                ),
                className="mb-3",
                style={"backgroundColor": _C["card_bg"]},
            ),
            # cameras
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            "Camera Ring",
                            className="small text-muted text-center mb-2",
                        ),
                        html.Div(id="cam-grid"),
                    ]
                ),
                className="mb-3",
                style={"backgroundColor": _C["card_bg"]},
            ),
            # BEV row
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(id="entropy-bev", config={"displaylogo": False})
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(id="radar-bev", config={"displaylogo": False})
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # 3-D row
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(id="full-3d", config={"displaylogo": False})
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Loading(
                            dcc.Graph(id="filt-3d", config={"displaylogo": False})
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # stats
            dbc.Card(
                dbc.CardBody(html.Div(id="stats-bar", className="text-center")),
                className="mb-4",
                style={"backgroundColor": _C["card_bg"]},
            ),
        ],
        fluid=True,
    )


# ── Callbacks ─────────────────────────────────────────────────────────


@app.callback(
    [
        Output("sample-sl", "max"),
        Output("sample-sl", "value"),
    ],
    Input("scene-dd", "value"),
)
def _on_scene(scene_token):
    if not scene_token:
        return 0, 0
    n = len(_viz.scene_samples.get(scene_token, []))
    return max(0, n - 1), 0


@app.callback(
    [
        Output("entropy-bev", "figure"),
        Output("radar-bev", "figure"),
        Output("full-3d", "figure"),
        Output("filt-3d", "figure"),
        Output("cam-grid", "children"),
        Output("stats-bar", "children"),
    ],
    [
        Input("sample-sl", "value"),
        Input("budget-sl", "value"),
    ],
    State("scene-dd", "value"),
)
def _update(sample_idx, budget_pct, scene_token):
    blank = (
        _empty_fig(),
        _empty_fig(),
        _empty_fig(),
        _empty_fig(),
        html.P("Select a scene.", className="text-muted text-center"),
        html.Span("–", className="text-muted"),
    )
    if not scene_token or scene_token not in _viz.scene_samples:
        return blank

    indices = _viz.scene_samples[scene_token]
    idx = min(sample_idx or 0, len(indices) - 1)
    info = _viz.val_infos[indices[idx]]
    token = info["sample_token"]
    cfg = _viz.config

    # ── cameras ───────────────────────────────────────────────────────
    cam_rows = []
    for row_cams in (_CAM_TOP, _CAM_BOT):
        cols = []
        for cam in row_cams:
            fpath = os.path.join(_viz.data_root, info["cam_infos"][cam]["filename"])
            src = _encode_image(fpath) if os.path.isfile(fpath) else ""
            cols.append(
                dbc.Col(_cam_card(cam.replace("CAM_", "").replace("_", " "), src), md=4)
            )
        cam_rows.append(dbc.Row(cols, className="mb-1"))

    # ── entropy & beam selection ──────────────────────────────────────
    entropy, belief = _viz.get_entropy(token, info)
    candidates = cfg.candidate_azimuths
    if entropy is not None and _viz.selector is not None:
        selected = _viz.selector.select_beams_from_maps(
            entropy, belief, budget_pct, candidates
        )
    else:
        selected = []

    # ── radar points ──────────────────────────────────────────────────
    pts_all = _viz.load_bev_points(info)
    pts_filt = filter_bev_points_by_beams(
        pts_all, selected, cfg.beam_width, cfg.min_range, cfg.max_range
    )

    # ── figures ───────────────────────────────────────────────────────
    fig_ent = create_entropy_bev(entropy, selected, cfg)
    fig_rbev = create_radar_bev(pts_all, pts_filt, selected, cfg)
    fig_3d_all = create_3d_scatter(pts_all, f"Full Radar  ({len(pts_all):,} pts)")
    fig_3d_f = create_3d_scatter(pts_filt, f"Filtered Radar  ({len(pts_filt):,} pts)")

    # ── stats ─────────────────────────────────────────────────────────
    n_all, n_filt = len(pts_all), len(pts_filt)
    pct = 100 * n_filt / n_all if n_all > 0 else 0
    stats = html.Div(
        [
            html.Span(f"Sample  {token[:16]}…   ", className="text-muted"),
            html.Span(
                f"Beams  {len(selected)} / {cfg.num_candidates}   |   ",
                style={"color": _C["accent"]},
            ),
            html.Span(
                f"Points  {n_filt:,} / {n_all:,}  ({pct:.1f}% retained)",
                style={"color": _C["accent"]},
            ),
        ],
        style={"fontSize": "0.9rem"},
    )

    return fig_ent, fig_rbev, fig_3d_all, fig_3d_f, cam_rows, stats


# ── CLI ───────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description="Beam-filtered radar viz app.")
    p.add_argument(
        "--data_root",
        required=True,
        help="Path to nuScenes root (contains nuscenes_infos_val.pkl).",
    )
    p.add_argument("--ensemble_ckpt", default="", help="Ensemble LSS checkpoint.")
    p.add_argument("--beam_budget_pct", type=float, default=20.0)
    p.add_argument("--beam_width", type=float, default=3.0)
    p.add_argument("--azimuth_fov", type=float, default=360.0)
    p.add_argument("--port", type=int, default=8050)
    return p.parse_args()


def main():
    args = _parse_args()
    config = BeamEvalConfig(
        beam_budget_pct=args.beam_budget_pct,
        beam_width=args.beam_width,
        azimuth_fov=args.azimuth_fov,
    )

    print(f"Loading data from: {args.data_root}")
    _viz.init(config, args.data_root, args.ensemble_ckpt)
    print(
        f"  {len(_viz.val_infos)} val samples  ·  "
        f"{len(_viz.scene_samples)} scenes"
    )
    if _viz.selector:
        print(f"  Ensemble loaded on {_viz.selector.device}")
    else:
        print("  No ensemble checkpoint – entropy disabled")

    app.layout = _build_layout()
    print(f"\n  → http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()

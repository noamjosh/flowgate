"""FlowGate for Google Colab — same engine, browser-based gating with Plotly.

Why a separate GUI: the PyQt app needs a desktop display, which Colab doesn't
have. Dash runs *inside* a Colab cell (``app.run(jupyter_mode="inline")``), and
Plotly's drawing tools let you draw a polygon directly on the density plot; we
read it back via ``relayoutData`` and store it on the shared gate. The core
engine (``flowgate.core``) is reused unchanged, so gating logic and the
apply-to-all-mice behaviour are identical to the desktop app.

Usage inside Colab (see the companion notebook):

    from flowgate.core import Session
    from flowgate.colab_app import build_app, init_gates_from_config
    sess = Session()
    sess.add_sample("Mouse1.fcs"); sess.add_sample("Mouse2.fcs")
    init_gates_from_config(sess)
    app = build_app(sess)
    app.run(jupyter_mode="inline", height=900)
"""
from __future__ import annotations

import re
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table

from .core import Session, PolygonGate, default_transform_for, pseudocolor_density
from .config import GATE_HIERARCHY, resolve_channel


# --------------------------------------------------------------------------- #
# helpers (pure functions — unit-testable without a browser)
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_svg_path(path: str) -> list[tuple[float, float]]:
    """Parse a Plotly 'drawclosedpath' SVG path ('M x,y L x,y ... Z') into
    a list of (x, y) vertices. Robust to spaces/commas and the trailing Z."""
    nums = [float(n) for n in _NUM.findall(path or "")]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def init_gates_from_config(session: Session) -> None:
    """Build the shared gate tree from GATE_HIERARCHY, resolving markers."""
    if session.tree.order:
        return
    mm = session.marker_map()
    chans = session.common_channels()
    for spec in GATE_HIERARCHY:
        try:
            xch = resolve_channel(spec.x, mm, chans)
            ych = resolve_channel(spec.y, mm, chans)
        except KeyError as e:
            print(f"[skip gate {spec.name}] {e}")
            continue
        session.tree.add(PolygonGate(
            name=spec.name, x_channel=xch, y_channel=ych, parent=spec.parent,
            vertices=[],
            x_transform=default_transform_for(xch, mm.get(xch, "")),
            y_transform=default_transform_for(ych, mm.get(ych, "")),
        ))


def _display_xy(session, sample_id, gate, max_points=120_000):
    s = session.samples[sample_id]
    x = gate.x_transform.forward(s.data[gate.x_channel].to_numpy())
    y = gate.y_transform.forward(s.data[gate.y_channel].to_numpy())
    if gate.parent and gate.parent != "root":
        pmask = session.masks_for(sample_id)[gate.parent]
        x, y = x[pmask], y[pmask]
    # subsample for plotting only (gating always uses all events)
    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), max_points, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def _axis_title(session, sample_id, channel, transform):
    marker = session.samples[sample_id].markers.get(channel, "")
    base = f"{channel} ({marker})" if marker and marker != channel else channel
    return base if transform.name == "linear" else f"{base} [{transform.name}]"


def density_figure(session, sample_id, gate, editable=True):
    """A FlowJo-style pseudocolor dot plot for (sample, gate) with draw tools."""
    x, y = _display_xy(session, sample_id, gate)
    fig = go.Figure()
    if len(x):
        xlo, xhi = np.percentile(x, [0.05, 99.5])
        ylo, yhi = np.percentile(y, [0.05, 99.5])
        xr, yr = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
        xlo, xhi = xlo - 0.03 * xr, xhi + 0.03 * xr
        ylo, yhi = ylo - 0.03 * yr, yhi + 0.03 * yr
        c = pseudocolor_density(x, y, (xlo, xhi), (ylo, yhi))
        order = np.argsort(c)
        fig.add_trace(go.Scattergl(
            x=x[order], y=y[order], mode="markers",
            marker=dict(color=c[order], colorscale="Jet", size=3, showscale=False),
            hoverinfo="skip",
        ))
        fig.update_xaxes(range=[xlo, xhi])
        fig.update_yaxes(range=[ylo, yhi])
    # show the existing gate polygon (closed) as a black outline
    if gate.is_defined():
        vx = [v[0] for v in gate.vertices] + [gate.vertices[0][0]]
        vy = [v[1] for v in gate.vertices] + [gate.vertices[0][1]]
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode="lines", line=dict(color="black", width=2),
            name="current gate", hoverinfo="skip",
        ))
    fig.update_layout(
        dragmode="drawclosedpath" if editable else "zoom",
        newshape=dict(line=dict(color="black", width=2)),
        margin=dict(l=50, r=10, t=40, b=50),
        title=f"{gate.name}  —  {sample_id}",
        xaxis_title=_axis_title(session, sample_id, gate.x_channel, gate.x_transform),
        yaxis_title=_axis_title(session, sample_id, gate.y_channel, gate.y_transform),
        height=560, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def compare_figure(session, gate):
    """One shared gate drawn on every sample, side by side, with % gated."""
    from plotly.subplots import make_subplots
    ids = list(session.samples.keys())
    n = len(ids)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    titles = []
    for sid in ids:
        masks = session.masks_for(sid)
        parent = gate.parent if gate.parent and gate.parent != "root" else None
        denom = masks[parent].sum() if parent else session.samples[sid].n_events()
        pct = 100.0 * masks[gate.name].sum() / max(1, denom)
        titles.append(f"{sid} — {gate.name}: {pct:.1f}%")
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for i, sid in enumerate(ids):
        r, c = i // cols + 1, i % cols + 1
        x, y = _display_xy(session, sid, gate)
        if len(x):
            fig.add_trace(go.Histogram2d(x=x, y=y, nbinsx=100, nbinsy=100,
                                         colorscale="Turbo", showscale=False),
                          row=r, col=c)
        if gate.is_defined():
            vx = [v[0] for v in gate.vertices] + [gate.vertices[0][0]]
            vy = [v[1] for v in gate.vertices] + [gate.vertices[0][1]]
            fig.add_trace(go.Scatter(x=vx, y=vy, mode="lines",
                                     line=dict(color="red", width=1.5),
                                     hoverinfo="skip"), row=r, col=c)
    fig.update_layout(height=320 * rows, showlegend=False,
                      margin=dict(l=30, r=10, t=40, b=30))
    fig.update_annotations(font_size=11)
    return fig


def _stats_records(session):
    session.apply()
    return session.stats_frame().to_dict("records")


# --------------------------------------------------------------------------- #
# the Dash app
# --------------------------------------------------------------------------- #
def build_app(session: Session) -> Dash:
    app = Dash(__name__)
    gate_names = session.tree.order
    sample_ids = list(session.samples.keys())

    app.layout = html.Div([
        html.H3("FlowGate — Colab"),
        html.Div([
            html.Div([
                html.Label("Sample (mouse)"),
                dcc.Dropdown(id="sample", options=sample_ids,
                             value=sample_ids[0] if sample_ids else None,
                             clearable=False),
            ], style={"flex": 1, "marginRight": "12px"}),
            html.Div([
                html.Label("Population (gate)"),
                dcc.Dropdown(id="gate", options=gate_names,
                             value=gate_names[0] if gate_names else None,
                             clearable=False),
            ], style={"flex": 1}),
        ], style={"display": "flex", "marginBottom": "8px"}),

        html.Div([
            html.Button("Apply drawn polygon → gate", id="apply", n_clicks=0),
            html.Button("Clear polygon", id="clear", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Button("Compare across samples", id="compare", n_clicks=0,
                        style={"marginLeft": "8px"}),
            html.Span(id="msg", style={"marginLeft": "12px", "color": "#555"}),
        ], style={"marginBottom": "8px"}),

        html.P("Draw a polygon: pick the pencil/'Draw closed freeform' tool in the "
               "plot toolbar, click to place vertices, close it, then click "
               "'Apply drawn polygon → gate'. The gate is applied to every sample.",
               style={"fontSize": "12px", "color": "#666"}),

        dcc.Graph(id="density",
                  config={"modeBarButtonsToAdd": ["drawclosedpath", "eraseshape"],
                          "displaylogo": False}),
        dcc.Store(id="drawn"),

        html.H4("Statistics (all samples × gates)"),
        dash_table.DataTable(
            id="stats",
            columns=[{"name": c, "id": c} for c in
                     ["sample", "gate", "events", "parent_events", "pct_of_parent"]],
            style_cell={"fontSize": "12px", "padding": "4px"},
            page_size=20,
        ),

        html.Div(id="compare-wrap"),
    ], style={"maxWidth": "980px", "margin": "0 auto", "fontFamily": "sans-serif"})

    # ---- redraw density on sample/gate change ----
    @app.callback(Output("density", "figure"),
                  Input("sample", "value"), Input("gate", "value"))
    def _draw(sample_id, gate_name):
        if not sample_id or not gate_name:
            return no_update
        session.apply()
        return density_figure(session, sample_id, session.tree.gates[gate_name])

    # ---- capture drawn shapes ----
    @app.callback(Output("drawn", "data"), Input("density", "relayoutData"),
                  prevent_initial_call=True)
    def _capture(relayout):
        if relayout and "shapes" in relayout and relayout["shapes"]:
            last = relayout["shapes"][-1]
            if last.get("type") == "path":
                return last.get("path")
        return no_update

    # ---- apply / clear ----
    @app.callback(
        Output("stats", "data"), Output("msg", "children"),
        Output("density", "figure", allow_duplicate=True),
        Input("apply", "n_clicks"), Input("clear", "n_clicks"),
        State("sample", "value"), State("gate", "value"), State("drawn", "data"),
        prevent_initial_call=True,
    )
    def _apply_or_clear(n_apply, n_clear, sample_id, gate_name, drawn_path):
        from dash import ctx
        gate = session.tree.gates[gate_name]
        if ctx.triggered_id == "clear":
            gate.vertices = []
            msg = f"Cleared '{gate_name}'."
        else:
            verts = parse_svg_path(drawn_path)
            if len(verts) < 3:
                return no_update, "Draw a polygon first (need ≥3 points).", no_update
            gate.vertices = verts
            msg = f"Applied '{gate_name}' ({len(verts)} vertices) to all samples."
        fig = density_figure(session, sample_id, gate)
        return _stats_records(session), msg, fig

    # ---- compare ----
    @app.callback(Output("compare-wrap", "children"),
                  Input("compare", "n_clicks"), State("gate", "value"),
                  prevent_initial_call=True)
    def _compare(n, gate_name):
        session.apply()
        return dcc.Graph(figure=compare_figure(session, session.tree.gates[gate_name]))

    # populate stats once at startup
    @app.callback(Output("stats", "data", allow_duplicate=True),
                  Input("sample", "value"), prevent_initial_call="initial_duplicate")
    def _init_stats(_):
        return _stats_records(session)

    return app

"""FlowGate GUI — a FlowJo-style interactive gating app built on the core engine.

Layout
------
  ┌─────────────┬───────────────────────────────┐
  │ Samples     │  Density plot for (sample,     │
  │ (mice)      │  gate) with an editable        │
  ├─────────────┤  polygon. Draw / edit /        │
  │ Populations │  apply the gate here.          │
  │ (gate tree) │                                │
  └─────────────┴───────────────────────────────┘
  │ Stats table: every sample × every gate       │
  └───────────────────────────────────────────────┘

The polygon you draw is stored on the shared gate and applied to ALL samples,
so re-gating and the "Compare across samples" grid update every mouse at once.
"""
from __future__ import annotations

import os
import sys
import numpy as np

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon as MplPolygon

from PyQt5 import QtWidgets, QtCore

from .core import (
    Session, PolygonGate, LinearAxis, AsinhAxis, default_transform_for,
    pseudocolor_density,
)
from .config import GATE_HIERARCHY, resolve_channel


# --------------------------------------------------------------------------- #
# A density plot that can host an editable polygon
# --------------------------------------------------------------------------- #
class DensityCanvas(FigureCanvas):
    """Renders a 2D density for one (sample, gate) and manages a PolygonSelector."""

    polygon_committed = QtCore.pyqtSignal(list)  # list of (x, y) in display coords

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._selector: PolygonSelector | None = None
        self._sample = None
        self._gate: PolygonGate | None = None
        self._parent_mask = None

    # ---- drawing ----
    def show_target(self, sample, gate: PolygonGate, parent_mask=None):
        """Render the density of the parent population and overlay the gate."""
        self._sample = sample
        self._gate = gate
        self._parent_mask = parent_mask
        self._deactivate_selector()
        self._redraw()

    def _display_points(self):
        g, s = self._gate, self._sample
        x = g.x_transform.forward(s.data[g.x_channel].to_numpy())
        y = g.y_transform.forward(s.data[g.y_channel].to_numpy())
        if self._parent_mask is not None:
            x, y = x[self._parent_mask], y[self._parent_mask]
        return x, y

    def _redraw(self):
        self.ax.clear()
        self.ax.set_facecolor("white")
        if self._sample is None or self._gate is None:
            self.draw_idle()
            return
        g = self._gate
        x, y = self._display_points()

        if len(x) > 0:
            # Tight, robust view limits so the population fills the frame
            # (FlowJo-style) instead of being squashed by rare outliers.
            xlo, xhi = np.percentile(x, [0.05, 99.5])
            ylo, yhi = np.percentile(y, [0.05, 99.5])
            xr, yr = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
            xlo, xhi = xlo - 0.03 * xr, xhi + 0.03 * xr
            ylo, yhi = ylo - 0.03 * yr, yhi + 0.03 * yr

            # Per-event local density -> pseudocolor dot plot (the FlowJo look).
            c = pseudocolor_density(x, y, (xlo, xhi), (ylo, yhi))
            order = np.argsort(c)  # draw dense points last, on top
            self.ax.scatter(
                x[order], y[order], c=c[order], s=3, cmap="jet",
                linewidths=0, rasterized=True,
            )
            self.ax.set_xlim(xlo, xhi)
            self.ax.set_ylim(ylo, yhi)

        self.ax.set_xlabel(self._axis_label(g.x_channel, g.x_transform))
        self.ax.set_ylabel(self._axis_label(g.y_channel, g.y_transform))
        self.ax.set_title(f"{g.name}  ({self._sample.sample_id})")

        # existing gate polygon: black so it's visible on the white background
        if g.is_defined():
            patch = MplPolygon(
                np.asarray(g.vertices), closed=True, fill=False,
                edgecolor="black", lw=1.6,
            )
            self.ax.add_patch(patch)
        self.draw_idle()

    def _axis_label(self, channel, transform):
        marker = self._sample.markers.get(channel, "") if self._sample else ""
        # don't show a redundant "(FSC-A)" when the marker equals the channel
        base = f"{channel} ({marker})" if marker and marker != channel else channel
        suffix = "" if isinstance(transform, LinearAxis) else f" [{transform.name}]"
        return base + suffix

    # ---- polygon editing ----
    def start_drawing(self):
        if self._gate is None:
            return
        self._deactivate_selector()

        def on_select(verts):
            # store; committing happens on button press so the user can refine
            self._pending = [tuple(map(float, v)) for v in verts]

        self._pending = None
        self._selector = PolygonSelector(
            self.ax, on_select, useblit=True,
            props=dict(color="black", linewidth=2),
        )
        self.draw_idle()

    def commit_polygon(self) -> bool:
        """Push the drawn polygon onto the gate. Returns True if a polygon existed."""
        verts = getattr(self, "_pending", None)
        if not verts or len(verts) < 3:
            # fall back to whatever the selector currently holds
            if self._selector is not None and len(self._selector.verts) >= 3:
                verts = [tuple(map(float, v)) for v in self._selector.verts]
            else:
                return False
        self._gate.vertices = verts
        self._deactivate_selector()
        self._redraw()
        self.polygon_committed.emit(verts)
        return True

    def clear_polygon(self):
        if self._gate is not None:
            self._gate.vertices = []
        self._deactivate_selector()
        self._redraw()
        self.polygon_committed.emit([])

    def _deactivate_selector(self):
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
                self._selector.set_visible(False)
            except Exception:
                pass
            self._selector = None


# --------------------------------------------------------------------------- #
# Comparison grid across all samples for one gate
# --------------------------------------------------------------------------- #
class CompareDialog(QtWidgets.QDialog):
    def __init__(self, session: Session, gate: PolygonGate, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Compare gate '{gate.name}' across samples")
        self.resize(1000, 700)
        lay = QtWidgets.QVBoxLayout(self)
        canvas = FigureCanvas(Figure(figsize=(10, 7), tight_layout=True))
        lay.addWidget(NavigationToolbar(canvas, self))
        lay.addWidget(canvas)

        ids = list(session.samples.keys())
        n = len(ids)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        for i, sid in enumerate(ids):
            ax = canvas.figure.add_subplot(rows, cols, i + 1)
            s = session.samples[sid]
            masks = session.masks_for(sid)
            pmask = masks[gate.parent] if gate.parent and gate.parent != "root" else None
            x = gate.x_transform.forward(s.data[gate.x_channel].to_numpy())
            y = gate.y_transform.forward(s.data[gate.y_channel].to_numpy())
            if pmask is not None:
                x, y = x[pmask], y[pmask]
            if len(x):
                xlo, xhi = np.percentile(x, [0.1, 99.9])
                ylo, yhi = np.percentile(y, [0.1, 99.9])
                h, xe, ye = np.histogram2d(
                    x, y, bins=120,
                    range=[[xlo, xhi], [ylo, yhi]],
                )
                ax.imshow(np.log1p(h.T), origin="lower", aspect="auto",
                          extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="turbo")
            if gate.is_defined():
                ax.add_patch(MplPolygon(np.asarray(gate.vertices), closed=True,
                                        fill=False, edgecolor="white", lw=1.5))
            pct = 100.0 * masks[gate.name].sum() / max(1, (pmask.sum() if pmask is not None else s.n_events()))
            ax.set_title(f"{sid}\n{gate.name}: {pct:.1f}%", fontsize=9)
            ax.tick_params(labelsize=7)
        canvas.draw_idle()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class GatingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowGate v2 — FlowJo-style plots (white background)")
        self.resize(1200, 850)
        self.session = Session()

        # --- central density canvas + controls ---
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        self.canvas = DensityCanvas()
        cv.addWidget(NavigationToolbar(self.canvas, self))
        cv.addWidget(self.canvas, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("① Load FCS files…")
        self.btn_load.setStyleSheet("font-weight: bold;")
        controls.addWidget(self.btn_load)
        self.btn_draw = QtWidgets.QPushButton("Draw / edit polygon")
        self.btn_apply = QtWidgets.QPushButton("Apply polygon → gate")
        self.btn_clear = QtWidgets.QPushButton("Clear polygon")
        self.btn_compare = QtWidgets.QPushButton("Compare across samples")
        for b in (self.btn_draw, self.btn_apply, self.btn_clear, self.btn_compare):
            controls.addWidget(b)
        cv.addLayout(controls)
        self.setCentralWidget(central)

        self.btn_load.clicked.connect(self._load_fcs)
        self.btn_draw.clicked.connect(lambda: self.canvas.start_drawing())
        self.btn_apply.clicked.connect(self._apply_polygon)
        self.btn_clear.clicked.connect(self.canvas.clear_polygon)
        self.btn_compare.clicked.connect(self._compare)

        # --- left dock: samples + populations ---
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel("Samples"))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.currentTextChanged.connect(self._refresh_canvas)
        lv.addWidget(self.sample_list)
        lv.addWidget(QtWidgets.QLabel("Populations (gate hierarchy)"))
        self.gate_tree = QtWidgets.QTreeWidget()
        self.gate_tree.setHeaderLabels(["Population"])
        self.gate_tree.currentItemChanged.connect(self._refresh_canvas)
        lv.addWidget(self.gate_tree)
        dock_l = QtWidgets.QDockWidget("Navigator", self)
        dock_l.setWidget(left)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_l)

        # --- bottom dock: stats ---
        self.stats_table = QtWidgets.QTableWidget()
        dock_b = QtWidgets.QDockWidget("Statistics (all samples × gates)", self)
        dock_b.setWidget(self.stats_table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_b)

        self._build_menu()

    # ---- menu / actions ----
    def _build_menu(self):
        mb = self.menuBar()
        # Keep the menu INSIDE the window. On macOS Qt otherwise moves it to the
        # top-of-screen menu bar; on some Linux setups the global menu hides it.
        mb.setNativeMenuBar(False)
        m = mb.addMenu("&File")
        m.addAction("Load FCS files…", self._load_fcs)
        m.addAction("Save gates…", self._save_gates)
        m.addAction("Load gates…", self._load_gates)
        m.addSeparator()
        m.addAction("Export stats CSV…", self._export_stats)
        g = mb.addMenu("&Gating")
        g.addAction("Re-gate all samples", self._regate)

        # A visible toolbar as well, so file loading is never hidden by a
        # missing/global menu bar regardless of platform.
        tb = self.addToolBar("Main")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        tb.addAction("Load FCS files…", self._load_fcs)
        tb.addAction("Load gates…", self._load_gates)
        tb.addAction("Save gates…", self._save_gates)
        tb.addAction("Re-gate all", self._regate)
        tb.addAction("Export stats CSV…", self._export_stats)

    def _load_fcs(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load FCS files", "", "FCS files (*.fcs)")
        if not paths:
            return
        for p in paths:
            self.session.add_sample(p)
        self._init_gates_from_config()
        self._refresh_sample_list()
        self._regate()

    def _init_gates_from_config(self):
        """Populate the shared tree with (initially empty) gates from config,
        resolving marker names to this panel's detectors."""
        if self.session.tree.order:
            return  # already built
        mm = self.session.marker_map()
        chans = self.session.common_channels()
        for spec in GATE_HIERARCHY:
            try:
                xch = resolve_channel(spec.x, mm, chans)
                ych = resolve_channel(spec.y, mm, chans)
            except KeyError as e:
                print(f"[skip gate {spec.name}] {e}")
                continue
            self.session.tree.add(PolygonGate(
                name=spec.name, x_channel=xch, y_channel=ych, parent=spec.parent,
                vertices=[],
                x_transform=default_transform_for(xch, mm.get(xch, "")),
                y_transform=default_transform_for(ych, mm.get(ych, "")),
            ))
        self._refresh_gate_tree()

    # ---- UI refreshers ----
    def _refresh_sample_list(self):
        self.sample_list.clear()
        for sid in self.session.samples:
            self.sample_list.addItem(sid)
        if self.sample_list.count() and self.sample_list.currentRow() < 0:
            self.sample_list.setCurrentRow(0)

    def _refresh_gate_tree(self):
        self.gate_tree.clear()
        items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for name in self.session.tree.order:
            gate = self.session.tree.gates[name]
            item = QtWidgets.QTreeWidgetItem([name])
            item.setData(0, QtCore.Qt.UserRole, name)
            items[name] = item
            if gate.parent and gate.parent in items:
                items[gate.parent].addChild(item)
            else:
                self.gate_tree.addTopLevelItem(item)
        self.gate_tree.expandAll()

    def _current_sample(self):
        it = self.sample_list.currentItem()
        return self.session.samples.get(it.text()) if it else None

    def _current_gate(self):
        it = self.gate_tree.currentItem()
        if not it:
            return None
        return self.session.tree.gates.get(it.data(0, QtCore.Qt.UserRole))

    def _refresh_canvas(self, *_):
        s, g = self._current_sample(), self._current_gate()
        if s is None or g is None:
            return
        masks = self.session.masks_for(s.sample_id)
        pmask = masks[g.parent] if g.parent and g.parent != "root" else None
        self.canvas.show_target(s, g, pmask)

    def _apply_polygon(self):
        if self.canvas.commit_polygon():
            self._regate()

    def _regate(self):
        self.session.apply()
        self._refresh_stats()
        self._refresh_canvas()

    def _compare(self):
        g = self._current_gate()
        if g is None:
            return
        self.session.apply()
        CompareDialog(self.session, g, self).exec_()

    def _refresh_stats(self):
        df = self.session.stats_frame()
        self.stats_table.setColumnCount(len(df.columns))
        self.stats_table.setHorizontalHeaderLabels(list(df.columns))
        self.stats_table.setRowCount(len(df))
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                self.stats_table.setItem(
                    r, c, QtWidgets.QTableWidgetItem(str(df.iloc[r, c])))
        self.stats_table.resizeColumnsToContents()

    def _save_gates(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save gates", "gates.json", "JSON (*.json)")
        if path:
            self.session.save_gates(path)

    def _load_gates(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load gates", "", "JSON (*.json)")
        if path:
            self.session.load_gates(path)
            self._refresh_gate_tree()
            self._regate()

    def _export_stats(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export stats", "stats.csv", "CSV (*.csv)")
        if path:
            self.session.stats_frame().to_csv(path, index=False)

    def load_paths(self, paths):
        """Load FCS files programmatically (e.g. from the command line)."""
        added = False
        for p in paths:
            if os.path.exists(p):
                self.session.add_sample(p)
                added = True
            else:
                print(f"[skip] file not found: {p}")
        if added:
            self._init_gates_from_config()
            self._refresh_sample_list()
            self._regate()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = GatingApp()
    # Any .fcs paths on the command line are loaded immediately, so you can
    # bypass the file dialog entirely:  python run.py file1.fcs file2.fcs
    paths = [a for a in sys.argv[1:] if a.lower().endswith(".fcs")]
    if paths:
        w.load_paths(paths)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

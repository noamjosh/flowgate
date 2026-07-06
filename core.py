"""FlowGate core engine — no GUI dependencies, fully testable.

Design in one paragraph
-----------------------
A :class:`Session` holds many :class:`FlowSample` objects (your mice) and ONE
shared :class:`GatingTree`. A gate is defined once (as a polygon in display
coordinates) and the *same* gate object is applied to every sample. That is the
whole trick behind requirement #8: edit a gate's vertices, call
``session.apply()`` again, and every mouse is re-gated and re-compared. Gates
form a hierarchy via ``parent`` — a child gate only sees the events its parent
kept.

Coordinate spaces
-----------------
Users draw polygons on a *display* plot whose axes may be transformed (linear for
scatter, arcsinh/biexponential for fluorescence). We store gate vertices in that
display space and, to apply a gate, we transform a sample's raw values into the
same display space and test point-in-polygon. This keeps "what you drew" and
"what gets gated" identical, regardless of sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.path import Path

import flowkit as fk


# --------------------------------------------------------------------------- #
# Axis transforms (forward only is needed for both plotting and gating)
# --------------------------------------------------------------------------- #
class AxisTransform:
    """Monotonic forward transform from raw channel value -> display value."""

    name = "identity"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)


class LinearAxis(AxisTransform):
    name = "linear"


class AsinhAxis(AxisTransform):
    """arcsinh transform — the well-behaved, invertible cousin of a log/biex axis.

    ``cofactor`` sets where the axis switches from ~linear (near 0, including
    negatives from compensation) to ~log. 150 is a sane default for fluorescence
    on an 18-bit scale; 262 or so also common.
    """

    name = "asinh"

    def __init__(self, cofactor: float = 150.0):
        self.cofactor = float(cofactor)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return np.arcsinh(x / self.cofactor)

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        return np.sinh(y) * self.cofactor


def default_transform_for(channel: str, marker: str = "") -> AxisTransform:
    """Scatter channels are linear; anything with a fluorophore/marker is asinh."""
    scatter_prefixes = ("FSC", "SSC", "TIME", "Time")
    if channel.upper().startswith(scatter_prefixes):
        return LinearAxis()
    return AsinhAxis()


def transform_to_dict(t: AxisTransform) -> dict:
    d = {"name": t.name}
    if isinstance(t, AsinhAxis):
        d["cofactor"] = t.cofactor
    return d


def transform_from_dict(d: dict) -> AxisTransform:
    if d["name"] == "asinh":
        return AsinhAxis(d.get("cofactor", 150.0))
    return LinearAxis()


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
class FlowSample:
    """One FCS file loaded into a DataFrame, with detector<->marker awareness."""

    def __init__(self, path: str, sample_id: Optional[str] = None):
        self.path = path
        self._fk = fk.Sample(path, sample_id=sample_id)
        self.sample_id = self._fk.id
        # DataFrame with raw values, columns = detector (PnN) labels
        df = self._fk.as_dataframe(source="raw")
        df.columns = [c[0] for c in df.columns]  # drop the (pnn, pns) multiindex
        self.data: pd.DataFrame = df.reset_index(drop=True)

        # detector -> marker map, e.g. {"BV421-A": "CD45"}
        self.markers: dict[str, str] = {}
        for pnn, pns in zip(self._fk.pnn_labels, self._fk.pns_labels):
            if pns:
                self.markers[pnn] = pns

    @property
    def channels(self) -> list[str]:
        return list(self.data.columns)

    def label(self, channel: str) -> str:
        """Human label: 'BV421-A (CD45)' when a marker is known."""
        m = self.markers.get(channel)
        return f"{channel} ({m})" if m else channel

    def n_events(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
@dataclass
class PolygonGate:
    """A named polygon gate on an (x, y) channel pair.

    Vertices are stored in *display* coordinates (after ``x_transform`` /
    ``y_transform``). ``parent`` is the name of the population this gate refines;
    ``None``/"root" means it operates on all ungated events.
    """

    name: str
    x_channel: str
    y_channel: str
    parent: Optional[str] = None
    vertices: list[tuple[float, float]] = field(default_factory=list)
    x_transform: AxisTransform = field(default_factory=LinearAxis)
    y_transform: AxisTransform = field(default_factory=LinearAxis)

    def is_defined(self) -> bool:
        return len(self.vertices) >= 3

    def display_xy(self, sample: FlowSample) -> np.ndarray:
        """Return this sample's events projected into the gate's display space."""
        x = self.x_transform.forward(sample.data[self.x_channel].to_numpy())
        y = self.y_transform.forward(sample.data[self.y_channel].to_numpy())
        return np.column_stack([x, y])

    def contains(self, sample: FlowSample) -> np.ndarray:
        """Boolean mask over ALL events of ``sample`` that fall inside the polygon.

        Note: this ignores parent membership; the GatingTree handles hierarchy by
        intersecting with the parent's mask.
        """
        if not self.is_defined():
            return np.ones(sample.n_events(), dtype=bool)
        pts = self.display_xy(sample)
        path = Path(np.asarray(self.vertices, dtype=float))
        return path.contains_points(pts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
            "parent": self.parent,
            "vertices": [list(map(float, v)) for v in self.vertices],
            "x_transform": transform_to_dict(self.x_transform),
            "y_transform": transform_to_dict(self.y_transform),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonGate":
        return cls(
            name=d["name"],
            x_channel=d["x_channel"],
            y_channel=d["y_channel"],
            parent=d.get("parent"),
            vertices=[tuple(v) for v in d.get("vertices", [])],
            x_transform=transform_from_dict(d["x_transform"]),
            y_transform=transform_from_dict(d["y_transform"]),
        )


# --------------------------------------------------------------------------- #
# Gating tree (shared across all samples)
# --------------------------------------------------------------------------- #
class GatingTree:
    """Ordered hierarchy of gates. One tree is shared by all samples in a Session."""

    def __init__(self):
        self.gates: dict[str, PolygonGate] = {}
        self.order: list[str] = []  # insertion order; parents precede children

    def add(self, gate: PolygonGate) -> PolygonGate:
        if gate.name in self.gates:
            raise ValueError(f"gate '{gate.name}' already exists")
        if gate.parent and gate.parent not in self.gates and gate.parent != "root":
            raise ValueError(
                f"parent '{gate.parent}' of gate '{gate.name}' not found"
            )
        self.gates[gate.name] = gate
        self.order.append(gate.name)
        return gate

    def children_of(self, name: Optional[str]) -> list[PolygonGate]:
        target = None if name in (None, "root") else name
        return [self.gates[n] for n in self.order if self.gates[n].parent == target]

    def apply(self, sample: FlowSample) -> dict[str, np.ndarray]:
        """Return {gate_name: boolean mask over sample events} respecting hierarchy.

        A "root" mask of all-True is available under the key ``"root"``.
        """
        masks: dict[str, np.ndarray] = {"root": np.ones(sample.n_events(), bool)}
        for name in self.order:
            gate = self.gates[name]
            parent_key = gate.parent if gate.parent else "root"
            parent_mask = masks.get(parent_key, masks["root"])
            masks[name] = parent_mask & gate.contains(sample)
        return masks


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class GateStat:
    sample_id: str
    gate: str
    n_in: int
    n_parent: int

    @property
    def pct_of_parent(self) -> float:
        return 100.0 * self.n_in / self.n_parent if self.n_parent else 0.0


class Session:
    """Holds all samples + the shared gating tree; applies gates to everyone."""

    def __init__(self):
        self.samples: dict[str, FlowSample] = {}
        self.tree = GatingTree()
        # cache: sample_id -> {gate_name: mask}
        self._masks: dict[str, dict[str, np.ndarray]] = {}

    # ---- samples ----
    def add_sample(self, path: str, sample_id: Optional[str] = None) -> FlowSample:
        s = FlowSample(path, sample_id=sample_id)
        self.samples[s.sample_id] = s
        return s

    def common_channels(self) -> list[str]:
        if not self.samples:
            return []
        sets = [set(s.channels) for s in self.samples.values()]
        common = set.intersection(*sets)
        # preserve first sample's order
        first = next(iter(self.samples.values()))
        return [c for c in first.channels if c in common]

    def marker_map(self) -> dict[str, str]:
        """Union of detector->marker maps across samples (first wins on conflict)."""
        out: dict[str, str] = {}
        for s in self.samples.values():
            for k, v in s.markers.items():
                out.setdefault(k, v)
        return out

    # ---- gating ----
    def apply(self) -> None:
        """(Re)gate every sample with the current tree. Call after any edit."""
        self._masks = {sid: self.tree.apply(s) for sid, s in self.samples.items()}

    def masks_for(self, sample_id: str) -> dict[str, np.ndarray]:
        if sample_id not in self._masks:
            self._masks[sample_id] = self.tree.apply(self.samples[sample_id])
        return self._masks[sample_id]

    def population(self, sample_id: str, gate_name: str) -> pd.DataFrame:
        """The DataFrame of events belonging to ``gate_name`` in one sample."""
        s = self.samples[sample_id]
        if gate_name in (None, "root"):
            return s.data
        mask = self.masks_for(sample_id)[gate_name]
        return s.data.loc[mask].reset_index(drop=True)

    def stats(self) -> list[GateStat]:
        """One row per (sample, gate): counts and % of parent."""
        rows: list[GateStat] = []
        for sid, s in self.samples.items():
            masks = self.masks_for(sid)
            for name in self.tree.order:
                gate = self.tree.gates[name]
                parent_key = gate.parent if gate.parent else "root"
                n_parent = int(masks[parent_key].sum())
                n_in = int(masks[name].sum())
                rows.append(GateStat(sid, name, n_in, n_parent))
        return rows

    # ---- persistence: thresholds/gates travel across sessions ----
    def save_gates(self, path: str) -> None:
        import json
        payload = {"gates": [self.tree.gates[n].to_dict() for n in self.tree.order]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_gates(self, path: str) -> None:
        import json
        with open(path) as f:
            payload = json.load(f)
        self.tree = GatingTree()
        for gd in payload["gates"]:
            self.tree.add(PolygonGate.from_dict(gd))
        self._masks = {}

    def stats_frame(self) -> pd.DataFrame:
        rows = self.stats()
        return pd.DataFrame(
            [
                {
                    "sample": r.sample_id,
                    "gate": r.gate,
                    "events": r.n_in,
                    "parent_events": r.n_parent,
                    "pct_of_parent": round(r.pct_of_parent, 2),
                }
                for r in rows
            ]
        )


# --------------------------------------------------------------------------- #
# Pseudocolor density (shared by both GUIs) — reproduces FlowJo's dot-plot shading
# --------------------------------------------------------------------------- #
def box_blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """Tiny dependency-free 5-point blur to smooth a 2D density grid."""
    for _ in range(passes):
        a = (a
             + np.pad(a[1:, :], ((0, 1), (0, 0)))
             + np.pad(a[:-1, :], ((1, 0), (0, 0)))
             + np.pad(a[:, 1:], ((0, 0), (0, 1)))
             + np.pad(a[:, :-1], ((0, 0), (1, 0)))) / 5.0
    return a


def pseudocolor_density(x, y, xlim, ylim, bins: int = 256) -> np.ndarray:
    """Per-event, log-scaled, smoothed local density for coloring a dot plot."""
    x = np.asarray(x); y = np.asarray(y)
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[list(xlim), list(ylim)])
    h = box_blur(h, passes=2)
    xi = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, bins - 1)
    return np.log1p(h[xi, yi])

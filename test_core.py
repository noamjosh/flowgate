"""Exercise the engine without a GUI: load all mice, build the hierarchy from
config, define polygons programmatically, apply to every sample, print stats.
"""
import glob
import numpy as np

from flowgate.core import (
    Session, PolygonGate, LinearAxis, AsinhAxis, default_transform_for,
)
from flowgate.config import GATE_HIERARCHY, resolve_channel


def rect(x0, x1, y0, y1):
    """Convenience: axis-aligned polygon (in display coords)."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def main():
    sess = Session()
    for p in sorted(glob.glob("data/*.fcs")):
        s = sess.add_sample(p)
        print(f"loaded {s.sample_id}: {s.n_events()} events, channels={s.channels}")
    print("markers:", sess.marker_map())
    print("common channels:", sess.common_channels())
    print()

    mm = sess.marker_map()
    chans = sess.common_channels()

    # Build gates from the config with simple demo polygons.
    for spec in GATE_HIERARCHY:
        xch = resolve_channel(spec.x, mm, chans)
        ych = resolve_channel(spec.y, mm, chans)
        xt = default_transform_for(xch, mm.get(xch, ""))
        yt = default_transform_for(ych, mm.get(ych, ""))

        # Choose a demo polygon per gate in DISPLAY coordinates.
        if spec.name == "Beads":
            verts = rect(150000, 262143, 150000, 262143)          # high FSC & SSC
        elif spec.name == "Real_Cells":
            verts = rect(25000, 130000, 15000, 110000)            # main cell cloud
        elif spec.name == "Real_Fwd_Cells":
            # FSC-A (lin) vs FSC-H (lin): singlets near diagonal
            verts = [(25000, 25000), (130000, 130000), (130000, 100000), (25000, 15000)]
        elif spec.name == "Real_Fwd_Side_Cells":
            verts = [(15000, 15000), (110000, 110000), (110000, 85000), (15000, 8000)]
        elif spec.name == "Live_Cells":
            # FSC-A (lin) vs Zombie (asinh): low zombie = live
            zlo, zhi = yt.forward(0), yt.forward(3000)
            verts = rect(25000, 130000, zlo, zhi)
        elif spec.name == "Category_Cells":
            # FSC-A (lin) vs CD45 (asinh): CD45 positive
            clo = yt.forward(3000)
            chi = yt.forward(262143)
            verts = rect(25000, 130000, clo, chi)
        else:
            verts = []

        sess.tree.add(PolygonGate(
            name=spec.name, x_channel=xch, y_channel=ych, parent=spec.parent,
            vertices=verts, x_transform=xt, y_transform=yt,
        ))

    sess.apply()
    print(sess.stats_frame().to_string(index=False))

    # Demonstrate requirement #8: edit ONE gate, re-apply, everyone updates.
    print("\n--- widen the Live gate (accept higher Zombie), re-apply ---")
    live = sess.tree.gates["Live_Cells"]
    yt = live.y_transform
    live.vertices = rect(25000, 130000, yt.forward(0), yt.forward(20000))
    sess.apply()
    sub = sess.stats_frame()
    print(sub[sub.gate.isin(["Live_Cells", "Category_Cells"])].to_string(index=False))


if __name__ == "__main__":
    main()

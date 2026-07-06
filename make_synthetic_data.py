"""Generate synthetic multi-mouse FCS files so the app can be tried without real data.

Each "mouse" gets slightly different population fractions and marker intensities,
which is what makes the cross-sample comparison view (Gate step #8) interesting.

Run:  python make_synthetic_data.py
Creates:  ./data/Mouse1.fcs ... Mouse4.fcs
"""
import os
import numpy as np
import flowio

CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "BV510-A", "BV421-A"]
# PnS marker names (the "stain"). Empty string means "scatter / unlabeled".
MARKERS = ["", "", "", "", "Zombie NIR", "CD45"]
MAX_VAL = 262143  # 18-bit, typical for modern cytometers


def make_mouse(seed, n=25000, live_frac=0.7, cd45_shift=0.0):
    rng = np.random.default_rng(seed)

    # Population fractions vary a little per mouse
    n_cells = int(n * 0.72)
    n_beads = int(n * 0.08)
    n_debris = n - n_cells - n_beads

    cells = rng.multivariate_normal([60000, 40000], [[8e7, 1e7], [1e7, 8e7]], n_cells)
    beads = rng.multivariate_normal([210000, 220000], [[4e7, 0], [0, 4e7]], n_beads)
    debris = rng.multivariate_normal([9000, 9000], [[2e7, 0], [0, 2e7]], n_debris)

    fsc_a = np.concatenate([cells[:, 0], beads[:, 0], debris[:, 0]])
    ssc_a = np.concatenate([cells[:, 1], beads[:, 1], debris[:, 1]])
    m = len(fsc_a)

    # Height correlates with area for singlets; add a doublet tail (low H/A ratio)
    ratio = rng.normal(0.95, 0.04, m)
    doublets = rng.random(m) < 0.08
    ratio[doublets] *= rng.uniform(0.55, 0.75, doublets.sum())
    fsc_h = fsc_a * ratio
    ssc_h = ssc_a * rng.normal(0.95, 0.04, m)

    # Viability (Zombie): live cells low, dead cells high. Only "cells" have meaningful signal.
    is_cell = np.arange(m) < n_cells
    live = is_cell & (rng.random(m) < live_frac)
    zombie = np.where(live, rng.lognormal(5.0, 0.8, m), rng.lognormal(9.5, 0.7, m))

    # CD45 (immune marker): a positive population among live cells, shifted per mouse
    cd45_pos = live & (rng.random(m) < 0.6)
    cd45 = np.where(cd45_pos,
                    rng.lognormal(9.5 + cd45_shift, 0.6, m),
                    rng.lognormal(6.0, 0.7, m))

    data = np.column_stack([fsc_a, fsc_h, ssc_a, ssc_h, zombie, cd45])
    data = np.clip(data, 0, MAX_VAL).astype("float32")
    return data


def main():
    os.makedirs("data", exist_ok=True)
    configs = [
        ("Mouse1", 1, 0.75, 0.0),
        ("Mouse2", 2, 0.68, 0.3),
        ("Mouse3", 3, 0.60, -0.2),
        ("Mouse4", 4, 0.80, 0.15),
    ]
    for name, seed, live_frac, cd45_shift in configs:
        data = make_mouse(seed, live_frac=live_frac, cd45_shift=cd45_shift)
        path = os.path.join("data", f"{name}.fcs")
        with open(path, "wb") as f:
            flowio.create_fcs(
                f, data.flatten(), channel_names=CHANNELS, opt_channel_names=MARKERS
            )
        print(f"wrote {path}  ({data.shape[0]} events)")


if __name__ == "__main__":
    main()

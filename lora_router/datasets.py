"""Dataset access for CircuitNet-N28 (source node) and CircuitNet-N14 (target node).

get(name) returns (names, load_placement(name, device), load_label(name)).
Labels follow CircuitNet's generate_training_set.py: cubic-resize horizontal and
vertical global-routing overflow to 256x256, sum, min-max scale to [0, 1].
"""
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .build import CF_ROOT

N28 = CF_ROOT / "dataset" / "CircuitNet"
N14 = CF_ROOT / "dataset" / "CircuitNet-N14"
N14_UM_PER_GCELL = 1.536  # measured: placement extent / label grid, Vortex-small


def _label(h, v):
    total = np.zeros((256, 256))
    for a in (h, v):
        total += ndimage.zoom(a, (256 / a.shape[0], 256 / a.shape[1]), order=3)
    rng = total.max() - total.min()
    return ((total - total.min()) / rng if total.max() != 0 else total).astype(np.float32)


def _to_batch(a, device):
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    y1, x1, y2, x2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]  # column order as in CircuitFormer's loader
    return [x1, y1, x2, y2, torch.tensor([a.shape[0]], dtype=torch.int32, device=device)]


def n28(unit="gcell"):
    names = [Path(l.split(",")[0]).stem for l in (CF_ROOT / "data" / "test.txt").read_text().split()]
    ov = N28 / "routability_features" / "congestion" / "overflow_based"

    def placement(name, device):
        d = np.load(N28 / "graph_features" / f"instance_placement_{unit}" / f"{name}.npy", allow_pickle=True).item()
        return _to_batch(np.array(list(d.values())), device)

    def label(name):
        return _label(*(np.load(ov / f"congestion_GR_{d}_overflow" / name) for d in ("horizontal", "vertical")))

    return names, placement, label


def n14(design="Vortex-small", unit="gcell", kind="util"):
    """N14 ships all-zero gcell placements, so gcell units are derived from micron.

    kind: "util" (utilization-based congestion, dense) or "overflow" (as N28; on
    Vortex-small it is almost empty and carries no learnable structure).
    """
    pl_dir = N14 / "graph_features" / design / "instance_placement_micron"
    ov = (N14 / "routability_features" / design / "congestion" / "congestion_global_routing"
          / ("utilization_based" if kind == "util" else "overflow_based"))
    cache = N14 / "cache" / design
    cache.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in pl_dir.glob("*.npz"))

    def placement(name, device):
        f = cache / f"{name}.place.npy"
        if not f.exists():  # the raw file is a 300k-entry dict: convert once
            d = np.load(pl_dir / f"{name}.npz", allow_pickle=True)["data"].item()
            # N14 lists filler cells (2/3 of all entries); N28, the source data, has none
            np.save(f, np.array([v for k, v in d.items() if not k.startswith("FILLER")], dtype=np.float32))
        a = np.load(f)
        if unit == "gcell":
            a = np.floor(a / N14_UM_PER_GCELL)
        return _to_batch(a, device)

    def label(name):
        f = cache / f"{name}.{kind}.npy"
        if not f.exists():
            np.save(f, _label(*(np.load(ov / f"congestion_GR_{d}_{kind}" / f"{name}.npz")["data"]
                                for d in ("horizontal", "vertical"))))
        return np.load(f)

    return names, placement, label


def get(name, unit="gcell"):
    return n28(unit) if name == "n28" else n14(name, unit)

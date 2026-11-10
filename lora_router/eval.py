"""Score a CircuitFormer model on a dataset.

    python -W ignore -m lora_router.eval --data n28 --unit gcell
    python -W ignore -m lora_router.eval --data Vortex-small --unit micron

Metrics follow CircuitFormer's metrics.py (per-layout correlation on the
[0,1]-clamped maps, averaged), plus SSIM and NRMSE.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torchmetrics.functional import structural_similarity_index_measure as ssim

from . import datasets
from .build import CF_ROOT, build_circuitformer

COLS = ("pearson", "spearman", "kendall", "ssim", "nrmse")


@torch.no_grad()
def evaluate(model, names, placement, label, device="cuda", csv=None, amp=True):
    model.eval()
    rows = []
    for name in names:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(placement(name, device))
        pred = pred.float().clamp(0, 1)  # [1,1,256,256]
        lab = torch.as_tensor(label(name), device=device)[None, None]
        p, l = pred.flatten().cpu().numpy(), lab.flatten().cpu().numpy()
        rows.append((stats.pearsonr(p, l)[0], stats.spearmanr(p, l)[0], stats.kendalltau(p, l)[0],
                     ssim(pred, lab, data_range=1.0).item(), float(np.sqrt(np.mean((p - l) ** 2)))))
    if csv:
        Path(csv).parent.mkdir(parents=True, exist_ok=True)
        with open(csv, "w") as f:
            f.write("name," + ",".join(COLS) + "\n")
            f.writelines(n + "," + ",".join(f"{v:.5f}" for v in r) + "\n" for n, r in zip(names, rows))
    return dict(zip(COLS, np.nanmean(np.array(rows), axis=0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="n28")
    ap.add_argument("--unit", choices=["gcell", "micron"], default="gcell")
    ap.add_argument("--ckpt", default=str(CF_ROOT / "ckpts" / "circuitformer.ckpt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = build_circuitformer(args.ckpt).cuda()
    names, placement, label = datasets.get(args.data, args.unit)
    names = names[: args.limit or None]
    res = evaluate(model, names, placement, label, csv=f"results/{args.data}_{args.unit}_zeroshot.csv")
    print(f"{args.data} {args.unit} n={len(names)}  " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()

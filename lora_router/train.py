"""Adapt CircuitFormer with one of three methods and report the cost.

    python -W ignore -m lora_router.train --mode lora --steps 100

modes:  lora  frozen base + LoRA on attention (add --decoder-head to also train the 1x1 output conv)
        head  frozen base, only the output conv trains
        full  everything trains (the baseline LoRA is compared against)

Trains on --n-train layouts of --data and scores on a fixed held-out set.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import random

from . import datasets
from .build import CF_ROOT, build_circuitformer
from .eval import evaluate
from .lora import apply_lora, lora_state_dict

HEAD = "decoder.segmentation_head"


def make_batch(samples, device):
    cols = [torch.cat([s[0][i] for s in samples]).to(device) for i in range(4)]
    offsets = torch.tensor([s[0][0].shape[0] for s in samples]).cumsum(0).int().to(device)
    labels = torch.stack([s[1] for s in samples])[:, None].to(device)
    return cols + [offsets], labels


def setup(model, mode, rank, decoder_head):
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        model.train()
    else:
        apply_lora(model, rank=rank if mode == "lora" else 0, alpha=2 * rank,
                   train_also=(HEAD,) if (mode == "head" or decoder_head) else ())
        model.eval()  # frozen base: keep BatchNorm statistics fixed
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lora", "head", "full"], required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--decoder-head", action="store_true")
    ap.add_argument("--data", default="Vortex-small")
    ap.add_argument("--unit", default="gcell")
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-amp", dest="amp", action="store_false")  # half precision on by default, as in their config
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="results/adapters")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    model = build_circuitformer(str(CF_ROOT / "ckpts" / "circuitformer.ckpt")).to(dev)
    trainable, total = setup(model, args.mode, args.rank, args.decoder_head)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)

    names, placement, label = datasets.get(args.data, args.unit)
    random.Random(0).shuffle(names)  # fixed split: held-out layouts are the same for every mode and seed
    test, pool = names[: args.n_test], names[args.n_test:]
    random.Random(args.seed).shuffle(pool)
    data = [(placement(n, "cpu"), torch.as_tensor(label(n))) for n in pool[: args.n_train]]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], None
    for step in range(args.steps):
        if step == 5:  # skip warm-up steps in the timing
            torch.cuda.synchronize(); t0 = time.time()
        idx = torch.randint(len(data), (args.batch,)).tolist()
        batch, labels = make_batch([data[i] for i in idx], dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            pred = model(batch)
        loss = 128 * torch.mean((pred.float() - labels) ** 2)  # their loss, without label-density weights
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(loss.item())
    torch.cuda.synchronize()
    sec = (time.time() - t0) / (args.steps - 5)
    peak = round(torch.cuda.max_memory_allocated() / 2**20)
    scores = evaluate(model, test, placement, label)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.mode + (f"_r{args.rank}" if args.mode == "lora" else "") + ("_head" if args.decoder_head else "")
    tag += f"_n{args.n_train}_s{args.seed}"
    f = out / f"{tag}.pt"
    torch.save(lora_state_dict(model), f)
    res = dict(mode=tag, trainable=trainable, total=total, pct=round(100 * trainable / total, 4),
               n_train=args.n_train, steps=args.steps, lr=args.lr, seed=args.seed,
               sec_per_step=round(sec, 4), train_min=round(sec * args.steps / 60, 2), peak_gpu_mb=peak,
               **{k: round(float(v), 4) for k, v in scores.items()},
               saved_mb=round(f.stat().st_size / 2**20, 3),
               loss_first10=round(sum(losses[:10]) / 10, 4), loss_last10=round(sum(losses[-10:]) / 10, 4))
    print(json.dumps(res))
    with open(f"results/{args.data}_runs.jsonl", "a") as fh:
        fh.write(json.dumps(res) + chr(10))


if __name__ == "__main__":
    main()

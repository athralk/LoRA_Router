"""Base Qwen vs Qwen + trained LoRA: each proposes knobs, OpenROAD routes, compare results.

Both policies sample n answers from the same prompt (temperature 1). Flow runs alternate
base/LoRA so slow drift in machine speed hits both equally.

    python -m flows.compare --design mac16 --n 6
"""
import argparse
import json
import statistics as st

import torch

from .rl import OUT, get_base, load_policy, parse, prompt_ids, reward, sample
from .run_flow import run

KEYS = ("ok", "route_s", "wns_ss", "wns_tt", "die_area", "drc", "wirelength", "power")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", default="mac16")
    p.add_argument("--adapter", default=None, help="default: results/rl/adapter_<design>.pt")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    base = get_base(a.design)
    tok, model = load_policy()
    ids = prompt_ids(tok, a.design, base)
    torch.manual_seed(a.seed)
    _, base_txt = sample(tok, model, ids, a.n)  # LoRA B is zero: this is the untouched model
    sd = torch.load(a.adapter or OUT / f"adapter_{a.design}.pt", weights_only=True)
    missing = set(sd) - set(model.state_dict())
    assert not missing, f"adapter keys not in model: {list(missing)[:3]}"
    model.load_state_dict(sd, strict=False)
    torch.manual_seed(a.seed)
    _, lora_txt = sample(tok, model, ids, a.n)

    res = {"base": [], "lora": []}
    for i in range(a.n):
        for who, txt in (("base", base_txt[i]), ("lora", lora_txt[i])):
            kn = parse(txt)
            rec = run(a.design, kn, note=f"compare {who} {i}") if kn else None
            row = {"mode": f"compare_{who}", "design": a.design, "i": i, "knobs": kn,
                   "reward": reward(rec, base), **({k: rec.get(k) for k in KEYS} if rec else {"ok": False})}
            res[who].append(row)
            with (OUT / "rl_runs.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(who, i, f"{row['reward']:+.3f}", row.get("route_s"), kn, flush=True)

    print(f"\ndefault flow: routing {base['route_s']:.1f} s")
    for who, rows in res.items():
        good = [r for r in rows if r["ok"] and r["reward"] > -0.5]  # passed DRC and timing guard
        same = [r for r in good if r["die_area"] <= base["die_area"] * 1.001]
        line = (f"{who}: valid {len(good)}/{len(rows)}, mean reward {st.mean(r['reward'] for r in rows):+.3f}")
        if good:
            line += f", median routing {st.median(r['route_s'] for r in good):.1f} s"
        if same:
            line += f", best same-area routing {min(r['route_s'] for r in same):.1f} s"
        print(line)


if __name__ == "__main__":
    main()

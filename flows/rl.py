"""REINFORCE on a LoRA adapter: Qwen proposes OpenROAD knobs, the flow returns the reward.

Only the LoRA weights train; the base model is frozen. Step 0 samples come from the
untouched model (LoRA B is zero at init), so they are the zero-shot baseline.

    python -m flows.rl --design mac16 --steps 6 --group 4
    python -m flows.rl --random --design mac16 --runs 24      # random-search control
"""
import argparse
import json
import math
import random
import re
import time

import torch
from torch.nn.utils import parametrize

from lora_router.lora import apply_lora, lora_state_dict
from .run_flow import KNOBS, REPO, run

MODEL = REPO / "third_party" / "models" / "Qwen2.5-1.5B-Instruct"
OUT = REPO / "results" / "rl"
BASE_FILE = OUT / "baselines.json"

SYSTEM = ("You are an expert physical-design engineer tuning the OpenROAD flow (OpenLane 2, "
          "sky130A, sky130_fd_sc_hd). Goal: shortest global + detailed routing runtime with zero "
          "DRC errors, without worse timing and without a larger die. Answer with one JSON object only.")


def knob_text():
    return "\n".join(f"- {k}: {t.__name__} in [{lo}, {hi}]" for k, (lo, hi, t) in KNOBS.items())


def user_msg(design, base):
    return (f"Design: {design}. Standard cells after placement: {base.get('cells', 'unknown')}. "
            f"Clock period: {base['clock']} ns.\n"
            f"Default flow: routing {base['route_s']:.0f} s, die area {base['die_area']:.0f} um^2, "
            f"setup WNS {base['wns_ss']:.2f} ns (slow corner), {base['wns_tt']:.2f} ns (typical).\n"
            f"Knobs (defaults: FP_CORE_UTIL 50, PL_TARGET_DENSITY_PCT about 62, GRT_ADJUSTMENT 0.3, "
            f"GRT_OVERFLOW_ITERS 50, DRT_OPT_ITERS 64, paddings 0):\n{knob_text()}\n"
            "The defaults are already measured. Propose a different setting that you expect to route "
            "faster. Return JSON with values for these knobs.")


def parse(text):
    """Every `"KNOB": number` pair anywhere in the answer (tolerates nesting and code fences)."""
    d = {m[1]: float(m[2]) for m in re.finditer(r'"([A-Z_]+)"\s*:\s*"?(-?\d+(?:\.\d+)?)', text)
         if m[1] in KNOBS}
    return d or None


def reward(rec, base, area_weight=1.0):
    """log routing speed-up, minus log die-area growth; failures and timing regressions are negative."""
    if rec is None:
        return -1.0  # unparseable answer
    if not rec.get("ok"):
        return -1.0  # flow failed or DRC errors left
    if rec["wns_tt"] < 0 or rec["wns_ss"] < base["wns_ss"] - 0.25:
        return -0.5  # bought speed with timing
    r = math.log(base["route_s"] / rec["route_s"])
    return r - area_weight * max(0.0, math.log(rec["die_area"] / base["die_area"]))


def load_policy(rank=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    tr, tot = apply_lora(model, rank, 16.0, linear_targets=("q_proj", "v_proj"),
                         adapter_dtype=torch.float32)
    print(f"Qwen2.5-1.5B + LoRA r{rank}: {tr:,} trainable of {tot:,} ({100 * tr / tot:.3f}%)")
    return tok, model


def prompt_ids(tok, design, base):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(design, base)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return (ids["input_ids"] if not torch.is_tensor(ids) else ids).cuda()


@torch.no_grad()
def sample(tok, model, ids, n, temperature=1.0, max_new=160):
    model.eval()
    with parametrize.cached():
        out = model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                             do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    comp = out[:, ids.shape[1]:]
    return comp, tok.batch_decode(comp, skip_special_tokens=True)


def seq_logprob(model, ids, comp, pad_id, eos_id):
    """Mean token log-prob of one completion [1, T] (tokens after the first EOS are masked).

    Only the completion's logits are materialised: the full 152k-vocab logits of the
    prompt would not fit an 8 GB GPU.
    """
    full = torch.cat([ids, comp], 1)
    logits = model(full, logits_to_keep=comp.shape[1] + 1).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
    end = (comp == eos_id) | (comp == pad_id)
    done = torch.cumsum(end, 1)
    mask = ((done == 0) | ((done == 1) & end)).float()
    return (lp * mask).sum(1) / mask.sum(1).clamp(min=1)


def reinforce_backward(model, ids, comp, adv, pad_id, eos_id):
    """Policy-gradient loss, one completion at a time to bound memory."""
    model.train()
    total = 0.0
    for i in range(comp.shape[0]):
        loss = -adv[i] * seq_logprob(model, ids, comp[i:i + 1], pad_id, eos_id).squeeze() / comp.shape[0]
        loss.backward()
        total += loss.item()
    return total


def get_base(design):
    bases = json.loads(BASE_FILE.read_text()) if BASE_FILE.exists() else {}
    if design not in bases:
        rec = run(design, {}, note="baseline")
        cfg = json.loads((REPO / "flows" / "designs" / design / "config.json").read_text())
        rec["clock"] = cfg["CLOCK_PERIOD"]
        bases[design] = rec
        OUT.mkdir(parents=True, exist_ok=True)
        BASE_FILE.write_text(json.dumps(bases, indent=1))
    return bases[design]


def log(rec):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rl_runs.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def train(a):
    base = get_base(a.design)
    tok, model = load_policy(a.rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    ids = prompt_ids(tok, a.design, base)
    for step in range(a.steps):
        comp, texts = sample(tok, model, ids, a.group, a.temperature)
        rs = []
        for i, t in enumerate(texts):
            kn = parse(t)
            rec = run(a.design, kn, note=f"rl step{step} s{i}") if kn else None
            r = reward(rec, base, a.area_weight)
            rs.append(r)
            log({"mode": "rl", "design": a.design, "step": step, "i": i, "text": t[:400], "knobs": kn,
                 "reward": r, **({k: rec.get(k) for k in ("ok", "route_s", "drt_s", "wns_ss", "wns_tt",
                                                          "die_area", "drc", "wirelength", "power")}
                                 if rec else {})})
            print(f"step {step} sample {i}: reward {r:+.3f} knobs {kn}", flush=True)
        r = torch.tensor(rs, device="cuda")
        if r.std() < 1e-6:
            print(f"step {step}: all rewards equal, no update")
            continue
        adv = (r - r.mean()) / (r.std() + 1e-6)
        opt.zero_grad()
        loss = reinforce_backward(model, ids, comp, adv, tok.pad_token_id, tok.eos_token_id)
        opt.step()
        print(f"step {step}: mean reward {r.mean():+.3f}, best {r.max():+.3f}, loss {loss:.4f}")
        torch.save(lora_state_dict(model), OUT / f"adapter_{a.design}.pt")


def random_search(a):
    base = get_base(a.design)
    rng = random.Random(a.seed)
    for i in range(a.runs):
        kn = {k: (rng.randint(lo, hi) if t is int else round(rng.uniform(lo, hi), 2))
              for k, (lo, hi, t) in KNOBS.items()}
        rec = run(a.design, kn, note=f"random {i}")
        r = reward(rec, base, a.area_weight)
        log({"mode": "random", "design": a.design, "i": i, "knobs": kn, "reward": r,
             **{k: rec.get(k) for k in ("ok", "route_s", "drt_s", "wns_ss", "wns_tt", "die_area", "drc",
                                        "wirelength", "power")}})
        print(f"random {i}: reward {r:+.3f} knobs {kn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--design", default="mac16")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--area-weight", type=float, default=1.0)
    p.add_argument("--random", action="store_true")
    p.add_argument("--runs", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry", action="store_true", help="load Qwen+LoRA and sample once, no flow runs")
    a = p.parse_args()
    if a.dry:
        tok, model = load_policy(a.rank)
        base = {"route_s": 128.2, "die_area": 53538.1, "wns_ss": -7.21, "wns_tt": 0.23, "clock": 8.0}
        ids = prompt_ids(tok, a.design, base)
        t0 = time.time()
        comp, texts = sample(tok, model, ids, a.group)
        print(f"sampled {a.group} in {time.time() - t0:.1f}s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        for t in texts:
            print("---", parse(t), "|", t[:200].replace("\n", " "))
        torch.cuda.reset_peak_memory_stats()
        reinforce_backward(model, ids, comp, torch.ones(a.group, device="cuda"),
                           tok.pad_token_id, tok.eos_token_id)
        print(f"backward ok, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    elif a.random:
        random_search(a)
    else:
        train(a)

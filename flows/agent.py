"""Command-level timing-closure agent: Qwen2.5-1.5B + LoRA, trained by RL in a live OpenROAD session.

Training (branch and compare): at each state, sample G actions, apply each to its own copy
of the state (parallel sessions from a checkpoint), score each by its step reward, update
the LoRA with group-normalised advantages, continue the episode from the best branch.
The step reward already prices tool time, so the policy learns the most progress per
second (fewer, cheaper iterations). This is a one-step (myopic) policy gradient.

    python -m flows.agent prep mac16 mac12 fir4 alu32 mac20 simd4
    python -m flows.agent train --designs mac16 mac12 fir4 alu32 --episodes 24
    python -m flows.agent eval --designs mac16 mac12 fir4 alu32 mac20 simd4 --policies default max_effort random qwen lora
"""
import argparse
import json
import random
import statistics as stats
import time

try:  # transformers imports sklearn lazily; loading it after torch/CUDA can crash natively on Windows
    import sklearn.metrics  # noqa: F401
except ImportError:
    pass
import torch

from lora_router.lora import apply_lora_hooks, lora_state_dict
from . import closure_env as E
from .rl import MODEL

LOG = E.OUT / "agent_runs.jsonl"
ADAPTER = E.OUT / "adapter_closure.pt"


def log(row):
    E.OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


class Policy:
    def __init__(self, rank=16, adapter=None, train=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
        tr, tot = apply_lora_hooks(self.model, rank, 2.0 * rank)
        self.model.lora_adapters.cuda()
        print(f"Qwen2.5-1.5B + LoRA r{rank} (attention + MLP): {tr:,} trainable of {tot:,} ({100 * tr / tot:.2f}%)")
        if adapter:
            missing = self.model.load_state_dict(torch.load(adapter, weights_only=True), strict=False)
            assert not [k for k in missing.unexpected_keys], missing.unexpected_keys[:3]
        if train:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-4)

    def ids(self, text):
        msgs = [{"role": "system", "content": E.SYSTEM}, {"role": "user", "content": text}]
        x = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        return (x["input_ids"] if not torch.is_tensor(x) else x).cuda()

    @torch.no_grad()
    def sample(self, ids, n, temperature=1.0, max_new=80):
        self.model.eval()
        out = self.model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                                  do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                  max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id, use_cache=True)
        comp = out[:, ids.shape[1]:]
        return comp, self.tok.batch_decode(comp, skip_special_tokens=True)

    def update(self, ids, comp, adv):
        self.model.train()
        self.opt.zero_grad()
        eos, pad = self.tok.eos_token_id, self.tok.pad_token_id
        total = 0.0
        for i in range(comp.shape[0]):
            c = comp[i:i + 1]
            full = torch.cat([ids, c], 1)
            logits = self.model(full, logits_to_keep=c.shape[1] + 1, use_cache=False).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, c.unsqueeze(-1)).squeeze(-1)
            end = (c == eos) | (c == pad)
            mask = ((torch.cumsum(end, 1) == 0) | ((torch.cumsum(end, 1) == 1) & end)).float()
            loss = -adv[i] * (lp * mask).sum() / mask.sum().clamp(min=1) / comp.shape[0]
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        return total


def forced(pol, comp, action):
    """Replace the last sampled completion with the tokens of a chosen action (guided exploration)."""
    t = pol.tok.encode(json.dumps(action), add_special_tokens=False) + [pol.tok.eos_token_id]
    width = max(comp.shape[1], len(t))
    pad = pol.tok.pad_token_id
    if width > comp.shape[1]:
        comp = torch.cat([comp, torch.full((comp.shape[0], width - comp.shape[1]), pad, device=comp.device)], 1)
    comp = comp.clone()
    comp[-1] = torch.tensor(t + [pad] * (width - len(t)), device=comp.device)
    return comp


def explore_action(ep, k, rng):
    """After the first move, half the explorations try stopping; the rest are random menu actions."""
    if k >= 1 and rng.random() < 0.5:
        return {"op": "finish"}
    return E.random_action(ep, rng)


def train(a):
    pol = Policy(a.rank, adapter=a.resume, train=True)
    preps = {d: E.load_prep(d) for d in a.designs}
    rng = random.Random(a.seed + a.start)
    for ep_i in range(a.start, a.start + a.episodes):
        d = a.designs[ep_i % len(a.designs)]
        ep = E.Episode(d, preps[d])
        t0, ret = time.time(), 0.0
        for k in range(E.MAX_ACTIONS):
            ids = pol.ids(E.prompt(ep))
            comp, texts = pol.sample(ids, a.group, a.temperature)
            acts = [E.parse_action(t) or {"op": "invalid", "raw": t[:60]} for t in texts]
            if a.explore:  # one of the G branches is a forced exploration action
                acts[-1] = explore_action(ep, k, rng)
                comp = forced(pol, comp, acts[-1])
            res = E.run_branches(ep, acts, f"tr{ep_i}_{k}")
            rs = torch.tensor([r for r, _, _ in res], device="cuda")
            if rs.std() > 1e-6:
                loss = pol.update(ids, comp, (rs - rs.mean()) / (rs.std() + 1e-6))
            else:
                loss = 0.0
            best = int(rs.argmax())
            for j, (r, info, br) in enumerate(res):
                log({"mode": "train", "episode": ep_i, "design": d, "k": k, "j": j, "action": acts[j],
                     "forced": bool(a.explore and j == len(acts) - 1),
                     "reward": round(r, 4), "valid": info.get("valid"), "P": info.get("P"),
                     "tool_s": info.get("tool_s"), "success": info.get("success"), "best": j == best,
                     "error": (info.get("error") or "")[:120]})
                if j != best:
                    br.close()
            ep.close()
            ep = res[best][2]
            ret += res[best][0]
            print(f"ep {ep_i} {d} step {k}: rewards {[round(float(x), 2) for x in rs]} best {acts[best]} "
                  f"P {ep.P:.2f} (target {preps[d]['target']}) loss {loss:+.3f}", flush=True)
            if res[best][1].get("done"):
                break
        log({"mode": "train_episode", "episode": ep_i, "design": d, "return": round(ret, 3), "P": ep.P,
             "success": ep.success() if ep.routed else None, "tool_s": round(ep.tool_s, 1),
             "actions": ep.k, "wall_s": round(time.time() - t0, 1)})
        ep.close()
        torch.save(lora_state_dict(pol.model), ADAPTER)


def episode(design, prep, chooser, label, seed):
    """Run one sequential episode with chooser(ep) -> action; returns the result row."""
    ep = E.Episode(design, prep)
    ret, info, acts, decide = 0.0, {}, [], []
    t_start = time.time()
    while True:
        t = time.time()
        a = chooser(ep)  # the model's (or script's) decision time, kept apart from OpenROAD's tool time
        decide.append(round(time.time() - t, 2))
        acts.append(a)
        r, info = ep.step(a)
        ret += r
        if info.get("done"):
            break
    row = {"mode": "eval", "policy": label, "design": design, "seed": seed, "success": info.get("success"),
           "P": round(ep.P, 3), "target": prep["target"], "hold": ep.m["hold"], "area": ep.m["area"],
           "power": info.get("power"), "tool_s": round(ep.tool_s, 1), "decide_s": round(sum(decide), 1),
           "decide_per_action": decide, "wall_s": round(time.time() - t_start, 1), "actions": ep.k,
           "return": round(ret, 3), "sequence": [x.get("op") for x in acts]}
    ep.close()
    log(row)
    print(f"{label:10s} {design:6s} seed {seed}: success {row['success']} P {row['P']:.2f}/{prep['target']} "
          f"actions {row['actions']} tool {row['tool_s']} s decide {row['decide_s']} s wall {row['wall_s']} s "
          f"{row['sequence']}", flush=True)
    return row


def evaluate(a):
    preps = {d: E.load_prep(d) for d in a.designs}
    pols = {}
    rows = []
    for label in a.policies:
        if label in ("qwen", "lora"):
            if pols:
                del pols["model"]
                torch.cuda.empty_cache()
            pols["model"] = Policy(a.rank, adapter=ADAPTER if label == "lora" else None)
        for d in a.designs:
            seeds = [0] if label in ("default", "max_effort") else list(range(a.seeds))
            for sd in seeds:
                rng = random.Random(sd)
                torch.manual_seed(sd)
                if label == "default":
                    script = iter(E.DEFAULT + [{"op": "finish"}])
                    choose = lambda ep, s=script: next(s)
                elif label == "max_effort":
                    script = iter(E.MAX_EFFORT + [{"op": "finish"}])
                    choose = lambda ep, s=script: next(s)
                elif label == "random":
                    choose = lambda ep, r=rng: E.random_action(ep, r)
                else:
                    pol = pols["model"]

                    def choose(ep, pol=pol):
                        _, texts = pol.sample(pol.ids(E.prompt(ep)), 1, a.temperature)
                        return E.parse_action(texts[0]) or {"op": "invalid", "raw": texts[0][:60]}
                rows.append(episode(d, preps[d], choose, label, sd))
    print("\npolicy      success  mean actions  mean tool s  mean decide s  mean wall s  mean P/target")
    for label in a.policies:
        rs = [r for r in rows if r["policy"] == label]
        print(f"{label:10s} {sum(bool(r['success']) for r in rs):3d}/{len(rs):<3d} "
              f"{stats.mean(r['actions'] for r in rs):9.1f} {stats.mean(r['tool_s'] for r in rs):11.1f} "
              f"{stats.mean(r['decide_s'] for r in rs):13.1f} {stats.mean(r['wall_s'] for r in rs):11.1f} "
              f"{stats.mean(r['P'] / r['target'] for r in rs):12.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prep", "train", "eval"])
    p.add_argument("names", nargs="*")
    p.add_argument("--designs", nargs="+", default=["mac16", "mac12", "fir4", "alu32"])
    p.add_argument("--policies", nargs="+", default=["default", "max_effort", "random", "qwen", "lora"])
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--start", type=int, default=0, help="first episode index (when resuming)")
    p.add_argument("--explore", action="store_true", help="force one exploration branch per step")
    a = p.parse_args()
    if a.cmd == "prep":
        for d in a.names:
            pr = E.prep(d, force=True)
            print(d, {k: pr[k] for k in ("P_start", "target", "default", "max_effort")}, flush=True)
    elif a.cmd == "train":
        train(a)
    else:
        evaluate(a)

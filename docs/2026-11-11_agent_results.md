# Command-level timing-closure agent: first results (2026-11-11)

Qwen2.5-1.5B-Instruct + LoRA (r16, attention + MLP, 18.5M trainable, 1.18%), trained by RL inside a live OpenROAD session (OpenLane 2.3.10 image, sky130A). Plan: [plan.md](plan.md). Earlier results today (1.69× routing time, +11.3% fmax from settings selection): [2026-11-11_openroad_rl.md](2026-11-11_openroad_rl.md).

**Headline:** the free-text RL agent does **not** beat OpenROAD's scripted flows. It met the timing target in 2 of 12 eval runs, against 5 of 6 for the max-effort script. 81% of its answers were invalid. This is a negative result with clear causes, listed below.

## Setup

- **Task:** take a placed design with its clock tree built (OpenLane's CTS output) and make it run as fast as possible: shortest clock period P at the slow corner (ss, 100 °C, 1.60 V), with hold ≥ 0 and cell area ≤ 1.25× start. At most 6 actions per episode.
- **Target per design:** the best P reached by either scripted flow (`default` = one `repair_timing`; `max_effort` = stronger `repair_timing` + `route_repair`), rounded up to 0.01 ns. Success = matching OpenROAD's best script.
- **What Qwen sees each step:** a ~1.2k-token text prompt built from the live database: targets, design summary (cell-type counts, high-fanout nets), current P/WNS/TNS/hold/area, past actions and their effect, the 5 worst paths from OpenSTA `report_checks` (per stage: instance, cell, fanout, load, cell and wire delay, x/y, flags WEAK/SLOWCELL/FANOUT/LONGWIRE/SPREAD), and the action menu.
- **What Qwen outputs:** one JSON action, e.g. `{"op": "resize", "cell": "_3931_", "size": "up"}`. Python validates it and turns it into OpenROAD Tcl (`replace_cell`, `repair_timing …`, `detailed_placement`, `global_route`).
- **Menu:** `repair_timing`, `repair_design`, `resize`, `upsize_path`, `pull_path`, `route_repair`, `finish`.
- **Step reward:** $r = (P_{prev} - P_{new})/\text{gap} - 0.003\,t_{tool}$; +1 at the end if successful; −0.1 invalid; −0.2 OpenROAD error. If the budget runs out, the design is globally routed and judged.
- **Training:** branch and compare. At each state, sample G = 4 actions, run each on its own copy of the state (parallel sessions from a checkpoint), use group-normalised advantages for a REINFORCE update, and continue from the best branch. AdamW lr 1e-4, grad clip 1.0.

## Designs (sky130, 8 ns clock request, 50% utilisation)

| Design | Role | P start (ns) | Target (ns) | Default P (tool s) | Max-effort P (tool s) |
|---|---|---|---|---|---|
| mac16 | train | 16.69 | 14.43 | 14.43 ✓ (108) | 14.44 ✗ (134) |
| mac12 | train | 14.83 | 13.01 | 13.19 ✗ (51) | 13.01 ✓ (94) |
| fir4 | train | 14.36 | 12.64 | 12.70 ✗ (15) | 12.63 ✓ (26) |
| alu32 | train | 12.99 | 11.36 | 11.55 ✗ (17) | 11.36 ✓ (32) |
| mac20 | held out | 17.48 | 14.87 | 14.97 ✗ (64) | 14.86 ✓ (95) |
| simd4 | held out | 14.66 | 13.24 | 13.35 ✗ (16) | 13.24 ✓ (40) |

## Training

**Run 1, episodes 0–9 (plain sampling).**
- 9 of 10 episodes met the target, helped by best-of-4 at every step.
- By episode 8 the policy had collapsed to `repair_timing` only. It never chose `finish`, so every episode used all 6 actions. All 4 samples were identical, so the advantages were zero and there was nothing to learn from.

**Run 2, episodes 10–21 (forced exploration, resumed from the episode-10 adapter).**
- One of the 4 branches at each step is forced: `finish` with probability 0.5 from step 1 onward, otherwise a random menu action.
- 11 of 12 episodes met the target. Mean actions fell from 6 to 4.4.
- Forced branches won 7 times as `finish`, plus `upsize_path` ×2, `resize` and `route_repair`.
- Qwen never wrote `finish` itself (0 times in ~250 sampled answers).
- **Best single result:** episode 19, alu32. A forced `route_repair` reached P 10.89 ns after the final route, against an 11.36 target (4.1% better than the best script), in 35 s of tool time against 32 s. This came from exploration, not from Qwen.
- **Late in training, the format drifted.** Updates on low-probability forced tokens gave losses of ±10–12, and the model began writing `"op": "timing repair"`, which is not a menu name.

## Eval (1 sample per step, temperature 1.0, 2 seeds per design)

| Policy | Met target | Mean actions | Mean tool s | Mean decide s | Mean P / target | Invalid answers |
|---|---|---|---|---|---|---|
| default script | 1 / 6 | 1 | 45 | 0 | ≈1.01 | 0 |
| max-effort script | **5 / 6** | 2 | 70 | 0 | ≈1.00 | 0 |
| random | 0 / 12 | 2.5 | 2 | 0 | 1.203 | 0 |
| base Qwen | 0 / 12 | 6 | 42 | not logged | 1.068 | 29% |
| **Qwen + LoRA** | **2 / 12** | 6 | 45 | **53** | **1.020** | **81%** |

The scripts are deterministic, so they ran once per design.

- The LoRA's first move was nearly always a sound `repair_timing`, so it ended closer to the target than base Qwen (1.020 vs 1.068). After that it wrote invalid answers until the budget ran out.
- Both successes were on mac16, the design where a single `repair_timing` meets the target.
- Held-out mac20 and simd4 behaved like the training designs: they matched the default script and missed the target.
- **Decision time is about 9 s per action** (80 generated tokens on the laptop GPU). This doubles the wall time: 100 s mean, against about 45 s for the default script.

## Why it failed

1. **Too few decisions.** About 250 sampled decisions across the whole run, each costing 1–2 min of OpenROAD time. That is far too little for RL on a 1.5B model.
2. **Every episode starts from the same state per design**, and `repair_timing` always helps first. The model can earn reward without reading the prompt.
3. **Free-text output.** Winning moves the model rarely writes (`finish`) need large, clipped updates on whole token strings. That caused the format drift.
4. **The step reward is myopic.** Moves that pay off later, such as `route_repair`, are under-valued.
5. **P before routing is about 0.2–0.3 ns optimistic**, so "met" is not really met until after the final route. Training episode 21 on mac12 missed by 0.02 ns this way.

## What is still worth keeping

- The live-session environment: <1 ms per command round trip, checkpoints, parallel branches, results matching OpenLane.
- The observation builder (worst paths with cell positions and flags) and the action-to-Tcl layer.
- About 600 logged (state, action, outcome) branches in `results/agent/agent_runs.jsonl`.
- Decision time and wall time are now logged per eval episode.

## Next (not started)

- **A. Choose from a list.** Score a list of 10–30 legal candidate actions with the LoRA model and sample from a softmax over them. No invalid answers; decision time about 1 s.
- **B. Expert iteration.** Offline search over the candidates, 2–3 steps deep, scored by final outcome. Supervised fine-tuning on the best moves, then RL only to polish.
- **C. Varied starting states:** different clock targets, utilisations, synthesis strategies and partial-flow starts, plus more small designs.
- **D. An honest state.** P after a quick global route (~0.3 s) at every step, a `MET: yes/no (margin)` line in the prompt, and a time-to-closure objective.
- **E. Episode-level scoring** from the search tree, not one step at a time.
- **F. Ablation.** Blank or shuffle the worst paths and check whether the choice changes.
- **Extra baseline:** the best fixed action sequence found by search. The adapter must beat it on held-out designs before we can claim it understands the design.
- **Plan B, if A–F fall short:** a predictor instead of an agent. Qwen + LoRA with a regression head predicts each candidate's resulting P and tool time; only the top candidate is run in OpenROAD. It is supervised learning on the logged branches, and it reduces OpenROAD runs directly (paper topic 1). The same predictor on a pretrained circuit or timing GNN backbone is paper topic 2.

## Files

- Code: `flows/session.py`, `flows/closure_env.py`, `flows/agent.py` (`--explore`, `--start`, `--resume`; decision and wall time in eval), `flows/pipeline.sh`.
- Logs: `results/agent/pipeline2.log` (run 1), `results/agent/train2.log` (run 2 + eval), `results/agent/agent_runs.jsonl` (every branch and episode), `results/agent/prep_<design>.json`.
- Adapters: `results/agent/adapter_closure_ep10.pt` (before forced exploration), `results/agent/adapter_closure.pt` (final).

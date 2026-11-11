# Hybrid routing: a learned model for critical nets, OpenROAD for the rest

Design decision, 2026-11-11.

## Finding

- **An LLM cannot do the detailed routing, on any hardware today.** A bigger GPU does not change this; memory is not the limit.
  - **Size:** even the small `mac16` test design has ~69,000 µm of wire in tens of thousands of segments; a real chip has millions. Writing them token by token takes hours to days; OpenROAD takes minutes.
  - **Exact rules:** every segment must meet spacing, width, via and short rules to the nanometre. One error is a DRC violation. Larger models make fewer mistakes, not zero.
  - **No published LLM**, at any size, produces DRC-clean detailed routes for a real design.
- **A learned model can do global routing** (which gcells and layers a net uses). With 40–80 GB of GPU memory it could route every net globally. On an 8 GB laptop, a few dozen nets is realistic.
- **What more hardware buys:** larger models (7B–70B), global routing of all nets by the model, and many parallel flow runs so RL gets more signal per hour.

## The split

| Step | Timing-critical nets (~20–50) | All other nets (thousands) |
|---|---|---|
| Global routing (gcells and layers) | **learned model** | OpenROAD |
| Detailed routing (exact wires) | OpenROAD | OpenROAD |

- **Critical** means *timing-critical*: the nets on the slowest paths, which set the maximum frequency. These are not always the most complex nets.
- **Order:** the model routes its nets first; OpenROAD routes the rest around them, so the two don't compete for the same space.
- **Guarantees:** OpenROAD keeps the result legal (DRC-clean wires) and measured (STA, which gives the reward).
- **RL ties them together:** OpenROAD's results (frequency, DRC, routing time) are the reward that trains the model.

## Why this is the paper's contribution

- Hybrid learned + conventional is how current research and commercial tools (Cadence, Synopsys) are built: ML guides or predicts for a conventional router; it does not replace it.
- **New here:** a small LoRA-adapted LLM that routes critical nets itself, trained by RL on real OpenROAD results, and tested on designs it never saw.
- This replaces the first version (2026-11-11 morning), where the model only chose flow settings. That is tuning, not routing.

## Plan

1. **Frequency reward (RL change).** Reward = ln(f / f_default), with f = 1 / (T_target − WNS_ss) at the slow corner. −1 if DRC errors remain or the flow fails; −0.5 if hold fails or routing time exceeds 1.1× default; minus a die-area penalty. New actions: clock target, synthesis strategy, timing-repair slack margins. The best result is re-run at its achieved period to confirm slack ≥ 0.
2. **Route-guide check.** Confirm this OpenROAD build can take externally written route guides for chosen nets, with global routing of the rest around them.
3. **Model routes critical nets.**
   - **State:** after placement, OpenROAD exports the critical paths and their nets (slack, fanout, pin locations) and a congestion grid around each net.
   - **Action:** for each of the top-K critical nets, a gcell path and metal layers, written as route guides.
   - **Model:** Qwen2.5-1.5B, LoRA rank 32 on all attention and MLP projections (~18M parameters); Qwen2.5-3B in 4-bit if needed.
4. **Training.** Imitation first (reproduce OpenROAD's own global routes, so paths are legal), then RL for frequency at 0 DRC and routing time no worse than default.
5. **Evidence the model understands routing.**
   - Gains on 2 held-out designs, beyond the 5–6 training designs.
   - An ablation: shuffled design state should collapse performance.
   - Its routes compared with OpenROAD's on the same nets (length, delay, layers).
   - Default and tuned runs interleaved in repeats: routing time varies by about ±15% between identical runs (128.2 s vs 145.8 s for the same default layout).

## Risks

- The model may only match OpenROAD on wirelength and win only on timing (critical nets on upper metal), if at all.
- Each episode costs a routing + STA run (~2–2.5 min from a saved placement), which limits RL steps.

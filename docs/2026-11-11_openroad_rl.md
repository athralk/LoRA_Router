# 2026-11-11: Qwen + LoRA, trained by RL, tunes OpenROAD

## Headline

- **Routing is 1.69× faster than the OpenROAD default flow on the same netlist, the same 8 ns clock target and the same die area, with 0 DRC errors.**
- The setting was found by a policy-gradient loop: Qwen2.5-1.5B with a LoRA adapter proposes OpenROAD settings, and OpenROAD routes each proposal to score it.
- **Timing cost:** setup slack is 0.04 ns worse at the slow corner and still positive at the typical corner. Wirelength is +9% and power +6.5%.

## Setup

- **Design:** `mac16`, a 16×16 signed multiply-accumulate with a 40-bit accumulator (`flows/designs/mac16`). 3,277 standard-cell instances after placement.
- **Flow:** OpenLane 2.3.10 (OpenROAD `edf00dff`, Yosys) in Docker on WSL2, sky130A, `sky130_fd_sc_hd`. The run ends at post-route STA (`OpenROAD.STAPostPNR`).
- **Machine:** 16 threads for OpenROAD (the default; a thread-count change can't fake the speed-up), RTX 4060 Laptop 8 GB for Qwen.
- **Routing time** = global routing + detailed routing step runtimes, as OpenLane reports them.

## Result

| | OpenROAD default | RL-found setting (3 repeats) |
|---|---|---|
| Routing time | 128.2 s (two runs: 128.3, 128.2) | **75.9 s** (76.1, 76.9, 74.7) |
| Speed-up | 1.00× | **1.69×** |
| Clock target | 8.0 ns | 8.0 ns |
| Setup WNS, slow corner (ss, 100 °C, 1.60 V) | −7.21 ns | −7.25 ns |
| Setup WNS, typical corner (tt, 25 °C, 1.80 V) | +0.23 ns | +0.11 ns |
| DRC errors after routing | 0 | 0 |
| Die area | 53,538 µm² | 53,538 µm² |
| Wirelength | 63,419 µm | 69,206 µm |
| Power | 4.94 mW | 5.26 mW |

- **Setting found:** `PL_TARGET_DENSITY_PCT=68`, `GRT_ADJUSTMENT=0.35`, `GRT_OVERFLOW_ITERS=45`, `GPL_CELL_PADDING=2`, `DPL_CELL_PADDING=2`; `FP_CORE_UTIL=50` and `DRT_OPT_ITERS=64` unchanged.
- **Why it is faster (our reading, not yet verified):** padding spreads the cells, and the higher density target keeps the die the same size. Detailed routing then starts with fewer conflicts.
- **Run-to-run noise:** the layout is identical on every repeat; routing time varies by up to ~15% (66.0 s in the training run, 74.7–76.9 s on the repeats). 1.69× uses the repeat mean.

## Policy and training

- **Policy** $\pi_\theta$: Qwen2.5-1.5B-Instruct, frozen, with LoRA on every `q_proj` and `v_proj`:

  $$W' = W + \frac{\alpha}{r} B A,\qquad r = 8,\ \alpha = 16,\qquad \theta = \{A, B\}$$

  1,089,536 trainable parameters (0.071% of 1.54B). $B = 0$ at the start, so step 0 is untouched Qwen.
- **State** $x$: a text prompt with the design, clock, default-flow results and the allowed range of each knob.
- **Action** $y$: a JSON answer with values for 7 knobs, clamped to their ranges: `FP_CORE_UTIL`, `PL_TARGET_DENSITY_PCT`, `GRT_ADJUSTMENT`, `GRT_OVERFLOW_ITERS`, `DRT_OPT_ITERS`, `GPL_CELL_PADDING`, `DPL_CELL_PADDING`.
- **Reward** $R(y)$, with $T$ = routing time, $A$ = die area, superscript $d$ = default flow:

  $$R(y)=\begin{cases}
  -1 & \text{answer unparseable, flow failed, or DRC} > 0\\
  -0.5 & \mathrm{WNS}_{tt} < 0 \ \text{or}\ \mathrm{WNS}_{ss} < \mathrm{WNS}^{d}_{ss} - 0.25\,\text{ns}\\
  \ln\dfrac{T^{d}}{T} - \lambda\,\max\!\left(0,\ \ln\dfrac{A}{A^{d}}\right) & \text{otherwise},\ \lambda = 1
  \end{cases}$$

- **Advantage** (group-normalised, $G = 4$ samples per step at temperature 1):

  $$\hat{A}_i = \frac{R_i - \operatorname{mean}(R_{1..G})}{\operatorname{std}(R_{1..G}) + 10^{-6}}$$

- **Loss** (REINFORCE, length-normalised log-probability of each answer):

  $$\mathcal{L}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \pi_\theta\!\left(y_{i,t} \mid x,\ y_{i,<t}\right)$$

- **Optimiser:** AdamW, learning rate 2e-4, 5 steps, so 20 OpenROAD runs (~4.5 min each, ~1.5 h in total).
- **Mean reward by step:** −0.53, −0.32, −0.21, +0.14, −0.40.

## What this does and does not show

- **Shown:** OpenROAD's default settings are far from runtime-optimal on this design. A Qwen + LoRA proposal loop found a setting that routes 1.69× faster at the same clock target and die area.
- **Not shown yet:** that the LoRA *learned* this. The reward did not rise steadily over 5 steps, and there is no random-search control yet.
- **Not shown:** a higher maximum frequency. The clock target was fixed at 8 ns and slack was only a guard. Achievable frequency is about the same (65.7 → 65.6 MHz slow corner, 128.7 → 126.7 MHz typical, period ≈ target − WNS).
- **Scope:** one small design, one PDK, one seed. OpenROAD's algorithms are unchanged; only its settings are chosen.

## Base Qwen vs Qwen + LoRA

- 6 sampled answers each, all routed by OpenROAD (`python -m flows.compare --n 6`).
- **Base:** 1/6 valid, mean reward −0.613. **LoRA:** 1/6 valid, mean reward −0.619. **No measurable difference.**
- Weak hint only: both LoRA runs that completed routed in 69–78 s; two of base Qwen's three took 150–160 s.
- **Conclusion:** 5 RL steps (20 runs) did not change the policy measurably. The 1.69× setting came from the search, not from a learned skill.

## Runtime noise

- A fresh default run took 145.8 s routing against 128.2 s earlier, with an identical layout (same wirelength and slack).
- Routing time varies by about ±15% between identical runs. The speed-up is 1.69× against the 128.2 s default and 1.92× against 145.8 s. Future comparisons interleave default and tuned repeats.

## Frequency objective (`--objective freq`)

- **Reward:** $R = \ln(P^{d}/P) - \lambda\max(0, \ln(A/A^{d}))$, with achievable period $P$ = clock target − WNS at the slow corner (ss, 100 °C, 1.60 V). −1 if the flow fails or DRC > 0; −0.5 if hold fails or routing takes more than 1.25× default. Fresh rank-8 adapter.
- **Extra actions (8):** `CLOCK_PERIOD`, `SYNTH_STRATEGY`, `SYNTH_ABC_BUFFERING`, `SYNTH_SIZING`, `PL_RESIZER_SETUP_SLACK_MARGIN`, `GRT_RESIZER_SETUP_SLACK_MARGIN`, `RUN_POST_GRT_RESIZER_TIMING`, `PL_RESIZER_SETUP_MAX_BUFFER_PCT`.
- **Training:** 6 steps × 4 runs. Mean reward by step −0.19, −0.52, −0.64, −0.50, −0.27, −0.30: **no learning trend**. The best answer came from untouched Qwen (step 0).
- **Best result at the same die area**, confirmed by re-timing the same layouts with `flows/sta_at.py`. At 6.57 ns the tool reproduces OpenLane's slack exactly, −7.1015 ns.

| | Default flow | Best frequency run |
|---|---|---|
| Shortest period meeting setup at ss | 15.22 ns | **13.68 ns** |
| Max frequency | 65.7 MHz | **73.1 MHz (+11.3%)** |
| Slack at 13.68 ns | −1.53 ns | +0.009 ns |
| Hold slack (ff, −40 °C, 1.95 V) | +0.123 ns | +0.118 ns |
| Die area | 53,538 µm² | 53,619 µm² |
| Power at 15.22 ns (tt) | 2.22 mW | 2.52 mW (+13.5%) |
| Power at own max frequency (tt) | 2.22 mW | 2.80 mW |

- **Setting:** clock target 6.57 ns, `SYNTH_STRATEGY="AREA 1"`, `PL_RESIZER_SETUP_SLACK_MARGIN=0`, `PL_RESIZER_SETUP_MAX_BUFFER_PCT=30`, `RUN_POST_GRT_RESIZER_TIMING=true`; placement and routing knobs at their defaults.
- **Our reading:** a tighter target makes synthesis and timing repair work harder, and post-route repair fixes paths the earlier steps missed. The cost is more buffers and larger cells, hence more power.
- **Larger-die runs** reached 11.23 ns (89 MHz) at 96,804 µm², over the 70,000 µm² limit.
- **Power note:** `flows/sta_at.py` reports typical-corner power with its own settings. It differs from OpenLane's `power__total` (4.22 vs 4.94 mW for the default at 8 ns), so only its numbers are compared with each other.
- **Not shown:** that this was learned (it was sampled before any update), or that it beats random search with the same budget.

## Open items

- Random-search control with the same number of runs (`python -m flows.rl --random`).
- Held-out design never seen in training.
- Change the objective to maximum frequency, with routing time no worse than default (adds clock-target, synthesis-strategy and repair-margin knobs).

## Reproduce

```bash
python -m flows.run_flow mac16                                  # default flow
python -m flows.run_flow mac16 PL_TARGET_DENSITY_PCT=68 GRT_ADJUSTMENT=0.35 GRT_OVERFLOW_ITERS=45 GPL_CELL_PADDING=2 DPL_CELL_PADDING=2
python -W ignore -m flows.rl --design mac16 --steps 5 --group 4  # RL training
python -W ignore -m flows.compare --design mac16 --n 6            # base vs LoRA
```

- **Raw data:** every flow run is in `results/flow_runs.jsonl`; RL and comparison runs are in `results/rl/rl_runs.jsonl`; the adapter is `results/rl/adapter_mac16.pt`.

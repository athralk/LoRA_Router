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

## Open items

- Base Qwen vs Qwen + LoRA, 6 sampled answers each, all routed (`python -m flows.compare`): running, results pending.
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

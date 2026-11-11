# LoRA Router

Low-rank adaptation (LoRA) for chip physical design, in two parts:

1. **[OpenROAD tuning by RL](#openroad-tuning-by-rl):** Qwen2.5-1.5B with a LoRA adapter proposes OpenROAD flow settings, and later issues step-by-step timing-closure commands; OpenROAD runs each one and the result is the reward.
2. **[Congestion-model adaptation](#congestion-model-adaptation):** a pretrained congestion predictor adapted to a new technology node.

## OpenROAD tuning by RL

**First result: routing 1.69× faster than the OpenROAD default flow** on the same netlist, at the same 8 ns clock target and the same die area, with 0 DRC errors. One design (`mac16`, sky130A), one seed. Full write-up, including the policy and reward equations: [docs/2026-11-11_openroad_rl.md](docs/2026-11-11_openroad_rl.md).

| | OpenROAD default | RL-found setting (3 repeats) |
|---|---|---|
| Routing time (global + detailed) | 128.2 s | **75.9 s** |
| Setup WNS, slow / typical corner | −7.21 / +0.23 ns | −7.25 / +0.11 ns |
| DRC errors | 0 | 0 |
| Die area | 53,538 µm² | 53,538 µm² |
| Wirelength / power | 63,419 µm / 4.94 mW | 69,206 µm / 5.26 mW |

**Frequency objective:** a sampled setting reaches **73.1 MHz vs 65.7 MHz (+11.3%)** at the slow corner, at the same die area, with hold met. That was confirmed by re-timing both layouts at equal clock periods. It costs 13.5% more power at equal frequency.

Not yet shown: that the LoRA learned either result rather than found it by search. Base Qwen and the trained LoRA scored the same, rewards showed no learning trend, and there is no random-search control yet.

### Command-level timing-closure agent

The goal is fewer iterations to timing closure. Qwen + LoRA (r16, attention + MLP) works inside a live OpenROAD session:

- **Each step:** it reads a text summary of the 5 worst timing paths, cell by cell, and replies with one JSON command: resize a cell, repair timing, pull a path together, route + repair, or finish.
- **Training:** RL with branch-and-compare on 4 designs (mac16, mac12, fir4, alu32). mac20 and simd4 are held out.

**Result so far: negative.** In eval it met the timing target in 2 of 12 runs, against 5 of 6 designs for OpenROAD's max-effort script. 81% of its answers were invalid, and each decision took about 9 s. Causes and next steps (candidate scoring, search-labelled fine-tuning, an honest post-route "met" signal): [docs/2026-11-11_agent_results.md](docs/2026-11-11_agent_results.md). Plan: [docs/plan.md](docs/plan.md).

| Path | Purpose |
|---|---|
| `flows/run_flow.py` | Runs OpenLane 2 / OpenROAD in WSL with given knobs; returns routing time, timing, DRC, area, power |
| `flows/rl.py` | REINFORCE on the LoRA weights of Qwen2.5-1.5B; `--objective route\|freq`; `--random` runs the random-search control |
| `flows/compare.py` | Base Qwen vs Qwen + trained LoRA, each answer routed by OpenROAD |
| `flows/sta_at.py` | Re-times a finished layout at other clock periods (setup at ss, hold at ff, power at tt) |
| `flows/session.py` | Live OpenROAD session (loads a design the way OpenLane does; Tcl round trip under 1 ms) |
| `flows/closure_env.py` | Agent environment: worst-path observation, action menu → Tcl, reward, checkpoints, parallel branches |
| `flows/agent.py` | Qwen + LoRA agent: `prep`, `train` (`--explore`, `--resume`, `--start`), `eval` (logs tool, decision and wall time) |
| `flows/pipeline.sh` | Prep → baselines → training → eval for all 6 designs |
| `flows/designs/` | mac16, mac12, mac20 (multiply-accumulate), fir4 (4-tap FIR), alu32, simd4 (4-lane 8-bit SIMD) |
| `docs/` | Dated write-ups and the plan |
| `results/flow_runs.jsonl`, `results/rl/` | Every flow run, RL logs, settings adapters (4.2 MB each) |
| `results/agent/` | Agent logs (`agent_runs.jsonl`: every branch and episode), prep files, agent adapters (74 MB each) |

**Setup (in addition to the Python requirements):**

1. WSL2 with Docker and the image `ghcr.io/efabless/openlane2:2.3.10`.
2. sky130 PDK installed by volare at `~/.volare` inside WSL (the version OpenLane 2.3.10 pins).
3. Qwen weights:
   ```bash
   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct', local_dir='third_party/models/Qwen2.5-1.5B-Instruct')"
   ```

Run directories are written to `~/lora_router_runs/` inside WSL and deleted after each run.

```bash
python -m flows.run_flow mac16                                   # default flow
python -W ignore -m flows.rl --design mac16 --steps 5 --group 4   # RL training (~1.5 h)
python -W ignore -m flows.compare --design mac16 --n 6             # base vs LoRA (~1 h)
python -W ignore -m flows.agent prep mac16                         # agent: baselines and target
python -W ignore -m flows.agent train --designs mac16 mac12 fir4 alu32 --episodes 24 --explore
python -W ignore -m flows.agent eval --designs mac20 simd4 --policies default max_effort lora
```

## Congestion-model adaptation

The base model is [CircuitFormer](https://github.com/hustvl/circuitformer) (NeurIPS 2023, 23.6M parameters), trained by its authors on CircuitNet-N28. This repo adapts it to CircuitNet-N14 by training a rank-8 adapter on the attention layers plus the final output layer: 8,257 parameters, 0.035% of the model.

**Status: early research code.** One design, one seed. See [Limitations](#limitations).

### First result

Vortex-small, CircuitNet-N14, utilization-based congestion. Trained on 16 layouts for 400 steps, scored on 40 held-out layouts, seed 0.

| Method | Trainable parameters | Pearson ↑ | Spearman ↑ | Kendall ↑ | SSIM ↑ | NRMSE ↓ |
|---|---|---|---|---|---|---|
| Original model, unchanged | 0 | 0.485 | 0.535 | – | 0.125 | 0.337 |
| Average of the 16 training labels | – | 0.565 | – | – | – | 0.180 |
| LoRA on attention only | 7,680 | 0.511 | 0.568 | 0.386 | 0.131 | 0.322 |
| Output layer only | 577 | 0.532 | 0.534 | 0.366 | 0.153 | 0.187 |
| **LoRA + output layer** | **8,257** | **0.617** | **0.610** | **0.421** | **0.182** | **0.172** |
| Full fine-tuning | 23,640,401 | 0.654 | 0.662 | 0.476 | 0.251 | 0.186 |

Raw numbers: `results/Vortex-small_runs.jsonl`. Adapter files: `results/adapters/`.

LoRA and full fine-tuning train at the same speed on this model (about 0.44 s per step). The benefit shown here is parameter and storage efficiency, not training time.

### Layout

| Path | Purpose |
|---|---|
| `lora_router/lora.py` | LoRA as a weight parametrization for `nn.MultiheadAttention` (packed q/k/v) and `nn.Linear`; model-agnostic, also used for Qwen |
| `lora_router/build.py` | Builds CircuitFormer from the clone without editing it; `python -m lora_router.build` runs a synthetic check |
| `lora_router/datasets.py` | CircuitNet-N28 and CircuitNet-N14 loaders and label construction |
| `lora_router/eval.py` | Scores a model on a dataset |
| `lora_router/train.py` | Adapts with `--mode lora`, `head` or `full` and scores on a fixed held-out split |
| `tests/test_lora.py` | Stand-alone LoRA smoke test, no CircuitFormer needed |
| `results/` | Per-layout scores and run logs; `overflow_attempt/` is a failed first attempt kept for the record |

### Setup

1. Install dependencies (see the notes at the top of `requirements.txt` for torch and torch-scatter).
2. Clone the base model. It has no licence, so it is not vendored here:
   ```bash
   git clone https://github.com/hustvl/circuitformer third_party/circuitformer
   ```
   Tested at commit `4beba89`.
3. Download the authors' checkpoint (link in the CircuitFormer README) to `third_party/circuitformer/ckpts/circuitformer.ckpt`.
4. Download data from [CircuitNet](https://github.com/circuitnet/CircuitNet) into `third_party/circuitformer/dataset/`:
   ```
   CircuitNet/                                   # N28
     graph_features/instance_placement_gcell/
     routability_features/congestion/overflow_based/congestion_GR_{horizontal,vertical}_overflow/
   CircuitNet-N14/
     graph_features/Vortex-small/instance_placement_micron/
     routability_features/Vortex-small/congestion/congestion_global_routing/utilization_based/
   ```

### Usage

```bash
python tests/test_lora.py
python -W ignore -m lora_router.build third_party/circuitformer/ckpts/circuitformer.ckpt
python -W ignore -m lora_router.eval --data n28
python -W ignore -m lora_router.eval --data Vortex-small
python -W ignore -m lora_router.train --mode lora --decoder-head --lr 1e-3 --steps 400 --n-train 16
python -W ignore -m lora_router.train --mode full --lr 1e-4 --steps 400 --n-train 16
```

Training uses half precision and batch size 1 by default so it fits an 8 GB GPU.

### Data notes

- **N28 baseline.** The released checkpoint scores Pearson 0.442 / Spearman 0.479 / Kendall 0.349 on the 921-layout test split with our pipeline; the paper reports 0.637 / 0.528 / 0.394. The gap is concentrated in the "-b" design variants and is unexplained. Per-layout scores are in `results/n28_test_gcell.csv`.
- **Units.** gcell placements score better than micron on N28 (`results/n28_test_micron.csv`).
- **N14 gcell placements are all zeros** in the published Vortex-small archive. gcell coordinates are derived from the micron files at 1.536 µm per gcell.
- **N14 lists filler cells; N28 does not.** Fillers (two-thirds of the entries) are dropped.
- **Label.** The overflow-based congestion label used on N28 is almost empty on Vortex-small and nothing learns it, full fine-tuning included (`results/overflow_attempt/`). N14 experiments use the utilization-based label.

### Limitations

- One target design, one seed, 40 test layouts: differences between the top rows are not statistically established.
- Averaging the training labels already scores Pearson 0.565, so the margin that shows per-layout learning is 0.617 vs 0.565.
- The source and target labels differ (overflow at N28, utilization at N14).
- The congestion predictions have not been fed to a router.

## Third-party

CircuitFormer and the CircuitNet datasets belong to their authors and are not redistributed here. Fully fine-tuned checkpoints are git-ignored because they are derived from CircuitFormer's weights. Qwen2.5-1.5B-Instruct (Apache 2.0) is downloaded, not redistributed; `results/rl/*.pt` and `results/agent/*.pt` hold only LoRA weights. OpenLane 2, OpenROAD and the sky130 PDK are used through their published Docker image and installer.

# Plan: command-level timing-closure agent (Qwen + LoRA + OpenROAD)

Agreed 2026-11-11. Starts after the frequency setting-selection run (`results/rl/train_mac16_freq.log`) finishes and is recorded.

## Goal

- **Paper 1 claim:** a LoRA-adapted Qwen, trained by RL, reaches a signed-off design that meets timing, area and power targets in fewer actions and less wall-clock time than the default flow, setting-only agents (ours and an ORFS-agent-style loop), base Qwen and random actions, including on designs it never saw.
- **How it differs from ORFS-agent** ([arXiv 2506.08332](https://arxiv.org/html/2506.08332v3)): ORFS-agent prompts frozen large LLMs to choose flow parameters and restarts the whole flow each time. Ours acts on specific cells and nets *inside one flow run*, and learns by RL.
- **Division of labour:** Qwen decides *what, where and which*. OpenROAD's placer, resizer and routers compute the exact geometry.

## 0. Once per design (before Qwen)

| # | Step | Tool | Output |
|---|---|---|---|
| 1 | Synthesis, RTL → sky130_fd_sc_hd cells | Yosys via OpenLane | gate-level netlist |
| 2 | Floorplan, I/O, power grid, tap cells | OpenROAD via OpenLane | empty die with rows, pins, straps |
| 3 | Save starting state | `write_db start.odb` | checkpoint, reloads in < 1 s |
| 4 | Design markers (§3) | Python over netlist + OpenSTA | ~300 tokens |

## 1. Environment

- **Live OpenROAD session** in the OpenLane Docker image, kept running. It loads the sky130 LEF and liberty (ss for setup, ff for hold, tt for reporting), the SDC with the clock target, and `start.odb`, then loops: read command → run → print result.
- **Commands are Tcl**, OpenROAD's command interface for every step (OpenLane drives it the same way). Qwen outputs a JSON action from a fixed menu. Python validates it and writes the one matching OpenROAD command. Qwen never writes free-form code.
- **Checkpoint after each action** (`write_db step_k.odb`): enables `undo`, and training can branch several actions from one state.

## 2. Getting the critical cells (after every action)

1. **Refresh timing:** `estimate_parasitics -placement` before global routing; `-global_routing` after it. Full extraction and signoff only at the end.
2. **Worst paths:** `report_checks -path_delay max -group_path_count 5 -endpoint_path_count 1 -fields {slew cap input_pins fanout} -digits 3`, plus `report_wns`, `report_tns`, hold `report_checks -path_delay min`, `report_design_area`, `report_power`.
3. **Positions:** each path instance's (x, y) from the database (`findInst` → `getLocation`).
4. **Compress** to a table per path (instance, cell, drive, fanout, load, stage delay, wire delay, x, y) with flags:

| Flag | Rule | Suggests |
|---|---|---|
| `WEAK` | drive `_1` with fanout > 4 or high load | resize up |
| `LONGWIRE` | wire delay into the stage > gate delay | pull cells together, wider wire |
| `FANOUT` | fanout > 8 | buffer / repair |
| `SPREAD` | path spans > 30% of die width | pull the path together |
| `SLOWCELL` | stage > 25% of path delay | resize or restructure |

## 3. Design markers (once per design, whole netlist)

- Cell counts by function (adder/majority, XOR/XNOR, AND/OR, mux, inverter/buffer, flip-flop), drive-strength mix, I/O count.
- Logic depth between flip-flops; fanout mean / p99 / top 3 nets.
- Clock target, flip-flop count.

## 4. Action menu

| JSON action | OpenROAD command | Stage |
|---|---|---|
| `place` mode timing/routability, density | `global_placement -timing_driven` / `-routability_driven -density`, `detailed_placement` | placement |
| `pad` cells, sites | `set_placement_padding -instances`, re-legalise | placement |
| `pull_path` path | `place_cell` toward the path centroid, `detailed_placement`, `improve_placement` | placement |
| `resize` cell, to | `replace_cell` (same-function variants only) | any |
| `repair_timing` margin | `repair_timing -setup …` (options to confirm) | any |
| `repair_design` | `repair_design` | any |
| `restructure` | `restructure` (options to confirm) | placement |
| `remove_buffers` | `remove_buffers` | early |
| `dont_touch` cells | `set_dont_touch` | any |
| `cts` | `clock_tree_synthesis`, `repair_clock_nets` | once |
| `wide_wire` net | `create_ndr` + `assign_ndr` | before routing |
| `route_margin` region, adj | `set_global_routing_region_adjustment` | before routing |
| `global_route` | `global_route`, STA with routed parasitics | once |
| `undo` | `read_db step_{k-1}.odb` | any |
| `finish` | detailed route → extraction → signoff STA (ss / tt / ff) | end |

All listed commands exist in this build (checked 2026-11-11). No direct buffer-insertion command: buffering goes through `repair_design` / `repair_timing`.

## 5. Prompt per step (~1.5–2.5k tokens, any design size)

- **System:** timing-closure engineer controlling OpenROAD (sky130_fd_sc_hd); one JSON action per turn; meet all targets in as few actions as possible; setup signed off at ss (100 °C, 1.60 V), hold at ff.
- **User:**
  - `TARGETS` (e.g. period ≤ 14.0 ns at ss, die area < 70,000 µm², power < 5.0 mW, hold ≥ 0, DRC 0)
  - `DESIGN` markers
  - `STAGE` (where in the flow, step k of 30, runtime so far)
  - `NOW` (WNS, achievable period, TNS, hold, area, power)
  - `HISTORY` (each past action and its effect)
  - `WORST PATHS` (5 compressed tables)
  - `MENU` (allowed actions now, with ranges)
- **Answer:** one JSON action, e.g. `{"op": "resize", "cell": "_1123_", "to": "xor2_4"}`.

## 6. Episode

```
read_db start.odb
up to 30 actions: state → Qwen → JSON → validate → OpenROAD (1–30 s) → refresh timing → markers → step reward
stop when targets met (pre-route estimate) and agent says finish
finish: detailed route → extraction → signoff STA → success only if signoff meets every target
```

## 7. Reward and training

- **Step:** ΔWNS/|WNS₀| + 0.5·ΔTNS/|TNS₀| − area and power penalties above target − 0.1 for an invalid action.
- **Terminal:** +1 if signoff meets every target, plus a bonus for fewer actions; −1 for DRC errors or a broken flow.
- **Branching:** at a state, sample 4 actions, run each from the same checkpoint, group-normalised advantage (as in `flows/rl.py`).
- **LoRA:** rank 16–32 on attention and MLP projections (~10–18M parameters) of Qwen2.5-1.5B-Instruct.
- **Designs:** 4–6 training (mac16 and variants, small RV32 core, SIMD/DSP block) and 2 held-out.

## 8. Benchmark (paper 1)

- **Metric:** actions and wall-clock time to a signed-off design meeting all targets; success rate within the budget.
- **Baselines:** OpenLane default flow; setting-selection agent (`flows/rl.py --objective freq`); base Qwen with the same prompt; ORFS-agent-style loop (base Qwen + history); random valid actions; Bayesian optimisation (Optuna TPE) over settings.
- **Proof of learning:** held-out designs; shuffled-state ablation; several seeds; default and agent runs interleaved (routing time varies ±15% between identical runs).

## 9. Risks

- Geometry stays with OpenROAD; the contribution is the decisions and the learning.
- Pre-route timing may not match signoff; large gaps are penalised at the end.
- Actions interact (later repair can undo a manual resize unless `dont_touch`).
- Options of `repair_timing` and `restructure` still to confirm.

## Status

- **Step 1 done (2026-11-11):** `flows/session.py` runs a live OpenROAD session in the OpenLane image.
  - **Setup:** loads the design the way OpenLane does (`io.tcl`, `resizer.tcl`, `set_rc.tcl`, the settings of every step of a kept default run, three resizer corners: tt, ss, ff).
  - **Overhead:** boot 2.8 s; Windows → WSL → Docker round trip under 1 ms per command (5 calls: 32 ms wall, 27 ms inside OpenROAD).
  - **Hand-run on mac16:**

    | Step | Time | Worst setup slack, ss (pre-route) | Hold |
    |---|---|---|---|
    | placement | 23.7 s | −14.01 ns | +0.204 ns |
    | design repair | 7.6 s | −9.24 ns | +0.135 ns |
    | clock tree | 9.6 s | −9.23 ns | +0.114 ns |
    | timing repair | 73.8 s | −6.20 ns | +0.114 ns |

  - **Matches OpenLane:** standard-cell area 23,933 µm² after placement (identical) and 22,060 vs 22,051 µm² after timing repair.
  - **`repair_timing` options confirmed:** `-setup|-hold -setup_margin -hold_margin -max_buffer_percent [-skip_gate_cloning] [-allow_setup_violations]`.
  - **Lesson:** a step's `_env.tcl` only holds that step's settings; the session merges all steps of a full default run, so every design needs one kept default run.
- **Steps 2–5 done (2026-11-11):** environment, baselines, RL training (22 episodes) and eval on 6 designs (2 held out).
  - **Result:** the free-text LoRA agent met the target in 2 of 12 eval runs, against 5 of 6 for the max-effort script, with 81% invalid answers. Full write-up: [2026-11-11_agent_results.md](2026-11-11_agent_results.md).
- **Next:** A (choose from a list) + D (honest post-route P, a "met" flag) first. Then B (search labels + fine-tuning) on the logged branches. Add the best fixed search sequence as a baseline. If the adapter does not beat it on held-out designs, switch to the predictor (Plan B in the write-up).

## 10. Build order

1. Live session and one hand-run episode (commands work, reports parse).
2. Marker extractor and prompt builder.
3. Default-flow and random-action baselines; set target numbers.
4. RL with LoRA and checkpoint branching.
5. More designs, held-out tests, runs-to-target benchmark.

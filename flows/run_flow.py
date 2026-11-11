"""Run the OpenLane 2 (OpenROAD) flow on a design with given knobs; return runtime and timing.

Runs in WSL through the OpenLane Docker image. Run directories live in WSL under
~/lora_router_runs/<design>/runs/<tag> and are deleted after metrics are read
(keep=True keeps them). Every result is appended to results/flow_runs.jsonl.

    python -m flows.run_flow mac16                       # default knobs
    python -m flows.run_flow mac16 GRT_ADJUSTMENT=0.4 FP_CORE_UTIL=40
"""
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESIGNS = REPO / "flows" / "designs"
LOG = REPO / "results" / "flow_runs.jsonl"
IMAGE = "ghcr.io/efabless/openlane2:2.3.10"
WSL_RUNS = "~/lora_router_runs"
STOP_AT = "OpenROAD.STAPostPNR"  # skip GDS streaming, DRC decks and LVS: not part of routing

# Knobs the policy may set. Numeric: (lo, hi, type), inclusive. bool. list: allowed values.
# Anything not set keeps the OpenLane default (DEFAULTS, as resolved for mac16).
ROUTE_KNOBS = {
    "FP_CORE_UTIL": (25, 65, int),
    "PL_TARGET_DENSITY_PCT": (35, 85, int),
    "GRT_ADJUSTMENT": (0.0, 0.7, float),
    "GRT_OVERFLOW_ITERS": (10, 100, int),
    "DRT_OPT_ITERS": (8, 64, int),
    "GPL_CELL_PADDING": (0, 4, int),
    "DPL_CELL_PADDING": (0, 4, int),
}
SYNTH_STRATEGIES = ["AREA 0", "AREA 1", "AREA 2", "AREA 3",
                    "DELAY 0", "DELAY 1", "DELAY 2", "DELAY 3", "DELAY 4"]
FREQ_KNOBS = {
    "CLOCK_PERIOD": (4.0, 20.0, float),  # target in ns; synthesis and timing repair aim for it
    "SYNTH_STRATEGY": SYNTH_STRATEGIES,
    "SYNTH_ABC_BUFFERING": bool,
    "SYNTH_SIZING": bool,
    "PL_RESIZER_SETUP_SLACK_MARGIN": (0.0, 1.0, float),
    "GRT_RESIZER_SETUP_SLACK_MARGIN": (0.0, 1.0, float),
    "RUN_POST_GRT_RESIZER_TIMING": bool,
    "PL_RESIZER_SETUP_MAX_BUFFER_PCT": (10, 100, int),
}
KNOBS = {**ROUTE_KNOBS, **FREQ_KNOBS}
DEFAULTS = {
    "FP_CORE_UTIL": 50, "PL_TARGET_DENSITY_PCT": "auto (62 for mac16)", "GRT_ADJUSTMENT": 0.3,
    "GRT_OVERFLOW_ITERS": 50, "DRT_OPT_ITERS": 64, "GPL_CELL_PADDING": 0, "DPL_CELL_PADDING": 0,
    "CLOCK_PERIOD": "the design's", "SYNTH_STRATEGY": "AREA 0", "SYNTH_ABC_BUFFERING": False,
    "SYNTH_SIZING": False, "PL_RESIZER_SETUP_SLACK_MARGIN": 0.05,
    "GRT_RESIZER_SETUP_SLACK_MARGIN": 0.025, "RUN_POST_GRT_RESIZER_TIMING": False,
    "PL_RESIZER_SETUP_MAX_BUFFER_PCT": 50,
}
METRICS = {
    "wns_ss": "timing__setup__ws__corner:max_ss_100C_1v60",
    "tns_ss": "timing__setup__tns__corner:max_ss_100C_1v60",
    "wns_tt": "timing__setup__ws__corner:nom_tt_025C_1v80",
    "hold_ws": "timing__hold__ws",
    "drc": "route__drc_errors",
    "wirelength": "route__wirelength",
    "die_area": "design__die__area",
    "cell_area": "design__instance__area__stdcell",
    "power": "power__total",
}


def clamp(knobs):
    """Coerce each value into its allowed range or set; drop values that cannot be read."""
    out = {}
    for k, v in knobs.items():
        spec = KNOBS.get(k)
        try:
            if isinstance(spec, list):
                out[k] = v if v in spec else spec[min(len(spec) - 1, max(0, int(float(v))))]
            elif spec is bool:
                out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")
            elif spec:
                lo, hi, t = spec
                out[k] = t(min(hi, max(lo, t(float(v)))))
        except (TypeError, ValueError):
            pass
    return out


def _sh(cmd, timeout=None):
    b64 = base64.b64encode(cmd.encode()).decode()  # avoids Windows -> WSL quoting issues
    return subprocess.run(["wsl", "-e", "bash", "-lc", f"echo {b64} | base64 -d | bash -l"],
                          capture_output=True, text=True, timeout=timeout)


def _secs(s):
    h, m, sec = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def run(design, knobs=None, tag=None, keep=False, timeout=1800, note=""):
    knobs = clamp(knobs or {})
    tag = tag or f"r{int(time.time() * 1000)}"
    cfg = json.loads((DESIGNS / design / "config.json").read_text())
    cfg.update(knobs)
    src = f"/mnt/{DESIGNS.drive[0].lower()}{DESIGNS.as_posix()[2:]}/{design}"
    work = f"{WSL_RUNS}/{design}"
    cfg_b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    cmd = (
        f"mkdir -p {work} && cp -r {src}/src {work}/ && cd {work} && "
        f"echo {cfg_b64} | base64 -d > cfg_{tag}.json && "
        f"docker run --rm -u $(id -u):$(id -g) -v $HOME:$HOME -e HOME=$HOME -w $PWD {IMAGE} "
        f"openlane --pdk-root $HOME/.volare --run-tag {tag} --overwrite --to {STOP_AT} cfg_{tag}.json "
        f"> log_{tag}.txt 2>&1; echo EXIT=$?; cd runs/{tag} && "
        "for d in */; do [ -f $d/runtime.txt ] && echo \"RT ${d%/} $(cat $d/runtime.txt)\"; done; "
        "f=$(ls -d */ | grep stapostpnr | tail -1); echo METRICS; cat ${f}state_out.json"
    )
    t0 = time.time()
    p = _sh(cmd, timeout)
    wall = time.time() - t0
    out = p.stdout
    rec = {"design": design, "tag": tag, "knobs": knobs, "clock": cfg["CLOCK_PERIOD"],
           "wall_s": round(wall, 1), "note": note, "ok": False}
    try:
        steps = {m[1]: _secs(m[2]) for m in re.finditer(r"^RT (\S+) (\S+)$", out, re.M)}
        grt = sum(v for k, v in steps.items() if k.endswith("openroad-globalrouting"))
        drt = sum(v for k, v in steps.items() if k.endswith("openroad-detailedrouting"))
        pnr = sum(v for k, v in steps.items() if int(k.split("-")[0]) >= 13)  # floorplan onward
        m = json.loads(out.split("METRICS", 1)[1])["metrics"]
        rec.update({"grt_s": round(grt, 2), "drt_s": round(drt, 2), "route_s": round(grt + drt, 2),
                    "pnr_s": round(pnr, 2), **{k: m.get(v) for k, v in METRICS.items()},
                    "slow_steps": {k: round(v, 1) for k, v in steps.items() if v > 5}})
        rec["ok"] = "EXIT=0" in out and rec["drc"] == 0
    except Exception as e:  # flow failed (e.g. placement over-utilised): record and move on
        rec["error"] = f"{type(e).__name__}: {e}; exit={'EXIT=0' in out}"
    if not keep:
        _sh(f"rm -rf {work}/runs/{tag}")
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    kn = dict(a.split("=", 1) for a in sys.argv[2:])
    r = run(sys.argv[1], kn, note="cli")
    print(json.dumps(r, indent=1))

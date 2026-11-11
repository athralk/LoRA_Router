"""Timing-closure environment for the command-level agent (docs/plan.md).

An episode starts from OpenLane's own post-CTS database of a design (default flow up to
clock-tree synthesis) in a live OpenROAD session. The agent then picks one action per
turn from a fixed menu (repair, resize, pull a path together, route + repair, finish)
until it finishes or runs out of actions. Timing is the session's estimate (placement
parasitics before routing, global-routing parasitics after); final success is judged
after a fresh global route. Full signoff (detailed route + extraction) is in signoff().

Target per design: the better of the default continuation (OpenLane's post-CTS repair
settings) and a max-effort script, both measured once in prep(). The agent wins by
reaching it in fewer actions / less tool time, or by beating it.
"""
import json
import math
import random
import re
import threading
import time

from .run_flow import REPO, _sh, run as run_flow
from .session import Session

OUT = REPO / "results" / "agent"
SS = "nom_ss_100C_1v60"
PREFIX = "sky130_fd_sc_hd__"
MAX_ACTIONS = 6
_HOME = []


def home():
    if not _HOME:
        _HOME.append(_sh("echo $HOME").stdout.strip())
    return _HOME[0]


def base_dir(design):
    return f"{home()}/lora_router_runs/{design}/runs/base"


# ---------------------------------------------------------------- measurements

def metrics(s, routed):
    """WNS (setup, worst corner), hold WNS, TNS, cell area, utilisation."""
    s.cmd(f"estimate_parasitics -{'global_routing' if routed else 'placement'}")
    out, _, _ = s.cmd("report_worst_slack -max -digits 4\nreport_worst_slack -min -digits 4\n"
                      "report_tns -digits 3\nreport_design_area")
    ws = [float(x) for x in re.findall(r"worst slack (-?[\d.]+)", out)]
    tns = re.search(r"tns (-?[\d.]+)", out)
    area = re.search(r"Design area ([\d.]+) u\^2 (\d+)% utilization", out)
    return {"wns": ws[0], "hold": ws[1], "tns": float(tns[1]) if tns else None,
            "area": float(area[1]), "util": int(area[2])}


def power_mw(s):
    out, _, _ = s.cmd("report_power -corner nom_tt_025C_1v80")
    m = re.search(r"^Total\s+\S+\s+\S+\s+\S+\s+([\d.eE+-]+)", out, re.M)
    return round(float(m[1]) * 1000, 3) if m else None


OUT_RE = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+[\^v]\s+(\S+)/(\S+)\s+\((\S+)\)")
IN_RE = re.compile(r"^\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+[\^v]\s+(\S+)/(\S+)\s+\((\S+)\)")


def worst_paths(s, n=5):
    """The n worst setup paths at the slow corner, one stage per driving cell."""
    out, _, _ = s.cmd(f"report_checks -path_delay max -group_count {n} -endpoint_count 1 "
                      f"-corner {SS} -fields {{fanout cap slew}} -digits 3")
    paths = []
    for blk in out.split("Startpoint:")[1:]:
        start = blk.split()[0]
        end = re.search(r"Endpoint: (\S+)", blk)
        slack = re.search(r"(-?[\d.]+)\s+slack \((?:VIOLATED|MET)\)", blk)
        stages, wire = [], {}
        for line in blk.split("data arrival time")[0].splitlines():
            if m := OUT_RE.match(line):
                fo, cap, slew, d, t, inst, pin, cell = m.groups()
                stages.append({"inst": inst, "cell": cell.replace(PREFIX, ""), "fo": int(fo),
                               "cap": float(cap) * 1000, "slew": float(slew), "delay": float(d),
                               "wire": wire.get(inst, 0.0), "pin": pin})
            elif m := IN_RE.match(line):
                wire[m[4]] = float(m[2])
        if stages and slack:
            paths.append({"start": start, "end": end[1] if end else "?", "slack": float(slack[1]),
                          "stages": stages})
    insts = sorted({st["inst"] for p in paths for st in p["stages"]})
    if insts:
        out, _, _ = s.cmd("foreach n {%s} { set i [[ord::get_db_block] findInst $n]; "
                          "if {$i ne \"NULL\"} { puts \"@P $n [$i getLocation]\" } }" % " ".join(insts))
        loc = {m[1]: (int(m[2]) / 1000, int(m[3]) / 1000) for m in re.finditer(r"@P (\S+) (\d+) (\d+)", out)}
        for p in paths:
            for st in p["stages"]:
                st["xy"] = loc.get(st["inst"])
    return paths


def drive(cell):
    m = re.search(r"_(\d+)$", cell)
    return int(m[1]) if m else 0


def flags(p, die):
    mean = sum(st["delay"] for st in p["stages"]) / len(p["stages"])
    for st in p["stages"]:
        f = []
        if drive(st["cell"]) <= 1 and (st["fo"] >= 4 or st["cap"] >= 15):
            f.append("WEAK")
        if st["wire"] >= 0.1 and st["wire"] > 0.5 * st["delay"]:
            f.append("LONGWIRE")
        if st["fo"] >= 8:
            f.append("FANOUT")
        if st["delay"] > 2 * mean:
            f.append("SLOWCELL")
        st["flags"] = f
    xy = [st["xy"] for st in p["stages"] if st.get("xy")]
    p["span"] = (max(x for x, _ in xy) - min(x for x, _ in xy), max(y for _, y in xy) - min(y for _, y in xy)) if xy else (0, 0)
    p["spread"] = max(p["span"][0] / die[0], p["span"][1] / die[1]) > 0.3


# ---------------------------------------------------------------- actions

MENU = {
    "repair_timing": "global setup+hold repair; margin 0-0.5 ns, buffer_pct 10-100, cloning true/false (slow: 10-90 s)",
    "repair_design": "fix slew, capacitance and fanout limits (~8 s)",
    "resize": "one cell on a worst path one drive strength up or down; cell=<instance>, size=up|down (~1 s)",
    "upsize_path": "every WEAK or SLOWCELL cell on path P1-P5 one size up (~1 s)",
    "pull_path": "move the cells of path P1-P5 halfway to their centre, then legalise (~1 s)",
    "route_repair": "global route, then setup+hold repair with routed wire parasitics; margin 0-0.5 (slow: 20-90 s)",
    "finish": "stop; the design is globally routed and judged",
}


class Variants:
    """Drive-strength variants per cell function, excluding the PDK's do-not-use cells."""

    def __init__(self, s):
        out, _, _ = s.cmd("foreach c [get_lib_cells *] { puts \"@L [get_name $c]\" }\nputs \"@X $::env(_PNR_EXCLUDED_CELLS)\"")
        excl = set(re.search(r"@X (.*)", out)[1].split())
        self.by_base = {}
        for name in set(re.findall(r"@L (\S+)", out)):
            if name in excl or not name.startswith(PREFIX):
                continue
            short = name.replace(PREFIX, "")
            m = re.match(r"(.+)_(\d+)$", short)
            if m:
                self.by_base.setdefault(m[1], set()).add(int(m[2]))

    def step(self, cell, up=True):
        m = re.match(r"(.+)_(\d+)$", cell)
        if not m or m[1] not in self.by_base:
            return None
        sizes = sorted(self.by_base[m[1]])
        cur = int(m[2])
        cand = [z for z in sizes if (z > cur if up else z < cur)]
        if not cand:
            return None
        return f"{m[1]}_{cand[0] if up else cand[-1]}"


def to_tcl(action, state, variants):
    """Validate one JSON action against the current state; return (tcl, routed_after) or raise ValueError."""
    op = action.get("op")
    paths = {f"P{i + 1}": p for i, p in enumerate(state["paths"])}
    est = f"estimate_parasitics -{'global_routing' if state['routed'] else 'placement'}\n"
    if op == "repair_timing":
        m = min(0.5, max(0.0, float(action.get("margin", 0.05))))
        b = int(min(100, max(10, float(action.get("buffer_pct", 50)))))
        clone = str(action.get("cloning", True)).lower() not in ("false", "0", "no")
        return (est + f"repair_timing -setup -setup_margin {m} -max_buffer_percent {b}"
                f"{'' if clone else ' -skip_gate_cloning'}\n"
                f"repair_timing -hold -setup_margin {m} -hold_margin 0.1 -max_buffer_percent 50\n"
                "detailed_placement"), False
    if op == "repair_design":
        return (est + "repair_design -max_wire_length $::env(DESIGN_REPAIR_MAX_WIRE_LENGTH) "
                "-slew_margin $::env(DESIGN_REPAIR_MAX_SLEW_PCT) -cap_margin $::env(DESIGN_REPAIR_MAX_CAP_PCT)\n"
                "detailed_placement"), False
    if op == "resize":
        cell = {st["inst"]: st for p in state["paths"] for st in p["stages"]}.get(str(action.get("cell")))
        if not cell:
            raise ValueError("cell not on a listed path")
        new = variants.step(cell["cell"], str(action.get("size", "up")).lower() != "down")
        if not new:
            raise ValueError(f"no {action.get('size', 'up')} size for {cell['cell']}")
        return f"replace_cell {cell['inst']} {PREFIX}{new}\ndetailed_placement", False
    if op in ("upsize_path", "pull_path"):
        p = paths.get(str(action.get("path", "P1")).upper())
        if not p:
            raise ValueError("unknown path")
        if op == "upsize_path":
            lines = []
            for st in p["stages"]:
                if {"WEAK", "SLOWCELL"} & set(st["flags"]) and (new := variants.step(st["cell"])):
                    lines.append(f"replace_cell {st['inst']} {PREFIX}{new}")
            if not lines:
                raise ValueError("no WEAK or SLOWCELL cell to upsize on this path")
            return "\n".join(lines) + "\ndetailed_placement", False
        xy = [(st["inst"], st["xy"]) for st in p["stages"] if st.get("xy")]
        cx = sum(x for _, (x, _) in xy) / len(xy)
        cy = sum(y for _, (_, y) in xy) / len(xy)
        lines = [f"[[ord::get_db_block] findInst {n}] setLocation {int(1000 * (x + cx) / 2)} {int(1000 * (y + cy) / 2)}"
                 for n, (x, y) in xy]
        return "\n".join(lines) + "\ndetailed_placement", False
    if op == "route_repair":
        m = min(0.5, max(0.0, float(action.get("margin", 0.025))))
        return ("source $::env(SCRIPTS_DIR)/openroad/common/grt.tcl\nestimate_parasitics -global_routing\n"
                f"repair_timing -setup -setup_margin {m} -max_buffer_percent 50\n"
                f"repair_timing -hold -setup_margin {m} -hold_margin 0.05 -max_buffer_percent 50\n"
                "detailed_placement"), False
    if op == "finish":
        return "source $::env(SCRIPTS_DIR)/openroad/common/grt.tcl", True
    raise ValueError(f"unknown op {op!r}")


def parse_action(text):
    """Every `"key": value` pair anywhere in the answer; the op is "op" (or "action"/"type") naming a menu entry."""
    pairs = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*(?:"([^"]*)"|(true|false)|(-?\d+(?:\.\d+)?))', text):
        k = m[1]
        v = m[2] if m[2] is not None else (m[3] == "true" if m[3] else float(m[4]))
        pairs.setdefault(k, v)
    op = next((pairs[k] for k in ("op", "action", "type", "name") if isinstance(pairs.get(k), str)
               and pairs[k] in MENU), None)
    if op is None:
        op = next((o for o in MENU if re.search(rf"\b{o}\b", text)), None)
    if op is None:
        return None
    a = {"op": op}
    for k in ("cell", "size", "path", "margin", "buffer_pct", "cloning"):
        if k in pairs:
            a[k] = pairs[k]
    return a


# ---------------------------------------------------------------- state, prompt, reward

def period(clock, wns):
    return clock - wns


class Episode:
    """One closure episode in one live session. Use .branch() to copy the state into new sessions."""

    def __init__(self, design, prep, odb=None, history=None, k=0, routed=False):
        self.design, self.prep = design, prep
        self.s = Session(design, "base", odb=odb or prep["cts_odb"])
        self.history = list(history or [])
        self.k, self.routed = k, routed
        self.variants = Variants(self.s) if not prep.get("_variants") else prep["_variants"]
        prep["_variants"] = self.variants
        self.tool_s = 0.0
        self.observe()

    def observe(self):
        self.m = metrics(self.s, self.routed)
        self.paths = worst_paths(self.s)
        for p in self.paths:
            flags(p, self.prep["die"])
        self.P = period(self.prep["clock"], self.m["wns"])
        return self.state()

    def state(self):
        return {"paths": self.paths, "routed": self.routed, "m": self.m, "P": self.P, "k": self.k}

    def step(self, action):
        """Apply one action. Returns reward and info; the episode is done after finish or MAX_ACTIONS."""
        p0, g = self.P, self.prep
        gap = max(0.05, g["P_start"] - g["target"])
        try:
            tcl, routed = to_tcl(action, self.state(), self.variants)
        except (ValueError, TypeError) as e:
            self.k += 1
            self.history.append((json.dumps(action)[:80], f"rejected: {e}"))
            if self.k >= MAX_ACTIONS:  # budget used up: judge the design as it stands
                r, info = self.step({"op": "finish"})
                self.k = MAX_ACTIONS
                return r - 0.1, {**info, "valid": False, "error": str(e)}
            return -0.1, {"valid": False, "error": str(e), "done": False}
        out, ok, ms = self.s.cmd(tcl)
        self.tool_s += ms / 1000
        self.k += 1
        self.routed = routed
        if not ok:
            self.history.append((json.dumps(action)[:80], "OpenROAD error"))
            if self.k >= MAX_ACTIONS and action.get("op") != "finish":
                r, info = self.step({"op": "finish"})
                self.k = MAX_ACTIONS
                return r - 0.2, {**info, "valid": False, "error": out[-300:]}
            return -0.2, {"valid": False, "error": out[-300:], "done": self.k >= MAX_ACTIONS}
        self.observe()
        done = action.get("op") == "finish" or self.k >= MAX_ACTIONS
        r = (p0 - self.P) / gap - 0.003 * ms / 1000
        info = {"valid": True, "done": done, "tool_s": ms / 1000, "P": self.P, "m": self.m}
        self.history.append((json.dumps(action)[:80],
                             f"period {p0:.2f} -> {self.P:.2f} ns, hold {self.m['hold']:+.3f}, "
                             f"area {self.m['area']:.0f}, {ms / 1000:.1f} s"))
        if done:
            if not self.routed:  # budget ran out before finish: judge after a global route
                self.s.cmd("source $::env(SCRIPTS_DIR)/openroad/common/grt.tcl")
                self.routed = True
                p1 = self.P
                self.observe()
                r += (p1 - self.P) / gap
            info["success"] = self.success()
            info["P"] = self.P
            info["power"] = power_mw(self.s)
            r += 1.0 if info["success"] else 0.0
        return r, info

    def success(self):
        g = self.prep
        return self.P <= g["target"] and self.m["hold"] >= 0 and self.m["area"] <= 1.25 * g["area_start"]

    def checkpoint(self, path):
        self.s.cmd(f"write_db {path}")
        return path

    def branch(self, path):
        """A new episode continuing from this exact state (new session from a checkpoint)."""
        e = Episode(self.design, self.prep, odb=self.checkpoint(path), history=self.history, k=self.k,
                    routed=self.routed)
        e.tool_s = self.tool_s
        return e

    def close(self):
        self.s.close()


SYSTEM = ("You are a timing-closure engineer controlling OpenROAD on the sky130_fd_sc_hd library. "
          "Each turn you see the design state and choose ONE action as JSON. Goal: reach the target "
          "clock period at the slow corner (ss, 100 C, 1.60 V) with hold met, in as few actions and as "
          "little tool time as possible.")


def prompt(ep):
    g, m = ep.prep, ep.m
    lines = [
        f"TARGETS  period <= {g['target']:.2f} ns at slow corner | hold >= 0 | cell area <= {1.25 * g['area_start']:.0f} um^2",
        f"DESIGN   {ep.design} | {g['markers']}",
        f"STAGE    after clock tree, {'globally routed' if ep.routed else 'placed, not routed'} | "
        f"action {ep.k + 1} of {MAX_ACTIONS} | tool time so far {ep.tool_s:.0f} s",
        f"NOW      period {ep.P:.2f} ns (clock {g['clock']} ns, WNS {m['wns']:.2f}) | TNS {m['tns']} | "
        f"hold {m['hold']:+.3f} | cell area {m['area']:.0f} um^2 ({m['util']}%)",
        "HISTORY  " + ("none" if not ep.history else ""),
    ]
    lines += [f"  {i + 1}. {a} -> {r}" for i, (a, r) in enumerate(ep.history)]
    lines.append("WORST PATHS (slow corner; cell delay and wire delay in ns, cap in fF, x,y in um)")
    for i, p in enumerate(ep.paths):
        lines.append(f" P{i + 1} slack {p['slack']:.2f} | {p['start']} -> {p['end']} | {len(p['stages'])} stages | "
                     f"span {p['span'][0]:.0f}x{p['span'][1]:.0f} um{' SPREAD' if p['spread'] else ''}")
        top = sorted(p["stages"], key=lambda st: -st["delay"])[:6] if i else p["stages"][:14]
        for st in top:
            xy = f"{st['xy'][0]:.0f},{st['xy'][1]:.0f}" if st.get("xy") else "?"
            lines.append(f"   {st['inst']} {st['cell']} fo{st['fo']} cap{st['cap']:.0f} d{st['delay']:.2f} "
                         f"w{st['wire']:.2f} @{xy} {' '.join(st['flags'])}")
    lines.append("MENU")
    lines += [f" {k}: {v}" for k, v in MENU.items()]
    lines.append('Answer with one JSON object, e.g. {"op": "resize", "cell": "_1123_", "size": "up"} '
                 'or {"op": "repair_timing", "margin": 0.1, "buffer_pct": 50, "cloning": true}.')
    return "\n".join(lines)


# ---------------------------------------------------------------- per-design preparation

def markers(s):
    out, _, _ = s.cmd("foreach i [[ord::get_db_block] getInsts] { puts \"@C [[$i getMaster] getName]\" }\n"
                      "foreach n [[ord::get_db_block] getNets] { if {[$n getSigType] in {POWER GROUND}} continue; "
                      "set k [llength [$n getITerms]]; if {$k > 8} { puts \"@F [$n getName] $k\" } }")
    groups = {"adder": 0, "xor": 0, "mux": 0, "flop": 0, "buf/inv": 0, "logic": 0}
    for name in re.findall(r"@C (\S+)", out):
        c = name.replace(PREFIX, "")
        if re.match(r"(tap|fill|decap|diode|conb)", c):
            continue
        key = ("adder" if re.match(r"(fa|ha|maj)", c) else "xor" if re.match(r"x(n)?or", c) else
               "mux" if c.startswith("mux") else "flop" if re.match(r"(df|dl|sdf|edf)", c) else
               "buf/inv" if re.match(r"(buf|inv|clkbuf|clkinv|dlygate|clkdly)", c) else "logic")
        groups[key] += 1
    fan = sorted(((int(k), n) for n, k in re.findall(r"@F (\S+) (\d+)", out)), reverse=True)[:3]
    total = sum(groups.values())
    return (f"{total} cells: " + ", ".join(f"{v} {k}" for k, v in groups.items()) +
            " | high fanout: " + (", ".join(f"{n} ({k})" for k, n in fan) or "none"))


def run_script(design, prep, actions):
    """Run a fixed action list in one episode; returns a result row."""
    ep = Episode(design, prep)
    total, info, used = 0.0, {}, []
    for a in actions:
        r, info = ep.step(a)
        total += r
        used.append(a)
        if info.get("done"):
            break
    if not info.get("done"):
        r, info = ep.step({"op": "finish"})
        total += r
    row = {"P": ep.P, "hold": ep.m["hold"], "area": ep.m["area"], "tool_s": round(ep.tool_s, 1),
           "actions": ep.k, "success": info.get("success"), "return": round(total, 3), "power": info.get("power")}
    ep.close()
    return row


DEFAULT = [{"op": "repair_timing", "margin": 0.05, "buffer_pct": 50, "cloning": True}]
MAX_EFFORT = [{"op": "repair_timing", "margin": 0.3, "buffer_pct": 100, "cloning": True},
              {"op": "route_repair", "margin": 0.2}]


def prep(design, force=False):
    """Default run (if missing), post-CTS checkpoint, markers, default and max-effort results, target."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"prep_{design}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    if _sh(f"ls -d {base_dir(design)}/*-openroad-cts").returncode != 0:
        run_flow(design, {}, tag="base", keep=True, note="agent base (default flow)")
    cts = _sh(f"ls {base_dir(design)}/*-openroad-cts/{design}.odb").stdout.strip()
    cfg = json.loads((REPO / "flows" / "designs" / design / "config.json").read_text())
    p = {"design": design, "cts_odb": cts, "clock": cfg["CLOCK_PERIOD"]}
    s = Session(design, "base", odb=cts)
    out, _, _ = s.cmd("set r [[ord::get_db_block] getDieArea]; puts \"@D [$r dx] [$r dy]\"")
    d = re.search(r"@D (\d+) (\d+)", out)
    p["die"] = (int(d[1]) / 1000, int(d[2]) / 1000)
    p["markers"] = markers(s)
    m = metrics(s, False)
    p["P_start"], p["area_start"] = period(p["clock"], m["wns"]), m["area"]
    s.close()
    p["target"] = p["P_start"]  # placeholder so the scripts' rewards are defined
    p["default"] = run_script(design, p, DEFAULT)
    p["max_effort"] = run_script(design, p, MAX_EFFORT)
    # Target = the best scripted result (rounded up to 10 ps), so the baselines just make it:
    # the agent wins by reaching it in fewer actions / less tool time, or by beating it.
    p["target"] = math.ceil(100 * min(p["max_effort"]["P"], p["default"]["P"])) / 100
    for k in ("default", "max_effort"):  # scripts ran with a placeholder target: recompute success
        p[k]["success"] = p[k]["P"] <= p["target"] and p[k]["hold"] >= 0 and p[k]["area"] <= 1.25 * p["area_start"]
        p[k].pop("return", None)
    p.pop("_variants", None)
    f.write_text(json.dumps(p, indent=1))
    return p


def load_prep(design):
    return json.loads((OUT / f"prep_{design}.json").read_text())


# ---------------------------------------------------------------- parallel branches

def run_branches(ep, actions, tag):
    """Apply each action to its own copy of ep's state, in parallel. Returns [(reward, info, episode)]."""
    copies = [ep.branch(f"{base_dir(ep.design)}/agent_tmp/{tag}_{i}.odb") for i in range(len(actions))]
    res = [None] * len(actions)

    def work(i):
        try:
            r, info = copies[i].step(actions[i])
        except Exception as e:  # a crashed session counts as an invalid action
            r, info = -0.2, {"valid": False, "error": repr(e)[:200], "done": True}
        res[i] = (r, info, copies[i])

    ts = [threading.Thread(target=work, args=(i,)) for i in range(len(actions))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


def random_action(ep, rng):
    op = rng.choice(list(MENU))
    if op == "repair_timing":
        return {"op": op, "margin": round(rng.uniform(0, 0.5), 2), "buffer_pct": rng.randint(10, 100),
                "cloning": rng.random() < 0.5}
    if op == "route_repair":
        return {"op": op, "margin": round(rng.uniform(0, 0.5), 2)}
    if op == "resize":
        st = rng.choice([st for p in ep.paths for st in p["stages"]])
        return {"op": op, "cell": st["inst"], "size": rng.choice(["up", "down"])}
    if op in ("upsize_path", "pull_path"):
        return {"op": op, "path": f"P{rng.randint(1, max(1, len(ep.paths)))}"}
    return {"op": op}

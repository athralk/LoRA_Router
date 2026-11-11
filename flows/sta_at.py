"""Re-time a finished layout at other clock periods (signoff check for the frequency claim).

Loads final/odb + the extracted parasitics of a kept run (run_flow.run(..., keep=True))
into OpenSTA with the ss / tt / ff liberty files, rewrites the SDC for each period P
(clock period P; input and output delays scaled with P, as OpenLane does: 20% of the
period), and reports setup slack (ss), hold slack (ff) and power (tt) at P. The same
layout is checked, so this verifies "achievable period = target - WNS".

    python -m flows.sta_at mac16 best_freq 6.57 13.67 13.8
"""
import base64
import re
import sys

from .run_flow import IMAGE, WSL_RUNS, _sh

LIB = ("$(ls -d $HOME/.volare/volare/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/lib | head -1)"
       "/sky130_fd_sc_hd__{}.lib")
CORNERS = {"ss": ("ss_100C_1v60", "max"), "tt": ("tt_025C_1v80", "nom"), "ff": ("ff_n40C_1v95", "min")}

TCL = r"""
define_corners ss tt ff
foreach c {ss tt ff} { read_liberty -corner $c $::env(LIB_$c) }
read_db $::env(RUN)/final/odb/$::env(DESIGN).odb
foreach {c s} {ss max tt nom ff min} { read_spef -corner $c $::env(RUN)/final/spef/$s/$::env(DESIGN).$s.spef }
set f [open $::env(RUN)/final/sdc/$::env(DESIGN).sdc]; set sdc [read $f]; close $f
regexp {create_clock[^\n]*-period ([0-9.]+)} $sdc -> p0
foreach P $::env(PERIODS) {
    set s [regsub -all -- {-add_delay } $sdc ""]
    set s [regsub {(create_clock[^\n]*-period )[0-9.]+} $s "\\1$P"]
    set out ""
    foreach line [split $s "\n"] {
        if {[regexp {^(set_(input|output)_delay )([0-9.]+)( .*)$} $line -> a - v b]} {
            set line "$a[format %.4f [expr {$v * $P / $p0}]]$b"
        }
        append out $line "\n"
    }
    set g [open /tmp/sta_$P.sdc w]; puts $g $out; close $g
    read_sdc /tmp/sta_$P.sdc
    puts "RESULT P=$P"
    puts "SETUP_SS"; report_checks -corner ss -path_delay max -digits 4
    puts "HOLD_FF"; report_checks -corner ff -path_delay min -digits 4
    puts "POWER_TT"; report_power -corner tt
}
"""


def sta_at(design, tag, periods):
    run = f"{WSL_RUNS}/{design}/runs/{tag}"
    tcl = base64.b64encode(TCL.encode()).decode()
    env = " ".join(f"-e LIB_{c}={LIB.format(lib)}" for c, (lib, _) in CORNERS.items())
    cmd = (f"echo {tcl} | base64 -d > {run}/sta_at.tcl && "
           f"docker run --rm -u $(id -u):$(id -g) -v $HOME:$HOME -e HOME=$HOME {env} "
           f"-e RUN={run} -e DESIGN={design} -e PERIODS='{' '.join(map(str, periods))}' {IMAGE} "
           f"openroad -no_splash -exit {run}/sta_at.tcl 2>&1")
    out = _sh(cmd).stdout
    res = []
    for blk in out.split("RESULT P=")[1:]:
        p = float(blk.split()[0])
        setup, hold, power = (blk.split(k, 1)[1] if k in blk else "" for k in ("SETUP_SS", "HOLD_FF", "POWER_TT"))
        setup = setup.split("HOLD_FF")[0]
        hold = hold.split("POWER_TT")[0]
        slack = lambda t: (float(m[1]) if (m := re.search(r"(-?[\d.]+)\s+slack \((?:MET|VIOLATED)\)", t)) else None)
        pw = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", power, re.M)
        res.append({"period": p, "setup_ws_ss": slack(setup), "hold_ws_ff": slack(hold),
                    "power_tt_mW": round(float(pw[4]) * 1000, 3) if pw else None})
    if not res or res[0]["setup_ws_ss"] is None:
        raise RuntimeError(out[-3000:])
    return res


if __name__ == "__main__":
    for r in sta_at(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]):
        print(r)

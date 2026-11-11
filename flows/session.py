"""A live OpenROAD session: the design stays loaded, commands are sent one at a time.

The session runs OpenROAD inside the OpenLane Docker image and loads the design exactly
as OpenLane's own scripts do (io.tcl, resizer corners, layer RC). Each command is sent
base64-encoded (no quoting issues) and the reply carries its output, return code and
OpenROAD-side runtime, so tool time can be separated from Windows -> WSL -> Docker overhead.

    s = Session("mac16", "base")           # a kept full default run: its IO-placement
                                           # checkpoint is the start, all step settings merged
    out, ok, ms = s.cmd("report_wns")
    s.close()
"""
import base64
import subprocess
import time

from .run_flow import IMAGE, WSL_RUNS

REPL = r"""
set ::env(_TCL_ENV_IN) $::env(STEP)/_env.tcl
source $::env(SCRIPTS_DIR)/openroad/common/io.tcl
# Each step's _env.tcl only holds that step's settings: merge all steps of the (full) run
foreach f [lsort [glob -nocomplain [file dirname $::env(STEP)]/*/_env.tcl]] { source $f }
set ::env(CURRENT_ODB) [expr {$::env(ODB) ne "" ? $::env(ODB) : "$::env(STEP)/$::env(DESIGN).odb"}]
set ::env(STEP_DIR) [file dirname $::env(STEP)]/agent_tmp
file mkdir $::env(STEP_DIR)
# Variables OpenLane's Python sets in the process environment (steps/openroad.py)
proc _libs_for {corner} {
    set out {}
    foreach {pat libs} $::env(LIB) { if {[string match $pat $corner]} { lappend out {*}$libs } }
    return $out
}
set ::env(_PNR_LIBS) [_libs_for $::env(DEFAULT_CORNER)]
set ::env(_MACRO_LIBS) ""
set ::env(_SDC_IN) $::env(FALLBACK_SDC_FILE)
set f [open $::env(PNR_EXCLUDED_CELL_FILE)]
set cells {}
foreach l [split [read $f] "\n"] {
    set l [string trim $l]
    if {$l ne "" && ![string match "#*" $l]} { lappend cells $l }
}
close $f
set ::env(_PNR_EXCLUDED_CELLS) $cells
set i 0
foreach c {nom_tt_025C_1v80 nom_ss_100C_1v60 nom_ff_n40C_1v95} {
    set ::env(RSZ_CORNER_$i) [concat $c [_libs_for $c]]
    incr i
}
if {![info exists ::env(RSZ_DONT_TOUCH_RX)]} { set ::env(RSZ_DONT_TOUCH_RX) {$^} }
source $::env(SCRIPTS_DIR)/openroad/common/resizer.tcl
load_rsz_corners
read_current_odb
set_dont_touch_objects
source $::env(SCRIPTS_DIR)/openroad/common/set_rc.tcl
fconfigure stdout -buffering line
puts "@@READY"
while {[gets stdin line] >= 0} {
    set t [clock milliseconds]
    set rc [catch {uplevel #0 [binary decode base64 $line]} res]
    puts "\n@@RES $rc [expr {[clock milliseconds] - $t}] [binary encode base64 $res]"
    flush stdout
}
"""


class Session:
    def __init__(self, design, tag, step_glob="*-openroad-ioplacement", odb=""):
        """odb: a checkpoint to start from instead of the IO-placement database (WSL path)."""
        run = f"{WSL_RUNS}/{design}/runs/{tag}"
        repl = base64.b64encode(REPL.encode()).decode()
        boot = (
            f"STEP=$(ls -d {run}/{step_glob} | tail -1) && echo {repl} | base64 -d > {run}/repl.tcl && "
            f"docker run -i --rm -u $(id -u):$(id -g) -v $HOME:$HOME -e HOME=$HOME "
            f"-e STEP=$STEP -e DESIGN={design} -e ODB={odb} {IMAGE} bash -c "
            "'export SCRIPTS_DIR=$(python3 -c \"import openlane,os;print(os.path.dirname(openlane.__file__))\")/scripts; "
            f"openroad -no_splash -exit {run}/repl.tcl' 2>&1"
        )
        b64 = base64.b64encode(boot.encode()).decode()
        t0 = time.time()
        # eval, not "| bash": the decoded command must inherit this process's stdin
        self.p = subprocess.Popen(["wsl", "-e", "bash", "-lc", f'eval "$(echo {b64} | base64 -d)"'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                  encoding="utf-8", errors="replace")
        self.boot_log = self._read_until("@@READY")
        self.boot_s = time.time() - t0
        self.tool_ms = 0      # total OpenROAD-side time
        self.wall_s = 0.0     # total round-trip time
        self.calls = 0

    def _read_until(self, marker):
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("OpenROAD session ended:\n" + "".join(lines[-40:]))
            if line.startswith(marker):
                return "".join(lines), line
            lines.append(line)

    def cmd(self, tcl):
        """Run one Tcl command (may be multi-line). Returns (output, ok, tool_ms)."""
        t0 = time.time()
        self.p.stdin.write(base64.b64encode(tcl.encode()).decode() + "\n")
        self.p.stdin.flush()
        out, res = self._read_until("@@RES")
        _, rc, ms, *val = res.split()
        val = base64.b64decode(val[0]).decode(errors="replace") if val else ""
        self.tool_ms += int(ms)
        self.wall_s += time.time() - t0
        self.calls += 1
        return out + val, rc == "0", int(ms)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()

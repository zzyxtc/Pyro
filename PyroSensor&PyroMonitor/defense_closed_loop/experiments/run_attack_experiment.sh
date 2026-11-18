#!/usr/bin/env bash
# Run one experiment configuration: E1-Covert_Channel attack (+ optional Pyro
# monitoring / mitigation) with archived outputs.
#
# Configurations (config matrix):
#   attack_only              : sender + receiver only (baseline, no Pyro)
#   attack_monitor           : open loop  -> telemetry + predictions, no actuation (CONTROL arm)
#   attack_monitor_mitigate  : closed loop -> clamp frequency on predicted thermal event (TREATMENT arm)
#
# attack_monitor and attack_monitor_mitigate are the two arms of the paired causal
# comparison. Running them with the same --seed makes the sender emit an identical
# bit sequence, so the only independent variable is whether actuation happened.
#
# All outputs land in <outdir>. Nothing is deleted outside <outdir>.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYRO_DIR="$REPO_ROOT/Pyro"
E1_DIR="$REPO_ROOT/E1-Covert_Channel"

# shellcheck source=env_lib.sh
. "$EXP_DIR/env_lib.sh"
PYTHON="$(resolve_python)"

usage() {
  cat <<'EOF'
Usage: run_attack_experiment.sh --config <name> --cycles N [options]

Required:
  --config NAME     attack_only | attack_monitor | attack_monitor_mitigate
  --cycles N        number of bits to transmit (each bit is ~10 s)

Options:
  --outdir DIR      output directory (default: ./outputs/<config>/<timestamp>)
  --tthres C        thermal-event LABEL threshold in Celsius (default: 90)
  --seed N          bit-sequence RNG seed; shared by both arms so the paired
                    control/treatment runs transmit an identical sequence (default: 12345)
  --sender-core N   pin the sender to CPU N so the heated core is reproducible
                    (default: 0; pass 'none' to leave scheduling to the kernel)
  --cpus LIST       FALLBACK mitigation CPUs, used only when the core that
                    crossed the threshold is unknown. Empty (default) = throttle
                    the triggering core itself.
  --attack-core N   informational: the CPU the receiver pins itself to
                    (must match VICTIM_CORE in receiver-*.c; default: 3)
  --receiver NAME   irq | throttle (default: irq)
  --duration S      telemetry duration; default: cycles*10 + 60

Mitigation strength (closed-loop configs only):
  --mitigate-margin C  actuate when the predicted temperature exceeds
                       (--tthres - C). Default 10 (i.e. start at 80 C). This is the
                       dominant lever for *coverage*: measured on real traces,
                       margin 0 -> 60% of the attack window covered, margin 10 -> 94%.
  --target MODE        'base' = stop turboing, 'min' = lowest allowed frequency.
                       Default: min. The concrete knob depends on the backend.
  --perf-pct N         explicit intel_pstate/max_perf_pct cap depth (17-100 on most
                       parts). Overrides the depth implied by --target and forces the
                       intel_pstate_pct backend. Use this to sweep how strong the cap
                       must be before the attack is actually suppressed.
  --cooldown S         how long a mitigation stays applied (default: 25)
  --min-dwell S        minimum hold before a throttle may be released (default: 10)
  --min-interval S     minimum time between two throttles (default: 0.5; measured
                       to have no effect on coverage -- the trigger rate dominates)

Mitigation backend (closed-loop configs only):
  --backend NAME       auto | intel_pstate_pct | none     (default: auto)
                       The controller uses intel_pstate/max_perf_pct, verified
                       UNDER LOAD so a writable-but-ineffective knob cannot
                       silently invalidate the experiment.
  --no-probe           skip the under-load capability probe (debugging only)
  --probe-cpu N        CPU used by the probe (default: first monitored CPU)
  --verify-interval S  read-back verification period in seconds (default: 2, 0 = off)
  --reassert-interval S  re-write the knob every S seconds to counter external
                       overwrites such as thermald (default: 2, 0 = off)

Warm-up:
  --no-warmup          do not hold the attacker back until the first prediction is
                       available. By default the sender starts only after the model
                       can act, so no bit window runs unprotected (the model needs
                       ~12 s of telemetry, which is more than one 10 s bit period).
  --warmup-timeout S   give up waiting for the first prediction after S s (default 90)

  --no-build        skip rebuilding the E1 binaries and the kernel module
EOF
}

CONFIG=""
CYCLES=""
OUTDIR=""
TTHRES=90
SEED=12345
SENDER_CORE=0
CPUS=""
ATTACK_CORE=3
RECEIVER="irq"
DURATION=""
TARGET="min"
PERF_PCT=""          # explicit intel_pstate/max_perf_pct cap depth (empty = derive from --target)
COOLDOWN=25
MIN_DWELL=10
MIN_INTERVAL=0.5
MITIGATE_MARGIN=10
BACKEND="auto"
NO_PROBE=0
PROBE_CPU=""
VERIFY_INTERVAL=2.0
REASSERT_INTERVAL=2.0
WARMUP=1
WARMUP_TIMEOUT=90
DO_BUILD=1

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --tthres) TTHRES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sender-core) SENDER_CORE="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --attack-core) ATTACK_CORE="$2"; shift 2 ;;
    --receiver) RECEIVER="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --perf-pct) PERF_PCT="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --min-dwell) MIN_DWELL="$2"; shift 2 ;;
    --min-interval) MIN_INTERVAL="$2"; shift 2 ;;
    --mitigate-margin) MITIGATE_MARGIN="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --no-probe) NO_PROBE=1; shift ;;
    --probe-cpu) PROBE_CPU="$2"; shift 2 ;;
    --verify-interval) VERIFY_INTERVAL="$2"; shift 2 ;;
    --reassert-interval) REASSERT_INTERVAL="$2"; shift 2 ;;
    --no-warmup) WARMUP=0; shift ;;
    --warmup-timeout) WARMUP_TIMEOUT="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$CONFIG" in
  attack_only|attack_monitor|attack_monitor_mitigate) ;;
  "") echo "error: --config is required" >&2; usage; exit 2 ;;
  *) echo "error: invalid --config '$CONFIG'" >&2; usage; exit 2 ;;
esac
if [ -z "$CYCLES" ]; then echo "error: --cycles is required" >&2; usage; exit 2; fi
case "$RECEIVER" in irq|throttle) ;; *) echo "error: --receiver must be irq or throttle" >&2; exit 2 ;; esac
case "$TARGET" in base|min) ;; *) echo "error: --target must be base or min" >&2; exit 2 ;; esac
if [ -n "$PERF_PCT" ]; then
  case "$PERF_PCT" in *[!0-9]*) echo "error: --perf-pct must be an integer" >&2; exit 2 ;; esac
fi

if [ -z "$OUTDIR" ]; then
  OUTDIR="$EXP_DIR/outputs/$CONFIG/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUTDIR"
# Make it absolute: later we run attack programs from inside E1_DIR in a
# subshell, and relative redirect targets would resolve against E1_DIR.
OUTDIR="$(cd "$OUTDIR" && pwd)"

if [ -z "$DURATION" ]; then
  # +90 s of slack: model load plus the warm-up wait for the first prediction.
  DURATION=$((CYCLES * 10 + 90))
fi
# NOTE: CPUS is intentionally NOT defaulted to the attack core. The E1 sender is
# unpinned in the original code, so the core that actually heats up is not the
# core the receiver sits on. Mitigation therefore targets the core that crossed
# the predicted threshold (handled inside attack_experiment.py); CPUS is only a
# fallback.

MONITOR=0
MODE="none"          # only attack_only leaves this unset; it starts no collector
case "$CONFIG" in
  attack_monitor) MONITOR=1; MODE="detect" ;;
  attack_monitor_mitigate) MONITOR=1; MODE="mitigate" ;;
esac

if [ "$MONITOR" -eq 1 ] && [ "$(id -u)" -ne 0 ]; then
  echo "error: config '$CONFIG' needs root (insmod/rmmod + cpufreq writes)" >&2
  exit 1
fi

echo "== Pyro attack experiment =="
echo "  config     : $CONFIG (mode=$MODE)"
echo "  cycles     : $CYCLES  (~$((CYCLES * 10)) s)"
echo "  receiver   : $RECEIVER  (pinned to CPU $ATTACK_CORE by the receiver itself)"
if [ -z "$SENDER_CORE" ] || [ "$SENDER_CORE" = "none" ]; then
  echo "  sender core: (unpinned)"
else
  echo "  sender core: CPU $SENDER_CORE (pinned)"
fi
echo "  mitig. cpus: ${CPUS:-(auto: the core that crossed the threshold; ignored by global backends)}"
echo "  mitigation : backend=$BACKEND target=$TARGET cooldown=${COOLDOWN}s dwell=${MIN_DWELL}s interval=${MIN_INTERVAL}s"
echo "               verify every ${VERIFY_INTERVAL}s, re-assert every ${REASSERT_INTERVAL}s"
echo "               trigger when predicted > $(awk "BEGIN{print $TTHRES-$MITIGATE_MARGIN}") C (tthres ${TTHRES} - margin ${MITIGATE_MARGIN})"
if [ "$WARMUP" -eq 1 ]; then
  echo "  warmup     : on (attacker held back until the first prediction)"
else
  echo "  warmup     : off"
fi
echo "  tthres     : $TTHRES C   seed: $SEED"
echo "  outdir     : $OUTDIR"

COLLECTOR_PID=""
cleanup() {
  if [ -n "$COLLECTOR_PID" ] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    kill -INT "$COLLECTOR_PID" 2>/dev/null || true
    wait "$COLLECTOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Build with a real `cd` into the target directory: Pyro/Makefile relies on
# `M=$(PWD)`, and `$(PWD)` is the directory make was *launched* from, not the
# one set by `make -C`. When running under sudo we build as SUDO_USER so the
# repository does not end up with root-owned object files.
build_in_dir() {  # build_in_dir <dir> <logfile>
  local dir="$1" log="$2"
  if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ] \
     && command -v runuser >/dev/null 2>&1; then
    runuser -u "$SUDO_USER" -- bash -c "cd \"$dir\" && make" >"$log" 2>&1
  else
    ( cd "$dir" && make ) >"$log" 2>&1
  fi
}

if [ "$DO_BUILD" -eq 1 ]; then
  echo "-- building E1 attack programs --"
  build_in_dir "$E1_DIR" "$OUTDIR/build-e1.log" \
    || { echo "error: E1 build failed (see $OUTDIR/build-e1.log)"; exit 1; }
  if [ "$MONITOR" -eq 1 ]; then
    echo "-- building Pyro kernel module --"
    build_in_dir "$PYRO_DIR" "$OUTDIR/build-pyro.log" \
      || { echo "error: Pyro build failed (see $OUTDIR/build-pyro.log)"; exit 1; }
  fi
fi

for bin in sender "$RECEIVER" err; do
  if [ ! -x "$E1_DIR/$bin" ]; then echo "error: $E1_DIR/$bin missing; build first" >&2; exit 1; fi
done

if [ "$MONITOR" -eq 1 ]; then
  if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "error: '$PYTHON' cannot import torch." >&2
    echo "       Set PYTHON_BIN=/path/to/python3 (e.g. your conda env) and retry." >&2
    exit 1
  fi
  echo "-- starting collector (mode=$MODE, duration=${DURATION}s, python=$PYTHON) --"
  EXTRA_ARGS=()
  [ "$NO_PROBE" -eq 1 ] && EXTRA_ARGS+=(--no-probe)
  [ -n "$PROBE_CPU" ] && EXTRA_ARGS+=(--probe-cpu "$PROBE_CPU")
  [ "$WARMUP" -eq 0 ] && EXTRA_ARGS+=(--no-warmup)
  [ -n "$PERF_PCT" ] && EXTRA_ARGS+=(--perf-pct "$PERF_PCT")
  # -u: unbuffered stdout, otherwise the readiness marker never reaches the log
  # until the process exits (stdout is block-buffered when redirected to a file).
  # -B: do not write .pyc files, otherwise running as root leaves root-owned
  # __pycache__ directories inside the repository.
  "$PYTHON" -u -B "$EXP_DIR/attack_experiment.py" \
    --mode "$MODE" \
    --duration "$DURATION" \
    --out "$OUTDIR/telemetry.jsonl" \
    --stats "$OUTDIR/telemetry.stats.json" \
    --tthres "$TTHRES" \
    --mitigate-margin "$MITIGATE_MARGIN" \
    --cpus "$CPUS" \
    --target "$TARGET" \
    --cooldown "$COOLDOWN" \
    --min-dwell "$MIN_DWELL" \
    --min-interval "$MIN_INTERVAL" \
    --backend "$BACKEND" \
    --verify-interval "$VERIFY_INTERVAL" \
    --reassert-interval "$REASSERT_INTERVAL" \
    --warmup-timeout "$WARMUP_TIMEOUT" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    >"$OUTDIR/collector.log" 2>&1 &
  COLLECTOR_PID=$!
  # Wait for the warm-up marker rather than the startup banner: the model needs
  # ~12 s of telemetry (more than one 10 s bit period) before it can act, and the
  # attacker must stay idle until then or the first bits run unprotected.
  WAIT_S=$(( WARMUP_TIMEOUT + 60 ))
  echo "-- waiting for warm-up (torch + model load + first prediction; attacker idle) --"
  ready=0
  for _ in $(seq 1 "$WAIT_S"); do
    if ! kill -0 "$COLLECTOR_PID" 2>/dev/null; then
      echo "error: collector exited early (see $OUTDIR/collector.log)" >&2
      exit 1
    fi
    if grep -q "attack_experiment: warmup complete" "$OUTDIR/collector.log" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo "warning: warm-up not confirmed within ${WAIT_S}s; continuing anyway" >&2
  fi
fi

if [ -n "$SENDER_CORE" ] && [ "$SENDER_CORE" != "none" ]; then
  echo "-- starting sender (background, pinned to CPU $SENDER_CORE) --"
  ( cd "$E1_DIR" && ./sender "$CYCLES" "$SEED" "$SENDER_CORE" >"$OUTDIR/sender.log" 2>&1 ) &
else
  echo "-- starting sender (background, unpinned) --"
  ( cd "$E1_DIR" && ./sender "$CYCLES" "$SEED" >"$OUTDIR/sender.log" 2>&1 ) &
fi
SENDER_PID=$!
sleep 1

echo "-- starting receiver (foreground, ~${CYCLES}0 s) --"
( cd "$E1_DIR" && ./"$RECEIVER" "$CYCLES" >"$OUTDIR/receiver.log" 2>&1 )
RECEIVER_RC=$?

echo "-- waiting for sender --"
wait "$SENDER_PID" 2>/dev/null || true

if [ "$MONITOR" -eq 1 ]; then
  echo "-- stopping collector --"
  cleanup
  COLLECTOR_PID=""
fi

echo "-- archiving raw signals --"
for f in send_signal receive_signal send_timeline.csv receive_log.csv; do
  if [ -f "$E1_DIR/$f" ]; then cp -f "$E1_DIR/$f" "$OUTDIR/$f"; fi
done

echo "-- computing BER --"
( cd "$E1_DIR" && ./err "$CYCLES" "$OUTDIR/send_signal" "$OUTDIR/receive_signal" ) \
  >"$OUTDIR/err.log" 2>&1 || true
cat "$OUTDIR/err.log"

# Hand the outputs back to the invoking user so cleanup does not need sudo.
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  chown -R "$SUDO_USER" "$OUTDIR" 2>/dev/null || true
fi

echo
echo "== finished =="
echo "  receiver rc : $RECEIVER_RC"
echo "  outputs     : $OUTDIR"
ls -1 "$OUTDIR" | sed 's/^/    /'
exit 0

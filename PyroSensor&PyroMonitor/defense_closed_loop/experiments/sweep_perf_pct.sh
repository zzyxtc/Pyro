#!/usr/bin/env bash
# Sweep the intel_pstate/max_perf_pct cap depth and measure how strongly
# PREDICTION-TRIGGERED mitigation suppresses the E1 covert channel.
#
# The question this answers: the controller decides WHEN to cap (that part is
# fixed by --mitigate-margin); the percentage decides HOW HARD. How does the
# suppression scale with the depth of the cap?
#
# Layout per repetition r:
#
#   control            (attack_monitor)      run ONCE -- no actuation, so it does
#                                            not depend on the cap depth
#   pct100/predict     (attack_monitor_mitigate --perf-pct 100)
#   pct080/predict     ...
#   ...
#
# Every predict run uses the SAME seed as the control run of its repetition, so
# each point on the curve is a paired comparison against an identical bit
# sequence. The control arm is shared across all percentages, which is what makes
# the curve comparable point-to-point.
#
# pct=100 is a useful null point: the knob is written but caps nothing, so it
# should reproduce the control arm's BER. If it does not, the sweep is suspect.
#
# Usage:
#   sudo ./sweep_perf_pct.sh [CYCLES] [REPEATS] [SEED_BASE] [SENDER_CORE] \
#                            [MARGIN] [COOLDOWN] ["PCT1 PCT2 ..."] [MIN_DWELL]
#   defaults: 15 2 7000 0 25 25 "100 80 58 35 20" 10
#
#   MARGIN   trigger threshold = tthres(90) - MARGIN, i.e. 25 -> fire above 65 C.
#            This is the "when" knob and should be fixed while sweeping the "how
#            hard" knob, otherwise the two effects are confounded.
#            MARGIN is forwarded to BOTH arms. The control arm (attack_monitor)
#            never actuates, but it does record `mitigation_triggers` at the same
#            threshold -- that is what makes "would the controller have fired here?"
#            answerable and the two arms comparable.
#            Note the trigger is evaluated on the PREDICTION only
#            (attack_experiment.py: any(prediction > tthres - margin)), never on
#            the measured temperature, so the lead time comes from the 20-step
#            prediction horizon; MARGIN only shifts the threshold along the
#            predicted trajectory.
#   COOLDOWN hold time REQUESTED for each cap. The realised hold is
#            max(COOLDOWN, MIN_DWELL), so lowering COOLDOWN below MIN_DWELL is
#            silently inert (cpufreq_mitigation releases only when BOTH
#            now >= active_until AND held >= min_dwell).
#   MIN_DWELL minimum hold before a throttle may be released, forwarded to
#            run_attack_experiment.sh as --min-dwell, whose default is 10. Kept at
#            10 by default here so runs stay comparable with the earlier sweeps
#            (which could not override it). Lower BOTH to shorten the duty cycle.
#   PCTS     space separated intel_pstate/max_perf_pct values. The kernel exposes
#            min_perf_pct as a floor (measured 17 on this machine), so values
#            below it are clamped -- the script reports the achieved percentage.
#
# Cost: (REPEATS x (1 + len(PCTS))) runs, each about CYCLES*10+90 s plus startup.
#
# Environment variables (both optional):
#   OUTDIR   output directory; default $EXP_DIR/outputs/sweep_perf_pct. A pre-existing
#            non-empty directory is archived to <OUTDIR>_<timestamp>, never deleted.
#   TAG      suffix appended to OUTDIR, e.g. TAG=m20 with the default OUTDIR gives
#            outputs/sweep_perf_pct_m20. Keeps different margins side by side.
#            The m<N> form is checked against MARGIN and the run is refused on a
#            mismatch (TAG=m20 with margin 5 would otherwise label 1.5 h of results
#            with the wrong configuration). Use any other spelling to opt out.
#
# Guards and diagnostics built in (added after the 2026-09 session, in which a stale
# kernel module and a silently broken predictor each wasted a full sweep):
#   * refuses to start while a stale Pyro.ko is loaded -- every collector would
#     otherwise die at insmod with "File exists", and that failure is self-locking
#     because rmmod only runs on the normal-exit path;
#   * refuses to start unless the deployed predictor carries the merge-window
#     smoothing fix (savgol_filter mode='interp'). Without it the online path
#     zero-pads every 30-sample window and under-predicts peaks by ~30 C, which
#     silently turns the depth sweep into noise;
#   * counts runs that produced no telemetry, reports them, and exits non-zero;
#   * predictor sanity check on the RECORDED predictions (core 0): pre-fix they
#     never exceeded 79.2 C, post-fix they should reach ~93-98 C;
#   * per-RUN diagnostic table and csv -- a single run with 0 triggers is invisible
#     in the per-percentage means this script prints by default.
#
# Artifacts, all inside OUTDIR:
#   sweep.config.json    authoritative parameters (the directory NAME is free text,
#                        so it must not be the only record of what was run)
#   sweep.log            full stdout/stderr transcript
#   sweep.summary.csv    the SWEEP SUMMARY table
#   sweep.per_run.csv    per-run diagnostics (cap_effective flags the pct=100 null
#                        point and the control arm, whose att_win is not a
#                        protection rate)
#   control/runN, pct<N>/predict/runN, pct<N>/eval/   raw experiment output
#
# Post-fix calibration: after the smoothing fix the crossing rate at margin 20
# (70 C) is ~50% per decision instant (pre-fix: 14%), so triggering saturates and
# coverage becomes ~100% for every pct -- which is exactly what isolates clamp
# DEPTH as the only remaining variable. To study trigger density instead, use
# MARGIN 0-5 (90-85 C), where the post-fix rate is 22.9%/31.4% and stays below the
# cooldown ceiling. A pct set spanning both the thermal plateau and the collapse
# region is recommended, e.g.
#   sudo TAG=m20 ./sweep_perf_pct.sh 15 3 7000 0 20 25 "100 80 60 40 20"

set -u

CYCLES="${1:-15}"
REPEATS="${2:-2}"
SEED_BASE="${3:-7000}"
SENDER_CORE="${4:-0}"
MARGIN="${5:-25}"
COOLDOWN="${6:-25}"
PCTS="${7:-100 80 58 35 20}"
MIN_DWELL="${8:-10}"     # 10 = run_attack_experiment.sh's default, kept for continuity
TTHRES=90
TARGET=min          # perf_pct forces the intel_pstate_pct backend anyway

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYRO_DIR="$EXP_DIR/../../../Pyro"
PREDICTOR="$PYRO_DIR/PyroNet_predict.py"

# Which checkpoint will the runs load? attack_experiment.py owns that decision (it is
# env-overridable via PYRONET_MODEL_DIR), so ask it instead of duplicating the path
# here -- otherwise the guard below could bless a stale directory. Falls back to the
# historical location if the import fails for any reason.
MODEL_DIR="$(PYRONET_MODEL_DIR="${PYRONET_MODEL_DIR:-}" python3 -c "
import os, sys
sys.path.insert(0, '$EXP_DIR')
import attack_experiment as ae
print(ae._DEFAULT_MODEL_DIR)
" 2>/dev/null)"
[ -n "$MODEL_DIR" ] || MODEL_DIR="$PYRO_DIR/PyroNet_model"
MODEL="$MODEL_DIR/PyroNet.pth"

OUTDIR="${OUTDIR:-$EXP_DIR/outputs/sweep_perf_pct}"
[ -n "${TAG:-}" ] && OUTDIR="${OUTDIR}_${TAG}"
LOG="$OUTDIR/sweep.log"
SUMMARY_CSV="$OUTDIR/sweep.summary.csv"
RUN_CSV="$OUTDIR/sweep.per_run.csv"
CONFIG_JSON="$OUTDIR/sweep.config.json"

# The m<N> tag convention encodes the margin in the directory name, which makes the
# name the only durable label unless the parameters are written down too. Catch the
# mismatch up front: `TAG=m20 ... 5 25` would otherwise produce 1.5 h of results in a
# directory that claims the wrong configuration.
case "${TAG:-}" in
  m[0-9]*)
    # Only the leading integer run after `m` is the margin, so suffixes that name
    # another swept knob stay legal: m5c10 = margin 5, cooldown 10.
    tag_margin=$(printf '%s' "${TAG#m}" | sed 's/[^0-9].*//')
    if [ "$tag_margin" != "$MARGIN" ]; then
      echo "error: TAG=$TAG encodes margin $tag_margin, but MARGIN=$MARGIN." >&2
      echo "       Use TAG=m$MARGIN (optionally suffixed, e.g. m${MARGIN}c10), or a" >&2
      echo "       TAG that does not start with m<N>." >&2
      exit 1
    fi
    ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "error: needs root (insmod/rmmod + cpufreq writes)" >&2
  exit 1
fi

# run_attack_experiment.sh only mkdir -p's its --outdir, so re-running in place
# would merge stale telemetry into the new averages. Archive, never delete.
if [ -d "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]; then
  ARCHIVE="${OUTDIR}_$(date +%Y%m%d-%H%M%S)"
  echo "== previous sweep archived to $ARCHIVE =="
  mv "$OUTDIR" "$ARCHIVE" || { echo "error: cannot archive $OUTDIR" >&2; exit 1; }
fi
mkdir -p "$OUTDIR"

# Transcribe everything from here on into $LOG. It lives inside the freshly created
# OUTDIR, so it travels with the archived results instead of being overwritten by
# the next run. The output-directory handling above is deliberately NOT logged: it
# has to happen before the log file exists.
exec > >(tee "$LOG") 2>&1

# --- guard 1: a stale Pyro.ko blocks every collector at insmod ----------------
# attack_experiment.py only rmmod's on its normal-exit path, so a leftover module
# survives a crashed run and then kills every subsequent run too -- silently, since
# run_one merely printed "WARNING: run failed" and the sweep carried on.
if lsmod 2>/dev/null | grep -q '^Pyro[[:space:]]'; then
  echo "== stale Pyro.ko detected, removing it (otherwise every run dies at insmod) =="
  rmmod Pyro || { echo "error: cannot rmmod Pyro; resolve manually" >&2; exit 1; }
fi

# --- guard 2: the deployed predictor must carry the smoothing fix -------------
# Without it the online path smooths each 30-sample window with np.convolve(...,
# mode='same'), whose zero padding keeps only 0.5754 of the SG kernel mass inside
# the window: the newest sample is scaled to ~0.58x and a spurious +8% overshoot
# appears one sample earlier. Measured effect: peak bias -4.4 C -> -32.2 C and
# recall@70 0.865 -> 0.390, i.e. the whole sweep would measure a broken controller.
if [ -f "$PREDICTOR" ]; then
  if grep -q "savgol_filter" "$PREDICTOR" && ! grep -q "np.convolve(feature_col" "$PREDICTOR"; then
    echo "== predictor: merge-window smoothing fix present =="
  else
    echo "error: $PREDICTOR does not carry the merge-window smoothing fix." >&2
    echo "       expected savgol_filter(..., mode='interp') instead of" >&2
    echo "       np.convolve(feature_col, kernel, mode='same')." >&2
    echo "       Running anyway would measure a controller that under-predicts" >&2
    echo "       peaks by ~28 C. Refusing to start." >&2
    exit 1
  fi
fi
# --- guard 3: the checkpoint the runs will actually load ------------------------
# All three files come from MODEL_DIR, so check the whole set: a missing scalers.json
# or feature_columns.json makes every collector die at model-load time.
for f in "$MODEL" "$MODEL_DIR/scalers.json" "$MODEL_DIR/feature_columns.json"; do
  [ -f "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done
MODEL_SHA1="$(python3 -c "
import hashlib, sys
print(hashlib.sha1(open(sys.argv[1], 'rb').read()).hexdigest()[:16])
" "$MODEL")"
MODEL_SIZE="$(stat -c %s "$MODEL")"
MODEL_MTIME="$(stat -c %y "$MODEL" | cut -c1-19)"

# --- machine-readable provenance ---------------------------------------------
# The directory name is free text (TAG), so it must not be the only record of what
# was run. This file is authoritative and survives any later renaming.
python3 - "$CONFIG_JSON" "${TAG:-}" "$CYCLES" "$REPEATS" "$SEED_BASE" "$SENDER_CORE" \
         "$MARGIN" "$COOLDOWN" "$MIN_DWELL" "$PCTS" "$TTHRES" "$TARGET" \
         "$MODEL_DIR" "$MODEL" "$MODEL_SHA1" "$MODEL_SIZE" "$MODEL_MTIME" <<'PY'
import datetime, json, os, sys

(path, tag, cycles, repeats, seed_base, sender_core, margin, cooldown, min_dwell,
 pcts, tthres, target, model_dir, model_pth, model_sha1, model_size,
 model_mtime) = sys.argv[1:18]
cfg = {
    "script": "sweep_perf_pct.sh",
    "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "outdir": os.path.dirname(os.path.abspath(path)),
    "tag": tag,
    "model_dir": model_dir,
    "model_pth": os.path.realpath(model_pth),
    "model_sha1_16": model_sha1,
    "model_bytes": int(model_size),
    "model_mtime": model_mtime,
    "cycles": int(cycles),
    "repeats": int(repeats),
    "seed_base": int(seed_base),
    "sender_core": int(sender_core),
    "tthres": int(tthres),
    "mitigate_margin": int(margin),
    "trigger_threshold_c": int(tthres) - int(margin),
    "cooldown_s": int(cooldown),
    "min_dwell_s": int(min_dwell),
    # What the state machine actually holds for: cpufreq_mitigation only releases
    # once BOTH the cooldown has elapsed and min_dwell has been satisfied.
    "hold_seconds": max(int(cooldown), int(min_dwell)),
    "target": target,
    "pcts": pcts.split(),
}
with open(path, "w") as handle:
    json.dump(cfg, handle, indent=1)
print("sweep config  : {}  (margin={} -> trigger above {} C)".format(
    path, margin, int(tthres) - int(margin)))
print("model         : {}  sha1 {}".format(os.path.realpath(model_pth), model_sha1))
PY

pct_dir() { printf 'pct%03d' "$1"; }        # pct_dir 35 -> pct035 (sortable names)

FIRST=1
FAILED=0
FAILED_RUNS=()
run_one() {                                 # run_one <rel-outdir> <config> [extra args...]
  local rel="$1" cfg="$2"
  shift 2
  local extra=()
  [ "$FIRST" -eq 1 ] || extra+=(--no-build)   # only the very first run rebuilds
  FIRST=0
  "$EXP_DIR/run_attack_experiment.sh" \
    --config "$cfg" \
    --cycles "$CYCLES" \
    --sender-core "$SENDER_CORE" \
    --tthres "$TTHRES" \
    --outdir "$OUTDIR/$rel" \
    "${extra[@]+"${extra[@]}"}" "$@" \
    || echo "   WARNING: run_attack_experiment.sh returned non-zero"
  # The real success criterion is that the collector produced telemetry. A run that
  # died at insmod both exits non-zero and leaves nothing behind, and previously it
  # vanished into the per-percentage means without leaving a visible trace.
  if [ ! -f "$OUTDIR/$rel/telemetry.stats.json" ]; then
    FAILED=$((FAILED + 1))
    FAILED_RUNS+=("$rel")
    echo "   WARNING: no telemetry produced for $rel -- this run is unusable"
  fi
}

echo "=== max_perf_pct sweep: prediction-triggered cap depth vs attack suppression ==="
echo "  cycles=$CYCLES repeats=$REPEATS seed_base=$SEED_BASE sender_core=$SENDER_CORE"
echo "  trigger : predicted > $((TTHRES - MARGIN)) C   (tthres $TTHRES - margin $MARGIN)"
HOLD="$COOLDOWN"
[ "$MIN_DWELL" -le "$HOLD" ] || HOLD="$MIN_DWELL"
echo "  cap     : max_perf_pct in [$PCTS], cooldown=${COOLDOWN}s dwell=${MIN_DWELL}s -> hold=${HOLD}s"
if [ "$COOLDOWN" -le "$MIN_DWELL" ]; then
  echo "  !! COOLDOWN=${COOLDOWN}s <= MIN_DWELL=${MIN_DWELL}s: cpufreq_mitigation releases"
  echo "     only once BOTH have elapsed, so COOLDOWN currently changes nothing."
  echo "     Lower MIN_DWELL too (8th positional argument) to actually shorten the cap."
fi
echo "  control : attack_monitor (shared across all percentages)"
echo "  model   : $MODEL"
echo "            sha1 $MODEL_SHA1  ($MODEL_SIZE B, mtime $MODEL_MTIME)"
echo "  outdir  : $OUTDIR"
echo

for r in $(seq 1 "$REPEATS"); do
  seed=$(( SEED_BASE + r ))
  echo "### repetition $r / $REPEATS (seed=$seed) ###"

  echo "-- control (attack_monitor) run$r"
  # Forward the SAME margin to the control arm. attack_experiment.py evaluates the
  # trigger in BOTH modes precisely so the two arms can be compared on "would the
  # controller have fired here?" (see the comment above `mitigate_threshold`).
  # Without this the control arm silently fell back to run_attack_experiment.sh's
  # default (margin 10 -> 80 C) while every treatment run used $MARGIN, which made
  # `mitigation_triggers` incomparable between the arms -- harmless today because
  # nothing downstream reads the control trigger count, but a trap for anyone who
  # later analyses "would the controller have fired?".
  run_one "control/run$r" attack_monitor --seed "$seed" --mitigate-margin "$MARGIN"

  for pct in $PCTS; do
    echo "-- predict (max_perf_pct=$pct) run$r"
    run_one "$(pct_dir "$pct")/predict/run$r" attack_monitor_mitigate \
      --seed "$seed" --target "$TARGET" --perf-pct "$pct" \
      --cooldown "$COOLDOWN" --min-dwell "$MIN_DWELL" --mitigate-margin "$MARGIN" \
      --no-probe --verify-interval 2
  done
  echo
done

if [ "$FAILED" -gt 0 ]; then
  echo
  echo "!! $FAILED run(s) produced no telemetry and are missing from this sweep:"
  for rel in "${FAILED_RUNS[@]+"${FAILED_RUNS[@]}"}"; do echo "     - $rel"; done
  echo "!! everything below is computed on the remaining runs only."
  echo
fi

echo "=== aligning timelines ==="
for d in "$OUTDIR"/control/run* "$OUTDIR"/pct*/predict/run*; do
  [ -d "$d" ] || continue
  [ -f "$d/telemetry.jsonl" ] || { echo "  skip (no telemetry): $d"; continue; }
  python3 "$EXP_DIR/align_timeline.py" \
    --telemetry "$d/telemetry.jsonl" \
    --send-timeline "$d/send_timeline.csv" \
    --receive-log "$d/receive_log.csv" \
    --outdir "$d" --tthres "$TTHRES" >/dev/null 2>&1 || echo "  align failed: $d"
done
echo

echo "=== per-percentage paired evaluation (control vs pct<N>) ==="
for pct in $PCTS; do
  dir="$(pct_dir "$pct")"
  echo "--- max_perf_pct=$pct ---"
  python3 "$EXP_DIR/evaluate_degradation.py" \
    --control "$OUTDIR/control" \
    --treatment "$OUTDIR/$dir/predict" \
    --outdir "$OUTDIR/$dir/eval" | sed 's/^/    /'
  echo
done

echo "=== SWEEP SUMMARY ==="
python3 - "$OUTDIR" "$PCTS" "$SUMMARY_CSV" "$RUN_CSV" "$MARGIN" <<'PY'
import csv
import glob
import json
import os
import sys

root, pcts, summary_csv, run_csv, margin = (sys.argv[1], sys.argv[2].split(),
                                            sys.argv[3], sys.argv[4], sys.argv[5])


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def read_ber(run_dir):
    try:
        with open(os.path.join(run_dir, "err.log")) as handle:
            for line in handle:
                if line.startswith("METRIC ber="):
                    return float(line.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return float("nan")


def cap_intervals(cpufreq, ended):
    out, opened = [], None
    for event in cpufreq.get("events") or []:
        if event.get("event") == "throttle":
            opened = event.get("ts_ns")
        elif event.get("event") == "restore" and opened is not None:
            out.append((opened, event["ts_ns"]))
            opened = None
    if opened is not None:
        out.append((opened, ended or opened))
    return out


def attack_window(run_dir):
    """[start_ns, end_ns] spanned by the transmitted bits."""
    path = os.path.join(run_dir, "bit_windows.csv")
    rows = list(csv.DictReader(open(path))) if os.path.isfile(path) else []
    if not rows:
        return None
    return (min(int(r["start_ns"]) for r in rows),
            max(int(r["end_ns"]) for r in rows))


def run_stats(run_dir):
    stats = load_json(os.path.join(run_dir, "telemetry.stats.json"))
    cpufreq = stats.get("cpufreq") or {}
    ended = stats.get("ended_ns")
    span = (ended or 0) - (stats.get("started_ns") or 0)
    intervals = cap_intervals(cpufreq, ended)
    active = sum(b - a for a, b in intervals)

    window = attack_window(run_dir)
    cov = float("nan")
    if window:
        covered = 0
        for a, b in intervals:
            lo, hi = max(a, window[0]), min(b, window[1])
            if hi > lo:
                covered += hi - lo
        cov = covered / float(window[1] - window[0])

    observed = [v["observed_khz"] for v in cpufreq.get("verifications") or []
                if isinstance(v.get("observed_khz"), (int, float))]
    return {
        "duty": (active / float(span)) if span else float("nan"),
        "coverage": cov,
        "caps": len(intervals),
        # SCOPE WARNING: `triggers` and `predictions` are accumulated over ALL 12
        # logical cores -- attack_experiment.py evaluates the trigger once per core
        # per batch, so both counters are (core, batch) tallies, NOT core-0 counts.
        # The core-0 view is `pred_maxima()` / `percentile()` further down.
        "triggers": stats.get("mitigation_triggers", 0),
        "actions": stats.get("mitigation_actions", 0),
        "predictions": stats.get("predictions", 0),
        "requested_pct": cpufreq.get("perf_pct"),
        "ceiling_khz": cpufreq.get("target_freq_khz"),
        "observed_khz": mean(observed),
        "observed_min": min(observed) if observed else float("nan"),
        "verify_ok": cpufreq.get("verify_ok", 0),
        "verify_fail": cpufreq.get("verify_fail", 0),
        "ber_tool": read_ber(run_dir),
    }


def arm_runs(pattern):
    dirs = sorted(d for d in glob.glob(pattern)
                  if os.path.isfile(os.path.join(d, "telemetry.stats.json")))
    return [run_stats(d) for d in dirs]


def eval_metric(eval_csv, arm, key):
    if not os.path.isfile(eval_csv):
        return float("nan")
    with open(eval_csv) as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == key:
                try:
                    return float(row.get("{}_mean".format(arm), "nan"))
                except ValueError:
                    return float("nan")
    return float("nan")


control = arm_runs(os.path.join(root, "control", "run*"))
if not control:
    sys.stderr.write("sweep: no control runs found under {}\n".format(root))
    sys.exit(1)

c_ber = eval_metric(os.path.join(root, "pct%03d" % int(pcts[0]), "eval",
                                 "degradation_summary.csv"), "control", "ber_effective")
if c_ber != c_ber:
    # fall back to the raw per-run BER if the eval CSV is missing
    c_ber = mean([r["ber_tool"] for r in control if r["ber_tool"] == r["ber_tool"]])

c_freq = [r["observed_khz"] for r in control if r["observed_khz"] == r["observed_khz"]]
print("  control (no actuation): duty {:.1f}%, BER {:.3f}, freq {}"
      .format(mean([r["duty"] for r in control]) * 100.0, c_ber,
              "{:.0f} kHz".format(mean(c_freq)) if c_freq
              else "n/a (detect arm has no actuator, so no frequency read-back)"))
print()
print("  {:>5} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>11} {:>9} {:>10} {:>10} {:>9}".format(
    "pct", "freq kHz", "caps", "duty", "att.win", "act", "BER", "capacity", "mut.info",
    "thermal", "TRM", "verify"))
summary_rows = []
for pct in pcts:
    dir_name = "pct%03d" % int(pct)
    runs = arm_runs(os.path.join(root, dir_name, "predict", "run*"))
    if not runs:
        print("  {:>5}  (no runs)".format(pct))
        continue
    ev = os.path.join(root, dir_name, "eval", "degradation_summary.csv")

    def m(key):
        return eval_metric(ev, "treatment", key)

    ber = m("ber_effective")
    if ber != ber:
        ber = mean([r["ber_tool"] for r in runs if r["ber_tool"] == r["ber_tool"]])
    obs = [r["observed_khz"] for r in runs if r["observed_khz"] == r["observed_khz"]]
    ceiling = [r["ceiling_khz"] for r in runs if r["ceiling_khz"]]
    # pct=100 writes the knob and accumulates cap time, but max_perf_pct=100 clamps
    # nothing -- mark it so its duty/att.win are not read as a protection rate.
    cap_eff = int(pct) < 100
    att = mean([r["coverage"] for r in runs]) * 100.0
    att_txt = "{:>8.1f}%".format(att) if cap_eff else "{:>7.1f}%*".format(att)
    print("  {:>5} {:>9} {:>8} {:>7.1f}% {} {:>9} {:>9.3f} {:>11.4f} {:>9.4f} {:>10.4g} {:>10.4g} {:>9}".format(
        pct,
        "{:.0f}".format(mean(obs)) if obs else "-",
        mean([r["caps"] for r in runs]),
        mean([r["duty"] for r in runs]) * 100.0,
        att_txt,
        mean([r["actions"] for r in runs]),
        ber, m("capacity_bps"), m("mutual_info_bits"),
        m("thermal_events"), m("trm_raw_sum"),
        "{:.0f}/{:.0f}".format(mean([r["verify_ok"] for r in runs]),
                               sum(r["verify_fail"] for r in runs))))
    summary_rows.append({
        "pct": pct,
        "freq_khz": int(mean(obs)) if obs else "",
        "caps": mean([r["caps"] for r in runs]),
        "duty_pct": mean([r["duty"] for r in runs]) * 100.0,
        "att_win_pct": mean([r["coverage"] for r in runs]) * 100.0,
        "actions": mean([r["actions"] for r in runs]),
        "ber": ber,
        "capacity_bps": m("capacity_bps"),
        "mutual_info_bits": m("mutual_info_bits"),
        "thermal_events": m("thermal_events"),
        "trm_raw_sum": m("trm_raw_sum"),
        "verify_ok": mean([r["verify_ok"] for r in runs]),
        "verify_fail": sum(r["verify_fail"] for r in runs),
    })
print()
print("  说明: att.win = cap 覆盖攻击窗口的时间比例; capacity/mut.info/thermal/TRM 取自")
print("        evaluate_degradation.py 的 treatment 均值; BER 优先取 err_tool(=./err)。")
print("        * pct=100 是空点: 旋钮被写入、cap 时间也被累计, 但 max_perf_pct=100 不产生")
print("          任何钳制, 因此该行的 duty / att.win 是\"cap 生效时间\"而非\"保护率\",")
print("          只有 pct<100 的行的 att.win 才可读作覆盖率意义上的保护比例。")
print("          该空点的自校验判据是 BER/capacity/mut.info 应接近 control。")

SUMMARY_FIELDS = ["pct", "freq_khz", "caps", "duty_pct", "att_win_pct", "actions",
                  "ber", "capacity_bps", "mutual_info_bits", "thermal_events",
                  "trm_raw_sum", "verify_ok", "verify_fail"]
if summary_rows:
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print("  summary csv -> {}".format(summary_csv))


# ---------------------------------------------------------------- per-run view
# The table above is a per-percentage mean. That hides the failure mode that bit
# this sweep twice: a run whose predictor never crossed the threshold engages no
# clamping at all, i.e. it is a control mixed into the treatment mean.
def percentile(values, frac):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(frac * len(values)))]


def pred_maxima(run_dir):
    """Max over the 20-step horizon of every RECORDED core-0 prediction."""
    path = os.path.join(run_dir, "telemetry.jsonl")
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("logical_core") != 0 or not rec.get("prediction"):
                continue
            try:
                out.append(max(float(v) for v in rec["prediction"]))
            except (TypeError, ValueError):
                pass
    return out


targets = [("control", d) for d in sorted(glob.glob(os.path.join(root, "control", "run*")))]
for d in sorted(glob.glob(os.path.join(root, "pct*", "predict", "run*"))):
    targets.append((os.path.basename(os.path.dirname(os.path.dirname(d))), d))

run_rows = []
for arm, run_dir in targets:
    if not os.path.isfile(os.path.join(run_dir, "telemetry.stats.json")):
        continue
    s = run_stats(run_dir)
    pm = pred_maxima(run_dir)
    n_allcore = s["predictions"]
    run_rows.append({
        "arm": arm,
        "run": os.path.basename(run_dir),
        "requested_pct": s["requested_pct"],
        "ceiling_khz": s["ceiling_khz"],
        # False for the control arm and for the pct=100 null point: both record cap
        # time but clamp nothing, so their duty/att_win are not protection rates.
        "cap_effective": (s["requested_pct"] is not None
                          and int(s["requested_pct"]) < 100),
        "caps": s["caps"],
        # --- all-core scope: summed over the 12 logical cores (see run_stats) -----
        "triggers_allcore": s["triggers"],
        "actions": s["actions"],
        "pred_allcore": n_allcore,
        "trig_frac_pct": (100.0 * s["triggers"] / n_allcore) if n_allcore else float("nan"),
        # --- run-wide ------------------------------------------------------------
        "duty_pct": s["duty"] * 100.0,
        "att_win_pct": s["coverage"] * 100.0,
        # --- core-0 scope only ---------------------------------------------------
        "n_pred_core0": len(pm),
        "pred_med_core0": percentile(pm, 0.5),
        "pred_p90_core0": percentile(pm, 0.9),
        # max(pm), NOT pm[-1]: percentile() sorts a local rebinding of the list and
        # therefore leaves the caller's list in recording order.
        "pred_max_core0": max(pm) if pm else float("nan"),
        "verify_ok": s["verify_ok"],
        "verify_fail": s["verify_fail"],
    })

if run_rows:
    print()
    print("=== PER-RUN DIAGNOSTICS (margin {} -> trigger above {} C) ===".format(
        margin, 90 - int(margin)))
    print("  {:<8} {:<5} {:>5} {:>9} {:>6} {:>8} {:>5} {:>8} {:>7} {:>8} {:>9} {:>9}".format(
        "arm", "run", "pct", "ceil kHz", "caps", "trigAll", "act", "trig%",
        "duty", "att.win", "predMed0", "predMax0"))
    for r in run_rows:
        print("  {:<8} {:<5} {:>5} {:>9} {:>6} {:>8} {:>5} {:>7.1f}% {:>6.1f}% {:>7.1f}% {:>9.1f} {:>9.1f}".format(
            r["arm"], r["run"], str(r["requested_pct"]), str(r["ceiling_khz"]),
            r["caps"], r["triggers_allcore"], r["actions"], r["trig_frac_pct"],
            r["duty_pct"], r["att_win_pct"], r["pred_med_core0"], r["pred_max_core0"]))
    print("  口径说明: trigAll / act 为全核累加(每个核每批次各评估一次触发);")
    print("            trig% = trigAll / 全核预测次数; duty / att.win / verify 为整轮;")
    print("            predMed0 / predMax0 只统计 core 0 的 20 步视界最大值。")
    print("            trigAll 与 predMax0 量纲不同, 不可相除或互推。")
    print("            pct=100 与 control 行的 att.win 不是保护率(cap 空操作),")
    print("            该判据在 CSV 的 cap_effective 列里是 False。")

    RUN_FIELDS = ["arm", "run", "requested_pct", "ceiling_khz", "cap_effective", "caps",
                  "triggers_allcore", "actions", "pred_allcore", "trig_frac_pct",
                  "duty_pct", "att_win_pct",
                  "n_pred_core0", "pred_med_core0", "pred_p90_core0", "pred_max_core0",
                  "verify_ok", "verify_fail"]
    with open(run_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)
    print("  per-run csv  -> {}".format(run_csv))

    zero = ["{} {}".format(r["arm"], r["run"]) for r in run_rows
            if r["arm"] != "control" and r["triggers_allcore"] == 0]
    if zero:
        print()
        print("  !! WARNING: {} predict run(s) never triggered, i.e. they applied no")
        print("     clamping at all and are controls mixed into the treatment means:")
        for name in zero:
            print("     - {}".format(name))
        print("     If that is more than ~1/3 of the predict runs, the trigger threshold")
        print("     is too high for this predictor -- or the predictor is not fixed.")

    preds = [r["pred_max_core0"] for r in run_rows
             if r["pred_max_core0"] == r["pred_max_core0"]]
    if preds:
        print()
        print("  == predictor sanity (core 0, max over each 20-step horizon) ==")
        print("     worst run         : {:.1f} C".format(max(preds)))
        print("     pre-fix reference : never above 79.2 C (zero-padded smoothing)")
        print("     post-fix expected : ~93-98 C on the hot core")
        if max(preds) < 85.0:
            print("     !! WARNING: predictions never reach 85 C, so the online predictor")
            print("        still looks handicapped. Check that the patched")
            print("        Pyro/PyroNet_predict.py is the module being imported.")
PY

echo
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  chown -R "$SUDO_USER" "$OUTDIR" 2>/dev/null || true
fi
echo "artifacts   : $OUTDIR"
echo "  config    : $CONFIG_JSON"
echo "  log       : $LOG"
echo "  summary   : $SUMMARY_CSV"
echo "  per run   : $RUN_CSV"

if [ "$FAILED" -gt 0 ]; then
  echo "!! $FAILED run(s) produced no telemetry (listed above)." >&2
  echo "!! Treat this sweep as incomplete; re-run the missing arms." >&2
  exit 1
fi

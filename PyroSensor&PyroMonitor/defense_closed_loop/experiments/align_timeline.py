#!/usr/bin/env python3
"""Align the experiment timelines into per-bit windows.

Inputs (all timestamps are CLOCK_MONOTONIC nanoseconds on the same host):

* ``send_timeline.csv``    -- one row per sent bit   (bit_index, start_ns, end_ns, value)
* ``receive_log.csv``      -- one row per decision   (decision_index, decision_ns, raw_sum, bit)
* ``telemetry.jsonl``      -- one row per sample     (recv_ts_ns, logical_core, coretemp, ...)

Outputs (written into ``--outdir``):

* ``bit_windows.csv``       -- per sent bit, the received bit, and the thermal
  activity observed inside that bit window. This is the only input of
  ``evaluate_degradation.py``: it carries both the channel outcome
  (sent/received bit) and the physical cause (TRM counts, temperatures).
* ``alignment_report.json`` -- coverage / sanity statistics so that a bad run is
  never silently averaged into the results.
"""

import argparse
import bisect
import csv
import json
import os
import sys
from collections import defaultdict


def _iter_jsonl(path):
    with open(path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stderr.write("align_timeline: skipping malformed line {}: {}\n"
                                 .format(line_no, exc))


def load_send_timeline(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "bit_index": int(row["bit_index"]),
                    "start_ns": int(row["start_ns"]),
                    "end_ns": int(row["end_ns"]),
                    "value": int(row["value"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["start_ns"])
    return rows


def load_receive_log(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "decision_index": int(row["decision_index"]),
                    "decision_ns": int(row["decision_ns"]),
                    "raw_sum": int(row["raw_sum"]),
                    "bit": int(row["bit"]),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r["decision_ns"])
    return rows


def build_bit_windows(telemetry, send_rows, recv_rows, attack_core, tthres, match_tol_ns):
    ordered = sorted(telemetry, key=lambda r: r["recv_ts_ns"])
    timestamps = [r["recv_ts_ns"] for r in ordered]

    recv_by_ns = [(r["decision_ns"], r) for r in recv_rows]
    used_decision = [False] * len(recv_by_ns)

    rows = []
    matched = 0
    for bit in send_rows:
        left = bisect.bisect_left(timestamps, bit["start_ns"])
        right = bisect.bisect_right(timestamps, bit["end_ns"])
        window = ordered[left:right]
        # Thermal statistics span ALL cores: the E1 sender heats whichever core
        # it runs on, which is not necessarily the core the receiver sits on.
        # Restricting this to a single core would hide the real thermal events.
        window_temps = [(r["logical_core"], r.get("coretemp", 0)) for r in window]
        max_coretemp = max((t for _, t in window_temps), default=None)
        hottest_cpu = max(window_temps, key=lambda ct: ct[1])[0] if window_temps else None
        hot_cpu_count = len({c for c, t in window_temps if t > tthres})
        attack_core_samples = sum(1 for r in window if r["logical_core"] == attack_core)

        # One-to-one assignment: every receiver decision may be consumed by at
        # most one bit window. Among the still-unused decisions inside the
        # tolerance band we take the earliest, which follows the receiver's own
        # ordering (decision k trails bit window k by roughly the sender/receiver
        # start offset). A "nearest to centre" rule would let a single decision
        # satisfy two neighbouring bits and silently corrupt the BER.
        lo = bit["start_ns"] - match_tol_ns
        hi = bit["end_ns"] + match_tol_ns
        choice_index = None
        for index, (decision_ns, _) in enumerate(recv_by_ns):
            if used_decision[index] or decision_ns < lo or decision_ns > hi:
                continue
            choice_index = index
            break

        decision = None
        margin_ns = ""
        if choice_index is not None:
            used_decision[choice_index] = True
            decision_ns, decision = recv_by_ns[choice_index]
            center_ns = (bit["start_ns"] + bit["end_ns"]) // 2
            margin_ns = abs(decision_ns - center_ns)
            matched += 1

        rows.append({
            "bit_index": bit["bit_index"],
            "sent_value": bit["value"],
            "start_ns": bit["start_ns"],
            "end_ns": bit["end_ns"],
            "window_s": (bit["end_ns"] - bit["start_ns"]) / 1e9,
            "received_bit": decision["bit"] if decision is not None else "",
            "received_raw_sum": decision["raw_sum"] if decision is not None else "",
            "margin_ns": margin_ns,
            "matched": decision is not None,
            "attack_core_samples": attack_core_samples,
            "all_core_samples": len(window),
            "max_coretemp": "" if max_coretemp is None else max_coretemp,
            "hottest_cpu": "" if hottest_cpu is None else hottest_cpu,
            "hot_cpu_count": hot_cpu_count,
            "thermal_event": hot_cpu_count > 0,
        })
    return rows, matched


def main(argv=None):
    parser = argparse.ArgumentParser(description="Align experiment timelines.")
    parser.add_argument("--telemetry", required=True, help="telemetry JSONL path")
    parser.add_argument("--send-timeline", default=None, help="send_timeline.csv path")
    parser.add_argument("--receive-log", default=None, help="receive_log.csv path")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--tthres", type=float, default=90.0,
                        help="temperature above which a core counts as thermally hot; "
                             "drives the per-bit thermal_event flag")
    parser.add_argument("--attack-core", type=int, default=3,
                        help="the receiver's CPU (VICTIM_CORE); only used for the "
                             "per-core sample count column - thermal statistics span "
                             "all cores")
    parser.add_argument("--match-tol-ns", type=int, default=2_000_000_000,
                        help="tolerance when matching a bit window to a decision")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    telemetry = list(_iter_jsonl(args.telemetry))
    send_rows = load_send_timeline(args.send_timeline)
    recv_rows = load_receive_log(args.receive_log)

    bit_rows, matched = build_bit_windows(
        telemetry, send_rows, recv_rows, args.attack_core, args.tthres, args.match_tol_ns)

    bit_path = os.path.join(args.outdir, "bit_windows.csv")
    bit_fields = ["bit_index", "sent_value", "start_ns", "end_ns", "window_s",
                  "received_bit", "received_raw_sum", "margin_ns", "matched",
                  "attack_core_samples", "all_core_samples", "max_coretemp",
                  "hottest_cpu", "hot_cpu_count", "thermal_event"]
    with open(bit_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bit_fields)
        writer.writeheader()
        for row in bit_rows:
            writer.writerow(row)

    per_core = defaultdict(int)
    for record in telemetry:
        per_core[str(record["logical_core"])] += 1

    report = {
        "telemetry_records": len(telemetry),
        "telemetry_per_core": dict(sorted(per_core.items(), key=lambda kv: int(kv[0]))),
        "bits_sent": len(send_rows),
        "receive_decisions": len(recv_rows),
        "bits_matched": matched,
        "bit_match_rate": (matched / len(send_rows)) if send_rows else None,
        "thermal_events": sum(1 for r in bit_rows if r["thermal_event"]),
        "tthres": args.tthres,
        "attack_core": args.attack_core,
        "outputs": {"bit_windows": bit_path},
    }
    report_path = os.path.join(args.outdir, "alignment_report.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("align_timeline: telemetry={} bits={} matched={} thermal_events={}".format(
        len(telemetry), len(send_rows), matched, report["thermal_events"]))
    if send_rows and matched < len(send_rows):
        print("align_timeline: WARNING: {} of {} bit windows had no receive decision "
              "within tolerance {}".format(len(send_rows) - matched, len(send_rows),
                                           args.match_tol_ns))
    print("align_timeline: wrote {} and {}".format(bit_path, report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

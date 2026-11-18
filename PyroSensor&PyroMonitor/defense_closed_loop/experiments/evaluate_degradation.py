#!/usr/bin/env python3
"""Quantify attack degradation via paired causal comparison.

The mitigation effect is measured at the *outcome* level, comparing runs that
differ only in whether actuation happened:

    control   : attack_monitor            (predictions, no actuation)
    treatment : attack_monitor_mitigate   (downclock on predicted thermal event)

Both arms must be run with the same seed so the sender transmits an identical bit
sequence; then the only independent variable is the actuation policy.

Input : ``bit_windows.csv`` (from ``align_timeline.py``), optionally ``err.log``.
Output: ``degradation_runs.csv``, ``degradation_summary.csv``, ``degradation.tex``.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

SUMMARY_METRICS = [
    "bits_sent", "bits_matched", "match_rate",
    "ber_effective", "ber_matched",
    "goodput_bps", "capacity_bps", "mutual_info_bits",
    "thermal_events", "thermal_event_rate", "trm_raw_sum",
]


def discover_runs(path):
    if os.path.isfile(os.path.join(path, "bit_windows.csv")):
        return [path]
    runs = sorted(p for p in glob.glob(os.path.join(path, "*"))
                  if os.path.isfile(os.path.join(p, "bit_windows.csv")))
    return runs


def load_bit_windows(path):
    rows = []
    with open(path, "r", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def load_err_metrics(path):
    metrics = {}
    log_path = os.path.join(path, "err.log")
    if not os.path.isfile(log_path):
        return metrics
    with open(log_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("METRIC "):
                key, _, value = line[len("METRIC "):].partition("=")
                metrics[key.strip()] = value.strip()
    return metrics


def _as_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _entropy2(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _mutual_information(sent, recv):
    """Empirical mutual information (bits) between sent and received bits."""
    n00 = n01 = n10 = n11 = 0
    for x, y in zip(sent, recv):
        if x == 0 and y == 0:
            n00 += 1
        elif x == 0 and y == 1:
            n01 += 1
        elif x == 1 and y == 0:
            n10 += 1
        else:
            n11 += 1
    total = n00 + n01 + n10 + n11
    if total == 0:
        return float("nan")
    px = [(n00 + n01) / total, (n10 + n11) / total]
    py = [(n00 + n10) / total, (n01 + n11) / total]
    joint = [[n00, n01], [n10, n11]]
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pxy = joint[i][j] / total
            if pxy > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                mi += pxy * math.log2(pxy / (px[i] * py[j]))
    return mi


def run_metrics(run_dir):
    rows = load_bit_windows(os.path.join(run_dir, "bit_windows.csv"))
    n_bits = len(rows)
    matched_rows = []
    errors_matched = 0
    errors_effective = 0
    thermal_events = 0
    trm_sum = 0
    sent = []
    recv = []
    t_start = None
    t_end = None

    for row in rows:
        sent_value = _as_int(row.get("sent_value"))
        is_matched = str(row.get("matched", "")).strip().lower() in ("true", "1")
        if str(row.get("thermal_event", "")).strip().lower() in ("true", "1"):
            thermal_events += 1
        raw = _as_int(row.get("received_raw_sum"))
        if raw is not None:
            trm_sum += raw
        start = _as_int(row.get("start_ns"))
        end = _as_int(row.get("end_ns"))
        if start is not None:
            t_start = start if t_start is None else min(t_start, start)
        if end is not None:
            t_end = end if t_end is None else max(t_end, end)

        if is_matched:
            recv_value = _as_int(row.get("received_bit"))
            if recv_value is None:
                is_matched = False

        if is_matched:
            matched_rows.append(row)
            if sent_value is not None and recv_value is not None:
                sent.append(sent_value)
                recv.append(recv_value)
                if sent_value != recv_value:
                    errors_matched += 1
                    errors_effective += 1
        else:
            errors_effective += 1

    n_matched = len(matched_rows)
    duration_s = ((t_end - t_start) / 1e9) if (t_start is not None and t_end is not None and t_end > t_start) else float("nan")

    ber_matched = (errors_matched / n_matched) if n_matched else float("nan")
    ber_effective = (errors_effective / n_bits) if n_bits else float("nan")

    correct_bits = n_matched - errors_matched
    goodput_bps = (correct_bits / duration_s) if duration_s == duration_s and duration_s > 0 else float("nan")
    bit_period_s = (duration_s / n_bits) if (duration_s == duration_s and n_bits) else 10.0
    capacity_bsc = 1.0 - _entropy2(ber_effective) if ber_effective == ber_effective else float("nan")
    capacity_bps = (capacity_bsc / bit_period_s) if (capacity_bsc == capacity_bsc and bit_period_s > 0) else float("nan")
    mi_bits = _mutual_information(sent, recv) if sent else float("nan")

    metrics = {
        "run_dir": run_dir,
        "bits_sent": n_bits,
        "bits_matched": n_matched,
        "match_rate": (n_matched / n_bits) if n_bits else float("nan"),
        "ber_effective": ber_effective,
        "ber_matched": ber_matched,
        "goodput_bps": goodput_bps,
        "capacity_bsc": capacity_bsc,
        "capacity_bps": capacity_bps,
        "mutual_info_bits": mi_bits,
        "thermal_events": thermal_events,
        "thermal_event_rate": (thermal_events / n_bits) if n_bits else float("nan"),
        "trm_raw_sum": trm_sum,
        "duration_s": duration_s,
    }
    err_metrics = load_err_metrics(run_dir)
    if "ber" in err_metrics:
        try:
            metrics["err_tool_ber"] = float(err_metrics["ber"])
        except ValueError:
            pass
    return metrics


def _aggregate(runs_metrics, metric):
    values = [m[metric] for m in runs_metrics
              if metric in m and isinstance(m[metric], (int, float))
              and m[metric] == m[metric]]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(value):
    if isinstance(value, float):
        if value != value:
            return "nan"
        return "{:.4f}".format(value)
    return str(value)


def build_summary(arms):
    """arms: dict name -> list of run metric dicts."""
    summary = {}
    for name, runs in arms.items():
        summary[name] = {}
        for metric in SUMMARY_METRICS:
            mean, std = _aggregate(runs, metric)
            summary[name][metric] = {"mean": mean, "std": std, "runs": len(runs)}
    return summary


def write_runs_csv(path, arms):
    fields = ["arm", "run_dir", "bits_sent", "bits_matched", "match_rate",
              "ber_effective", "ber_matched", "err_tool_ber", "goodput_bps",
              "capacity_bsc", "capacity_bps", "mutual_info_bits",
              "thermal_events", "thermal_event_rate", "trm_raw_sum", "duration_s"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, runs in arms.items():
            for metrics in runs:
                row = {"arm": name}
                for field in fields[2:]:
                    row[field] = fmt(metrics.get(field, "nan"))
                writer.writerow(row)


def write_summary_csv(path, summary, order):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + ["{}_mean".format(n) for n in order]
                        + ["{}_std".format(n) for n in order])
        for metric in SUMMARY_METRICS:
            means = [fmt(summary[n][metric]["mean"]) for n in order]
            stds = [fmt(summary[n][metric]["std"]) for n in order]
            writer.writerow([metric] + means + stds)


def write_latex(path, summary, order, reference="control"):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired causal comparison of the E1 covert channel with and without "
        r"prediction-triggered actuation (mean $\pm$ std over repetitions).}",
        r"\label{tab:degradation}",
        r"\begin{tabular}{l" + "r" * len(order) + "}",
        r"\toprule",
        " & ".join(["Metric"] + order) + r" \\",
        r"\midrule",
    ]
    for metric in SUMMARY_METRICS:
        cells = []
        for name in order:
            entry = summary[name][metric]
            cells.append("${:.4f} \\pm {:.4f}$".format(entry["mean"], entry["std"])
                         if entry["mean"] == entry["mean"] else "nan")
        lines.append(" & ".join([metric] + cells) + r" \\")
    if reference in order:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{Relative change vs %s} \\" % (len(order) + 1, reference))
        base_cap = summary[reference]["capacity_bps"]["mean"]
        base_ber = summary[reference]["ber_effective"]["mean"]
        for name in order:
            if name == reference:
                continue
            cap = summary[name]["capacity_bps"]["mean"]
            ber = summary[name]["ber_effective"]["mean"]
            cap_delta = ((cap - base_cap) / base_cap * 100.0) if base_cap else float("nan")
            ber_delta = ((ber - base_ber) / base_ber * 100.0) if base_ber else float("nan")
            lines.append("{} & \\multicolumn{{{}}}{{l}}{{capacity {:+.1f}\\%, BER {:+.1f}\\%}} \\\\"
                         .format(name, len(order), cap_delta, ber_delta))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired causal attack-degradation metrics.")
    parser.add_argument("--control", required=True,
                        help="run dir or parent of run dirs for attack_monitor")
    parser.add_argument("--treatment", required=True,
                        help="run dir or parent of run dirs for attack_monitor_mitigate")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    arm_paths = [("control", args.control), ("treatment", args.treatment)]

    arms = {}
    for name, path in arm_paths:
        runs = discover_runs(path)
        if not runs:
            sys.stderr.write("evaluate_degradation: no runs found under '{}' for arm '{}'\n"
                             .format(path, name))
            arms[name] = []
            continue
        arms[name] = [run_metrics(run) for run in runs]

    order = [name for name, _ in arm_paths]
    summary = build_summary(arms)

    for name in order:
        if not arms[name]:
            continue
        print("== {} ({} run(s)) ==".format(name, len(arms[name])))
        for metric in ("bits_sent", "bits_matched", "ber_effective", "ber_matched",
                       "goodput_bps", "capacity_bps", "mutual_info_bits",
                       "thermal_events", "trm_raw_sum"):
            mean, std = summary[name][metric]["mean"], summary[name][metric]["std"]
            print("   {:<18} {}{}".format(
                metric, fmt(mean), "" if std == 0 else "  +/- {}".format(fmt(std))))

    runs_csv = os.path.join(args.outdir, "degradation_runs.csv")
    summary_csv = os.path.join(args.outdir, "degradation_summary.csv")
    tex_path = os.path.join(args.outdir, "degradation.tex")
    write_runs_csv(runs_csv, arms)
    write_summary_csv(summary_csv, summary, order)
    write_latex(tex_path, summary, order)

    if len(arms.get("control", [])) < 1 or len(arms.get("treatment", [])) < 1:
        sys.stderr.write("evaluate_degradation: WARNING: missing runs for control/treatment; "
                         "results are incomplete.\n")

    print("evaluate_degradation: wrote {}, {} and {}".format(runs_csv, summary_csv, tex_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

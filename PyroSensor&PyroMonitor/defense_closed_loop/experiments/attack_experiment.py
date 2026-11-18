#!/usr/bin/env python3
"""Collect Pyro telemetry while the E1-Covert_Channel attack runs, and predict
thermal events with PyroNet.

Modes
-----
detect   : collect telemetry + predictions, never actuate (open loop). This is the
           control arm of the degradation comparison.
mitigate : when the predicted temperature crosses ``tthres - margin``, pull the
           frequency down for a cooldown window (closed loop). This is the
           treatment arm.

Both modes emit the same telemetry, so the ONLY difference between the two arms is
whether actuation happened -- which is what gives the comparison its causal
interpretation.

The script does not modify ``Pyro/PyroNet_predict.py``. It imports it and only
subclasses the predictor to redirect the model directory and the kernel module
path, so the original paper workflow is untouched.

Output: one JSON line per telemetry sample plus a small stats JSON.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTROLLER_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_CONTROLLER_ROOT))
_PYRO_DIR = os.path.join(_REPO_ROOT, "Pyro")
# Inference checkpoint. The online experiments load <model_dir>/PyroNet.pth together
# with its scalers.json and feature_columns.json, so switching models means switching
# this DIRECTORY, not just a filename.
#
# Default: the E1-fine-tuned model (2026-10-01). The original model is kept intact at
# PyroNet_model/ and can be selected per run without editing anything:
#   PYRONET_MODEL_DIR="$PWD/Pyro/PyroNet_model" ./sweep_perf_pct.sh ...
# sweep_perf_pct.sh records the resolved directory (and the .pth sha1) in
# sweep.config.json, so a run's artifacts always say which checkpoint produced them.
_DEFAULT_MODEL_DIR = (os.environ.get("PYRONET_MODEL_DIR")
                      or os.path.join(_PYRO_DIR, "PyroNet_model_ft"))
_DEFAULT_MODULE_PATH = os.path.join(_PYRO_DIR, "Pyro.ko")

MODES = ("detect", "mitigate")

# Feature order used by PyroNet.preprocess_data. Kept in sync on purpose: the
# scaler parameters loaded from scalers.json follow the same order.
FEATURE_KEYS = [
    "coretemp", "pkg_energy", "dram_energy", "cpu_freq", "cpu_voltage",
    "cpu_load", "instruction_diff", "cycle_diff", "cache_diff",
    "cache_miss_diff", "bus_cycle_diff", "branch_instruction_diff",
    "branch_insMiss_diff", "task_clock_diff", "l1_data_read_diff",
    "l1_data_write_diff", "l1_data_miss_diff", "l1_ins_miss_diff",
    "l2_read_miss_diff", "context_switches_diff", "page_faults_diff",
]


def _import_pyro():
    """Import PyroNet_predict with the Pyro/ directory on sys.path."""
    if _PYRO_DIR not in sys.path:
        sys.path.insert(0, _PYRO_DIR)
    try:
        import numpy as np
        import torch
        import PyroNet_predict as pp
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "attack_experiment: cannot import Pyro/PyroNet dependencies ({})\n"
            "  PYRO dir: {}\n"
            "  Hint: install torch, numpy, scipy, scikit-learn and joblib."
            .format(exc, _PYRO_DIR))
    return np, torch, pp


def _build_predictor(pp, np, torch, model_dir, module_path, do_insmod, keep_module):
    class _ExperimentPredictor(pp.HybridTemperaturePredictor):
        def load_kernel_module(self):
            pid = os.getpid()
            if not do_insmod:
                print("attack_experiment: skipping insmod (module assumed loaded), "
                      "user_pid={}".format(pid))
                return
            print("attack_experiment: inserting {} with user_pid={}".format(module_path, pid))
            argv = ["insmod", module_path, "user_pid={}".format(pid)]
            if os.geteuid() != 0:
                argv.insert(0, "sudo")
            try:
                subprocess.run(argv, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                sys.stderr.write(
                    "attack_experiment: insmod failed (rc={})\nstdout: {}\nstderr: {}\n"
                    .format(exc.returncode, exc.stdout, exc.stderr))
                sys.exit(2)

        def unload_kernel_module(self):
            if not do_insmod or keep_module:
                print("attack_experiment: leaving kernel module loaded")
                return
            print("attack_experiment: removing kernel module")
            argv = ["rmmod", "Pyro"]
            if os.geteuid() != 0:
                argv.insert(0, "sudo")
            subprocess.run(argv, capture_output=True, text=True)

        def load_models(self):
            model = pp.EnhancedCNNSeq2Seq(
                input_size=pp.CONFIG["num_features"],
                hidden_size=pp.CONFIG["hidden_size"],
                cnn_channels=pp.CONFIG["cnn_channels"],
                num_layers=pp.CONFIG["num_layers"],
                dropout=pp.CONFIG["dropout"],
                pred_length=pp.CONFIG["pred_length"],
            ).to(pp.DEVICE)
            model.load_state_dict(torch.load(
                os.path.join(model_dir, "PyroNet.pth"), map_location=pp.DEVICE))
            if pp.DEVICE.type == "cuda":
                model = model.half()
            model.eval()
            try:
                example = torch.randn(1, pp.CONFIG["seq_length"], pp.CONFIG["num_features"],
                                      dtype=self.model_dtype, device=pp.DEVICE)
                model = torch.jit.trace(model, example)
                print("attack_experiment: TorchScript optimization applied")
            except Exception as exc:
                print("attack_experiment: TorchScript conversion failed ({}); "
                      "using eager model".format(exc))
            with open(os.path.join(model_dir, "scalers.json"), "r") as handle:
                scaler_data = json.load(handle)
            scalers = {}
            for feature, params in scaler_data.items():
                scaler = pp.MinMaxScaler()
                scaler.min_ = np.array(params["min_"])
                scaler.scale_ = np.array(params["scale_"])
                scaler.data_min_ = np.array(params["data_min_"])
                scaler.data_max_ = np.array(params["data_max_"])
                scaler.feature_range = tuple(params["feature_range"])
                scalers[feature] = scaler
            with open(os.path.join(model_dir, "feature_columns.json"), "r") as handle:
                feature_columns = json.load(handle)
            return model, scalers, feature_columns

    return _ExperimentPredictor()


class TelemetryCollector:
    def __init__(self, predictor, np, args, throttler):
        self.p = predictor
        self.np = np
        self.args = args
        self.throttler = throttler
        self.seq_length = int(args.seq_length)
        self.batch_id = 0
        self.stats = {
            "mode": args.mode,
            "tthres": args.tthres,
            "mitigate_margin": args.mitigate_margin,
            "mitigate_threshold": args.tthres - args.mitigate_margin,
            "duration_s": args.duration,
            "batches": 0,
            "samples": 0,
            "predictions": 0,
            "mitigation_triggers": 0,
            "mitigation_actions": 0,
            "mitigation_scope": (throttler.scope if throttler is not None else None),
            "mitigation_backend": (throttler.backend.name if throttler is not None
                                   and throttler.backend is not None else None),
            "samples_per_core": {},
            "started_ns": time.monotonic_ns(),
            "ended_ns": None,
        }
        self._warmup_done = False
        self._warmup_deadline_ns = self.stats["started_ns"] + int(args.warmup_timeout * 1e9)

    # ------------------------------------------------------------------ helpers
    def _features(self, sample):
        return [sample.get(key, 0) for key in FEATURE_KEYS]

    def _check_warmup(self, now_ns):
        """Announce that the prediction pipeline is able to act.

        The orchestrator starts the attacker only after this marker. Otherwise the
        first ~12 s (more than one 10 s bit period) run unprotected simply because
        the model has not accumulated its 30-sample window yet.
        """
        if self._warmup_done:
            return
        if self.stats["predictions"] >= 1:
            self._warmup_done = True
            print("attack_experiment: warmup complete after {:.1f}s "
                  "(predictions={})".format(
                      (now_ns - self.stats["started_ns"]) / 1e9,
                      self.stats["predictions"]), flush=True)
        elif now_ns >= self._warmup_deadline_ns:
            self._warmup_done = True
            print("attack_experiment: WARNING: warmup timed out after {:.0f}s "
                  "(predictions={}); continuing anyway".format(
                      self.args.warmup_timeout, self.stats["predictions"]), flush=True)

    def handle_message(self, data, handle):
        temps = self.p.parse_temperatures(data)
        if not temps:
            return
        self.batch_id += 1
        self.stats["batches"] += 1
        recv_ts_ns = time.monotonic_ns()

        by_core = {}
        for sample in temps:
            by_core.setdefault(sample["logical_core"], []).append(sample)

        for core, samples in by_core.items():
            buffer = self.p.core_buffers.setdefault(core, [])
            buffer.extend(samples)
            if len(buffer) > self.seq_length:
                del buffer[:-self.seq_length]

            prediction = None
            if len(buffer) >= self.seq_length:
                prediction = self.p.predict_temperature(core, buffer[-self.seq_length:])
                if prediction is not None:
                    self.stats["predictions"] += 1

            # The actuation policy fires earlier than the thermal-event *label*
            # (tthres): the controller needs lead time to actually cover the
            # attack window. `--mitigate-margin` is how many degrees below tthres
            # actuation starts. The trigger is evaluated in BOTH modes so that the
            # control arm reports the same trigger count -- that is what lets the
            # two arms be compared on "would the controller have fired here?".
            mitigate_threshold = self.args.tthres - self.args.mitigate_margin
            mitigate_trigger = bool(prediction is not None
                                    and self.np.any(self.np.asarray(prediction)
                                                    > mitigate_threshold))
            if mitigate_trigger:
                self.stats["mitigation_triggers"] += 1

            mitigated = False
            target_khz = None
            mitigated_cpus = []
            if mitigate_trigger and self.args.mode == "mitigate" and self.throttler is not None:
                # The intel_pstate_pct backend is system-wide: the cap applies to
                # every core at once, so cpus must stay None and the triggering
                # core is informational only.
                if self.throttler.scope == "global":
                    target_cpus = None
                else:
                    target_cpus = [core] if self.throttler.base_khz(core) is not None else None
                applied = self.throttler.throttle(cpus=target_cpus)
                if applied:
                    self.stats["mitigation_actions"] += 1
                    mitigated = True
                    target_khz = applied[0][1]
                    mitigated_cpus = [cpu for cpu, _ in applied]

            for index, sample in enumerate(samples):
                self.stats["samples"] += 1
                self.stats["samples_per_core"][str(core)] = \
                    self.stats["samples_per_core"].get(str(core), 0) + 1
                record = {
                    "recv_ts_ns": recv_ts_ns,
                    "batch_id": self.batch_id,
                    "sample_index": index,
                    "logical_core": sample["logical_core"],
                    "physical_core": sample["physical_core"],
                    "coretemp": sample["coretemp"],
                    "features": self._features(sample),
                    "prediction": None,
                    "pred_mean": None,
                    "mitigated": False,
                    "mitigation_target_khz": None,
                    "mitigation_cpus": [],
                }
                if index == len(samples) - 1 and prediction is not None:
                    record["prediction"] = [float(v) for v in self.np.asarray(prediction).reshape(-1)]
                    record["pred_mean"] = float(self.np.mean(prediction))
                    record["mitigated"] = mitigated
                    record["mitigation_target_khz"] = target_khz
                    record["mitigation_cpus"] = mitigated_cpus
                handle.write(json.dumps(record) + "\n")
            handle.flush()

    # -------------------------------------------------------------------- run
    def run(self):
        sock = self.p.sock
        sock.settimeout(1.0)
        deadline_ns = time.monotonic_ns() + int(self.args.duration * 1e9)

        if not self.args.warmup:
            print("attack_experiment: warmup complete (disabled)", flush=True)
            self._warmup_done = True

        out_dir = os.path.dirname(os.path.abspath(self.args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        telemetry_mode = "a" if self.args.append else "w"
        with open(self.args.out, telemetry_mode) as handle:
            while time.monotonic_ns() < deadline_ns:
                now_ns = time.monotonic_ns()
                try:
                    data = sock.recv(1024 * 1024)
                except socket.timeout:
                    data = None
                except OSError as exc:
                    sys.stderr.write("attack_experiment: socket error: {}\n".format(exc))
                    break
                if data:
                    self.handle_message(data, handle)
                self._check_warmup(now_ns)
                if self.throttler is not None:
                    self.throttler.tick(now_ns)

        self.stats["ended_ns"] = time.monotonic_ns()
        if self.throttler is not None:
            self.stats["cpufreq"] = self.throttler.summary()
        return self.stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect Pyro telemetry under the E1-Covert_Channel attack.")
    parser.add_argument("--mode", choices=MODES, default="detect",
                        help="detect = open loop (control arm), "
                             "mitigate = closed loop (treatment arm)")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="collection duration in seconds")
    parser.add_argument("--out", type=str, default=None,
                        help="JSONL output path (default: outputs/<mode>/telemetry.jsonl)")
    parser.add_argument("--stats", type=str, default=None,
                        help="stats JSON path (default: <out without .jsonl>.stats.json)")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing telemetry file instead of truncating it")
    parser.add_argument("--tthres", type=float, default=None,
                        help="thermal event threshold in Celsius (default: PyroNet heat_threshold)")
    parser.add_argument("--seq-length", type=int, default=30,
                        help="fallback sequence length if PyroNet CONFIG is unavailable")
    parser.add_argument("--model-dir", type=str, default=_DEFAULT_MODEL_DIR)
    parser.add_argument("--module-path", type=str, default=_DEFAULT_MODULE_PATH)
    parser.add_argument("--cpus", type=str, default="",
                        help="fallback CPUs to throttle when the triggering core is "
                             "unknown (default: auto - throttle the core that crossed "
                             "the threshold)")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--min-dwell", type=float, default=3.0)
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--target", choices=("base", "min"), default="base",
                        help="'base' = stop turboing, 'min' = lowest allowed "
                             "frequency; the concrete knob depends on the backend")
    parser.add_argument("--perf-pct", type=int, default=None,
                        help="explicit intel_pstate/max_perf_pct cap depth (17-100 on "
                             "most parts). Overrides the depth implied by --target and "
                             "forces the intel_pstate_pct backend. Use this to sweep how "
                             "mitigation strength affects attack suppression")
    parser.add_argument("--backend", default="auto",
                        help="mitigation backend: auto | intel_pstate_pct | none")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the under-load capability probe, which verifies "
                             "that writing max_perf_pct actually lowers the frequency")
    parser.add_argument("--probe-cpu", type=int, default=None,
                        help="CPU used by the capability probe (default: first monitored CPU)")
    parser.add_argument("--verify-interval", type=float, default=2.0,
                        help="seconds between read-back verifications (0 = off)")
    parser.add_argument("--reassert-interval", type=float, default=2.0,
                        help="seconds between re-writes of the mitigation knob, to counter "
                             "external overwrites such as thermald (0 = off)")
    parser.add_argument("--mitigate-margin", type=float, default=0.0,
                        help="start actuation when the predicted temperature exceeds "
                             "(tthres - margin). A margin of ~10 C makes the controller "
                             "act early enough to actually cover the attack window; "
                             "the thermal-event *label* still uses tthres")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="do not wait for the first prediction before signalling "
                             "readiness. By default the attacker is held back until the "
                             "model can actually act, so no bit window runs unprotected")
    parser.add_argument("--warmup-timeout", type=float, default=90.0,
                        help="give up waiting for the first prediction after this many seconds")
    parser.add_argument("--module-loaded", action="store_true",
                        help="do not insmod; assume the orchestrator already loaded it")
    parser.add_argument("--keep-module", action="store_true",
                        help="do not rmmod on exit")
    args = parser.parse_args(argv)

    if args.out is None:
        args.out = os.path.join(_HERE, "outputs", args.mode, "telemetry.jsonl")
    if args.stats is None:
        base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
        args.stats = base + ".stats.json"
    if args.tthres is None:
        args.tthres = 90.0
    args.cpus = [int(c) for c in args.cpus.split(",") if c.strip()]
    return args


def main(argv=None):
    args = parse_args(argv)
    np, torch, pp = _import_pyro()
    if args.tthres is None or args.tthres == 90.0:
        args.tthres = float(pp.CONFIG.get("heat_threshold", args.tthres))
    args.seq_length = int(pp.CONFIG.get("seq_length", args.seq_length))

    throttler = None
    if args.mode == "mitigate":
        # PyroController keeps the controller one level above experiments/.
        sys.path.insert(0, _CONTROLLER_ROOT)
        from cpufreq_mitigation import FrequencyThrottler
        throttler = FrequencyThrottler(
            cpus=args.cpus,
            cooldown_s=args.cooldown,
            min_dwell_s=args.min_dwell,
            min_interval_s=args.min_interval,
            target=args.target,
            backend=args.backend,
            probe=not args.no_probe,
            probe_cpu=args.probe_cpu,
            verify_interval_s=args.verify_interval,
            reassert_interval_s=args.reassert_interval,
            perf_pct=args.perf_pct)
        if throttler.available:
            print("attack_experiment: mitigation backend={} scope={} target={} "
                  "expected_max={} kHz{}".format(
                      throttler.backend.name, throttler.scope, throttler.target_mode,
                      throttler.target_khz(),
                      "" if throttler.perf_pct is None
                      else " (requested max_perf_pct={})".format(throttler.perf_pct)))
        else:
            print("attack_experiment: WARNING: no effective frequency mitigation "
                  "backend; mode 'mitigate' will behave like open loop")

    predictor = _build_predictor(pp, np, torch, args.model_dir, args.module_path,
                                 do_insmod=not args.module_loaded,
                                 keep_module=args.keep_module)

    print("attack_experiment: mode={} duration={}s tthres={} out={}".format(
        args.mode, args.duration, args.tthres, args.out))
    print("attack_experiment: warmup={} actuate when predicted > {} C "
          "(tthres {} - margin {})".format(
              args.warmup, args.tthres - args.mitigate_margin,
              args.tthres, args.mitigate_margin), flush=True)
    collector = TelemetryCollector(predictor, np, args, throttler)

    stats = collector.stats
    try:
        stats = collector.run()
    except KeyboardInterrupt:
        print("\nattack_experiment: interrupted, finalizing...")
        stats["ended_ns"] = time.monotonic_ns()
    finally:
        try:
            predictor.sock.close()
        except Exception:
            pass
        if throttler is not None:
            throttler.close()
            stats["cpufreq"] = throttler.summary()
        try:
            predictor.unload_kernel_module()
        except Exception as exc:
            sys.stderr.write("attack_experiment: failed to unload module: {}\n".format(exc))

    with open(args.stats, "w") as handle:
        json.dump(stats, handle, indent=2)
    print("attack_experiment: wrote {} and {}".format(args.out, args.stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

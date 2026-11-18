#!/usr/bin/env python3
"""CPU mitigation controller for the Pyro closed-loop experiment.

The controller caps the package frequency through
``/sys/devices/system/cpu/intel_pstate/max_perf_pct`` (system-wide, continuous).
It verifies the *observed* frequency instead of trusting sysfs writability,
because a writable knob is not necessarily enforced by the hardware.

* **Capability probe**: on startup the knob is tested *under load*. If the
  observed frequency does not actually drop, the backend is rejected. This
  prevents a "written but ineffective" mitigation from silently invalidating
  the experiment.
* **Runtime read-back verification** and **re-assertion** so that an external
  agent (e.g. ``thermald``) overwriting the knob is detected and countered.

The module has no dependency on PyTorch and can be imported standalone.
"""

import glob
import os
import subprocess
import time

DEFAULT_COOLDOWN_S = 5.0
DEFAULT_MIN_DWELL_S = 3.0
DEFAULT_MIN_INTERVAL_S = 2.0
DEFAULT_VERIFY_INTERVAL_S = 2.0
DEFAULT_REASSERT_INTERVAL_S = 2.0
DEFAULT_PROBE_LOAD_S = 3.0
DEFAULT_PROBE_SETTLE_S = 3.0
# Observed frequency must land within this factor of the expected ceiling for a
# backend to be considered effective (HWP/thermal drift makes an exact match
# unrealistic, but "did not move at all" must be rejected).
PROBE_TOLERANCE = 1.6

TARGET_MODES = ("base", "min")
BACKENDS = ("intel_pstate_pct",)

INTEL_PSTATE_DIR = "/sys/devices/system/cpu/intel_pstate"
CPUFREQ_ROOT = "/sys/devices/system/cpu"


# --------------------------------------------------------------------- helpers
def discover_cpus():
    """Return the CPUs that expose a cpufreq policy directory."""
    cpus = []
    for path in glob.glob(os.path.join(CPUFREQ_ROOT, "cpu[0-9]*", "cpufreq")):
        name = os.path.basename(os.path.dirname(path))
        try:
            cpus.append(int(name[3:]))
        except ValueError:
            continue
    return sorted(cpus)


def _read_int(path, default=None):
    try:
        with open(path, "r") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return default


def _write_int(path, value):
    with open(path, "w") as handle:
        handle.write(str(int(value)))


def _sysfs_cpu(cpu, name):
    return os.path.join(CPUFREQ_ROOT, "cpu{}".format(cpu), "cpufreq", name)


def _first_cpu_value(name, cpus):
    for cpu in cpus:
        value = _read_int(_sysfs_cpu(cpu, name))
        if value is not None:
            return value
    return None


def _spawn_load(cpu):
    """Start a busy loop pinned to ``cpu``; returns a Popen or None."""
    if cpu is None:
        return None
    cmd = ["taskset", "-c", str(int(cpu)), "bash", "-c", "while :; do :; done"]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _stop_load(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# -------------------------------------------------------------------- backends
class _IntelPstatePct:
    """``intel_pstate/max_perf_pct`` - continuous, system wide."""

    name = "intel_pstate_pct"
    scope = "global"
    supports = ("base", "min")

    def __init__(self, cpus, target_mode, verbose=True, perf_pct=None):
        self.cpus = cpus
        self.target_mode = target_mode
        self.verbose = verbose
        self.path = os.path.join(INTEL_PSTATE_DIR, "max_perf_pct")
        self._max_khz = _first_cpu_value("cpuinfo_max_freq", cpus)
        self._base_khz = _first_cpu_value("base_frequency", cpus)
        self._min_pct = _read_int(os.path.join(INTEL_PSTATE_DIR, "min_perf_pct"))
        # Explicit cap depth for sweeps, instead of one derived from target_mode
        # ("min" -> min_perf_pct, "base" -> base_frequency as a % of cpuinfo_max).
        self.explicit_pct = None if perf_pct is None else int(perf_pct)
        self.original = None
        self.requested = None

    @property
    def available(self):
        return bool(self._max_khz) and os.access(self.path, os.W_OK)

    def _pct(self):
        if self.explicit_pct is not None:
            # Never go below min_perf_pct: the kernel would clamp it and the
            # read-back verification compares against expected_khz().
            floor = self._min_pct if self._min_pct else 1
            return max(floor, min(100, self.explicit_pct))
        if self.target_mode == "min":
            return self._min_pct if self._min_pct else 20
        if self._base_khz:
            pct = int(round(self._base_khz * 100.0 / self._max_khz))
            return max(1, min(100, pct))
        return 60

    def expected_khz(self):
        if not self._max_khz:
            return None
        return int(self._max_khz * self._pct() / 100.0)

    def apply(self, cpus=None, now_ns=None):
        self.original = _read_int(self.path)
        self.requested = self._pct()
        _write_int(self.path, self.requested)
        return {"requested": self.requested, "original": self.original,
                "cpus": list(self.cpus), "expected_khz": self.expected_khz()}

    def raw_value(self):
        return _read_int(self.path)

    def revert(self):
        if self.original is not None:
            _write_int(self.path, self.original)
            self.original = None

    def observed_khz(self):
        return _first_cpu_value("scaling_cur_freq", self.cpus)

    def describe(self):
        return {"name": self.name, "scope": self.scope, "path": self.path,
                "target_mode": self.target_mode, "max_khz": self._max_khz,
                "base_khz": self._base_khz, "min_perf_pct": self._min_pct,
                "request_value": self._pct(), "expected_khz": self.expected_khz()}


_BACKEND_CLASSES = {
    "intel_pstate_pct": _IntelPstatePct,
}

AUTO_ORDER = {
    "base": ("intel_pstate_pct",),
    "min": ("intel_pstate_pct",),
}


def build_backend(name, cpus, target_mode, verbose=True, perf_pct=None):
    cls = _BACKEND_CLASSES.get(name)
    if cls is None or target_mode not in cls.supports:
        return None
    return cls(cpus, target_mode, verbose=verbose, perf_pct=perf_pct)


# ------------------------------------------------------------------- controller
class FrequencyThrottler:
    """Apply a mitigation knob, verify that it works, and keep it applied.

    Parameters
    ----------
    backend
        ``auto`` (probe candidates in preference order), an explicit backend
        name, or ``none`` to disable mitigation.
    probe
        When true, each candidate backend is tested under load before use.
    """

    def __init__(self, cpus=None, cooldown_s=DEFAULT_COOLDOWN_S,
                 min_dwell_s=DEFAULT_MIN_DWELL_S, min_interval_s=DEFAULT_MIN_INTERVAL_S,
                 target="base", backend="auto", probe=True, probe_cpu=None,
                 probe_load_s=DEFAULT_PROBE_LOAD_S, probe_settle_s=DEFAULT_PROBE_SETTLE_S,
                 verify_interval_s=DEFAULT_VERIFY_INTERVAL_S,
                 reassert_interval_s=DEFAULT_REASSERT_INTERVAL_S,
                 perf_pct=None, verbose=True):
        self.cpus = list(cpus) if cpus else discover_cpus()
        self.cooldown_s = float(cooldown_s)
        self.min_dwell_s = float(min_dwell_s)
        self.min_interval_s = float(min_interval_s)
        self.target_mode = target if target in TARGET_MODES else "base"
        self.perf_pct = None if perf_pct is None else int(perf_pct)
        self.requested_backend = backend
        self.verbose = verbose
        self.probe_load_s = float(probe_load_s)
        self.probe_settle_s = float(probe_settle_s)
        self.verify_interval_s = float(verify_interval_s)
        self.reassert_interval_s = float(reassert_interval_s)
        self.probe_cpu = probe_cpu if probe_cpu is not None else (
            self.cpus[0] if self.cpus else None)

        self.events = []
        self.errors = []
        self.verifications = []
        self.reassert_count = 0
        self.verify_ok = 0
        self.verify_fail = 0

        self._active_since_ns = 0
        self._active_until_ns = 0
        self._last_trigger_ns = 0
        self._last_verify_ns = 0
        self._last_reassert_ns = 0
        self._active_state = None
        self._expected_khz = None
        self._active = False

        self.backend = None
        self.probe_result = {"performed": False, "ok": None, "reason": "not attempted"}
        self.backend_candidates = []

        self._select_backend(target, backend, probe)
        self.available = self.backend is not None and self._expected_khz is not None

    # ------------------------------------------------------------------ setup
    def _warn(self, message):
        if self.verbose:
            print("[cpufreq] WARNING: {}".format(message))

    def _info(self, message):
        if self.verbose:
            print("[cpufreq] {}".format(message))

    def _candidates(self, target, backend):
        if backend == "none":
            return []
        if backend and backend != "auto":
            return [backend]
        return list(AUTO_ORDER.get(target, ()))

    def _select_backend(self, target, backend, probe):
        if not self.cpus:
            self._warn("no CPU exposes a cpufreq policy; mitigation disabled")
            return
        for name in self._candidates(target, backend):
            candidate = build_backend(name, self.cpus, target, verbose=self.verbose,
                                      perf_pct=self.perf_pct)
            if candidate is None:
                self.backend_candidates.append({"name": name, "ok": False,
                                                "reason": "unsupported for target"})
                continue
            if not candidate.available:
                self.backend_candidates.append({"name": name, "ok": False,
                                                "reason": "unavailable"})
                continue
            result = self._capability_probe(candidate) if probe else \
                {"performed": False, "ok": True, "reason": "probe disabled"}
            self.backend_candidates.append({"name": name, "ok": bool(result.get("ok")),
                                            "reason": result.get("reason"),
                                            "observed_khz": result.get("observed_khz"),
                                            "expected_khz": result.get("expected_khz")})
            if result.get("ok"):
                self.backend = candidate
                self.probe_result = result
                self._expected_khz = candidate.expected_khz()
                self._info("backend selected: {} (scope={}){}".format(
                    candidate.name, candidate.scope,
                    "" if not probe else ", probe verified"))
                return
            self._warn("backend '{}' rejected: {}".format(name, result.get("reason")))
        self._warn("no effective mitigation backend on this machine "
                   "(tried: {})".format(", ".join(b["name"] for b in self.backend_candidates)
                                        or "none"))

    def _capability_probe(self, candidate):
        """Apply the knob under load and check that the frequency really drops."""
        if self.probe_cpu is None:
            return {"performed": False, "ok": False, "reason": "no probe cpu"}
        load = _spawn_load(self.probe_cpu)
        state = None
        try:
            if load is None:
                return {"performed": False, "ok": False,
                        "reason": "could not start probe load"}
            time.sleep(self.probe_load_s)
            before = candidate.observed_khz()
            try:
                state = candidate.apply()
            except OSError as exc:
                return {"performed": True, "ok": False, "reason": "write failed: {}".format(exc)}
            time.sleep(self.probe_settle_s)
            after = candidate.observed_khz()
            expected = candidate.expected_khz()
            if after is None or expected is None:
                return {"performed": True, "ok": False, "reason": "unreadable frequency"}
            ok = after <= expected * PROBE_TOLERANCE
            return {"performed": True, "ok": bool(ok),
                    "reason": None if ok else
                              "observed {} kHz stayed above expected ceiling {} kHz".format(
                                  after, expected),
                    "before_khz": before, "observed_khz": after, "expected_khz": expected,
                    "requested": state.get("requested")}
        finally:
            try:
                candidate.revert()
            except Exception:
                pass
            _stop_load(load)
            # let the frequency settle back so consecutive probes are comparable
            time.sleep(1.0)

    # ----------------------------------------------------------------- queries
    @property
    def scope(self):
        return self.backend.scope if self.backend else "none"

    def base_khz(self, cpu=None):
        """Base frequency in kHz, or None when unknown."""
        if self.backend is not None and hasattr(self.backend, "base_freq"):
            if cpu is not None:
                return self.backend.base_freq.get(cpu)
            return self.backend.expected_khz()
        return _first_cpu_value("base_frequency", self.cpus)

    def target_khz(self, cpu=None):
        """Frequency ceiling this controller applies, in kHz."""
        return self._expected_khz

    def is_active(self):
        return self._active

    # ----------------------------------------------------------------- actions
    def throttle(self, cpus=None, now_ns=None):
        """Apply the mitigation. Returns ``[(cpu, ceiling_khz), ...]``.

        The list reflects which CPUs are affected: for global backends all
        monitored CPUs are reported, because the knob is system wide.
        """
        if not self.available:
            return []
        now_ns = now_ns if now_ns is not None else time.monotonic_ns()
        if self.is_active():
            return []
        if now_ns - self._last_trigger_ns < int(self.min_interval_s * 1e9):
            return []

        try:
            state = self.backend.apply(cpus=cpus, now_ns=now_ns)
        except OSError as exc:
            self.errors.append({"event": "throttle_failed", "backend": self.backend.name,
                                "error": str(exc), "ts_ns": now_ns})
            return []

        affected = state.get("cpus") or []
        if not affected:
            return []

        ceiling = state.get("expected_khz")
        self._active_state = state
        self._active = True
        self._active_since_ns = now_ns
        self._active_until_ns = now_ns + int(self.cooldown_s * 1e9)
        self._last_trigger_ns = now_ns
        self._last_reassert_ns = now_ns
        self._last_verify_ns = now_ns
        self.events.append({
            "event": "throttle",
            "ts_ns": now_ns,
            "backend": self.backend.name,
            "scope": self.backend.scope,
            "requested": state.get("requested"),
            "original": state.get("original"),
            "expected_khz": ceiling,
            "cpus": affected,
            "target_mode": self.target_mode,
            "cooldown_s": self.cooldown_s,
        })
        return [(cpu, ceiling) for cpu in affected]

    def restore(self):
        """Revert the knob to its original value."""
        if not self._active or self.backend is None:
            return
        now_ns = time.monotonic_ns()
        try:
            self.backend.revert()
        except OSError as exc:
            self.errors.append({"event": "restore_failed", "backend": self.backend.name,
                                "error": str(exc), "ts_ns": now_ns})
        self.events.append({"event": "restore", "ts_ns": now_ns,
                            "backend": self.backend.name, "scope": self.backend.scope})
        self._active = False
        self._active_state = None
        self._active_since_ns = 0
        self._active_until_ns = 0

    def _reassert(self, now_ns):
        """Rewrite the knob when an external agent changed it behind our back."""
        if self.backend is None or self.target_mode is None:
            return
        state = self._active_state or {}
        raw = self.backend.raw_value()
        request = state.get("requested")
        if request in (None, "per-cpu"):
            return
        if raw is None or raw == request:
            return
        try:
            _write_int(self.backend.path, request)
            self.reassert_count += 1
            self.events.append({"event": "reassert", "ts_ns": now_ns,
                                "backend": self.backend.name,
                                "found": raw, "rewritten": request})
        except OSError as exc:
            self.errors.append({"event": "reassert_failed", "backend": self.backend.name,
                                "error": str(exc), "ts_ns": now_ns})

    def _verify(self, now_ns):
        """Read back the effective frequency and record whether it is capped."""
        if self.backend is None:
            return
        observed = self.backend.observed_khz()
        expected = self._expected_khz
        ok = (observed is not None and expected is not None
              and observed <= expected * PROBE_TOLERANCE)
        if ok:
            self.verify_ok += 1
        else:
            self.verify_fail += 1
        self.verifications.append({
            "ts_ns": now_ns, "observed_khz": observed, "expected_khz": expected,
            "ok": bool(ok),
        })

    def tick(self, now_ns=None):
        """Keep the mitigation applied, then release it after the cooldown."""
        if not self._active:
            return False
        now_ns = now_ns if now_ns is not None else time.monotonic_ns()

        if self.reassert_interval_s > 0 and \
                now_ns - self._last_reassert_ns >= int(self.reassert_interval_s * 1e9):
            self._reassert(now_ns)
            self._last_reassert_ns = now_ns
        if self.verify_interval_s > 0 and \
                now_ns - self._last_verify_ns >= int(self.verify_interval_s * 1e9):
            self._verify(now_ns)
            self._last_verify_ns = now_ns

        if now_ns < self._active_until_ns:
            return False
        if now_ns - self._active_since_ns < int(self.min_dwell_s * 1e9):
            return False
        self.restore()
        return True

    # ----------------------------------------------------------------- summary
    def summary(self):
        throttles = [e for e in self.events if e["event"] == "throttle"]
        restores = [e for e in self.events if e["event"] == "restore"]
        return {
            "available": self.available,
            "backend": self.backend.name if self.backend else None,
            "scope": self.scope,
            "backend_candidates": self.backend_candidates,
            "probe": self.probe_result,
            "requested_backend": self.requested_backend,
            "cpus": list(self.cpus),
            "target_mode": self.target_mode,
            "perf_pct": self.perf_pct,
            "target_freq_khz": self._expected_khz,
            "cooldown_s": self.cooldown_s,
            "min_dwell_s": self.min_dwell_s,
            "min_interval_s": self.min_interval_s,
            "verify_interval_s": self.verify_interval_s,
            "reassert_interval_s": self.reassert_interval_s,
            "probe_cpu": self.probe_cpu,
            "backend_detail": self.backend.describe() if self.backend else None,
            "throttle_count": len(throttles),
            "restore_count": len(restores),
            "reassert_count": self.reassert_count,
            "verify_ok": self.verify_ok,
            "verify_fail": self.verify_fail,
            "verifications": self.verifications,
            "error_count": len(self.errors),
            "events": self.events,
            "errors": self.errors,
        }

    def close(self):
        """Context-manager style cleanup: always restore before leaving."""
        try:
            self.restore()
        except Exception as exc:  # pragma: no cover - defensive
            self._warn("failed to restore mitigation on close: {}".format(exc))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


if __name__ == "__main__":  # pragma: no cover - manual smoke helper
    import argparse
    import json
    import signal

    parser = argparse.ArgumentParser(
        description="Probe and exercise the CPU mitigation backends")
    parser.add_argument("--cpus", type=str, default="", help="comma separated CPU ids")
    parser.add_argument("--apply", action="store_true", help="actually throttle/restore")
    parser.add_argument("--hold", type=float, default=0.0,
                        help="keep the mitigation applied for N seconds, then restore "
                             "(restores on SIGINT/SIGTERM too)")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--target", choices=TARGET_MODES, default="base",
                        help="'base' = stop turboing, 'min' = lowest allowed frequency")
    parser.add_argument("--perf-pct", type=int, default=None,
                        help="explicit intel_pstate/max_perf_pct cap depth (1-100). "
                             "Overrides the depth implied by --target and forces the "
                             "intel_pstate_pct backend")
    parser.add_argument("--backend", default="auto",
                        help="auto | {} | none".format(" | ".join(BACKENDS)))
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the under-load capability probe")
    parser.add_argument("--probe-cpu", type=int, default=None)
    args = parser.parse_args()

    cpu_list = [int(c) for c in args.cpus.split(",") if c.strip()] or None
    cooldown = max(args.cooldown, args.hold + 5.0) if args.hold > 0 else args.cooldown
    throttler = FrequencyThrottler(
        cpu_list, cooldown_s=cooldown, target=args.target,
        backend=args.backend, probe=not args.no_probe, probe_cpu=args.probe_cpu,
        perf_pct=args.perf_pct)

    def _handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("available:", throttler.available)
    print("backend  :", throttler.backend.name if throttler.backend else None)
    print("scope    :", throttler.scope)
    print("probe    :", json.dumps(throttler.probe_result))
    print("candidates:", json.dumps(throttler.backend_candidates))

    try:
        if args.apply or args.hold > 0:
            print("applied:", throttler.throttle(), flush=True)
        if args.hold > 0:
            deadline = time.monotonic() + args.hold
            while time.monotonic() < deadline:
                time.sleep(0.5)
                throttler.tick()
        elif args.apply:
            time.sleep(min(args.cooldown, 5.0))
            throttler.tick()
    except KeyboardInterrupt:
        print("mitigation holder: interrupted, restoring")
    finally:
        throttler.close()
    print(json.dumps(throttler.summary(), indent=2))

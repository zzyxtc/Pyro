#!/usr/bin/env bash
# Shared helpers for the attack-eval orchestration scripts.
#
# The experiment must run as root (insmod/rmmod + cpufreq writes), but PyTorch
# is usually installed in the *invoking user's* conda/venv, which root's default
# `python3` (e.g. /usr/bin/python3) does not see. resolve_python() therefore
# picks an interpreter that can actually `import torch`.

resolve_python() {
  # 1) explicit override
  if [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  local user_home="" candidates=()
  if [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != "root" ]; then
    user_home="$(getent passwd "${SUDO_USER}" 2>/dev/null | cut -d: -f6)"
  fi
  [ -z "$user_home" ] && user_home="$HOME"

  candidates+=(
    "$user_home/miniconda3/bin/python3"
    "$user_home/anaconda3/bin/python3"
    "$user_home/miniforge3/bin/python3"
    "$user_home/mambaforge/bin/python3"
    "$user_home/.venv/bin/python3"
    "$user_home/venv/bin/python3"
  )
  local onpath
  onpath="$(command -v python3 2>/dev/null || true)"
  [ -n "$onpath" ] && candidates+=("$onpath")

  local cand
  for cand in "${candidates[@]}"; do
    if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -c 'import torch' >/dev/null 2>&1; then
      printf '%s\n' "$cand"
      return 0
    fi
  done

  printf '%s\n' "${onpath:-python3}"
}

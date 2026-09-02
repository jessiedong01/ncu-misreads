#!/usr/bin/env bash
# Profile the five metrics that matter, plus achieved occupancy and duration.
set -euo pipefail

ARCH="${1:-sm_90}"          # sm_90 H100, sm_89 RTX 4090, sm_100 B200

METRICS=$(cat <<'M' | tr -d '\n'
sm__throughput.avg.pct_of_peak_sustained_elapsed,
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,
smsp__issue_active.avg.pct_of_peak_sustained_active,
sm__warps_active.avg.pct_of_peak_sustained_active,
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,
gpu__time_duration.sum,
launch__occupancy_limit_registers,
launch__registers_per_thread
M
)

echo "=== step 1: is ncu here and can it read the counters ==="
# Nsight Compute ships separately from the CUDA toolkit on most images, and
# when it is installed it usually is not on PATH.
if ! command -v ncu >/dev/null 2>&1; then
  FOUND=$(find /opt/nvidia /usr/local/cuda* /usr/local -maxdepth 4 -name ncu -type f 2>/dev/null | head -1)
  if [ -n "$FOUND" ]; then
    export PATH="$(dirname "$FOUND"):$PATH"
    echo "found ncu at $FOUND"
  else
    echo "ncu not installed. on Ubuntu with the CUDA repo:"
    echo "    sudo apt-get update && sudo apt-get install -y nsight-compute"
    echo "then run this script again."
    exit 1
  fi
fi
ncu --version | head -3

echo
echo "=== step 2: build ==="
nvcc -O3 -arch="$ARCH" -lineinfo ncu_demos.cu -o ncu_demos
echo "built"

echo
echo "=== step 3: permission smoke test ==="
# Write scratch next to the repo, not /tmp. With fs.protected_regular set, root
# cannot write a /tmp file owned by another user, and the failed redirect leaves
# you reading a stale log from the previous attempt.
SMOKE="$(mktemp "${PWD}/.ncu_smoke.XXXXXX")"
trap 'rm -f "$SMOKE"' EXIT
# The failure everyone hits: counters are admin-only by default. On a machine
# you own, running ncu under sudo is usually enough, so try that before telling
# anyone to reboot.
NCU=$(command -v ncu)
SUDO=""
if ! "$NCU" --metrics sm__cycles_elapsed.avg ./ncu_demos >"$SMOKE" 2>&1; then
  if command -v sudo >/dev/null && sudo "$NCU" --metrics sm__cycles_elapsed.avg ./ncu_demos >"$SMOKE" 2>&1; then
    SUDO="sudo"
    echo "counters readable under sudo, using it for the profile"
  fi
fi
if [ -z "$SUDO" ] && ! ncu --metrics sm__cycles_elapsed.avg --target-processes all ./ncu_demos >"$SMOKE" 2>&1; then
  echo "PROFILING FAILED. first 20 lines:"
  head -20 "$SMOKE"
  echo
  echo "If you see ERR_NVGPUCTRPERM, the GPU performance counters are locked to"
  echo "admin. Fixes, in order of how likely they are to work:"
  echo "  1. run as root, or with sudo"
  echo "  2. on your own machine, set the driver to allow all users:"
  echo "     https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters"
  echo "  3. in a container, it needs --cap-add=SYS_ADMIN at start time."
  echo "     You cannot fix this from inside a running pod. Pick another host."
  exit 1
fi
[ -z "$SUDO" ] && echo "counters readable"

echo
echo "=== step 4: profile ==="
$SUDO "$NCU" --csv --metrics "$METRICS" ./ncu_demos | tee ncu-results.csv

echo
echo "=== done. results in ncu-results.csv ==="

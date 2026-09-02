#!/usr/bin/env bash
# Profile the five metrics that matter, plus achieved occupancy and duration.
set -euo pipefail

ARCH="${1:-sm_89}"          # sm_89 RTX 4090, sm_90 H100, sm_100 B200

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
command -v ncu >/dev/null || { echo "ncu not on PATH. install Nsight Compute."; exit 1; }
ncu --version | head -3

echo
echo "=== step 2: build ==="
nvcc -O3 -arch="$ARCH" -lineinfo ncu_demos.cu -o ncu_demos
echo "built"

echo
echo "=== step 3: permission smoke test ==="
# The failure everyone hits: counters are admin-only by default.
if ! ncu --metrics sm__cycles_elapsed.avg --target-processes all ./ncu_demos >/tmp/ncu_smoke.txt 2>&1; then
  echo "PROFILING FAILED. first 20 lines:"
  head -20 /tmp/ncu_smoke.txt
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
echo "counters readable"

echo
echo "=== step 4: profile ==="
ncu --csv --metrics "$METRICS" ./ncu_demos | tee ncu-results.csv

echo
echo "=== done. results in ncu-results.csv ==="

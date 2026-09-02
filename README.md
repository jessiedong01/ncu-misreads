# ncu-misreads

Four kernels whose Nsight Compute reports are each misleading in a specific way.

The point is not that these kernels are interesting. It is that reading their
reports the usual way gives you the wrong answer, and reading five metrics in
one particular order gives you the right one.

## The five metrics

| metric | question |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | is compute busy |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | is memory busy |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | is anything issuing |
| `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` | stalled on what |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | is traffic wasted |

Achieved occupancy is collected too, but as evidence rather than as a target.

## Reading order

Start with the first two. If compute is high, find the saturated pipe. If dram
is high, go to sectors per request. If both are low, the kernel is latency
bound, so check issue slot utilization and then the dominant stall.

## The kernels

`fma_dependent` and `fma_ilp` do the same number of fused multiply-adds.
The first gives every thread one dependent chain, so a warp has one instruction
in flight and needs many warps to stay busy. The second gives every thread eight
independent chains, which costs registers and lowers occupancy.

`mem_covered` issues coalesced loads with enough warps to cover them. The warp
stall breakdown is dominated by long scoreboard while the issue slots stay busy.

`gather` runs twice over the same data, at stride 1 and stride 32. A warp
loading 32 contiguous 4 byte values touches 128 bytes, and a sector is 32 bytes,
so 4 sectors per request is the floor.

## Run

```bash
./run_ncu.sh sm_89
```

Use `sm_90` for an H100 or `sm_100` for a B200.

The script checks that Nsight Compute can actually read the performance counters
before profiling anything. That check exists because the counters are restricted
to admin by default, and inside most cloud containers the profile silently
collects nothing. If it fails there, no amount of rerunning helps; the container
needed `--cap-add=SYS_ADMIN` when it was created.

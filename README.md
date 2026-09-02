## More stuff on NCU!!

- [Using Nsight Compute to Inspect Your Kernels](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)
- [Better Performance at Lower Occupancy](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## A useful first pass

| Measurement | Question |
|---|---|
| SM throughput | How busy is the processing hardware? |
| DRAM throughput | How much of the off-chip memory bandwidth is being used? |
| Issue activity | How often does the GPU start a new instruction? |
| Warp stalls | What keeps groups of threads from continuing? |
| Sectors per request | Is each memory load using extra memory sectors? |

Start with SM and DRAM throughput. If neither is high, check that the kernel
launched enough work, then look at issue activity and warp stalls. Occupancy is
still useful context, but it is not a performance score.

## Results

### Memory reads

The same `gather` kernel loaded one FP32 value per thread. Only the distance
between the input addresses changed.

| Read pattern | Duration | Achieved occupancy | DRAM throughput | Sectors/request |
|---|---:|---:|---:|---:|
| Nearby reads (stride 1) | 24.800 us | 74.33% | 55.60% | 4 |
| Spread-out reads (stride 32) | 62.816 us | 85.66% | 24.11% | 32 |

The spread-out version was 2.53x slower even though its achieved occupancy was
higher. A warp requested 128 useful bytes in both cases. The second pattern
used 32 memory sectors instead of 4.

### Occupancy

These kernels performed the same total number of fused multiply-adds. One used
many warps with one dependent calculation per thread. The other used fewer
warps with 16 independent calculations per thread.

| Kernel | Duration | Achieved occupancy | SM throughput |
|---|---:|---:|---:|
| Many warps, one dependent chain | 248.800 us | 88.67% | 96.33% |
| Fewer warps, 16 independent chains | 250.336 us | 12.07% | 95.67% |

Occupancy was more than 7x lower, but runtime changed by less than 1%. The
independent calculations gave the GPU enough other work to do while an earlier
calculation was still finishing.

### Warp stalls

`long scoreboard` means warps were waiting for data from memory. Its value is
not a percentage of runtime.

| Kernel | Long scoreboard | DRAM throughput | Sectors/request |
|---|---:|---:|---:|
| In-order memory reads | 97.85 | 90.13% | 4 |
| Spread-out gather reads | 107.53 | 24.11% | 32 |

The first kernel was already near the DRAM bandwidth limit. The second used
eight times as many sectors because its reads were spread out. Almost the same
stall value pointed to a different thing to fix.

The complete measurements are in [`a100-results.csv`](a100-results.csv).

## Run

```bash
./run_ncu.sh sm_80   # A100
```

Use `sm_89` for an RTX 4090, `sm_90` for an H100, or `sm_100` for a B200.
The script builds the kernels, checks whether NCU can read the GPU performance
counters, and writes the output to `ncu-results.csv`. Some systems require
`sudo`; containers may need to be started with `SYS_ADMIN` access.

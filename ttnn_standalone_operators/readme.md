# TT-NN Standalone Operator Benchmarks

This folder contains standalone TT-NN operator benchmark scripts that
can be executed directly on WH or BH devices.

Each script allows controlled micro-benchmarking of a single TT-NN
operator and supports Tracy profiling.

------------------------------------------------------------------------

## Example Operator

File:

    standalone_binary_add.py

This script benchmarks the TT-NN `add` operator using tensors of shape
(N, C, H, W).

------------------------------------------------------------------------

## Basic Run Command

``` bash
python standalone_binary_add.py --batch 1 --c 64 --h 32 --w 32 --dtype fp16 --warmup 5 --iters 20 --device-id 0
```

------------------------------------------------------------------------

## Argument Description

### Tensor Shape

-   `--batch` → Batch size (N)
-   `--c` → Channels (C)
-   `--h` → Height (H)
-   `--w` → Width (W)

These define the tensor shape:

    (N, C, H, W)

Example:

    (1, 64, 32, 32)

------------------------------------------------------------------------

### Data Type

-   `--dtype fp16` → Host tensors created as `torch.float16`
-   `--dtype bf16` → Host tensors created as `torch.bfloat16`

Note: Device dtype is `ttnn.bfloat16`.

------------------------------------------------------------------------

### Performance Parameters

-   `--warmup` → Number of warmup iterations (not timed)
-   `--iters` → Number of timed iterations
-   `--device-id` → Selects TT-NN device (default: 0)

The script reports: - Average time per iteration (ms) - Approximate
bytes per operation - Effective bandwidth (GB/s)

------------------------------------------------------------------------

## Tracy Profiling

To run with Tracy profiler:

``` bash
python -m tracy -p -r -v standalone_binary_add.py --batch 1 --c 64 --h 32 --w 32 --dtype fp16 --warmup 5 --iters 20 --device-id 0
```

This generates `ops_perf_results_*.csv` which can be used with:

    ttnn_operator_analysis.py

for per-operator summary analysis.

------------------------------------------------------------------------

## Purpose

These standalone benchmarks are used for:

-   Operator-level micro-benchmarking
-   WH vs BH performance comparison
-   Tracy profiler analysis
-   Bandwidth validation
-   Kernel-level performance investigation

------------------------------------------------------------------------

Each operator (add, matmul, conv2d, reshape, softmax, transpose, etc.)
follows the same structure and CLI interface for consistency.

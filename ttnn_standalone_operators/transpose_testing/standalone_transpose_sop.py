Here is the SOP for the transpose benchmark using the updated script.

---

standalone_transpose_sop.txt

python standalone_transpose.py 1024 2048 --dtype bfloat8_b --fidelity HiFi4 --device-id 0 --iters 10

File name  
standalone_transpose.py

Description  
This script benchmarks TTNN transpose on a 2D tensor. It:

1. Creates a random host tensor A of shape `[M, N]`.  
2. Moves A to the device in TILE layout and the chosen dtype.  
3. Runs `ttnn.transpose(A, 0, 1)` on the device multiple times.  
4. Copies the result back to host, checks correctness vs PyTorch, and reports average time per transpose.

Positional arguments (matrix dimensions)

1024  
This is `M`: the number of rows of the input tensor A.

2048  
This is `N`: the number of columns of the input tensor A.

So:

- Input shape A: `[M, N] = [1024, 2048]`  
- Transposed shape Aᵀ: `[N, M] = [2048, 1024]`

Optional arguments

--dtype bfloat8_b  
Sets the **device data type** for A.

Accepted values in this script:

- `bf16` / `bfloat16` → `ttnn.bfloat16`  
- `fp32` / `float32` → `ttnn.float32`  
- `bfp8` / `bfloat8_b` / `bfp8_b` → `ttnn.bfloat8_b` (BFP8/BFLOAT8 type)

Host tensor A is always created as `torch.bfloat16` and then converted to the selected TTNN dtype on device.

In this example, `bfloat8_b` means the transpose is computed on a BFP8/BFLOAT8 tensor on device.

--fidelity HiFi4  
Maps to a math fidelity enum (LoFi, HiFi2, HiFi3, HiFi4) via `ttnn.MathFidelity`, but **plain transpose doesn’t actually use this**—it is kept only for interface consistency with your matmul tests and is printed in logs.

Allowed values (if present in your build):

- `LoFi`  
- `HiFi2`  
- `HiFi3`  
- `HiFi4`

--device-id 0  
Opens TTNN **device 0**. If your system has multiple Tenstorrent devices, you can change this (for example, `--device-id 1`) to run the benchmark on a different chip.

--iters 10  
Number of **timed transpose iterations** to run.

The script:

1. Runs 2 warmup transposes (not timed).  
2. Runs `iters` transposes under timing.  
3. Reports the **average time per transpose** in milliseconds.  

It also performs a correctness check against PyTorch’s `a_torch.transpose(0, 1)` and prints the maximum absolute difference.

Behavior summary

- Host A: `[M, N]` in `torch.bfloat16`.  
- Device A: `ttnn.from_torch(..., dtype=<device dtype>, layout=TILE_LAYOUT)`.  
- Transpose: `ttnn.transpose(a_tt, 0, 1)` → `[N, M]` on device.  
- Back to host: `ttnn.to_torch(a_T_tt)`.  
- Correctness: compare with `a_torch.transpose(0, 1)`; print `max abs diff`.  
- Performance: average time over `--iters` transposes.

Tracy profiling command (same arguments, wrapped with tracy)

python -m tracy -p -r -v standalone_transpose.py 1024 2048 --dtype bfloat8_b --fidelity HiFi4 --device-id 0 --iters 10

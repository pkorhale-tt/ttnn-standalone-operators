import sys
import math
import argparse
import time
import torch
import ttnn

def parse_dtype(dtype_str: str):
    s = dtype_str.lower()
    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32
    if s in ("bfp8", "bfloat8_b", "bfp8_b"):
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype string: {dtype_str}")

def parse_math_fidelity(fidelity_str: str):
    s = fidelity_str.lower()
    if s == "lofi":
        return ttnn.MathFidelity.LoFi
    if s == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if s == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if s == "hifi4":
        return ttnn.MathFidelity.HiFi4
    raise ValueError(f"Unsupported math fidelity: {fidelity_str}")

def run_matmul_reshape_test(
    M: int,
    K: int,
    N: int,
    target_shape: tuple[int, ...],
    num_cores: int | None,
    dtype_str: str,
    fidelity_str: str,
    num_runs: int = 10,
    device_id: int = 0,
):
    # Validate dimensions
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("M, K, N must be positive integers")

    total_elements = M * N
    if math.prod(target_shape) != total_elements:
        raise ValueError(
            f"Target shape {target_shape} has {math.prod(target_shape)} elements, "
            f"but M*N = {total_elements}. They must match."
        )

    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)

    print(f"\n=== Matmul + Reshape Test ===")
    print(f"M, K, N         : {M}, {K}, {N}")
    print(f"Result shape    : ({M}, {N})")
    print(f"Target reshape  : {target_shape}")
    print(f"Data type       : {dtype_str}")
    print(f"Math fidelity   : {fidelity_str}")
    print(f"Device ID       : {device_id}")
    print(f"Num runs        : {num_runs}")

    print(f"\nOpening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        grid_size = device.compute_with_storage_grid_size()
        total_cores = grid_size.x * grid_size.y

        if num_cores is None:
            target_num_cores = total_cores
        else:
            if num_cores <= 0:
                raise ValueError("num_cores must be positive")
            target_num_cores = min(num_cores, total_cores)

        if target_num_cores <= grid_size.x:
            core_grid = ttnn.CoreGrid(x=target_num_cores, y=1)
        else:
            x = min(grid_size.x, int(math.sqrt(target_num_cores)))
            y = min(grid_size.y, math.ceil(target_num_cores / x))
            core_grid = ttnn.CoreGrid(x=x, y=y)

        print(f"Device grid     : {grid_size.x} x {grid_size.y} = {total_cores} cores")
        print(f"Using cores     : {target_num_cores}")
        print(f"Matmul core_grid: {core_grid.x} x {core_grid.y} "
              f"(={core_grid.x * core_grid.y} cores)")

        # Host tensors
        print(f"\nCreating A[{M}, {K}] and B[{K}, {N}] (host bf16 → device {dtype_str})...")
        a_torch = torch.randn((M, K), dtype=torch.bfloat16)
        b_torch = torch.randn((K, N), dtype=torch.bfloat16)

        print("Moving tensors to device...")
        a_tt = ttnn.to_device(
            ttnn.from_torch(a_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b_tt = ttnn.to_device(
            ttnn.from_torch(b_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        print("Requested math_fidelity:", compute_config.math_fidelity)

        def matmul_then_reshape(a_dev, b_dev):
            c_dev = ttnn.matmul(
                a_dev,
                b_dev,
                compute_kernel_config=compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=core_grid,
            )
            c_reshaped = ttnn.reshape(c_dev, target_shape)
            return c_reshaped

        # Warmup
        for _ in range(num_runs):
            _ = matmul_then_reshape(a_tt, b_tt)

        ttnn.synchronize_device(device)

        # Timed runs
        start = time.time()
        for _ in range(num_runs):
            out = matmul_then_reshape(a_tt, b_tt)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_ms = (end - start) * 1000.0 / num_runs
        flops = 2 * M * K * N
        tflops = flops / (avg_time_ms * 1e-3) / 1e12

        print("\nMatmul + reshape completed.")
        print(f"Average time       : {avg_time_ms:.3f} ms")
        print(f"Result reshaped to : {target_shape}")
        print(f"Achieved throughput: {tflops:.2f} TFLOPs")

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)

def cli_parse_args(argv):
    parser = argparse.ArgumentParser("Matmul + Reshape test framework for Tenstorrent ttnn")
    parser.add_argument("M", type=int, help="Rows of A")
    parser.add_argument("K", type=int, help="Cols of A / Rows of B")
    parser.add_argument("N", type=int, help="Cols of B")

    parser.add_argument(
        "--target-shape",
        type=int,
        nargs="+",
        required=True,
        help="Target reshape shape for C (must have M*N elements)",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of cores to use (default: all available)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat8_b",
        help="Data type on device (e.g., bfloat8_b, bfp8, bf16, float32)",
    )
    parser.add_argument(
        "--fidelity",
        type=str,
        default="HiFi4",
        help="Math fidelity (e.g., LoFi, HiFi2, HiFi3, HiFi4)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of matmul+reshape iterations to run",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to open",
    )

    return parser.parse_args(argv)

def main():
    args = cli_parse_args(sys.argv[1:])
    try:
        run_matmul_reshape_test(
            M=args.M,
            K=args.K,
            N=args.N,
            target_shape=tuple(args.target_shape),
            num_cores=args.num_cores,
            dtype_str=args.dtype,
            fidelity_str=args.fidelity,
            num_runs=args.num_runs,
            device_id=args.device_id,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

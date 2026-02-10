import sys
import math
import argparse
import torch
import ttnn

# -------- Helpers to map strings to ttnn enums / dtypes --------


def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Extend this as needed based on what your ttnn build supports.
    """
    s = dtype_str.lower()

    # Common aliases
    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32

    # BFP8/BFLOAT8 variants – adapt these to your actual ttnn symbols
    if s in ("bfp8", "bfloat8_b", "bfp8_b"):
        # If your build uses a different symbol (e.g. ttnn.bfloat8, ttnn.bfp8_b),
        # change it here.
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def parse_math_fidelity(fidelity_str: str):
    """
    Map a string like 'LoFi', 'HiFi2', 'HiFi3', 'HiFi4' to ttnn.MathFidelity.
    Only HiFi4 is shown in your code; extend with what your SDK exposes.
    """
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


# -------- Core test runner --------


def run_matmul_test(
    M: int,
    K: int,
    N: int,
    num_cores: int | None,
    dtype_str: str,
    fidelity_str: str,
    num_runs: int = 10,
    device_id: int = 0,
):
    # Validate inputs
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("M, K, N must be positive integers")

    # Map string options to ttnn types/enums
    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)

    print(f"\n=== Matmul Test ===")
    print(f"M, K, N        : {M}, {K}, {N}")
    print(f"Data type      : {dtype_str}")
    print(f"Math fidelity  : {fidelity_str}")
    print(f"Device ID      : {device_id}")
    print(f"Num runs       : {num_runs}")

    print(f"\nOpening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # Build grid
        grid_size = device.compute_with_storage_grid_size()
        total_cores = grid_size.x * grid_size.y

        if num_cores is None:
            target_num_cores = total_cores  # default: all cores
        else:
            if num_cores <= 0:
                raise ValueError("num_cores must be positive")
            target_num_cores = min(num_cores, total_cores)

        print(f"Device grid    : {grid_size.x} x {grid_size.y} = {total_cores} cores")
        print(f"Using cores    : {target_num_cores}")

        # Simple rectangular grid selection
        if target_num_cores <= grid_size.x:
            core_grid = ttnn.CoreGrid(x=target_num_cores, y=1)
        else:
            x = min(grid_size.x, int(math.sqrt(target_num_cores)))
            y = min(grid_size.y, math.ceil(target_num_cores / x))
            core_grid = ttnn.CoreGrid(x=x, y=y)

        print(f"Matmul core_grid: {core_grid.x} x {core_grid.y} " f"(={core_grid.x * core_grid.y} cores)")

        # Create host tensors in higher-precision for initialization
        # (bfloat16 is a good default; can switch based on dtype if you want)
        print(f"\nCreating A[{M}, {K}] and B[{K}, {N}] (host bf16 → device {dtype_str})...")
        a_torch = torch.randn((M, K), dtype=torch.bfloat16)
        b_torch = torch.randn((K, N), dtype=torch.bfloat16)

        print("Moving tensors to device...")
        a_tt = ttnn.to_device(
            ttnn.from_torch(
                a_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        b_tt = ttnn.to_device(
            ttnn.from_torch(
                b_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Compute kernel config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        print("Requested math_fidelity:", compute_config.math_fidelity)

        # Run matmul multiple times
        for i in range(num_runs):
            print(
                f"Run {i+1}/{num_runs}: matmul ({dtype_str}, {fidelity_str}) "
                f"on {core_grid.x * core_grid.y} cores..."
            )
            output_tensor = ttnn.matmul(
                a_tt,
                b_tt,
                compute_kernel_config=compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=core_grid,
            )

        # Bring result back for inspection
        result_torch = ttnn.to_torch(output_tensor)

        print("\nMatmul completed.")
        print(f"A shape: {a_torch.shape}")
        print(f"B shape: {b_torch.shape}")
        print(f"C = A @ B shape: {result_torch.shape}")
        print(f"C dtype on host: {result_torch.dtype}")

        # Optional: basic correctness check vs. PyTorch on CPU (in higher precision)
        # Note: this can be expensive for large matrices; enable only when needed.
        # ref = (a_torch.to(torch.float32) @ b_torch.to(torch.float32))
        # diff = (result_torch.to(torch.float32) - ref).abs().max()
        # print(f"Max abs diff vs PyTorch fp32: {diff.item()}")

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)


# -------- CLI entrypoint --------


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Matmul test framework for Tenstorrent ttnn")
    parser.add_argument("M", type=int, help="Rows of A")
    parser.add_argument("K", type=int, help="Cols of A / Rows of B")
    parser.add_argument("N", type=int, help="Cols of B")

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
        help="Number of matmul iterations to run",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to open",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    try:
        run_matmul_test(
            M=args.M,
            K=args.K,
            N=args.N,
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

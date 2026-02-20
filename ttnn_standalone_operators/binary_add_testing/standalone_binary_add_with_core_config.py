import sys
import math
import argparse
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

def build_sub_core_grids(device, target_num_cores: int) -> ttnn.CoreRangeSet:
    """
    Build a CoreRangeSet selecting a contiguous x-by-y block of cores.
    NOTE: ttnn.add expects sub_core_grids: CoreRangeSet, NOT a list.
    """
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y

    if target_num_cores <= 0:
        raise ValueError("target_num_cores must be positive")

    target_num_cores = min(target_num_cores, total_cores)

    # Choose a simple rectangular region (0..x-1, 0..y-1)
    if target_num_cores <= grid_size.x:
        x = target_num_cores
        y = 1
    else:
        x = min(grid_size.x, int(math.sqrt(target_num_cores)))
        y = min(grid_size.y, math.ceil(target_num_cores / x))

    start = ttnn.CoreCoord(0, 0)
    end = ttnn.CoreCoord(x - 1, y - 1)

    core_range = ttnn.CoreRange(start, end)
    core_range_set = ttnn.CoreRangeSet([core_range])

    print(
        f"Sub-core grid (CoreRangeSet): [{start.x},{start.y}] -> [{end.x},{end.y}] "
        f"= {x} x {y} = {x*y} cores"
    )

    return core_range_set

def run_add_test(
    M: int,
    N: int,
    num_cores: int | None,
    dtype_str: str,
    fidelity_str: str,
    num_runs: int = 10,
    device_id: int = 0,
):
    if M <= 0 or N <= 0:
        raise ValueError("M, N must be positive integers")

    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)

    print(f"\n=== Eltwise Add Test (with sub_core_grids) ===")
    print(f"M, N           : {M}, {N}")
    print(f"Data type      : {dtype_str}")
    print(f"Math fidelity  : {fidelity_str}")
    print(f"Device ID      : {device_id}")
    print(f"Num runs       : {num_runs}")

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

        print(f"Device grid    : {grid_size.x} x {grid_size.y} = {total_cores} cores")
        print(f"Requested cores: {target_num_cores}")

        sub_core_grids = build_sub_core_grids(device, target_num_cores)

        print(f"\nCreating X[{M}, {N}] and Y[{M}, {N}] (host bf16 → device {dtype_str})...")
        x_torch = torch.randn((M, N), dtype=torch.bfloat16)
        y_torch = torch.randn((M, N), dtype=torch.bfloat16)

        print("Moving tensors to device...")
        x_tt = ttnn.to_device(
            ttnn.from_torch(
                x_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        y_tt = ttnn.to_device(
            ttnn.from_torch(
                y_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
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

        out_tt = None
        for i in range(num_runs):
            print(
                f"Run {i+1}/{num_runs}: add ({dtype_str}, {fidelity_str}) "
                f"on {target_num_cores} core(s)..."
            )

            # Key fix: sub_core_grids is a CoreRangeSet, NOT a list
            out_tt = ttnn.add(
                x_tt,
                y_tt,
                sub_core_grids=sub_core_grids,
            )

        result_torch = ttnn.to_torch(out_tt)

        print("\nEltwise add completed.")
        print(f"X shape: {x_torch.shape}")
        print(f"Y shape: {y_torch.shape}")
        print(f"Z = X + Y shape: {result_torch.shape}")
        print(f"Z dtype on host: {result_torch.dtype}")

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Eltwise add test with sub_core_grids")
    parser.add_argument("M", type=int, help="Rows")
    parser.add_argument("N", type=int, help="Cols")

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
        help="Number of add iterations to run",
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
        run_add_test(
            M=args.M,
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



'''
Command to run:
(Example for 1024 1024 matrix)

python -m tracy -p -r -v standalone_add.py 1024 1024 \
  --num-cores 1 \
  --dtype bfloat8_b \
  --fidelity HiFi4 \
  --num-runs 10 \
  --device-id 0

'''

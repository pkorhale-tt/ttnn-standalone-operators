import sys
import math
import torch
import ttnn


def main():
    # Usage: python matmul_wh_hifi4_bfp8.py M K N [num_cores]
    if len(sys.argv) not in (4, 5):
        print("Usage: python matmul_wh_hifi4_bfp8.py M K N [num_cores]")
        sys.exit(1)

    try:
        M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    except ValueError:
        print("Error: M, K, N must be integers.")
        sys.exit(1)

    if M <= 0 or K <= 0 or N <= 0:
        print("Error: M, K, N must be positive.")
        sys.exit(1)

    # Optional 4th arg: number of cores to use
    user_num_cores = None
    if len(sys.argv) == 5:
        try:
            user_num_cores = int(sys.argv[4])
        except ValueError:
            print("Error: num_cores must be an integer.")
            sys.exit(1)
        if user_num_cores <= 0:
            print("Error: num_cores must be positive.")
            sys.exit(1)

    print(f"\nOpening device 0...")
    device = ttnn.open_device(device_id=0)

    # Build a rectangular core_grid for matmul
    grid_size = device.compute_with_storage_grid_size()
    total_cores = grid_size.x * grid_size.y

    if user_num_cores is None:
        target_num_cores = total_cores  # default: use all cores
    else:
        target_num_cores = min(user_num_cores, total_cores)

    print(f"Device grid size: {grid_size.x} x {grid_size.y} = {total_cores} cores")
    print(f"Using {target_num_cores} cores for matmul")

    # Simple strategy: 1 row of target_num_cores, as long as it fits in X
    if target_num_cores <= grid_size.x:
        core_grid = ttnn.CoreGrid(x=target_num_cores, y=1)
    else:
        # Make a roughly square grid within device limits
        x = min(grid_size.x, int(math.sqrt(target_num_cores)))
        y = min(grid_size.y, math.ceil(target_num_cores / x))
        core_grid = ttnn.CoreGrid(x=x, y=y)

    print(f"Matmul core_grid: {core_grid.x} x {core_grid.y} (={core_grid.x * core_grid.y} cores)")

    try:
        print(f"Creating A[{M}, {K}] and B[{K}, {N}] (BFP8 target)...")
        # Host tensors in bfloat16 (better for random init), will be quantized to BFP8 on device
        a_torch = torch.randn((M, K), dtype=torch.bfloat16)
        b_torch = torch.randn((K, N), dtype=torch.bfloat16)

        print("Moving tensors to L1 on device as BFP8...")
        # NOTE: replace `ttnn.bfp8` below with the exact BFP8 dtype symbol
        # used in your ttnn version, e.g. ttnn.bfloat8_b, ttnn.bfp8_b, etc.
        a_tt = ttnn.to_device(
            ttnn.from_torch(
                a_torch,
                dtype=ttnn.bfloat8_b,  # <--- BFP8 input
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        b_tt = ttnn.to_device(
            ttnn.from_torch(
                b_torch,
                dtype=ttnn.bfloat8_b,  # <--- BFP8 input
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Explicit HiFi4 compute kernel config
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # <--- HiFi4
            math_approx_mode=False,  # HiFi usually non-approx
            fp32_dest_acc_en=False,  # keep dest in BFP8 path
            packer_l1_acc=True,
        )

        print("Requested math_fidelity:", compute_config.math_fidelity)

        # Run matmul 10 times
        for i in range(10):
            print(f"Run {i+1}/10: BFP8 HiFi4 matmul on {core_grid.x * core_grid.y} cores...")
            output_tensor = ttnn.matmul(
                a_tt,
                b_tt,
                compute_kernel_config=compute_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=core_grid,
            )

        result_torch = ttnn.to_torch(output_tensor)
        print("\nMatmul successful (BFP8 × BFP8 → BFP8 with HiFi4).")
        print(f"A shape: {a_torch.shape}")
        print(f"B shape: {b_torch.shape}")
        print(f"C = A @ B shape: {result_torch.shape}")
        print(f"C dtype on host: {result_torch.dtype}")

    finally:
        print("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

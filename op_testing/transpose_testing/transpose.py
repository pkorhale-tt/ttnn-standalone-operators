import torch
import ttnn


def run_transpose_example(M: int, N: int, device_id: int = 0):
    print(f"\nRunning transpose for tensor of shape [{M}, {N}]...")

    print(f"Opening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # 1. Create host tensor (PyTorch)
        print(f"Creating host tensor A[{M}, {N}] in bf16...")
        a_torch = torch.randn((M, N), dtype=torch.bfloat16)

        # 2. Move to device as tiles
        print("Moving A to device...")
        a_tt = ttnn.to_device(
            ttnn.from_torch(
                a_torch,
                dtype=ttnn.bfloat16,  # or bfloat8_b, etc.
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # 3. Transpose on device
        print("Running ttnn.transpose...")
        a_T_tt = ttnn.transpose(a_tt, dim0=0, dim1=1)

        # 4. Bring back to host and check
        a_T_torch = ttnn.to_torch(a_T_tt)

        print("Original shape (host):", a_torch.shape)
        print("Transposed shape (host):", a_T_torch.shape)

        # Optional correctness check vs PyTorch
        ref = a_torch.transpose(0, 1)
        max_diff = (a_T_torch.to(torch.float32) - ref.to(torch.float32)).abs().max()
        print("Max abs diff vs PyTorch transpose:", max_diff.item())

    finally:
        print("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    # You can change 256, 512, and device_id here
    run_transpose_example(256, 512, device_id=0)

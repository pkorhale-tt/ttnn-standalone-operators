import sys
import argparse
import torch
import ttnn

# -------- Helpers to map strings to ttnn enums / dtypes --------


def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Extend based on what your ttnn build actually supports.
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
    Transpose doesn't actually use this, but it's here to mirror matmul interface.
    """
    s = fidelity_str.lower()
    if hasattr(ttnn.MathFidelity, "LoFi") and s == "lofi":
        return ttnn.MathFidelity.LoFi
    if hasattr(ttnn.MathFidelity, "HiFi2") and s == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if hasattr(ttnn.MathFidelity, "HiFi3") and s == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if hasattr(ttnn.MathFidelity, "HiFi4") and s == "hifi4":
        return ttnn.MathFidelity.HiFi4

    raise ValueError(f"Unsupported math fidelity: {fidelity_str}")


# -------- Core transpose runner --------


def run_transpose_test(
    M: int,
    N: int,
    dtype_str: str,
    fidelity_str: str,
    device_id: int = 0,
):
    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers")

    # Map CLI strings to ttnn types
    dtype = parse_dtype(dtype_str)
    math_fidelity = parse_math_fidelity(fidelity_str)
    # Note: math_fidelity not used in plain transpose, but kept for consistency
    print(f"\n=== Transpose Test ===")
    print(f"Input shape     : {M} x {N}")
    print(f"Data type       : {dtype_str}")
    print(f"Math fidelity   : {fidelity_str} (not used by plain transpose)")
    print(f"Device ID       : {device_id}")

    print(f"\nOpening device {device_id}...")
    device = ttnn.open_device(device_id=device_id)

    try:
        # 1. Create host tensor
        print(f"Creating host tensor A[{M}, {N}] in bf16...")
        a_torch = torch.randn((M, N), dtype=torch.bfloat16)

        # 2. Move to device
        print("Moving A to device...")
        a_tt = ttnn.to_device(
            ttnn.from_torch(
                a_torch,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # 3. Transpose on device: swap dim 0 and dim 1
        print("Running ttnn.transpose...")
        # Correct signature is transpose(input_tensor, dim1, dim2, *, ...)
        a_T_tt = ttnn.transpose(a_tt, 0, 1)

        # 4. Bring back to host and verify
        a_T_torch = ttnn.to_torch(a_T_tt)

        print("Original shape (host):", a_torch.shape)
        print("Transposed shape (host):", a_T_torch.shape)

        # Basic correctness check vs PyTorch
        ref = a_torch.transpose(0, 1)  # [N, M]
        max_diff = (a_T_torch.to(torch.float32) - ref.to(torch.float32)).abs().max()
        print("Max abs diff vs PyTorch transpose:", max_diff.item())

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)


# -------- CLI handling --------


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Transpose test framework for Tenstorrent ttnn")
    parser.add_argument("M", type=int, help="Rows of input tensor")
    parser.add_argument("N", type=int, help="Cols of input tensor")

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
        help="Math fidelity (e.g., LoFi, HiFi2, HiFi3, HiFi4) – kept for consistency",
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
        run_transpose_test(
            M=args.M,
            N=args.N,
            dtype_str=args.dtype,
            fidelity_str=args.fidelity,
            device_id=args.device_id,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import time
import math
import torch
import ttnn

# -----------------------------
# Argument parsing
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser("Standalone TTNN Reshape benchmark")

    # Input tensor shape: N, C, H, W
    parser.add_argument("--batch", type=int, required=True, help="Batch size (N)")
    parser.add_argument("--c", type=int, required=True, help="Channels (C)")
    parser.add_argument("--h", type=int, required=True, help="Height (H)")
    parser.add_argument("--w", type=int, required=True, help="Width (W)")

    parser.add_argument(
        "--target-shape",
        type=int,
        nargs="+",
        required=True,
        help="Target reshape shape as a list of ints, e.g. --target-shape 1 1 2048 32",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Host dtype (device uses bfloat16)",
    )

    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations")

    parser.add_argument("--device-id", type=int, default=0)

    return parser.parse_args()


# -----------------------------
# Main benchmark
# -----------------------------


def main():
    args = parse_args()

    N = args.batch
    C = args.c
    H = args.h
    W = args.w

    target_shape = tuple(args.target_shape)

    # Check element counts match
    in_elems = N * C * H * W
    out_elems = math.prod(target_shape)
    if in_elems != out_elems:
        raise ValueError(
            f"Input elements ({in_elems}) != target elements ({out_elems}). "
            f"Shapes must have same number of elements."
        )

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    print("\n=== Standalone TTNN Reshape ===")
    print(f"Input shape       : (N={N}, C={C}, H={H}, W={W})")
    print(f"Target shape      : {target_shape}")
    print(f"Total elements    : {in_elems}")
    print(f"Host DType        : {args.dtype} (device uses bfloat16)")
    print(f"Warmup            : {args.warmup}")
    print(f"Iters             : {args.iters}")
    print(f"Device ID         : {args.device_id}\n")

    # -----------------------------
    # Device
    # -----------------------------
    print(f"Opening device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)

    try:
        # -----------------------------
        # Host tensor
        # -----------------------------
        x_torch = torch.randn((N, C, H, W), dtype=torch_dtype)

        # Move to device (TILE layout like other tests)
        x = ttnn.from_torch(
            x_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
        )

        # -----------------------------
        # Reshape op wrapper
        # -----------------------------
        def reshape_op(t):
            return ttnn.reshape(t, target_shape)

        # -----------------------------
        # Warmup
        # -----------------------------
        for _ in range(args.warmup):
            _ = reshape_op(x)

        ttnn.synchronize_device(device)

        # -----------------------------
        # Timed runs
        # -----------------------------
        start = time.time()
        for _ in range(args.iters):
            out = reshape_op(x)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_ms = (end - start) * 1000.0 / args.iters
        elems_per_s = in_elems / (avg_time_ms * 1e-3)

        print(f"Average time       : {avg_time_ms:.3f} ms")
        print(f"Elements per op    : {in_elems}")
        print(f"Throughput         : {elems_per_s / 1e9:.3f} Gelems/s\n")

    finally:
        print("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

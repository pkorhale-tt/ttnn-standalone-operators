import argparse
import time
import torch
import ttnn

# -----------------------------
# Argument parsing
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser("Standalone TTNN Softmax benchmark")

    # Tensor shape: N, C, H, W
    parser.add_argument("--batch", type=int, required=True, help="Batch size (N)")
    parser.add_argument("--c", type=int, required=True, help="Channels (C)")
    parser.add_argument("--h", type=int, required=True, help="Height (H)")
    parser.add_argument("--w", type=int, required=True, help="Width (W)")

    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        help="Axis for softmax (default: -1 for last dim, i.e., W)",
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

    axis = args.axis
    if axis < 0:
        # Convert negative axis to positive index for shape [N, C, H, W]
        axis = 4 + axis  # 4 dims total

    if axis not in (0, 1, 2, 3):
        raise ValueError("axis must be one of 0, 1, 2, 3 or negative index for those")

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    num_elements = N * C * H * W

    print("\n=== Standalone TTNN Softmax ===")
    print(f"Input shape       : (N={N}, C={C}, H={H}, W={W})")
    print(f"Softmax axis      : {axis} (0=N, 1=C, 2=H, 3=W)")
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
        # Softmax op wrapper
        # -----------------------------
        def softmax_op(t):
            # TTNN softmax typically takes axis/dim; use axis from CLI
            return ttnn.softmax(t, dim=axis)

        # -----------------------------
        # Warmup
        # -----------------------------
        for _ in range(args.warmup):
            _ = softmax_op(x)

        ttnn.synchronize_device(device)

        # -----------------------------
        # Timed runs
        # -----------------------------
        start = time.time()
        for _ in range(args.iters):
            out = softmax_op(x)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_ms = (end - start) * 1000.0 / args.iters
        elems_per_s = num_elements / (avg_time_ms * 1e-3)

        print(f"Average time       : {avg_time_ms:.3f} ms")
        print(f"Elements per op    : {num_elements}")
        print(f"Throughput         : {elems_per_s / 1e9:.3f} Gelems/s\n")

    finally:
        print("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

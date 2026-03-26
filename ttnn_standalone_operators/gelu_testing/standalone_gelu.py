import argparse
import time
import torch
import ttnn


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("Standalone TTNN GELU benchmark")

    parser.add_argument("--batch", type=int, required=True, help="Batch size (N)")
    parser.add_argument("--c", type=int, required=True, help="Channels (C)")
    parser.add_argument("--h", type=int, required=True, help="Height (H)")
    parser.add_argument("--w", type=int, required=True, help="Width (W)")

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

    parser.add_argument(
        "--fast-approx",
        action="store_true",
        help="Use fast and approximate GELU mode",
    )

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

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    num_elements = N * C * H * W
    bytes_per_element = 2  # bf16/fp16 = 2 bytes

    # GELU is unary → read + write
    approx_bytes_per_op = 2 * num_elements * bytes_per_element

    print("\n=== Standalone TTNN GELU ===")
    print(f"Input shape       : (N={N}, C={C}, H={H}, W={W})")
    print(f"Host DType        : {args.dtype} (device uses bfloat16)")
    print(f"Warmup            : {args.warmup}")
    print(f"Iters             : {args.iters}")
    print(f"Fast Approx       : {args.fast_approx}")
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
        a_torch = torch.randn((N, C, H, W), dtype=torch_dtype)

        a = ttnn.from_torch(
            a_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
        )

        # -----------------------------
        # GELU op
        # -----------------------------
        def gelu_op(x):
            return ttnn.gelu(
                x,
                fast_and_approximate_mode=args.fast_approx,
            )

        # -----------------------------
        # Warmup
        # -----------------------------
        for _ in range(args.warmup):
            _ = gelu_op(a)

        ttnn.synchronize_device(device)

        # -----------------------------
        # Timed runs
        # -----------------------------
        start = time.time()
        for _ in range(args.iters):
            out = gelu_op(a)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_ms = (end - start) * 1000.0 / args.iters
        gb_per_s = (approx_bytes_per_op / (avg_time_ms * 1e-3)) / 1e9

        print(f"Average time       : {avg_time_ms:.3f} ms")
        print(f"Approx bytes/op    : {approx_bytes_per_op / 1e6:.3f} MB")
        print(f"Effective bandwidth: {gb_per_s:.2f} GB/s\n")

    finally:
        print("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

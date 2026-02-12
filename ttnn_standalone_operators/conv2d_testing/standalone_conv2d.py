import argparse
import time
import math
import torch
import ttnn
from loguru import logger

# -----------------------------
# Argument parsing
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser("Standalone TTNN Conv2D benchmark (matching ttnn_basic_conv.py)")

    # Input tensor shape: N, C_in, H, W (PyTorch / BCHW)
    parser.add_argument("--batch", type=int, required=True, help="Batch size (N)")
    parser.add_argument("--c-in", type=int, required=True, help="Input channels (C_in)")
    parser.add_argument("--c-out", type=int, required=True, help="Output channels (C_out)")
    parser.add_argument("--h", type=int, required=True, help="Input height (H)")
    parser.add_argument("--w", type=int, required=True, help="Input width (W)")

    # Kernel; stride/padding are fixed to (1,1) like the tutorial
    parser.add_argument("--kernel-h", type=int, default=3, help="Kernel height (K_h)")
    parser.add_argument("--kernel-w", type=int, default=3, help="Kernel width (K_w)")

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Logical dtype for FLOP accounting; device uses bfloat16",
    )

    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Number of timed iterations")

    parser.add_argument("--device-id", type=int, default=0)

    return parser.parse_args()


# -----------------------------
# FLOPs estimation helper
# -----------------------------


def compute_conv2d_flops(N, C_in, C_out, H_in, W_in, K_h, K_w, stride_h=1, stride_w=1, pad_h=1, pad_w=1):
    H_out = math.floor((H_in + 2 * pad_h - K_h) / stride_h) + 1
    W_out = math.floor((W_in + 2 * pad_w - K_w) / stride_w) + 1

    flops = N * H_out * W_out * C_out * (C_in * K_h * K_w * 2)
    return flops, H_out, W_out


# -----------------------------
# Main benchmark
# -----------------------------


def main():
    args = parse_args()

    N = args.batch
    C_in = args.c_in
    C_out = args.c_out
    H = args.h
    W = args.w

    K_h = args.kernel_h
    K_w = args.kernel_w
    kernel_size = (K_h, K_w)

    # Tutorial uses stride=(1,1), padding=(1,1) hard-coded
    stride = (1, 1)
    padding = (1, 1)

    # FLOPs for reporting
    flops, H_out, W_out = compute_conv2d_flops(N, C_in, C_out, H, W, K_h, K_w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)

    print("\n=== Standalone TTNN Conv2D (matching ttnn_basic_conv.py style) ===")
    print(f"Input BCHW        : (N={N}, C_in={C_in}, H={H}, W={W})")
    print(f"Kernel            : (C_out={C_out}, C_in={C_in}, K_h={K_h}, K_w={K_w})")
    print(f"Stride            : {stride}")
    print(f"Padding           : {padding}")
    print(f"Estimated output  : (N={N}, C_out={C_out}, H_out={H_out}, W_out={W_out})")
    print(f"Logical DType     : {args.dtype} (device uses bfloat16)")
    print(f"Warmup            : {args.warmup}")
    print(f"Iters             : {args.iters}")
    print(f"Device ID         : {args.device_id}\n")

    # -----------------------------
    # Device (EXACTLY as tutorial)
    # -----------------------------
    logger.info(f"Opening device {args.device_id} with l1_small_size=8192 ...")
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=8192)

    try:
        # -----------------------------
        # Create input/weights/bias on device, like tutorial
        # -----------------------------
        # Input: BCHW -> created as TTNN tensor directly (ROW_MAJOR on device)
        x = ttnn.rand(
            (N, C_in, H, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Weights: (C_out, C_in, K_h, K_w)
        W = ttnn.rand(
            (C_out, C_in, *kernel_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Bias: [1, 1, 1, C_out]
        B = ttnn.zeros(
            (1, 1, 1, C_out),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # -----------------------------
        # Forward conv matching tutorial
        # -----------------------------
        def forward(
            input_tensor: ttnn.Tensor,
            weight_tensor: ttnn.Tensor,
            bias_tensor: ttnn.Tensor,
            out_channels: int,
            kernel_size: tuple,
            device: ttnn.Device,
        ) -> ttnn.Tensor:
            # BCHW -> NHWC
            permuted_input = ttnn.permute(input_tensor, (0, 2, 3, 1))

            # B, H, W, C (after permute)
            B_size, H_size, W_size, C_size = permuted_input.shape

            # Reshape to (1, 1, B*H*W, C)
            reshaped_input = ttnn.reshape(
                permuted_input,
                (1, 1, B_size * H_size * W_size, C_size),
            )

            conv_config = ttnn.Conv2dConfig(weights_dtype=weight_tensor.dtype)

            out = ttnn.conv2d(
                input_tensor=reshaped_input,
                weight_tensor=weight_tensor,
                bias_tensor=bias_tensor,
                in_channels=C_size,
                out_channels=out_channels,
                device=device,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=(1, 1),
                batch_size=1,
                input_height=1,
                input_width=B_size * H_size * W_size,
                conv_config=conv_config,
                groups=0,
            )

            return out

        # -----------------------------
        # Warmup
        # -----------------------------
        logger.info("Running warmup iterations...")
        for _ in range(args.warmup):
            _ = forward(x, W, B, C_out, kernel_size, device)

        ttnn.synchronize_device(device)

        # -----------------------------
        # Timed runs
        # -----------------------------
        logger.info("Running timed iterations...")
        start = time.time()
        for _ in range(args.iters):
            out = forward(x, W, B, C_out, kernel_size, device)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_ms = (end - start) * 1000.0 / args.iters
        tflops_per_s = flops / (avg_time_ms * 1e-3) / 1e12

        print(f"\nAverage time       : {avg_time_ms:.3f} ms")
        print(f"Estimated FLOPs    : {flops / 1e9:.3f} GFLOPs per conv")
        print(f"Achieved throughput: {tflops_per_s:.2f} TFLOPs/s")
        logger.info(f"Conv2D TTNN output shape: {out.shape}")

    finally:
        logger.info("Closing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

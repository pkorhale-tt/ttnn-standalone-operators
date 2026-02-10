import argparse
import time
import math
import torch
import ttnn

# -----------------------------
# Device helper
# -----------------------------


def open_default_device(device_id: int = 0):
    """
    Open TTNN device the same way as your working matmul code.
    """
    return ttnn.open_device(device_id=device_id)


# -----------------------------
# Argument parsing
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser("Standalone TTNN Conv2D benchmark")

    # Input tensor shape: N, C_in, H, W
    parser.add_argument("--batch", type=int, required=True, help="Batch size (N)")
    parser.add_argument("--c-in", type=int, required=True, help="Input channels (C_in)")
    parser.add_argument("--c-out", type=int, required=True, help="Output channels (C_out)")
    parser.add_argument("--h", type=int, required=True, help="Input height (H)")
    parser.add_argument("--w", type=int, required=True, help="Input width (W)")

    # Kernel + stride + padding
    parser.add_argument("--kernel-h", type=int, default=3, help="Kernel height (K_h)")
    parser.add_argument("--kernel-w", type=int, default=3, help="Kernel width (K_w)")
    parser.add_argument("--stride-h", type=int, default=1, help="Stride height")
    parser.add_argument("--stride-w", type=int, default=1, help="Stride width")
    parser.add_argument("--pad-h", type=int, default=1, help="Padding height")
    parser.add_argument("--pad-w", type=int, default=1, help="Padding width")

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Host input/output dtype (device will use bfloat16)",
    )

    parser.add_argument(
        "--math-fidelity",
        type=str,
        default="HiFi2",
        choices=["LoFi", "HiFi2", "HiFi4"],
        help="Math fidelity",
    )

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)

    parser.add_argument("--device-id", type=int, default=0)

    return parser.parse_args()


# -----------------------------
# FLOPs estimation helper
# -----------------------------


def compute_conv2d_flops(N, C_in, C_out, H_in, W_in, K_h, K_w, stride_h, stride_w, pad_h, pad_w):
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
    stride_h = args.stride_h
    stride_w = args.stride_w
    pad_h = args.pad_h
    pad_w = args.pad_w

    # Host dtypes
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    # Device dtype: use bfloat16
    ttnn_dtype = ttnn.bfloat16

    math_fidelity_map = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }
    math_fidelity = math_fidelity_map[args.math_fidelity]

    flops, H_out, W_out = compute_conv2d_flops(N, C_in, C_out, H, W, K_h, K_w, stride_h, stride_w, pad_h, pad_w)

    print("\n=== Standalone TTNN Conv2D ===")
    print(f"Input shape       : (N={N}, C_in={C_in}, H={H}, W={W})")
    print(f"Kernel shape      : (C_out={C_out}, C_in={C_in}, K_h={K_h}, K_w={K_w})")
    print(f"Stride            : ({stride_h}, {stride_w})")
    print(f"Padding           : ({pad_h}, {pad_w})")
    print(f"Output shape      : (N={N}, C_out={C_out}, H_out={H_out}, W_out={W_out})")
    print(f"Host DType        : {args.dtype} (device uses bfloat16)")
    print(f"Math Fidelity     : {args.math_fidelity}")
    print(f"Warmup            : {args.warmup}")
    print(f"Iters             : {args.iters}")
    print(f"Device ID         : {args.device_id}\n")

    # -----------------------------
    # Device
    # -----------------------------
    device = open_default_device(args.device_id)

    # -----------------------------
    # Tensors
    # -----------------------------
    # Host tensors
    x_torch = torch.randn((N, C_in, H, W), dtype=torch_dtype)
    w_torch = torch.randn((C_out, C_in, K_h, K_w), dtype=torch_dtype)

    # Input to device in TILE layout (like your matmul code)
    x = ttnn.from_torch(
        x_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )

    # Weights stay on host in ROW_MAJOR layout (required by conv2d)
    w = ttnn.from_torch(
        w_torch,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # -----------------------------
    # Compute config (for math fidelity)
    # -----------------------------
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # -----------------------------
    # Conv call wrapper
    # -----------------------------
    def conv2d_op(x_in, w_in):
        return ttnn.conv2d(
            input_tensor=x_in,
            weight_tensor=w_in,
            device=device,
            in_channels=C_in,
            out_channels=C_out,
            batch_size=N,
            input_height=H,
            input_width=W,
            kernel_size=(K_h, K_w),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(1, 1),
            groups=1,
            dtype=ttnn_dtype,
            bias_tensor=None,
            conv_config=None,
            compute_config=compute_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            slice_config=None,
            return_output_dim=False,
            return_weights_and_bias=False,
        )

    # -----------------------------
    # Warmup
    # -----------------------------
    for _ in range(args.warmup):
        _ = conv2d_op(x, w)

    ttnn.synchronize_device(device)

    # -----------------------------
    # Timed runs
    # -----------------------------
    start = time.time()
    for _ in range(args.iters):
        out = conv2d_op(x, w)

    ttnn.synchronize_device(device)
    end = time.time()

    avg_time_ms = (end - start) * 1000.0 / args.iters
    tflops_per_s = flops / (avg_time_ms * 1e-3) / 1e12

    print(f"Average time       : {avg_time_ms:.3f} ms")
    print(f"Estimated FLOPs    : {flops / 1e9:.3f} GFLOPs per conv")
    print(f"Achieved throughput: {tflops_per_s:.2f} TFLOPs/s\n")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()

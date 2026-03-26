# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice


# -------- Helpers to map strings to ttnn dtypes --------

def parse_dtype(dtype_str: str):
    """
    Map a string like 'bfloat8_b', 'bfp8', 'bf16', 'float32' to ttnn dtype.
    Adjust symbols to match your TTNN build if needed.
    """
    s = str(dtype_str).lower()

    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32
    if s in ("bfp8", "bfloat8_b", "bfp8_b", "bf8"):
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string for TTNN ReLU workload: {dtype_str}")


# -------- Polaris TTNN workload entry point --------

def run_relu_test(wlname: str, device: TTNNDevice, cfg: dict):
    """
    Polaris TTNN workload entry for a simple ReLU sweep.

    Args:
        wlname: Workload identifier from YAML (e.g., "ReluTensor").
        device: TTNN device managed by Polaris (do NOT open/close here).
        cfg:    Merged configuration dict from YAML + CLI overrides.

    Expected cfg keys (with defaults):
        bs        : int, batch size (required by Polaris infra, default 1)
        M         : int, rows of input tensor (default 1024)
        N         : int, cols of input tensor (default 1024)
        dtype     : str, device dtype ("bfloat8_b", "bf16", "float32", ...) (default "bfloat8_b")
        num_runs  : int, how many times to run relu (default 10)

    Returns:
        TTNN tensor result of the last ReLU, for Polaris graph capture.
    """
    # 1) Read config with safe defaults
    batchSize = int(cfg.get("bs", 1))

    M = int(cfg.get("M", cfg.get("m", 1024)))
    N = int(cfg.get("N", cfg.get("n", 1024)))

    numRuns = int(cfg.get("num_runs", cfg.get("runs", 10)))
    dtypeStr = cfg.get("dtype", "bfloat8_b")

    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers")

    logger.info(f"=== TTNN Polaris ReLU Workload: {wlname} ===")
    logger.info(f"M, N           : {M}, {N}")
    logger.info(f"Data type      : {dtypeStr}")
    logger.info(f"Num runs       : {numRuns}")
    logger.info(f"Batch size     : {batchSize}")

    # 2) Map dtype string to TTNN dtype
    dtype = parse_dtype(dtypeStr)

    # 3) Create input tensor directly on device
    # ReLU should ideally see both negative and positive values.
    # If your build has only _rand(), this will still run, but results may not exercise negatives well.
    logger.info(
        f"Creating TTNN input tensor X[{M}, {N}] on device with dtype={dtypeStr}..."
    )

    inputTensor = ttnn._rand(
        (M, N),
        dtype=dtype,
        device=device,
    )

    # 4) Run ReLU multiple times; result of last run is returned
    outputTensor = None
    for i in range(numRuns):
        logger.info(f"Run {i + 1}/{numRuns}: relu ({dtypeStr})")
        outputTensor = ttnn.relu(
            inputTensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    # 5) Optional: log result shape
    try:
        outShape = outputTensor.shape
    except Exception:
        outShape = "unknown"

    logger.info(f"ReLU completed; output shape (TTNN): {outShape}")
    logger.info("Returning output tensor for Polaris graph capture.")

    return outputTensor


'''
Workload:

  - api: TTNN
    name: ReluTensor
    basedir: tests/Relutest
    module: run_relu_test@ttnn_functional_relu.py
    instances:
      default: { bs: 1, M: 1024, N: 1024, dtype: bfloat8_b, num_runs: 10 }

Command:
    python polaris.py \
    --archspec  config/tt_wh.yaml \
    --wlspec    config/all_workloads.yaml \
    --wlmapspec config/wl2archmapping.yaml \
    --filterwl  ReluTensor \
    --filterwli default \
    --filterarch n150 \
    --study     RELU_WH \
    --odir      __RELU_WH \
    --dump_stats_csv
  
'''

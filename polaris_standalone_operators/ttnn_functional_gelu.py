# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice


def parse_dtype(dtype_str: str):
    s = str(dtype_str).lower()

    if s in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if s in ("fp32", "float32"):
        return ttnn.float32
    if s in ("bfp8", "bfloat8_b", "bfp8_b", "bf8"):
        return ttnn.bfloat8_b

    raise ValueError(f"Unsupported dtype string for TTNN GELU workload: {dtype_str}")


def run_gelu_test(wlname: str, device: TTNNDevice, cfg: dict):
    batchSize = int(cfg.get("bs", 1))

    M = int(cfg.get("M", cfg.get("m", 1024)))
    N = int(cfg.get("N", cfg.get("n", 1024)))

    numRuns = int(cfg.get("num_runs", cfg.get("runs", 10)))
    dtypeStr = cfg.get("dtype", "bfloat8_b")
    approximateMode = cfg.get("approximate", "none")

    if M <= 0 or N <= 0:
        raise ValueError("M and N must be positive integers")

    logger.info(f"=== TTNN Polaris GELU Workload: {wlname} ===")
    logger.info(f"M, N           : {M}, {N}")
    logger.info(f"Data type      : {dtypeStr}")
    logger.info(f"Num runs       : {numRuns}")
    logger.info(f"Batch size     : {batchSize}")
    logger.info(f"Approx mode    : {approximateMode}")

    dtype = parse_dtype(dtypeStr)

    logger.info(f"Creating TTNN input tensor X[{M}, {N}] on device with dtype={dtypeStr}...")

    inputTensor = ttnn._rand(
        (M, N),
        dtype=dtype,
        device=device,
    )

    outputTensor = None
    for i in range(numRuns):
        logger.info(f"Run {i + 1}/{numRuns}: gelu ({dtypeStr})")

        if approximateMode == "tanh":
            outputTensor = ttnn.gelu(
                inputTensor,
                approximate="tanh",
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            outputTensor = ttnn.gelu(
                inputTensor,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

    try:
        outShape = outputTensor.shape
    except Exception:
        outShape = "unknown"

    logger.info(f"GELU completed; output shape (TTNN): {outShape}")
    logger.info("Returning output tensor for Polaris graph capture.")

    return outputTensor


'''
Workload:

  - api: TTNN
    name: GeluTensor
    basedir: tests/Gelutest
    module: run_gelu_test@ttnn_functional_gelu.py
    instances:
      default: { bs: 1, M: 1024, N: 1024, dtype: bfloat8_b, num_runs: 10, approximate: tanh }

Command:

python polaris.py \
  --archspec  config/tt_wh.yaml \
  --wlspec    config/all_workloads.yaml \
  --wlmapspec config/wl2archmapping.yaml \
  --filterwl  GeluTensor \
  --filterwli default \
  --filterarch n150 \
  --study     GELU_WH \
  --odir      __GELU_WH \
  --dump_stats_csv
'''

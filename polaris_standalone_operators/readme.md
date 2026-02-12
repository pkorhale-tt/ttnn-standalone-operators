
# Polaris TTNN Standalone Operator Testing

This folder contains standalone TTNN operator workloads that can be
executed:

-   Individually on Polaris
-   As part of a workload configuration
-   For WH / BH architecture studies
-   For single-operator performance analysis

Each script defines a Polaris workload entry that can be mapped via
`all_workloads.yaml`.

------------------------------------------------------------------------

## Available Operators

-   `add`
-   `conv2d`
-   `matmul`
-   `reshape`
-   `softmax`
-   `transpose`

Each file exposes a workload function that Polaris can invoke.

------------------------------------------------------------------------

## Example: Binary Add Workload

``` yaml
Workload:
  - api: TTNN
    name: BinaryAdd
    basedir: tests/BinaryAdd
    module: run_binary_add@ttnn_functional_add.py
    instances:
      default:
        bs: 1
        N: 1
        C: 1
        H: 120
        W: 120
        dtype: fp16
        warmup: 5
        iters: 10
```

------------------------------------------------------------------------

## Running on Polaris

``` bash
python polaris.py   --archspec  config/tt_wh.yaml   --wlspec    config/all_workloads.yaml   --wlmapspec config/wl2archmapping.yaml   --filterwl  BinaryAdd   --filterwli default   --filterarch n150   --study     BINARY_ADD_WH   --odir      __BINARY_check_WH   --dump_stats_csv
```

------------------------------------------------------------------------

## Output

When executed with `--dump_stats_csv`, Polaris generates:

-   `*-opstats.csv`
-   `*-opstats.json`

These files can be used with:

`polaris_operator_analysis.py`

to generate per-operator summaries.

------------------------------------------------------------------------

## Purpose

These standalone operator workloads are designed for:

-   Operator-level benchmarking
-   WH vs BH comparison
-   Polaris performance validation
-   Micro-benchmarking of individual TTNN ops

# Operator Summary Tools

Small Python utilities for per-operator performance analysis.

## Files

- `polaris_operator_analysis.py` – Processes Polaris `*-opstats.csv`
- `ttnn_operator_analysis.py` – Processes TT-NN profiler `ops_perf_results_*.csv`

## Input Files

### Polaris

When running Polaris, it generates an `*-opstats.csv` file inside the `STATS/` directory.

Example:
`n150-TTNN-ReshapeTest-default-b1-opstats.csv`

This generated CSV file should be passed to:
`polaris_operator_analysis.py`


### TT-NN (WH / BH)

When running a model on WH or BH with the TT-NN profiler (Tracy) enabled, it generates:
`ops_perf_results_*.csv`

This profiler-generated CSV file should be passed to:
`ttnn_operator_analysis.py`


## Features

- Aggregates by operator
- Computes:
  - Total Call Count
  - Total Time (ms)
  - % of Total Runtime
- Prints summary to console
- Writes `summary_*.csv` next to the input file

## Usage

### Polaris
```bash
python3 polaris_operator_analysis.py <path-to-opstats.csv>

Example:
python3 polaris_operator_analysis.py STATS/n150-TTNN-ReshapeTest-default-b1-opstats.csv
```
### TT-NN
```bash
python3 ttnn_operator_analysis.py <path-to-ops_perf_results.csv>

Example:
python3 ttnn_operator_analysis.py ops_perf_results_2026_02_12_05_10_17.csv
```

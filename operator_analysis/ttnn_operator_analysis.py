#!/usr/bin/env python3
import sys
import os
import csv
from collections import defaultdict

def main():
    # --- CLI args ---
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ops_perf_results.csv>")
        sys.exit(1)

    input_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # --- Aggregate per operator ---
    agg = defaultdict(lambda: {"count": 0, "total_ns": 0.0})

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)

        op_field = "OP CODE"
        count_field = "GLOBAL CALL COUNT"
        dur_field = "DEVICE KERNEL DURATION [ns]"

        if op_field not in reader.fieldnames:
            print(f"Error: column '{op_field}' not found in CSV header")
            print("Header columns:", reader.fieldnames)
            sys.exit(1)
        if count_field not in reader.fieldnames:
            print(f"Error: column '{count_field}' not found in CSV header")
            print("Header columns:", reader.fieldnames)
            sys.exit(1)
        if dur_field not in reader.fieldnames:
            print(f"Error: column '{dur_field}' not found in CSV header")
            print("Header columns:", reader.fieldnames)
            sys.exit(1)

        for row in reader:
            op = (row.get(op_field) or "").strip()
            if not op:
                continue

            try:
                call_count = int(row.get(count_field, 0))
            except (ValueError, TypeError):
                call_count = 0

            try:
                dur_ns = float(row.get(dur_field, 0.0))
            except (ValueError, TypeError):
                dur_ns = 0.0

            agg[op]["count"] += call_count
            agg[op]["total_ns"] += dur_ns

    if not agg:
        print("Warning: no operator rows parsed; nothing to summarize.")
        sys.exit(0)

    # --- Build summary rows ---
    total_ns_all_ops = sum(v["total_ns"] for v in agg.values())
    total_ms_all_ops = total_ns_all_ops / 1e6 if total_ns_all_ops > 0 else 0.0

    summary_rows = []
    for op, v in agg.items():
        total_ms = v["total_ns"] / 1e6
        pct = (total_ms / total_ms_all_ops * 100.0) if total_ms_all_ops > 0 else 0.0
        summary_rows.append((op, v["count"], total_ms, pct))

    summary_rows.sort(key=lambda x: x[2], reverse=True)

    # --- Write CSV summary file (VERY explicit) ---
    base = os.path.basename(input_path)
    out_name = f"summary_{base.replace('.csv', '')}-opstats.csv"
    out_dir = os.path.dirname(input_path) or os.getcwd()
    out_path = os.path.join(out_dir, out_name)

    try:
        with open(out_path, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(
                ["Operator", "Count", "Total msec", "Percentage of time in the network"]
            )
            for op, count, total_ms, pct in summary_rows:
                writer.writerow([op, count, f"{total_ms:.6f}", f"{pct:.2f}"])
            writer.writerow(
                [
                    "Grand Total",
                    sum(r[1] for r in summary_rows),
                    f"{total_ms_all_ops:.6f}",
                    "100.00",
                ]
            )
    except Exception as e:
        print("Error while writing CSV summary:", e)
        print("Tried path:", out_path)
        sys.exit(1)

    # --- Pretty terminal output ---
    print(f"Summary saved to: {out_path}\n")
    print(f"Summary for: {base}")

    col_op = 20
    col_count = 10
    col_total_ms = 14
    col_pct = 50

    line_width = col_op + col_count + col_total_ms + col_pct
    print("-" * line_width)

    header_op = "Operator"
    header_count = "Count"
    header_total_ms = "Total msec"
    header_pct = "Percentage of time in the network"

    print(
        f"{header_op:<{col_op}}"
        f"{header_count:>{col_count}}"
        f"{header_total_ms:>{col_total_ms}}"
        f"{header_pct:>{col_pct}}"
    )

    for op, count, total_ms, pct in summary_rows:
        print(
            f"{op:<{col_op}}"
            f"{count:>{col_count}d}"
            f"{total_ms:>{col_total_ms}.6f}"
            f"{pct:>{col_pct}.2f}%"
        )

    print("-" * line_width)
    print(
        f"{'Grand Total':<{col_op}}"
        f"{sum(r[1] for r in summary_rows):>{col_count}d}"
        f"{total_ms_all_ops:>{col_total_ms}.6f}"
        f"{100.0:>{col_pct}.2f}%"
    )

if __name__ == "__main__":
    main()
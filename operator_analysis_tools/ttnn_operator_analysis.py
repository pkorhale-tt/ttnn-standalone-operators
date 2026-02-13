#!/usr/bin/env python3
import sys
import os
import csv
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ops_perf_results.csv>")
        sys.exit(1)

    input_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    agg = defaultdict(lambda: {
        "count": 0,
        "total_ns": 0.0,
        "cores": None
    })

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)

        op_field = "OP CODE"
        dur_field = "DEVICE KERNEL DURATION [ns]"
        core_field = "CORE COUNT"

        if op_field not in reader.fieldnames:
            print(f"Error: column '{op_field}' not found")
            sys.exit(1)

        if dur_field not in reader.fieldnames:
            print(f"Error: column '{dur_field}' not found")
            sys.exit(1)

        has_core_info = core_field in reader.fieldnames

        for row in reader:
            op = (row.get(op_field) or "").strip()
            if not op:
                continue

            try:
                dur_ns = float(row.get(dur_field, 0.0))
            except:
                dur_ns = 0.0

            agg[op]["count"] += 1
            agg[op]["total_ns"] += dur_ns

            if has_core_info and agg[op]["cores"] is None:
                try:
                    agg[op]["cores"] = int(row.get(core_field, 0))
                except:
                    agg[op]["cores"] = 0

    if not agg:
        print("No operator rows parsed.")
        sys.exit(0)

    total_ns_all = sum(v["total_ns"] for v in agg.values())
    total_ms_all = total_ns_all / 1e6 if total_ns_all > 0 else 0.0
    total_count_all = sum(v["count"] for v in agg.values())
    grand_avg_ms = total_ms_all / total_count_all if total_count_all > 0 else 0.0

    summary_rows = []
    for op, v in agg.items():
        total_ms = v["total_ns"] / 1e6
        count = v["count"]
        avg_ms = total_ms / count if count > 0 else 0.0
        pct = (total_ms / total_ms_all * 100.0) if total_ms_all > 0 else 0.0
        cores = v["cores"] if v["cores"] is not None else "N/A"

        summary_rows.append((op, count, total_ms, avg_ms, pct, cores))

    # Sort by total time descending
    summary_rows.sort(key=lambda x: x[2], reverse=True)

    # Write CSV
    base = os.path.basename(input_path)
    out_name = f"summary_{base.replace('.csv', '')}-opstats.csv"
    out_path = os.path.join(os.path.dirname(input_path), out_name)

    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "Operator",
            "Count",
            "Total msec",
            "Avg msec",
            "% Time",
            "Cores"
        ])

        for row in summary_rows:
            writer.writerow([
                row[0],
                row[1],
                f"{row[2]:.6f}",
                f"{row[3]:.6f}",
                f"{row[4]:.2f}",
                row[5]
            ])

        # Grand Total row
        writer.writerow([
            "Grand Total",
            total_count_all,
            f"{total_ms_all:.6f}",
            f"{grand_avg_ms:.6f}",
            "100.00",
            "-"
        ])

    # Pretty Print (Operator column width = 40)
    print(f"\nSummary saved to: {out_path}\n")
    print("-" * 130)
    print(f"{'Operator':<40}{'Count':>10}{'Total msec':>15}{'Avg msec':>15}{'% Time':>12}{'Cores':>10}")

    for op, count, total_ms, avg_ms, pct, cores in summary_rows:
        print(f"{op:<40}{count:>10}{total_ms:>15.6f}{avg_ms:>15.6f}{pct:>12.2f}%{str(cores):>10}")

    print("-" * 130)
    print(f"{'Grand Total':<40}{total_count_all:>10}{total_ms_all:>15.6f}{grand_avg_ms:>15.6f}{100.0:>12.2f}%{'-':>10}")
    print("-" * 130)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Generate operator-wise time table from Polaris CSV."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input Polaris opstats CSV file"
    )
    args = parser.parse_args()

    # Configuration
    OP_COL = "optype"
    TIME_COL = "msecs"

    # Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Validate columns
    if OP_COL not in df.columns or TIME_COL not in df.columns:
        print(f"Error: CSV must contain '{OP_COL}' and '{TIME_COL}' columns.")
        return

    # Convert time column to numeric
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0.0)

    # Group by operator
    grouped = df.groupby(OP_COL).agg(
        Count=(OP_COL, 'count'),
        Total_msec=(TIME_COL, 'sum')
    ).reset_index()

    # Calculate averages
    grouped["Avg msec"] = grouped["Total_msec"] / grouped["Count"]

    # Calculate total network time
    total_network_time = grouped["Total_msec"].sum()

    if total_network_time > 0:
        grouped["% Time"] = (
            grouped["Total_msec"] / total_network_time * 100
        )
    else:
        grouped["% Time"] = 0.0

    # Rename columns
    grouped = grouped.rename(columns={
        OP_COL: "Operator",
        "Total_msec": "Total msec"
    })

    # Reorder columns
    grouped = grouped[[
        "Operator",
        "Count",
        "Total msec",
        "Avg msec",
        "% Time"
    ]]

    # Sort alphabetically
    grouped = grouped.sort_values("Operator", ignore_index=True)

    # Grand Total row
    total_count = grouped["Count"].sum()
    total_msec = grouped["Total msec"].sum()
    grand_avg = total_msec / total_count if total_count > 0 else 0.0

    grand_total_row = pd.DataFrame({
        "Operator": ["Grand Total"],
        "Count": [total_count],
        "Total msec": [total_msec],
        "Avg msec": [grand_avg],
        "% Time": [100.0]
    })

    final_table = pd.concat([grouped, grand_total_row], ignore_index=True)

    # Pretty print (formatted like you requested)
    print(f"\nSummary for: {csv_path.name}")
    print("-" * 95)

    print(final_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
        formatters={
            "% Time": lambda x: f"{x:.2f}%"
        }
    ))

    print("-" * 95)

    # Save to CSV
    output_name = f"summary_{csv_path.stem}.csv"
    final_table.to_csv(output_name, index=False)

    print(f"Summary saved to: {output_name}")


if __name__ == "__main__":
    main()

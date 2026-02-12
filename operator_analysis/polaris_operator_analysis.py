#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path

def main():
    # 1) Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate operator-wise time table from Polaris CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the input Polaris opstats CSV file")
    args = parser.parse_args()

    # 2) Configuration
    OP_COL = "optype"
    TIME_COL = "msecs"

    # 3) Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 4) Data Cleaning
    if OP_COL not in df.columns or TIME_COL not in df.columns:
        print(f"Error: CSV must contain '{OP_COL}' and '{TIME_COL}' columns.")
        return

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0.0)

    # 5) Grouping Logic
    grouped = df.groupby(OP_COL).agg(
        Count=(OP_COL, 'count'),
        Total_msec=(TIME_COL, 'sum')
    ).reset_index()

    # Calculate Total Network Time for Percentage
    total_network_time = grouped["Total_msec"].sum()
    
    # Calculate Percentage
    grouped["Percentage"] = (grouped["Total_msec"] / total_network_time * 100) if total_network_time > 0 else 0

    # Rename for final output labels
    grouped = grouped.rename(columns={OP_COL: "Operator", "Total_msec": "Total msec", "Percentage": "Percentage of time in the network"})

    # Sort by operator name
    grouped = grouped.sort_values("Operator", ignore_index=True)

    # 6) Add Grand Total row
    grand_total_row = pd.DataFrame({
        "Operator": ["Grand Total"],
        "Count": [grouped["Count"].sum()],
        "Total msec": [grouped["Total msec"].sum()],
        "Percentage of time in the network": [100.0]
    })

    final_table = pd.concat([grouped, grand_total_row], ignore_index=True)

    # 7) Print Final Table
    print(f"\nSummary for: {csv_path.name}")
    print("-" * 80)
    print(final_table.to_string(
        index=False, 
        float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else x,
        formatters={
            "Percentage of time in the network": lambda x: f"{x:.2f}%"
        }
    ))
    print("-" * 80)

    # Optional: save to a result file automatically
    output_name = f"summary_{csv_path.stem}.csv"
    final_table.to_csv(output_name, index=False)
    print(f"Summary saved to: {output_name}")

if __name__ == "__main__":
    main()
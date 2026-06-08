#!/usr/bin/env python3

import json
import re
from collections import defaultdict

import pandas as pd


def load_mean_rmse(jsonl_file):
    rmse_values = defaultdict(list)

    with open(jsonl_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            for run in record["runs"]:
                label = run["label"]

                if label.startswith("arch"):
                    arch = int(label.replace("arch", ""))
                else:
                    # NAS architecture (aee7a1, etc.)
                    arch = 8

                rmse_values[arch].append(run["rmse"])

    return {
        arch: sum(vals) / len(vals)
        for arch, vals in rmse_values.items()
    }


def extract_arch(filename):
    """
    Examples:
        left_wall_dist_arch1.pt -> 1
        heading_error_arch7.pt -> 7
        track_width_arch8_trial6.pt -> 8
    """
    match = re.search(r"arch(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def main(parameter_csv, jsonl_file):
    df = pd.read_csv(parameter_csv)

    mean_rmse = load_mean_rmse(jsonl_file)

    df["arch"] = df["file"].apply(extract_arch)
    df["rmse"] = df["arch"].map(mean_rmse)

    # place rmse after file column
    cols = list(df.columns)
    cols.remove("rmse")
    cols.insert(1, "rmse")
    df = df[cols]

    output_file = parameter_csv.replace(".csv", "_with_rmse.csv")
    df.to_csv(output_file, index=False)

    print(f"Saved {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="parameter_summary.csv",
    )

    parser.add_argument(
        "--jsonl",
        required=True,
        help="evaluation jsonl file",
    )

    args = parser.parse_args()

    main(args.csv, args.jsonl)
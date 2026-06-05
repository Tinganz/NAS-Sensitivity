#!/usr/bin/env python3

from pathlib import Path
import torch
import pandas as pd


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_and_count(pt_path: Path):
    try:
        model = torch.jit.load(str(pt_path), map_location="cpu")
        total, trainable = count_parameters(model)
        return total, trainable, None
    except Exception as e:
        return None, None, str(e)


def main(base_dir: str):
    base_dir = Path(base_dir)

    targets = ["left_wall_dist", "track_width", "heading_error"]
    archs = range(1, 8)

    rows = []

    for target in targets:
        for arch in archs:
            pt_file = base_dir / f"{target}_arch{arch}.pt"

            total, trainable, err = load_and_count(pt_file)

            rows.append({
                "file": pt_file.name,
                "exists": pt_file.exists(),
                "total_params": total,
                "trainable_params": trainable,
                "error": err
            })

    df = pd.DataFrame(rows)
    out_path = Path("parameter_summary.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")

    print("\n=== Parameter Count Summary ===\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing .pt model files",
    )

    args = parser.parse_args()
    main(args.dir)
#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_DIR = Path("safety-nas/new_baseline/models")          # change this
DATASET = Path("safety-nas/datasets/combined_all.npz")
OUTPUT_CSV = "model_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

data = np.load(DATASET, allow_pickle=True)

scans = np.clip(data["scans"].astype(np.float32) / 10.0, 0, 1)
x = torch.from_numpy(scans).unsqueeze(1).to(DEVICE)


def get_target(name):
    if "heading" in name:
        return "heading_error"

    if "left_wall" in name:
        return "left_wall_dist"

    if "track_width" in name:
        return "track_width"

    raise RuntimeError(f"Cannot determine target from {name}")


results = []


# -------------------------------------------------
# Evaluate every model
# -------------------------------------------------

for model_path in sorted(MODEL_DIR.glob("*.pt")):

    target = get_target(model_path.name)

    if target == "track_width" and "track_width" not in data:
        y = (
            data["left_wall_dist"].astype(np.float32)
            + data["right_wall_dist"].astype(np.float32)
        )
    else:
        y = data[target].astype(np.float32)

    y = torch.from_numpy(y).to(DEVICE)

    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()

    with torch.no_grad():
        pred = model(x).squeeze()

    rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item()

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results.append(
        {
            "file": model_path.name,
            "target": target,
            "params": params,
            "trainable_params": trainable,
            "rmse": rmse,
        }
    )

    print(
        f"{model_path.name:<40}"
        f" Params={params:8d}"
        f" RMSE={rmse:.6f}"
    )


# -------------------------------------------------
# Save summary
# -------------------------------------------------

df = pd.DataFrame(results)
df = df.sort_values(["target", "params"])

print("\n")
print(df)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved results to {OUTPUT_CSV}")
import json
import re
import subprocess
import sys

def test_cnn_arch(left_wall_dist_filepath: str = "data/models/left_wall_dist_arch1.pt",
                track_width_filepath: str = "data/models/track_width_arch1.pt", 
                heading_error_filepath: str = "data/models/heading_error_arch1.pt") -> float:
    """returns RMSE using .pt weights as input"""
    
    CMD = [
        sys.executable,
        "packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py",
        "--planner",
        "dnn",
        "--map",
        "data/maps/F1/Spa/Spa_map",
        "--waypoints",
        "data/maps/F1/Spa/Spa_centerline.tsv",
        "--render-mode",
        "None",
        "--max-laps",
        "2",
        "--left-wall-model",
        left_wall_dist_filepath, # modified based on training
        "--track-width-model",
        track_width_filepath, # modified based on training
        "--heading-model",
        heading_error_filepath, # modified based on training
    ]

    proc = subprocess.run(CMD, capture_output=True, text=True, check=False)

    # extract RMSE logic
    match = re.search(r"(\{.*?\})\s*---", proc.stdout, re.DOTALL)
    if not match:
        # Fallback: try parsing everything from first { to last }
        match = re.search(r"(\{.*\})", proc.stdout, re.DOTALL)

    if not match:
        print("Could not find JSON in output:")
        print(proc.stdout)
        sys.exit(1)

    try:
        summary = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print("Raw output:")
        print(proc.stdout)
        sys.exit(1)

    rmse = summary["Cross-Track Error"]["crosstrack_rmse_m"]
    print(f"Cross-Track RMSE: {rmse:.4f} m")
    
    return rmse
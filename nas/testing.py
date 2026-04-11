import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Tuple

TrackConfig = Tuple[str, str]


def _run_single_track(
    map_filepath: str,
    waypoints_filepath: str,
    left_wall_dist_filepath: str,
    track_width_filepath: str,
    heading_error_filepath: str,
) -> float:
    """Execute the simulator for a single track and return its RMSE."""
    cmd = [
        sys.executable,
        "packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py",
        "--planner",
        "dnn",
        "--map",
        map_filepath,
        "--waypoints",
        waypoints_filepath,
        "--render-mode",
        "None",
        "--max-laps",
        "2",
        "--left-wall-model",
        left_wall_dist_filepath,
        "--track-width-model",
        track_width_filepath,
        "--heading-model",
        heading_error_filepath,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    match = re.search(r"(\{.*?\})\s*---", proc.stdout, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", proc.stdout, re.DOTALL)

    if not match:
        print("Could not find JSON in output:")
        print(proc.stdout)
        sys.exit(1)

    try:
        summary = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        print(f"JSON parse error: {exc}")
        print("Raw output:")
        print(proc.stdout)
        sys.exit(1)

    rmse = summary["Cross-Track Error"]["crosstrack_rmse_m"]
    print(f"[{map_filepath}] Cross-Track RMSE: {rmse:.4f} m")
    return rmse


def test_cnn_arch(
    left_wall_dist_filepath: str,
    track_width_filepath: str,
    heading_error_filepath: str,
    track_configs: Sequence[TrackConfig],
) -> tuple[float, list[float]]:
    """
    Run the CNN checkpoints against multiple tracks concurrently and return
    the aggregate RMSE statistics.
    """
    if not track_configs:
        raise ValueError("track_configs must include at least one map/waypoints pair")

    rmses: list[float] = [0.0 for _ in track_configs]
    with ThreadPoolExecutor(max_workers=len(track_configs)) as executor:
        futures: list[tuple[int, any]] = []
        for idx, (map_filepath, waypoints_filepath) in enumerate(track_configs):
            futures.append(
                (
                    idx,
                    executor.submit(
                        _run_single_track,
                        map_filepath,
                        waypoints_filepath,
                        left_wall_dist_filepath,
                        track_width_filepath,
                        heading_error_filepath,
                    ),
                )
            )

        for idx, future in futures:
            rmses[idx] = future.result()

    average_rmse = sum(rmses) / len(rmses)
    print(f"Average Cross-Track RMSE ({len(track_configs)} tracks): {average_rmse:.4f} m")
    return average_rmse, rmses


if __name__ == "__main__":
    test_cnn_arch(
        "data/models/left_wall_dist_arch8.pt",
        "data/models/track_width_arch8.pt",
        "data/models/heading_error_arch8.pt",
        [
            (
                "data/maps/F1/Sepang/Sepang_map",
                "data/maps/F1/Sepang/Sepang_centerline.tsv",
            ),
            (
                "data/maps/F1/YasMarina/YasMarina_map",
                "data/maps/F1/YasMarina/YasMarina_centerline.tsv",
            ),
            (
                "data/maps/F1/Austin/Austin_map",
                "data/maps/F1/Austin/Austin_centerline.tsv",
            ),
            (
                "data/maps/F1/Sakhir/Sakhir_map",
                "data/maps/F1/Sakhir/Sakhir_centerline.tsv",
            ),
            (
                "data/maps/F1/Melbourne/Melbourne_map",
                "data/maps/F1/Melbourne/Melbourne_centerline.tsv",
            ),
        ],
    )

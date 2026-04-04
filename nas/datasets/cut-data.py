from pathlib import Path

import numpy as np

def split_dataset(
    dataset_path: Path,
    output_dir: Path | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """
    Split an NPZ dataset into train/trial/test subsets.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    out_dir = output_dir or dataset_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": out_dir / f"{dataset_path.stem}_train.npz",
        "trial": out_dir / f"{dataset_path.stem}_trial.npz",
        "test": out_dir / f"{dataset_path.stem}_test.npz",
    }

    data = np.load(dataset_path)
    num_samples = data["scans"].shape[0]
    ratios_arr = np.array(ratios, dtype=np.float64)
    ratios_arr = ratios_arr / ratios_arr.sum()

    n_train = int(num_samples * ratios_arr[0])
    n_trial = int(num_samples * ratios_arr[1])
    n_test = num_samples - n_train - n_trial

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    split_indices = {
        "train": indices[:n_train],
        "trial": indices[n_train : n_train + n_trial],
        "test": indices[n_train + n_trial :],
    }

    for split_name, path in splits.items():
        idx = split_indices[split_name]
        arrays = {key: data[key][idx] for key in data.files}
        np.savez(path, **arrays)

    data.close()

def main() -> None:
    split_dataset("combined_all.npz")

if __name__ == "__main__":
    main()
# Standard Search

This folder runs the supervised comparison for the LiDAR CNN search.

The CTE search in `nas/` selects a triplet from simulator cross-track RMSE. This
search selects each LiDAR predictor independently from validation RMSE:

- `left_wall_dist`
- `track_width`
- `heading_error`

Set the values at the top of each script, then create fixed row splits once:

```bash
python standard-search/split_dataset.py
```

Run the three target searches together:

```bash
python standard-search/control-logic.py
```

Trial logs land in `standard-search/dnn-output/standard_trials_*.jsonl`.

Set the three `TARGET_FILES` entries in `test-best.py`, then retrain the
selected models on `train_validation.npz` and report their test RMSE:

```bash
python standard-search/test-best.py
```

The retrained `.pt` files beside the exported YAML files form the triplet to add
to `nas/compare-track.py`.

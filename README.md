# Safety NAS

Neural architecture search experiments for LiDAR-based F1TENTH reactive planners.
The NAS code trains three small 1D CNNs that estimate:

- `left_wall_dist`
- `track_width`
- `heading_error`

Those checkpoints are evaluated together in the F1TENTH simulator through the
DNN reactive planner.

## Repository Layout

- `nas/`: Optuna search, best-trial retraining, evaluation, and comparison scripts.
- `packages/f110_gym/`: local Gymnasium F1TENTH simulator package.
- `packages/f110_planning/`: planners, metrics, model utilities, and simulation helpers.
- `packages/f110_scripts/`: data generation, training, RL, and simulator entry scripts.
- `data/maps/`: map images, YAML metadata, and centerline waypoint files.
- `data/models/`: baseline trained checkpoint files.

## Setup

Use Python 3.12. Large maps and checkpoints may require Git LFS when cloning.

```bash
git lfs install
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is not available on your system, install Python 3.12 first or use
your environment manager of choice. The root `.python-version` file is included
for tools such as `pyenv`.

## NAS Workflow

Run an Optuna architecture search:

```bash
python nas/control-logic.py --track MELBOURNE --n-trials 120
```

The search writes JSONL results to:

```text
nas/dnn-output/nas_trials_*.jsonl
```

Pick the best trial from a NAS run and export training configs:

```bash
python nas/training.py --trials-file nas/dnn-output/nas_trials_<run>.jsonl
```

Export configs and retrain the best trial:

```bash
python nas/training.py --trials-file nas/dnn-output/nas_trials_<run>.jsonl --train
```

Evaluate the default checkpoint triple:

```bash
python nas/testing.py
```

Retrain or evaluate the configured best trials in `nas/test-best.py`:

```bash
python nas/test-best.py
```

Compare checkpoint triples across selected maps:

```bash
python nas/compare-track.py
```

## Data And Artifacts

Expected input dataset path for NAS training:

```text
nas/datasets/combined_all.npz
```

Common generated outputs:

- `nas/dnn-output/`
- `nas/dnn-output/trial_artifacts/`
- `nas/dnn-output/test-best-runs*/`
- `data/models/lightning_logs/`
- `data/models/checkpoints/`

These generated directories are intentionally ignored by Git. Keep important
trial JSONL files, final checkpoints, and datasets in a backed-up artifact store
or Git LFS if another machine needs to reproduce the same run.

## Quick Verification

After setup, this should parse the project metadata and import the local packages:

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); import f110_gym, f110_planning, f110_scripts"
```

To run tests:

```bash
pytest packages/f110_gym/tests packages/f110_planning/tests packages/f110_scripts/tests
```

## Notes

- The NAS search is computationally expensive because each trial trains three
  models and evaluates them in simulation.
- `nas/control-logic.sl`, `nas/test-best.sl`, and `nas/compare-track.sl` are
  Slurm wrappers for cluster runs.
- Some scripts contain experiment configuration directly in Python constants.
  Check the top of `nas/test-best.py` and `nas/compare-track.py` before running
  large batches.

## Attribution

The NAS work in this repository was created by Zayah Cortright in collaboration
with the Design Automation to X Lab at the University of North Carolina at
Chapel Hill Department of Computer Science.

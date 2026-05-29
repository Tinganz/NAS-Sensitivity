import argparse

import optuna

from cnn import EvaluationTrack, objective


def main(track: str | None = None, n_trials: int = 120) -> None:
    track_names = None
    if track:
        track_names = [EvaluationTrack[track.strip().upper()]]

    study = optuna.create_study(direction="minimize")
    if track_names:
        study.optimize(lambda t: objective(t, track_names=track_names, max_params=[200_000, 400_000, 800_000],), n_trials=n_trials)
    else:
        study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=120)
    args = parser.parse_args()
    main(track=args.track, n_trials=args.n_trials)

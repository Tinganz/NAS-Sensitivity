import optuna
# import optuna.visualization as vis
from cnn import objective

import logging
import warnings

def main() -> None:
    training_data: str = "f1tenth_ng_zc/nas/datasets/combined_train.npz"

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=75)

    # TODO fix figure output & following creation

if __name__ == "__main__":
    main()

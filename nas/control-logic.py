import optuna
# import optuna.visualization as vis
from cnn import objective

import logging
import warnings

def _print_info(best_rmse, best_params) -> str:
    print("OPTUNA TRIAL RESULTS: ----------------")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("Best params:", best_params)
    print("----------------")

def main() -> None:
    training_data: str = "f1tenth_ng_zc/nas/datasets/combined_train.npz"

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    _print_info(study.best_trial.value, study.best_trial.params)

    # TODO fix figure output & following creation

if __name__ == "__main__":
    main()

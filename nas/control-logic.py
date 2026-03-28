import optuna
import optuna.visualization as vis
from cnn import objective

import logging
import warnings

def main() -> None:
    # split set into 
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    print(f"Best RMSE: {study.best_trial.value:.4f}")
    print("Best params:", study.best_trial.params)
    
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html("./figures/fig1-poh.html")
    fig2 = vis.plot_param_importances(study)
    fig2.write_html("./figures/fig2-ppi.html")
    fig3 = vis.plot_parallel_coordinate(study)
    fig3.write_html("./figures/fig3-ppc.html")

if __name__ == "__main__":
    main()


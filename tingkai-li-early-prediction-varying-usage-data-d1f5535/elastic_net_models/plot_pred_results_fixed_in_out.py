import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_pred_results_fixed_in_out(training_error, true_pred_train,
                                    test_error_in, true_pred_in,
                                    test_error_out, true_pred_out,
                                    title_):

    plt.figure(figsize=(5, 5))
    plt.scatter(true_pred_train[:, 0], true_pred_train[:, 1], label="Training", color="#2E86C1", s=50, marker='o', edgecolor='k')
    plt.scatter(true_pred_in[:, 0], true_pred_in[:, 1], label="High DoD", color="#F39C12", s=50, marker='D', edgecolor='k')
    plt.scatter(true_pred_out[:, 0], true_pred_out[:, 1], label="Low DoD", color="#2ECC71", s=50, marker='s', edgecolor='k')

    lims = [3, 70]
    plt.plot(lims, lims, 'k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xticks([10, 20, 40, 60])
    plt.yticks([10, 20, 40, 60])
    plt.xlabel("True lifetime [weeks]")
    plt.ylabel("Predicted lifetime [weeks]")
    plt.title(title_)
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    T = pd.DataFrame({
        'MAE': [training_error[0], test_error_in[0], test_error_out[0]],
        'MAPE': [training_error[1], test_error_in[1], test_error_out[1]],
        'RMSE': [training_error[2], test_error_in[2], test_error_out[2]],
    }, index=['Training', 'In-distribution', 'Out-of-distribution'])

    print(T)

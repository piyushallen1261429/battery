import pandas as pd
import numpy as np
from fit_elastic_fixed_train_2_test import fit_elastic_fixed_train_2_test
from plot_pred_results_fixed_in_out import plot_pred_results_fixed_in_out

# Read feature files
DATA_DIR = 'tingkai-li-early-prediction-varying-usage-data-d1f5535/feature_extraction/'
train_table = pd.read_csv(DATA_DIR + 'training_discharge.csv',index_col=0)
# train_table = pd.read_csv("training.csv")
test_table_in = pd.read_csv(DATA_DIR + 'test_in_discharge.csv',index_col=0)
test_table_out = pd.read_csv(DATA_DIR + 'test_out_discharge.csv',index_col=0)

Y_train = np.log(train_table["Lifetime"].values)
Y_test_in = np.log(test_table_in["Lifetime"].values)
Y_test_out = np.log(test_table_out["Lifetime"].values)
# print(train_table.keys())
X_train = np.column_stack([
    train_table["Q_ini"].values,
    np.log(np.abs(train_table[["min_deltaQ", "var_deltaQ", "kurt_deltaQ", "skew_deltaQ"]].values))
])
X_test_in = np.column_stack([
    test_table_in["Q_ini"].values,
    np.log(np.abs(test_table_in[["min_deltaQ", "var_deltaQ", "kurt_deltaQ", "skew_deltaQ"]].values))
])
X_test_out = np.column_stack([
    test_table_out["Q_ini"].values,
    np.log(np.abs(test_table_out[["min_deltaQ", "var_deltaQ", "kurt_deltaQ", "skew_deltaQ"]].values))
])

print("Discharge Model")
train_error, true_pred_train, test_error, true_pred_test_in, true_pred_test_out, B, FitInfo = \
    fit_elastic_fixed_train_2_test(X_train, Y_train, X_test_in, Y_test_in, X_test_out, Y_test_out,
                                    n_sim=50, cv_=5, standardize_X=True, min_max_X=False,
                                    log_target=True, min_MSE_selection=False)

plot_pred_results_fixed_in_out(train_error, true_pred_train,
                               test_error[0], true_pred_test_in,
                               test_error[1], true_pred_test_out,
                               "Discharge Model")
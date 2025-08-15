import pandas as pd
import numpy as np

from fit_dummy import fit_dummy
from fit_elastic_fixed_train_2_test import fit_elastic_fixed_train_2_test
from plot_pred_results_fixed_in_out import plot_pred_results_fixed_in_out



# Load feature files
DATA_DIR = 'tingkai-li-early-prediction-varying-usage-data-d1f5535/feature_extraction/'
train_table = pd.read_csv(DATA_DIR + 'training.csv',index_col=0)
# train_table = pd.read_csv("training.csv")
test_table_in = pd.read_csv(DATA_DIR + 'test_in.csv',index_col=0)
test_table_out = pd.read_csv(DATA_DIR + 'test_out.csv',index_col=0)

# Log-transform lifetime
Y_train = np.log(train_table["Lifetime"].values)
Y_test_in = np.log(test_table_in["Lifetime"].values)
Y_test_out = np.log(test_table_out["Lifetime"].values)

# === Dummy model ===
_, train_error_dummy, true_pred_train = fit_dummy(Y_train, Y_train, True)
_, test_error_dummy_in, true_pred_test_in = fit_dummy(Y_train, Y_test_in, True)
_, test_error_dummy_out, true_pred_test_out = fit_dummy(Y_train, Y_test_out, True)

print("Dummy Model")
plot_pred_results_fixed_in_out(
    train_error_dummy, true_pred_train,
    test_error_dummy_in, true_pred_test_in,
    test_error_dummy_out, true_pred_test_out,
    "Dummy Model"
)

# === Condition model ===
# print(train_table)
X_train_cond = train_table[["Chg C-rate", "Dchg C-rate", "DoD"]].values
X_test_in_cond = test_table_in[["Chg C-rate", "Dchg C-rate", "DoD"]].values
X_test_out_cond = test_table_out[["Chg C-rate", "Dchg C-rate", "DoD"]].values

train_error_cond, true_pred_train_cond, test_error_cond, true_pred_test_in_cond, true_pred_test_out_cond, _, _ = \
    fit_elastic_fixed_train_2_test(
        X_train_cond, Y_train, X_test_in_cond, Y_test_in,
        X_test_out_cond, Y_test_out, 50, 5, True, False, True, False
    )

print("Condition Model")
plot_pred_results_fixed_in_out(
    train_error_cond, true_pred_train_cond,
    test_error_cond[0], true_pred_test_in_cond,
    test_error_cond[1], true_pred_test_out_cond,
    "Condition Model"
)

# === Degradation-informed model (2 features) ===
features_di2 = ["mean_dqdv_dchg_mid_3_0", "delta_CV_time_3_0"]
X_train_di2 = np.log(np.abs(train_table[features_di2].values))
X_test_in_di2 = np.log(np.abs(test_table_in[features_di2].values))
X_test_out_di2 = np.log(np.abs(test_table_out[features_di2].values))

train_error_di2, true_pred_train_di2, test_error_di2, true_pred_test_in_di2, true_pred_test_out_di2, _, _ = \
    fit_elastic_fixed_train_2_test(
        X_train_di2, Y_train, X_test_in_di2, Y_test_in,
        X_test_out_di2, Y_test_out, 50, 5, True, False, True, False
    )

print("Degradation-Informed Model (2 features)")
plot_pred_results_fixed_in_out(
    train_error_di2, true_pred_train_di2,
    test_error_di2[0], true_pred_test_in_di2,
    test_error_di2[1], true_pred_test_out_di2,
    "Degradation-Informed Model (2)"
)

# === Degradation-informed model (3 features) ===
X_train_di3 = np.column_stack((train_table["DoD"].values, X_train_di2))
X_test_in_di3 = np.column_stack((test_table_in["DoD"].values, X_test_in_di2))
X_test_out_di3 = np.column_stack((test_table_out["DoD"].values, X_test_out_di2))

train_error_di3, true_pred_train_di3, test_error_di3, true_pred_test_in_di3, true_pred_test_out_di3, _, _ = \
    fit_elastic_fixed_train_2_test(
        X_train_di3, Y_train, X_test_in_di3, Y_test_in,
        X_test_out_di3, Y_test_out, 50, 5, True, False, True, False
    )

print("Degradation-Informed Model (3 features)")
plot_pred_results_fixed_in_out(
    train_error_di3, true_pred_train_di3,
    test_error_di3[0], true_pred_test_in_di3,
    test_error_di3[1], true_pred_test_out_di3,
    "Degradation-Informed Model (3)"
)

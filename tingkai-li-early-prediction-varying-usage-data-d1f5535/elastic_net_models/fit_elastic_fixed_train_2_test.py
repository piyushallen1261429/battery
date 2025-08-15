import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import time

def compute_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    mdape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    rmdse = np.sqrt(np.median((y_true - y_pred) ** 2))
    return np.array([mae, mape, rmse, mdae, mdape, rmdse])

def fit_elastic_fixed_train_2_test(X_train, Y_train, X_test_1, Y_test_1, X_test_2, Y_test_2,
                                    n_sim=50, cv_=5, standardize_X=False, min_max_X=True,
                                    log_target=True, min_MSE_selection=False):

    start = time.time()

    if log_target:
        Y_train = np.log(Y_train)
        Y_test_1 = np.log(Y_test_1)
        Y_test_2 = np.log(Y_test_2)

    if min_max_X:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_1 = scaler.transform(X_test_1)
        X_test_2 = scaler.transform(X_test_2)

    if standardize_X:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_1 = scaler.transform(X_test_1)
        X_test_2 = scaler.transform(X_test_2)

    model = ElasticNetCV(
        l1_ratio=np.linspace(1e-5, 1, 101),
        alphas=np.linspace(1e-4, 5, 501),
        cv=cv_,
        n_jobs=-1,
        random_state=0,
        max_iter=10000
    )
    model.fit(X_train, Y_train)

    B = model.coef_
    intercept = model.intercept_

    y_hat_train = intercept + np.dot(X_train, B)
    y_hat_test_1 = intercept + np.dot(X_test_1, B)
    y_hat_test_2 = intercept + np.dot(X_test_2, B)

    if log_target:
        y_pred_train = np.exp(y_hat_train)
        y_true_train = np.exp(Y_train)

        y_pred_test_1 = np.exp(y_hat_test_1)
        y_true_test_1 = np.exp(Y_test_1)

        y_pred_test_2 = np.exp(y_hat_test_2)
        y_true_test_2 = np.exp(Y_test_2)
    else:
        y_pred_train = y_hat_train
        y_true_train = Y_train

        y_pred_test_1 = y_hat_test_1
        y_true_test_1 = Y_test_1

        y_pred_test_2 = y_hat_test_2
        y_true_test_2 = Y_test_2

    true_pred_train = np.column_stack((y_true_train, y_pred_train))
    true_pred_test_1 = np.column_stack((y_true_test_1, y_pred_test_1))
    true_pred_test_2 = np.column_stack((y_true_test_2, y_pred_test_2))

    training_error = compute_errors(y_true_train, y_pred_train)
    test_error_1 = compute_errors(y_true_test_1, y_pred_test_1)
    test_error_2 = compute_errors(y_true_test_2, y_pred_test_2)

    test_error = np.vstack((test_error_1, test_error_2))

    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")

    return training_error, true_pred_train, test_error, true_pred_test_1, true_pred_test_2, B, model

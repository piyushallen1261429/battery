import numpy as np

def fit_dummy(Y_train, Y_test, log_target=True):
    training_error = np.zeros(6)
    test_error = np.zeros(6)

    mean_train = np.mean(Y_train)
    y_hat_train = np.full(Y_train.shape, mean_train)
    y_hat_test = np.full(Y_test.shape, mean_train)

    if log_target:
        y_pred_train = np.exp(y_hat_train)
        y_true_train = np.exp(Y_train)
        true_pred_train = np.column_stack((y_true_train, y_pred_train))

        training_error[0] = np.mean(np.abs(y_true_train - y_pred_train))
        training_error[1] = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train))
        training_error[2] = np.sqrt(np.mean((y_true_train - y_pred_train) ** 2))
        training_error[3] = np.median(np.abs(y_true_train - y_pred_train))
        training_error[4] = np.median(np.abs((y_true_train - y_pred_train) / y_true_train))
        training_error[5] = np.sqrt(np.median((y_true_train - y_pred_train) ** 2))

        y_pred_test = np.exp(y_hat_test)
        y_true_test = np.exp(Y_test)
        true_pred_test = np.column_stack((y_true_test, y_pred_test))

        test_error[0] = np.mean(np.abs(y_true_test - y_pred_test))
        test_error[1] = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
        test_error[2] = np.sqrt(np.mean((y_true_test - y_pred_test) ** 2))
        test_error[3] = np.median(np.abs(y_true_test - y_pred_test))
        test_error[4] = np.median(np.abs((y_true_test - y_pred_test) / y_true_test))
        test_error[5] = np.sqrt(np.median((y_true_test - y_pred_test) ** 2))

    else:
        true_pred_train = np.column_stack((Y_train, y_hat_train))
        training_error[0] = np.mean(np.abs(Y_train - y_hat_train))
        training_error[1] = np.mean(np.abs((Y_train - y_hat_train) / Y_train))
        training_error[2] = np.sqrt(np.mean((Y_train - y_hat_train) ** 2))
        training_error[3] = np.median(np.abs(Y_train - y_hat_train))
        training_error[4] = np.median(np.abs((Y_train - y_hat_train) / Y_train))
        training_error[5] = np.sqrt(np.median((Y_train - y_hat_train) ** 2))

        true_pred_test = np.column_stack((Y_test, y_hat_test))
        test_error[0] = np.mean(np.abs(Y_test - y_hat_test))
        test_error[1] = np.mean(np.abs((Y_test - y_hat_test) / Y_test)) * 100
        test_error[2] = np.sqrt(np.mean((Y_test - y_hat_test) ** 2))
        test_error[3] = np.median(np.abs(Y_test - y_hat_test))
        test_error[4] = np.median(np.abs((Y_test - y_hat_test) / Y_test))
        test_error[5] = np.sqrt(np.median((Y_test - y_hat_test) ** 2))

    return training_error, test_error,true_pred_test

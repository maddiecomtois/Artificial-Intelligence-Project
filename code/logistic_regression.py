import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def logistic_regression_algorithm(x_train, x_test, y_train, kfold_data):

    # test_penalty_values(kfold_data)
    lr_model = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=200)
    lr_model.fit(x_train, y_train)

    lr_preds = lr_model.predict(x_test)
    return lr_preds, lr_model


def use_cross_validation(kfold_data, C):
    # This function uses k-fold cross validation to get the average ROC AUC
    # among the k tests, as well as the standard deviation among the costs.
    means = []
    # A model is trained using k-1 training segments and tested on the 1 remaining segment.
    for data in kfold_data:
        x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
        model = LogisticRegression(penalty='l2', C=C,
                                   solver='lbfgs', max_iter=200).fit(x_train, y_train)
        preds = model.predict_proba(x_test)
        means.append(roc_auc_score(y_test, preds, multi_class="ovo"))
    mean = np.mean(means)
    std = np.std(means)
    return mean, std


def plot_mean_and_std(x_range, x_name, mean_val, stds):
    # This function plots the mean ROC AUC and standard deviation of a model over a range of values.
    plt.errorbar(x_range, mean_val, yerr=stds)
    plt.xlabel(x_name)
    plt.ylabel('ROC AUC')
    plt.show()


def test_penalty_values(kfold_data):
    # This function tests different C penalty values on a
    # kernalised ridge regression model using cross validation.
    means = []
    stds = []
    C_vals = [0.001, 0.01, 0.1, 1, 10]
    for C in C_vals:
        mean, std = use_cross_validation(kfold_data, C=C)
        print("Average roc auc with C penalty value = {}: {} (+/− {})".format(
            C, mean, std))
        means.append(mean)
        stds.append(std)
    plot_mean_and_std(C_vals, "C", means, stds)

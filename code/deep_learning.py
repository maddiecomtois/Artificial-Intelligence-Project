from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from matplotlib import pyplot as plt


# run the MLP classifier on the training data and make predictions
def deep_learning_algorithm(x_train, x_test, y_train, kfold_data):
    # instantiate the classifier:
    # - 1 hidden layer of 20 nodes
    # - relu activation function
    # - adam is solver for weight optimisation

    #test_penalty_values(kfold_data)
    #test_values_for_layers(kfold_data)

    mlp = MLPClassifier(hidden_layer_sizes=20, activation='relu', solver='adam', max_iter=300, alpha=1.0 / 5)
    mlp.fit(x_train, y_train)

    # generate predictions
    predict_test = mlp.predict(x_test)
    return predict_test, mlp


def use_cross_validation_for_alpha(kfold_data, C):
    # This function uses k-fold cross validation to get the average ROC AUC
    # among the k tests depending on different alpha values, as well as the standard deviation among the costs.
    # The model was trained using a hidden layer of 2 nodes
    means = []
    # A model is trained using k-1 training segments and tested on the 1 remaining segment.
    i = 1
    for data in kfold_data:
        print("Kfold: ", i)
        x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
        model = MLPClassifier(hidden_layer_sizes=2, activation='relu', solver='adam', max_iter=300, alpha=1.0 / C).fit(x_train, y_train)
        preds = model.predict_proba(x_test)
        means.append(roc_auc_score(y_test, preds, multi_class="ovo"))
        i = i + 1
    mean = np.mean(means)
    std = np.std(means)
    return mean, std


def use_cross_validation_for_layers(kfold_data, layer_range):
    # This function uses k-fold cross validation to get the average ROC AUC
    # among the k tests for different number of hidden layers, as well as the standard deviation among the costs.
    means = []
    i = 1
    # A model is trained using k-1 training segments and tested on the 1 remaining segment.
    for data in kfold_data:
        print("Kfold: ", i)
        x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
        model = MLPClassifier(hidden_layer_sizes=layer_range, activation='relu', solver='adam', max_iter=300, alpha=1.0 / 5).fit(x_train, y_train)
        preds = model.predict_proba(x_test)
        means.append(roc_auc_score(y_test, preds, multi_class="ovo"))
        i = i + 1
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
    # This function tests different C penalty values on an
    # MLP classifier using cross validation.
    means = []
    stds = []

    # check different penalty values to use for alpha
    C_vals = [1, 5, 10, 20, 100]

    for C in C_vals:
        mean, std = use_cross_validation_for_alpha(kfold_data, C=C)
        print("Average roc auc with C penalty value = {}: {} (+/??? {})".format(
            C, mean, std))
        means.append(mean)
        stds.append(std)
    plot_mean_and_std(C_vals, "C", means, stds)


def test_values_for_layers(kfold_data):
    # This function tests different hidden layer values on an
    # MLP classifier using cross validation.
    means = []
    stds = []

    # check with one layer and different number of nodes
    hidden_layer_range = [2, 5, 10, 20, 25, 40, 50, 75, 100]
    # check with different layers and different number of nodes
    (layer_val1, layer_val2, layer_val3) = [(150, 100, 50), [40, 20], (10, 3), (50, 15, 10)]

    for layer_range in hidden_layer_range:
        mean, std = use_cross_validation_for_layers(kfold_data, layer_range=layer_range)
        print("Average roc auc with hidden layer range = {}: {} (+/??? {})".format(
            layer_range, mean, std))
        means.append(mean)
        stds.append(std)
    plot_mean_and_std(hidden_layer_range, "C", means, stds)

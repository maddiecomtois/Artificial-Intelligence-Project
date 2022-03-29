import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

def knn_algorithm(x_train, x_test, y_train, kfold_data):

    knn_model = KNeighborsClassifier(n_neighbors = 225, weights = 'distance')
    knn_model.fit(x_train, y_train)

    knn_preds = knn_model.predict(x_test)
    return knn_preds, knn_model


"""
knn_model = KNeighborsClassifier(n_neighbors=150, weights='uniform')
knn_model.fit(x_train, y_train)
"""

# CROSS VALIDATION CODE FROM HERE DOWN --> chose k = 225 with 'distance' weighting
import pandas as pd
from algorithms import preprocess_input
from algorithms import get_kfold_data

def use_cross_validation(kfold_data, k):
    # This function uses k-fold cross validation to get the average ROC AUC
    # among the k tests, as well as the standard deviation among the costs.
    means = []
    # A model is trained using k-1 training segments and tested on the 1 remaining segment.
    for data in kfold_data:
        x_train, y_train, x_test, y_test = data[0], data[1], data[2], data[3]
        model = KNeighborsClassifier(n_neighbors = k, weights = 'distance').fit(x_train, y_train)
        preds = model.predict_proba(x_test)
        means.append(roc_auc_score(y_test, preds, multi_class = "ovo"))
    mean = np.mean(means)
    std = np.std(means)
    return mean, std


def plot_mean_and_std(x_range, x_name, mean_val, stds):
    # This function plots the mean ROC AUC and standard deviation of a model over a range of values.
    plt.errorbar(x_range, mean_val, yerr = stds)
    plt.xlabel(x_name)
    plt.ylabel('ROC AUC')
    plt.show()


def test_k_values(kfold_data):
    # This function tests different k values 
    # on a knn model using cross validation.
    means = []
    stds = []
    k_values = [50, 100, 150, 200, 225, 250, 300]    
    for k in k_values:
        mean, std = use_cross_validation(kfold_data, k)
        print("Average roc auc with k = {}: {} (+/− {})".format(
            k, mean, std))
        means.append(mean)
        stds.append(std)
    plot_mean_and_std(k_values, "k", means, stds)


def main():
    print("Reading in training data...")
    # read in training set
    df_train = pd.read_json('../data/pre_train.json')

    print("Reading in test data...")
    # read in test set
    df_test = pd.read_json('../data/pre_test.json')

    train_x1 = np.array(df_train['posts'])
    train_x2 = np.array([df_train['genders']])
    y_train = np.array(df_train['group_ages'])

    test_x1 = np.array(df_test['posts'])
    test_x2 = np.array([df_test['genders']])
    y_test = np.array(df_test['group_ages'])

    print("Processing data...")
    # preprocess posts and gender input data
    x_train, x_test = preprocess_input(train_x1, train_x2, test_x1, test_x2)
    
    kfold_data = get_kfold_data(train_x1, train_x2, y_train)
    
    test_k_values(kfold_data)


if __name__ == "__main__":
    main()

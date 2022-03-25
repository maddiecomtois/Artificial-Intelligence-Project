import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             multilabel_confusion_matrix, roc_curve)
from sklearn.model_selection import KFold

from deep_learning import deep_learning_algorithm
from knn import knn_algorithm
from logistic_regression import logistic_regression_algorithm


def plot_roc(x_test, y_test):
    # algorithm 1
    fpr, tpr, _ = roc_curve(y_test, logisticModel.decision_function(x_test))
    plt.plot(fpr, tpr, label="Logistic")
    print("\nAUC Logistic:", auc(fpr, tpr))

    # algorithm 2
    y_scores_knn = knn_model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores_knn[:, 1])
    plt.plot(fpr, tpr, label="KNN")
    print("AUC KNN:", auc(fpr, tpr))

    # algorithm 3
    y_scores_deep = deep_model.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores_deep[:, 1])
    plt.plot(fpr, tpr, label="Deep Learning")
    print("AUC Deep Learning:", auc(fpr, tpr))

    plt.title('ROC - Logistic, KNN, and Deep Learning')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()


def preprocess_input(train_text, train_gender, test_text, test_gender):
    train_gender = np.swapaxes(train_gender, 0, 1)
    test_gender = np.swapaxes(test_gender, 0, 1)

    # use Bag of Words to transform text into numbers
    count_vect = CountVectorizer()
    train_text_counts = count_vect.fit_transform(train_text)
    train_text_counts = sparse.hstack((train_text_counts, train_gender))

    test_text_counts = count_vect.transform(test_text)
    test_text_counts = sparse.hstack((test_text_counts, test_gender))

    # use tf-idf to give weights to the words
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(train_text_counts)
    x_test_tfidf = tfidf_transformer.transform(test_text_counts)

    return x_train_tfidf, x_test_tfidf


def get_kfold_data(text_data, gender_data, y_data):
    kfold_data = []
    k_fold = KFold(n_splits=10)
    for train, test in k_fold.split(text_data, y_data):
        train_text = text_data[train]
        test_text = text_data[test]

        train_gender = gender_data[:, train]
        test_gender = gender_data[:, test]
        train_data, test_data = preprocess_input(train_text, train_gender,
                                                 test_text, test_gender)

        train_output = y_data[train]
        test_output = y_data[test]
        kfold_data.append((train_data, train_output, test_data, test_output))
    return kfold_data


if __name__ == "__main__":

    # read in training set
    df_train = pd.read_json('../data/pre_train.json')

    # read in test set
    df_test = pd.read_json('../data/pre_test.json')

    train_x1 = np.array(df_train['posts'])
    train_x2 = np.array([df_train['genders']])
    y_train = np.array(df_train['group_ages'])

    test_x1 = np.array(df_test['posts'])
    test_x2 = np.array([df_test['genders']])
    y_test = np.array(df_test['group_ages'])

    # preprocess posts and gender input data
    x_train, x_test = preprocess_input(train_x1, train_x2, test_x1, test_x2)

    kfold_data = get_kfold_data(train_x1, train_x2, y_train)

    # algorithm 1 predictions
    predictionsLogistic, logisticModel = logistic_regression_algorithm(
        x_train, x_test, y_train, kfold_data)

    # algorithm 2 predictions
    predictionsKnn, knn_model = knn_algorithm(x_train, x_test, y_train,
                                              kfold_data)

    # algorithm 3 predictions
    predictionsDeepLearning, deep_model = deep_learning_algorithm(
        x_train, x_test, y_train, kfold_data)

    # evaluating the three algorithms...

    # accuracy score
    accLogistic = accuracy_score(y_test, predictionsLogistic)
    print(f"Accuracy of the LG classifier is: {accLogistic}")

    accKnn = accuracy_score(y_test, predictionsKnn)
    print(f"Accuracy of the kNN classifier is: {accKnn}")

    accDeepL = accuracy_score(y_test, predictionsDeepLearning)
    print(f"Accuracy of the Deep Learning classifier is: {accDeepL}")

    # confusion matrices
    cmLogistic = multilabel_confusion_matrix(y_test, predictionsLogistic)
    print("\nLogistic Regression Confusion Matrix\n", cmLogistic)

    cmKnn = multilabel_confusion_matrix(y_test, predictionsKnn)
    print("\nKNN Confusion Matrix:\n", cmKnn)

    cmDeepL = multilabel_confusion_matrix(y_test, predictionsDeepLearning)
    print("\nDeep Learning Confusion Matrix:\n", cmDeepL)

    # Print the precision and recall, among other metrics
    print("\nLogistic Regression Classification Report:\n")
    print(classification_report(y_test, predictionsLogistic, digits=3))

    print("\nKNN Classification Report:\n")
    print(classification_report(y_test, predictionsKnn, digits=3))

    print("\nDeep Learning Classification Report:\n")
    print(classification_report(y_test, predictionsDeepLearning, digits=3))

    # roc curve
    #plot_roc(x_test, y_test)

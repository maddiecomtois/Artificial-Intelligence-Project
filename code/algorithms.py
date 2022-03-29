from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             multilabel_confusion_matrix, roc_curve, mean_squared_error, f1_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

from deep_learning import deep_learning_algorithm
from knn import knn_algorithm
from logistic_regression import logistic_regression_algorithm


def plot_roc(x_test, y_test, model, algorithm):
    y_score = None
    y = label_binarize(y_test, classes=[1, 2, 3])
    if algorithm == "logistic":
        y_score = model.decision_function(x_test)
    else:
        y_score = model.predict_proba(x_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in [1, 2, 3]:
        fpr[i], tpr[i], _ = roc_curve(y[:, i - 1], y_score[:, i - 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in [1, 2, 3]:
        print("AUC", algorithm, " for age group ", i, ":", roc_auc[i])
        plt.figure()
        plt.plot(
            fpr[i],
            tpr[i],
            color="darkorange",
            label="ROC curve for %f (area = %0.2f)" % (i, roc_auc[i]),
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
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


def run_algorithm(algorithm):
    if algorithm == "logistic":
        predictions, model = logistic_regression_algorithm(x_train, x_test, y_train, kfold_data)
    elif algorithm == "knn":
        predictions, model = knn_algorithm(x_train, x_test, y_train, kfold_data)
    elif algorithm == "deep learning":
        predictions, model = deep_learning_algorithm(x_train, x_test, y_train, kfold_data)
    else:
        model = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = multilabel_confusion_matrix(y_test, predictions)

    print(f"Accuracy of {algorithm} classifier is: {accuracy}")
    print("\n", algorithm, " Confusion Matrix\n", cm)
    print("\n", algorithm, " Classification Report:\n")
    print(classification_report(y_test, predictions, digits=3))
    return predictions, model

def model_comparison(model_predictions, y_true):
    models = ['logistic', 'knn', 'deep learning']
    x_labels = np.arange(len(models))
    colors = ['r', 'g', 'b']
    
    # mean squared error
    mse_scores = [mean_squared_error(y_true, p) for p in model_predictions]

    plt.bar(x_labels, mse_scores, color=colors, width=0.3)
    plt.xticks(x_labels, models)
    plt.title('MSE Comparison of Models')
    plt.ylabel("MSE")
    plt.xlabel("Models")
    plt.show()
    
    # accuracy
    accuracy_scores = [accuracy_score(y_true, pred) for pred in model_predictions]
    
    plt.bar(x_labels, accuracy_scores, color=colors, width=0.3)
    plt.xticks(x_labels, models)
    plt.title('Accuracy of Models')
    plt.ylabel("Accuracy Score")
    plt.xlabel("Models")
    plt.show()
    
    # F1
    classes = ('Younger than 17', 'Aged 17 to 27', "Older than 27")
    f1_scores = [f1_score(y_true, p, average=None) for p in model_predictions]
    scores = np.stack(f1_scores)

    class1 = scores[:, 0]
    class2 = scores[:, 1]
    class3 = scores[:, 2]
    width = 0.2

    bar1 = plt.bar(x_labels, class1, width, color='r')
    bar2 = plt.bar(x_labels + width, class2, width, color='g')
    bar3 = plt.bar(x_labels + width*2, class3, width, color='b')

    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('F1-Score of Models in Each Class')
    plt.xticks(x_labels+width, models)
    plt.legend((bar1, bar2, bar3), classes)
    plt.show()



if __name__ == "__main__":

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
    _, model = run_algorithm("logistic")
    plot_roc(x_test, y_test, model, "logistic")

    # comparisons -- Accuracy, Log-loss, F1
    predictions = []
    for algo in ['logistic', 'knn', 'deep learning']:
        p, m = run_algorithm(algo)
        predictions.append(p)
    
    model_comparison(predictions, y_test)
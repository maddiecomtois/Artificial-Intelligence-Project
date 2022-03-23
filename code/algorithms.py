import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_learning import deep_learning_algorithm
from logistic_regression import logistic_regression_algorithm
from knn import knn_algorithm
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

def plot_roc(x_test, y_test):
    # algorithm 1
    fpr, tpr, _ = roc_curve(y_test, logisticModel.decision_function(x_test))
    plt.plot(fpr, tpr, label="Logistic")
    print("\nAUC Logistic:", auc(fpr, tpr))

    # algorithm 2
    y_scores_knn = knnModel.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores_knn[:, 1])
    plt.plot(fpr, tpr, label="KNN")
    print("AUC KNN:", auc(fpr, tpr))

    # algorithm 3
    y_scores_deepL = deepLModel.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores_deepL[:, 1])
    plt.plot(fpr, tpr, label="Deep Learning")
    print("AUC Deep Learning:", auc(fpr, tpr))

    plt.title('ROC - Logistic, KNN, and Deep Learning')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()

# read in data and pass into each algorithm or each algorithm calls it?

if __name__ == "__main__":
    
    # read in training set
    df_train = pd.read_json('../data/pre_train.json')
    
    x1 = np.array(df_train['posts'])
    x2 = np.array(df_train['genders'])
    y_train = np.array(df_train['group_ages'])

    # use Bag of Words to transform text into numbers
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(x1)

    # read in data json
    df_test = pd.read_json('../data/pre_test.json')

    x1 = np.array(df_test['posts'])
    x2 = np.array(df_test['genders'])
    y_test = np.array(df_test['group_ages'])

    X_test_counts = count_vect.transform(x1)
    
    # algorithm 1 predictions
    predictionsLogistic, logisticModel = logistic_regression_algorithm(X_train_counts, y_train, X_test_counts, y_test)
    
    # algorithm 2 predictions
    predictionsKnn, knnModel = knn_algorithm(X_train_counts, X_test_counts, y_train, y_test)

    # algorithm 3 predictions
    predictionsDeepLearning, deepLModel = deep_learning_algorithm(X_train_counts, X_test_counts, y_train, y_test)

    
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

    print("\nLogistic Regression Classification Report:\n")
    print(classification_report(y_test, predictionsKnn, digits=3))

    print("\nDeep Learning Classification Report:\n")
    print(classification_report(y_test, predictionsDeepLearning, digits=3))

    # roc curve
    #plot_roc(x_test, y_test)

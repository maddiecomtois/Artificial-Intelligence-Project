import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import main
from deep_learning import deep_learning_algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error

def plot_roc():
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
    
    # preprocess data
    #main()

    # read in data json
    df = pd.read_json('../data/example_output.json')
    x1 = np.array(df['posts'])
    x2 = np.array(df['genders'])
    y = np.array(df['ages'])

    # split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x1, y, train_size=0.75, shuffle=False)

    # algorithm 1 predictions
    predictionsLogistic, logisticModel = LogisticRegressionFunction(x_train, x_test, y_train, y_test)
    
    # algorithm 2 predictions
    predictionsKnn, knnModel = KnnFunction(x_train, x_test, y_train, y_test)

    # algorithm 3 predictions
    predictionsDeepLearning, deepLModel = deep_learning_algorithm(x_train, x_test, y_train, y_test)

    
    # evaluating the three algorithms...
    
    # mean squared error
    mseLogistic = mean_squared_error(y_test, predictionsLogistic)
    mseKnn = mean_squared_error(y_test, predictionsKnn)
    mseDeepL = mean_squared_error(y_test, predictionsDeepLearning)

    print("\nLogistic Regression MSE\n", mseLogistic)
    print("\nKNN Confusion MSE:\n", mseKnn)
    print("\nKNN Confusion MSE:\n", mseDeepL)

    # confusion matrix
    cmLogistic = confusion_matrix(y_test, predictionsLogistic)
    cmKnn = confusion_matrix(y_test, predictionsKnn)
    cmDeepL = confusion_matrix(y_test, predictionsDeepLearning)

    print("\nLogistic Regression Confusion Matrix\n", cmLogistic)
    print("\nKNN Confusion Matrix:\n", cmKnn)
    print("\nDeep Learning Confusion Matrix:\n", cmDeepL)

    # roc curve
    plot_roc()


    pass
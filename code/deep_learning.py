import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split



def deep_learning_algorithm():
    # read in the data
    df = pd.read_csv('') 
    print(df.shape)
    df.describe().transpose()

    # column we want to predict
    target_column = [''] 
    # set of features (minus target)
    predictors = list(set(list(df.columns))-set(target_column))
    # normalise the features
    df[predictors] = df[predictors]/df[predictors].max()
    # values are scaled between 0 and 1
    df.describe().transpose()

    # create the x and y variables
    X = df[predictors].values
    y = df[target_column].values

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    print(X_train.shape); print(X_test.shape)

    # instantiate the classifier: 3 layers (set count to number of features, e.g. is 8), relu activation function, adam is solver for weight optimisation
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train,y_train)

    # generate predictions 
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)


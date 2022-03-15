import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def deep_learning_algorithm(X_train, X_test, y_train, y_test):
    # read in the data
    '''
    df = data
    print(df.shape)
    df.describe().transpose()

    # column we want to predict
    target_column = ['age']
    # set of features (minus target)
    predictors = list(set(list(df.columns))-set(target_column))
    print(predictors)

    # normalise the features
    #df[predictors] = df[predictors]/df[predictors].max()
    # values are scaled between 0 and 1
    df.describe().transpose()

    # create the x and y variables
    X = df[predictors].values
    y = df[target_column].values

    print(X)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    print(X_train.shape); print(X_test.shape)
    '''

    # use Bag of Words to transform text into numbers
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape
    X_test_counts = count_vect.fit_transform(X_test)
    X_test_counts.shape

    # use tf-idf to give weights to the words
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape
    X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
    X_test_tfidf.shape

    # instantiate the classifier: 3 layers (set count to number of features, e.g. is 8), relu activation function, adam is solver for weight optimisation
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train_tfidf,y_train)

    # generate predictions 
    predict_train = mlp.predict(X_train_tfidf)
    #predict_test = mlp.predict(X_test_tfidf)



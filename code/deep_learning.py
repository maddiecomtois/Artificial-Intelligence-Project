from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer

def deep_learning_algorithm(X_train, X_test, y_train, y_test):
    # use tf-idf to give weights to the words
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_test_tfidf = tfidf_transformer.fit_transform(X_test)

    # instantiate the classifier: 3 layers (set count to number of features, e.g. is 8), relu activation function, adam is solver for weight optimisation
    mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train_tfidf, y_train)

    # generate predictions 
    predict_test = mlp.predict(X_test_tfidf)
    return predict_test, mlp



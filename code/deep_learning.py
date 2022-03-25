from sklearn.neural_network import MLPClassifier


def deep_learning_algorithm(x_train, x_test, y_train, kfold_data):

    # instantiate the classifier:
    # - 3 layers (set count to number of features, e.g. is 8)
    # - relu activation function
    # - adam is solver for weight optimisation
    mlp = MLPClassifier(hidden_layer_sizes=(2, 2, 2),
                        activation='relu',
                        solver='adam',
                        max_iter=500)
    mlp.fit(x_train, y_train)

    # generate predictions
    predict_test = mlp.predict(x_test)
    return predict_test, mlp

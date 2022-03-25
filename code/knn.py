from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline


def knn_algorithm(X_train, X_test, y_train, y_test):
    # use tf-idf to give weights to the words
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)
    X_test_tfidf = tfidf_transformer.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=150, weights='uniform')
    knn_model.fit(X_train_tfidf, y_train)
    
    knn_preds = knn_model.predict(X_test_tfidf)
    return knn_preds, knn_model


"""
# cross validation for KNN
mean_sq_errors = []; std_devs = []
k_range = [25,50,75,100,125,150]
# can test these parameters as well
#min_df_range = [1,2,3,4,5,6,7,8,9,10] 
#max_df_range = (0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0) 
# ngram_range = (x, y) -- could test this too
for k in k_range:
    clf = make_pipeline(TfidfVectorizer(ngram_range = (1,2), min_df = 1, max_df = 0.5), KNeighborsClassifier(n_neighbors = k, weights = 'uniform'))
    scores = cross_validate(clf, texts, age_groups, scoring = ['roc_auc'], cv = 5)
    # scores = cross_validate(clf, texts, genders, scoring = ['roc_auc'], cv = 5)
    mean_sq_errors.append(np.mean(scores['test_roc_auc']))
    std_devs.append(np.std(scores['test_roc_auc']))
    
plt.errorbar(k_range, mean_sq_errors, yerr = std_devs, elinewidth = 2.5)
plt.xlabel('k'); plt.ylabel('AUC')
plt.show()
"""

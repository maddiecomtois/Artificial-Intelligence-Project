from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline


def logistic_regression_algorithm(x_train, x_test, y_train):

    lr_model = LogisticRegression(penalty='l2', C=50, solver='lbfgs')
    lr_model.fit(x_train, y_train)

    lr_preds = lr_model.predict(x_test)
    return lr_preds, lr_model


# crossvalidation code
"""
mean_sq_errors = []; std_devs = []
C_range = [1, 3, 5, 10, 25, 50, 75]
# can test these parameters as well
#min_df_range = [1,2,3,4,5,6,7,8,9,10]
#max_df_range = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# ngram_range = (x, y) -- could test this too
for C in C_range:    
    clf = make_pipeline(TfidfVectorizer(ngram_range = (1,2), min_df = 1, max_df = 0.6), LogisticRegression(penalty = 'l2', C = C, solver = 'lbfgs'))
    scores = cross_validate(clf, texts, age_groups, scoring = ['roc_auc'], cv = 5)     
    mean_sq_errors.append(np.mean(scores['test_roc_auc']))
    std_devs.append(np.std(scores['test_roc_auc']))
    
plt.errorbar(C_range, mean_sq_errors, yerr = std_devs, elinewidth = 2.5)
plt.xlabel('C'); plt.ylabel('AUC')
plt.show()
"""

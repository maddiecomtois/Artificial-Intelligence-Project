import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

# read in dataset
df = pd.read_csv("")
texts = df.iloc[:, 0]
age_groups = df.iloc[:, 1]
genders = df.iloc[:, 2]

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

# --- after we choose parameters, evaluate the models ---
"""
# plot ROC curve
plt.rc('font', size = 15)
plt.rcParams['figure.constrained_layout.use'] = True
xtrain, xtest, ytrain, ytest = train_test_split(texts, age_groups, test_size = 0.2)
# xtrain, xtest, ytrain, ytest = train_test_split(texts, genders, test_size = 0.2)

vec = TfidfVectorizer(ngram_range = (1,2), min_df = 1, max_df = 0.3)
X_train = vec.fit_transform(xtrain)
X_test = vec.transform(xtest)
knn_model = KNeighborsClassifier(n_neighbors = 150, weights = 'uniform').fit(X_train,ytrain)
knn_preds = knn_model.predict(X_test)
knn_scores = knn_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(ytest, knn_scores[:, 1])
plt.plot(fpr, tpr, color = 'red')
"""

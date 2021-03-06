####################################################
#
# SVD
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def create_search():
    clf = ExtraTreesClassifier(n_jobs=6, random_state=42, n_estimators=500)
    svd = TruncatedSVD()

    hyperparameters = {'etc__max_depth': [None, 10, 5], 'etc__max_features': ['auto', 'log2'],
                       'etc__min_samples_split': [1, 2, 3], 'etc__criterion': ['entropy', 'gini'],
                       'svd__n_components': np.arange(2, 15, 1)
                       }
    pipe = Pipeline([('svd', svd), ('etc', clf)])

    return RandomizedSearchCV(pipe, hyperparameters, cv=5, scoring='roc_auc')


df = pd.read_csv("./work_dir/boruta_filtered_train_split.csv")
y = df['y']
X = df.drop(['y'], axis=1)

print("Fitting RandomizedSearch.  Please Wait (awhile)...")
search = create_search()
search.fit(X, y)
print("Done!")
print("Best Parameters: ")
print(search.best_params_)
print("Best Score:")
print(search.best_score_)

# Create Submission
kaggle_test = pd.read_csv("./work_dir/my_midterm_kaggle_test.csv")
selected_features = pd.read_csv("./work_dir/feature_support.csv")
kaggle_test_selected = kaggle_test.ix[:, selected_features['0'].values]  # trim to the boruta features

prediction = pd.DataFrame(search.best_estimator_.predict_proba(kaggle_test_selected)[:, 1])
prediction.columns = ['y']
prediction.to_csv("svd_prediction.csv", index_label="Id")


# Best Parameters:
# {'max_features': 'log2', 'min_samples_split': 1, 'max_depth': None, 'criterion': 'entropy'}
# Best Score:
# 0.982666422664

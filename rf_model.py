####################################################
#
# RF Model
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV


def create_search():
    clf = RandomForestClassifier(n_jobs=6, random_state=42, n_estimators=500)
    hyperparameters = {'max_depth': [None, 10, 5], 'max_features': ['auto', 'log2'],
                       'min_samples_split': [2, 3, 5], 'criterion': ['entropy', 'gini']
                       }
    return RandomizedSearchCV(clf, hyperparameters, cv=5, scoring='roc_auc')


df = pd.read_csv("./data/boruta_filtered_train.csv")
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
kaggle_test = pd.read_csv("./work_dir/my_midterm_kaggle_submission.csv")
selected_features = pd.read_csv("./work_dir/feature_support.csv")
kaggle_test_selected = kaggle_test.ix[:, selected_features['0'].values]  # trim to the boruta features

prediction = pd.DataFrame(search.best_estimator_.predict_proba(kaggle_test_selected)[:, 1])
prediction.columns = ['y']
prediction.to_csv("extra_model_prediction.csv", index_label="Id")


# Best Parameters:
# {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 5, 'max_features': 'auto'}
# Best Score:
# 0.975445967774
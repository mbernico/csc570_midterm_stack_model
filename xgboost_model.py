####################################################
#
# XGBoost Model
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import numpy as np
import pandas as pd
import xgboost
from sklearn.grid_search import RandomizedSearchCV


def create_search():
    xgb = xgboost.XGBClassifier(nthread=6, n_estimators=300)
    hyperparameters = {'colsample_bytree': [.3, .4, .5, 1], 'max_depth': [10, 12, 14, 16],
                       'learning_rate': np.arange(0.01, 0.4, .1)}
    return RandomizedSearchCV(xgb, hyperparameters, cv=3, scoring='roc_auc', n_iter=15)


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
prediction.to_csv("xgboost_model_prediction.csv", index_label="Id")

# {'max_depth': 14, 'learning_rate': 0.11, 'colsample_bytree': 1}
# Best Score:
# 0.9808915808045747

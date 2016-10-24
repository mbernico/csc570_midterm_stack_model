####################################################
#
# Train Second Level Learners
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import glob as glob
import logging

import numpy as np
import pandas as pd
import xgboost
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


def load_data():
    """
    Loads all the level_1 data
    :return: X, y, S
    """
    # training data y
    df = pd.read_csv("./base_data/boruta_filtered_stacked_train.csv")
    y = df['y']

    # level_1 OOB predictions
    level_1_oobs = glob.glob('./level_1_OOB/*.csv')
    level_1_oob_df_list = []
    for file in level_1_oobs:
        level_1_oob_df_list.append(pd.read_csv(file))
    level_1_oob_df = pd.concat(level_1_oob_df_list, axis=1)

    # level_1 submission yhats
    level_1_yhats = glob.glob('./level_1_yhat/*.csv')
    level_1_yhat_df_list = []
    for file in level_1_yhats:
        level_1_yhat_df_list.append(pd.read_csv(file))
    level_1_yhat_df = pd.concat(level_1_yhat_df_list, axis=1)

    return level_1_oob_df, y, level_1_yhat_df


def config_logging():
    """
    Basic logging configuration
    :return: None
    """
    logging.basicConfig(filename="level_2_learners.log", level=logging.DEBUG,
                        format='%(levelname)s::%(asctime)s %(message)s')


def get_classifiers():
    """
    Creates a list of level 1 learners
    :return: a list of level 1 learners
    """
    etc = ExtraTreesClassifier(n_jobs=6, n_estimators=500, max_features='auto', min_samples_split=1,
                               max_depth=10, criterion='gini')
    xgb = xgboost.XGBClassifier(nthread=6, n_estimators=300, learning_rate=0.01, max_depth=14, colsample_bytree=0.5)
    rfc = RandomForestClassifier(n_jobs=6, random_state=42, n_estimators=500, max_depth=10, max_features='auto',
                                 min_samples_split=2, criterion='gini')
    # searched params
    # rfc {'criterion': 'entropy', 'min_samples_split': 1, 'max_features': 'auto', 'max_depth': 10}
    # extra 'criterion': 'entropy', 'min_samples_split': 3, 'max_features': 'auto', 'max_depth': 10}
    # xgb  {'learning_rate': 0.01, 'colsample_bytree': 1, 'max_depth': 16}
    return {'extra_tree': etc, 'xgboost': xgb, 'rfc': rfc}  # 'deep_learn': dnn}


def fit_classifier(clf_name, clf, X, y, S, n_folds):
    """
    Step 1. Fit Classifier with n_folds folds. K-Fold and log performance
    Step 2. Fit Classifier on X,y.  Predict on S.   Save that for third? level prediction
    :param clf_name: name of the classifier
    :param clf: classifier instance
    :param X: training independent
    :param y: training dependent
    :param S: kaggle submission
    :param n_folds:
    :return: None
    """
    X = X.values
    y = y.values
    S = S.values
    # Step 1 Above
    logging.info("Step 1. K-Fold ")
    OOB_predictions = np.zeros((X.shape[0]))
    skf = StratifiedKFold(y, n_folds=n_folds, random_state=42)
    # kfold seed must be the same for every level 1 CLF or INFO LEAK!!!
    for i, (train, test) in enumerate(skf):
        logging.debug("Fold " + str(i))
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        # y_test = y[test]
        clf.fit(X_train, y_train)
        OOB_predictions[test] = clf.predict_proba(X_test)[:, 1]
    clf_score = roc_auc_score(y, OOB_predictions)
    logging.info(clf_name + " done training.  AUC = " + str(clf_score))
    OOB_df = pd.DataFrame(data=OOB_predictions, columns=[clf_name], index=None)
    OOB_df.to_csv("./level_2_OOB/" + clf_name + "_oob.csv", index=False)

    # Step 2 from above
    logging.info("Step 2. Submission Scoring")
    clf.fit(X, y)
    y_hat = clf.predict_proba(S)[:, 1]
    y_hat_df = pd.DataFrame(data=y_hat, columns=[clf_name], index=None)
    y_hat_df.to_csv("./level_2_yhat/" + clf_name + "_yhat.csv", index=False)


def search_classifier(clf_name, X, y):
    """
    random search on a classifier
    :param clf_name: name of the classifier
    :param X: training independent
    :param y: training dependent
    :return: None
    """
    if clf_name == 'extra_tree':
        clf = ExtraTreesClassifier(n_jobs=6, random_state=42, n_estimators=100)
        hyperparameters = {'max_depth': [None, 10, 5], 'max_features': ['auto', 'log2'],
                           'min_samples_split': [1, 2, 3], 'criterion': ['entropy', 'gini']
                           }
        search = RandomizedSearchCV(clf, hyperparameters, cv=5, scoring='roc_auc')
    if clf_name == 'xgboost':
        xgb = xgboost.XGBClassifier(nthread=6, n_estimators=100)
        hyperparameters = {'colsample_bytree': [.3, .4, .5, 1], 'max_depth': [10, 12, 14, 16],
                           'learning_rate': np.arange(0.01, 0.4, .1)}
        search = RandomizedSearchCV(xgb, hyperparameters, cv=3, scoring='roc_auc', n_iter=15)
    if clf_name == 'rfc':
        clf = RandomForestClassifier(n_jobs=6, random_state=42, n_estimators=100)
        hyperparameters = {'max_depth': [None, 10, 5], 'max_features': ['auto', 'log2'],
                           'min_samples_split': [1, 2, 3], 'criterion': ['entropy', 'gini']
                           }
        search = RandomizedSearchCV(clf, hyperparameters, cv=5, scoring='roc_auc')

    X = X.values
    y = y.values
    logging.info("GridSearch started for " + clf_name)
    search.fit(X, y)
    logging.info("GridSearch done for " + clf_name)
    logging.info(search.best_params_)
    logging.info(search.best_score_)


def main(grid_search=False):
    config_logging()
    logging.info("Loading Data")
    X, y, S = load_data()
    logging.debug("X Shape = " + str(X.shape) + " y Shape = " + str(y.shape) + " S Shape = " + str(S.shape))
    logging.info("Brewing Classifiers")
    clfs = get_classifiers()
    logging.debug(str(len(clfs)) + " Classifiers Brewed")
    for k, clf in clfs.items():
        if grid_search:
            logging.info("Searching Classifier:" + str(k))
            search_classifier(k, X, y)
        else:
            logging.info("Fitting Classifier:" + str(k))
            fit_classifier(k, clf, X, y, S, n_folds=5)

    logging.info("Done")


if __name__ == "__main__":
    main(grid_search=False)

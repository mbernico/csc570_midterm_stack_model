####################################################
#
# Train Third Level Learners
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import glob as glob
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV


def load_data():
    """
    Loads all the level_1 data
    :return: X, y, S
    """
    # training data y
    df = pd.read_csv("./base_data/boruta_filtered_stacked_train.csv")
    y = df['y']

    # level_2 OOB predictions
    level_2_oobs = glob.glob('./level_2_OOB/*.csv')
    level_2_oob_df_list = []
    for file in level_2_oobs:
        level_2_oob_df_list.append(pd.read_csv(file))
    level_2_oob_df = pd.concat(level_2_oob_df_list, axis=1)

    # level_1 submission yhats
    level_2_yhats = glob.glob('./level_2_yhat/*.csv')
    level_2_yhat_df_list = []
    for file in level_2_yhats:
        level_2_yhat_df_list.append(pd.read_csv(file))
    level_2_yhat_df = pd.concat(level_2_yhat_df_list, axis=1)

    return level_2_oob_df, y, level_2_yhat_df


def config_logging():
    """
    Basic logging configuration
    :return: None
    """
    logging.basicConfig(filename="level_3_model.log", level=logging.DEBUG,
                        format='%(levelname)s::%(asctime)s %(message)s')


def main(grid_search=False):
    config_logging()
    logging.info("Loading Data")
    X, y, S = load_data()
    logging.debug("X Shape = " + str(X.shape) + " y Shape = " + str(y.shape) + " S Shape = " + str(S.shape))
    logging.info("Searching for a final logit model...")
    # clf = LogisticRegression(random_state=42, n_jobs=6)
    # hyperparameters = {'C': np.arange(.01, 10, .1)}
    # search = RandomizedSearchCV(clf, hyperparameters, n_iter=10, scoring='roc_auc', random_state=42)
    clf = RandomForestClassifier(n_jobs=6, random_state=42, n_estimators=100)
    hyperparameters = {'max_depth': [None, 10, 5], 'max_features': ['auto', 'log2'],
                       'min_samples_split': [1, 2, 3], 'criterion': ['entropy', 'gini']
                       }
    search = RandomizedSearchCV(clf, hyperparameters, cv=5, scoring='roc_auc')
    search.fit(X, y)
    logging.info("Done Searching")
    logging.info("Best Parameters: " + str(search.best_params_))
    logging.info("Best Score: " + str(search.best_score_))
    logging.info("Writing Final Score")
    y_hat = search.best_estimator_.predict_proba(S)[:, 1]
    # y_hat = X.mean(axis=1)
    y_hat_df = pd.DataFrame(data=y_hat, columns=['y'], index=None)
    y_hat_df.to_csv("./level_3_submission/submission.csv", index_label="Id")

    logging.info("Done")


if __name__ == "__main__":
    main(grid_search=False)

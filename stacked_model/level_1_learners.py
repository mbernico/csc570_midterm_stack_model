####################################################
#
# Train First Level Learners
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import logging

import numpy as np
import pandas as pd
import xgboost
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_data():
    """
    Loads all the data
    :return: X, y, S
    """
    # training data
    df = pd.read_csv("./base_data/boruta_filtered_stacked_train.csv")
    y = df['y']
    X = df.drop(['y'], axis=1)
    # Create Submission
    kaggle_submission = pd.read_csv("./base_data/my_midterm_kaggle_submission.csv")
    selected_features = pd.read_csv("./base_data/feature_support.csv")
    kaggle_submission = kaggle_submission.ix[:, selected_features['0'].values]  # trim to the boruta features
    return X, y, kaggle_submission


def config_logging():
    """
    Basic logging configuration
    :return: None
    """
    logging.basicConfig(filename="level_1_learners.log", level=logging.DEBUG,
                        format='%(levelname)s::%(asctime)s %(message)s')


def create_keras_model(dropout=0.1, optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=20, init='glorot_uniform', activation='relu'))
    model.add(Dense(10, init='glorot_uniform', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='glorot_uniform', activation='relu'))
    model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_classifiers():
    """
    Creates a list of level 1 learners
    :return: a list of level 1 learners
    """
    etc = ExtraTreesClassifier(n_jobs=6, n_estimators=500, max_features='log2', min_samples_split=1,
                               max_depth=None, criterion='entropy')
    xgb = xgboost.XGBClassifier(nthread=6, n_estimators=300, learning_rate=0.31, max_depth=16, colsample_bytree=1)
    rfc = RandomForestClassifier(n_jobs=6, random_state=42, n_estimators=500, max_depth=None, max_features='auto',
                                 min_samples_split=2, criterion='entropy')
    dnn = KerasClassifier(build_fn=create_keras_model, verbose=0, nb_epoch=150)

    return {'extra_tree': etc, 'xgboost': xgb, 'rfc': rfc, 'deep_learn': dnn}


def fit_classifier(clf_name, clf, X, y, S, n_folds):
    """
    Step 1. Fit Classifier with n_folds folds.  Predict on and retain the OOB predictions which will train the
    second level classifiers.
    Step 2. Fit Classifier on X,y.  Predict on S.   Save that for second level prediction
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
    logging.info("Step 1. K-Fold OOB Creation")
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
    OOB_df.to_csv("./level_1_OOB/" + clf_name + "_oob.csv", index=False)

    # Step 2 from above
    logging.info("Step 2. Submission Scoring")
    clf.fit(X, y)
    y_hat = clf.predict_proba(S)[:, 1]
    y_hat_df = pd.DataFrame(data=y_hat, columns=[clf_name], index=None)
    y_hat_df.to_csv("./level_1_yhat/" + clf_name + "_yhat.csv", index=False)


def main():
    config_logging()
    logging.info("Loading Data")
    X, y, S = load_data()
    logging.debug("X Shape = " + str(X.shape) + "y Shape = " + str(y.shape) + "S Shape = " + str(S.shape))
    logging.info("Brewing Classifiers")
    clfs = get_classifiers()
    logging.debug(str(len(clfs)) + " Classifiers Brewed")
    for k, clf in clfs.items():
        logging.info("Fitting Classifier:" + str(k))
        fit_classifier(k, clf, X, y, S, n_folds=5)

    logging.info("Done")


if __name__ == "__main__":
    main()

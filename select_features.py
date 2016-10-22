####################################################
#
# Midterm Feature Selection Script
# Using boruta_py for all relevant feature selection
#  Mike Bernico CS570 10/12/2016
#
####################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy


def main():
    print("Begin Feature Selection Step...")
    print('-' * 60)
    print('Loading Data...')
    df = pd.read_csv("./stacked_model/stacked_train.csv")
    y = df['y']
    X = df.drop(['y'], axis=1)

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2)
    print("Fitting Boruta...")
    # find all relevant features
    feat_selector.fit(X.values, y)

    print("Selected Features:")
    # check selected features
    print(feat_selector.support_)
    support = pd.DataFrame(feat_selector.support_)

    print("Selected Feature Rank:")
    # check ranking of features
    print(feat_selector.ranking_)
    ranking = pd.DataFrame(feat_selector.support_)
    # call transform() on X to filter it down to selected features
    print("Transforming X...")
    X_filtered = X.ix[:, feat_selector.support_]
    print("Writing Data...")
    support.to_csv("./work_dir/feature_support.csv", index=False)
    ranking.to_csv("./work_dir/feature_ranking.csv", index=False)
    combined_df = pd.concat([X_filtered, y], axis=1)
    combined_df.to_csv("./stacked_model/boruta_filtered_stacked_train.csv", index=False)


if __name__ == "__main__":
    main()

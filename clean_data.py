####################################################
#
# Midterm Data Cleaning Script
#
#  Mike Bernico CS570 10/12/2016
#
####################################################

import pandas as pd


def load_data(df_path):
    """
    Loads test and train datasets, returns dfs
    :return: train_df, test_df
    """
    df = pd.read_csv(df_path)
    return df


def make_dummies(df, dummy_cols):
    """
    Created indicator variables for cols in dummy_cols
    :param df: dataframe
    :param dummy_cols: list of columns to dummy
    :return: dummied dataframe
    """
    df = df.copy()
    df2 = pd.get_dummies(df, columns=dummy_cols)
    return df2


def remove_extra_characters(df, cols):
    """
    Cleans columns that have $ and % in them.
    :param df: dataframe to clean
    :param cols: columns to clean
    :return: cleaned df
    """
    df = df.copy()
    for col in cols:
        df[col] = df[col].str.replace('$', '').str.replace('%', '').astype(float)
    return df


def fix_x19(df):
    """
    X29 is really ugly, and I'd like to fix that before I dummy it.
    :param df:
    :return:
    """
    df = df.copy()
    cleaning_dict = {'July': 'july', 'Jun': 'june', 'Aug': 'august', 'May': 'may', 'sept.': 'september', 'Apr': 'april',
                     'Oct': 'october', 'Mar': 'march', 'Nov': 'november', 'Feb': 'february', 'Dev': 'december',
                     'January': 'january'}
    df.x19 = df.x19.map(cleaning_dict)
    return df


def handle_missing(df):
    df = df.copy()
    df = df.fillna(value=-9999)
    return df


def write_data(df, out_df_path, train=True):
    if train:
        # random.seed = 42
        # train_df, test_df = train_test_split(df)
        df.to_csv(out_df_path + "my_midterm_train.csv", index=False)
        # test_df.to_csv(out_df_path+"my_midterm_test_split.csv", index=False)
    else:
        df.to_csv(out_df_path + "my_midterm_kaggle_submission.csv", index=False)


def main():
    train_name = "./data/midterm_train.csv"
    kaggle_test_name = "./data/midterm_test.csv"
    out_df_path = "./work_dir/"
    df_train = load_data(train_name)
    df_test = load_data(kaggle_test_name)
    frames = {"train": df_train, "kaggle_test": df_test}

    for key, val in frames.items():
        print("Begin Clean Step for " + key)
        df = val
        print('-' * 60)
        print("Handling X19")
        df = fix_x19(df)
        print("Removing Extra Characters from x9 and x44")
        clean_cols = ['x9', 'x44']
        df = remove_extra_characters(df, clean_cols)
        print("Create Dummies")
        col_to_dummy = ['x16', 'x19', 'x43']
        df = make_dummies(df, col_to_dummy)
        print("Handle Missing Values")
        df = handle_missing(df)
        print("Writing Train/Test Data")
        if key == 'kaggle_test':
            write_data(df, out_df_path, False)
        else:
            write_data(df, out_df_path)


if __name__ == "__main__":
    main()

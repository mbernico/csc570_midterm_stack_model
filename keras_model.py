####################################################
#
# XGBoost Model
#
#  Mike Bernico CS570 10/12/2016
#
####################################################


import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import RandomizedSearchCV


def create_model(dropout=0.2, optimizer='rmsprop'):
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=24, init='glorot_uniform', activation='relu'))
    model.add(Dense(10, init='glorot_uniform', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='glorot_uniform', activation='relu'))
    model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
    # sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, verbose=2, nb_epoch=100)

df = pd.read_csv("./data/boruta_filtered_train.csv")
y = df['y']
X = df.drop(['y'], axis=1)

# grid search epochs, batch size and optimizer
optimizer = ['rmsprop', 'adam']
# lr = [0.1, 0.01, 0.05, 0.001]
#batches = [5, 10, 20, 50]
batches = [50, 64]
# dropout = [0.1, 0.2, 0.5]
dropout = [0.0, 0.1, 0.2]
param_grid = dict(batch_size=batches, dropout=dropout, optimizer=optimizer)
grid = RandomizedSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_iter=8)
grid_result = grid.fit(X.values, y.values)

print("Done!")
print("Best Parameters: ")
print(grid_result.best_params_)
print("Best Score:")
print(grid_result.best_score_)

# Create Submission
kaggle_test = pd.read_csv("./work_dir/my_midterm_kaggle_submission.csv")
selected_features = pd.read_csv("./work_dir/feature_support.csv")
kaggle_test_selected = kaggle_test.ix[:, selected_features['0'].values]  # trim to the boruta features

prediction = pd.DataFrame(grid_result.best_estimator_.predict_proba(kaggle_test_selected.values)[:, 1])
prediction.columns = ['y']
prediction.to_csv("keras_model_prediction.csv", index_label="Id")


# Best Parameters:
# {'optimizer': 'adam', 'batch_size': 50, 'dropout': 0.0}
# Best Score:
# 0.9881298237622679

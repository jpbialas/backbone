import backbone
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from skimage import io
import os
import numpy as np
import csv

train_x = []
train_y = []
test_x = []
test_y = []

with open('thesistrain.csv', 'r') as fin:
   train = np.array(list(csv.reader(fin))).astype('float')
train_x = train[:, 0:3]
train_y = train[:, 3]
with open('thesistest.csv', 'r') as fin:
   test = np.array(list(csv.reader(fin))).astype('float')
test_x = test[:, 0:3]
test_y = test[:, 3]

# use a full grid over all parameters
param_grid = {"n_estimators": [50, 150],
              "max_depth": [9, 15],
              "max_features": [1, 3],
              "min_samples_split": [1, 7],
              "min_samples_leaf": [1, 3],
              "bootstrap": [False],
              "criterion": ["entropy"]}

clf = RandomForestClassifier()

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs = -1, verbose=2)
grid_search.fit(train_x, train_y)
print(grid_search.best_params_)
print grid_search.best_score_

#use the validation data to compute labels, compare with validation labels
preds = grid_search.predict(test_x)
print(metrics.classification_report(test_y, preds))
print(metrics.confusion_matrix(test_y, preds))

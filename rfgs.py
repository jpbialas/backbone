from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from skimage import io
import os
import numpy as np
import csv
import sys

# grid search for random forest

f_train = sys.argv[1]
f_test = sys.argv[2]
out_img = sys.argv[3]

with open(f_train, 'r') as fin:
   train = np.array(list(csv.reader(fin))).astype('float')
train_x = train[:, 0:3]
train_y = train[:, 3]
with open(f_test, 'r') as fin:
   test = np.array(list(csv.reader(fin))).astype('float')
test_x = test[:, 0:3]
test_y = test[:, 3]

param_grid = {"n_estimators": [100, 500],
              "max_depth": [7, 12],
              "max_features": [1, 2, 3],
              "min_samples_split": [1, 5],
              "bootstrap": [False],
              "criterion": ["gini", "entropy"]}

clf = RandomForestClassifier()

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs = -1, verbose=2)
grid_search.fit(train_x, train_y)
print(grid_search.best_params_)
print grid_search.best_score_

#use the validation data to compute labels, compare with validation labels
preds = grid_search.predict(test_x)
print(metrics.classification_report(test_y, preds))
conf = metrics.confusion_matrix(test_y, preds)
print conf
print 'false positive rates'
for i in range(8):
   print float(np.sum(conf[i]) - conf[i, i]) / np.sum(conf[i])

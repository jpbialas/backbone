from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from skimage import io
import os
import numpy as np
import csv
import sys

f_train = sys.argv[1]
f_test = sys.argv[2]
out_img = sys.argv[3]

# given csv data, predict using a random forest

raw = io.imread('1-0003-0002.tif')
labelled = io.imread('1-0003-0002_3.tif')

with open(f_train, 'r') as fin:
   train = np.array(list(csv.reader(fin))).astype('float')
train_x = train[:, 0:3]
train_y = train[:, 3]
with open(f_test, 'r') as fin:
   test = np.array(list(csv.reader(fin))).astype('float')
test_x = test[:, 0:3]
test_y = test[:, 3]

clf = RandomForestClassifier(n_estimators = 100, max_depth = 9, max_features = 3, n_jobs = -1, verbose = 4)
clf.fit(train_x, train_y)

train_preds = clf.predict(train_x)
test_preds = clf.predict(test_x)
# print metrics
print(metrics.classification_report(test_y, test_preds))
conf = metrics.confusion_matrix(test_y, test_preds)
print conf
print 'false positive rates'
for i in range(2):
   print float(np.sum(conf[i]) - conf[i, i]) / np.sum(conf[i])

# write predicted labels to image
rf_labels = np.copy(raw)
for i in range(train_x.shape[0]):
   r = train[i][4]
   c = train[i][5]
   pixel = [255, 0, 0]
   if train_preds[i] == 0:
      pixel = [0, 255, 0]
   rf_labels[r][c][:3] = pixel
for i in range(test_x.shape[0]):
   r = test[i][4]
   c = test[i][5]
   pixel = [255, 0, 0]
   if test_preds[i] == 0:
      pixel = [0, 255, 0]
   rf_labels[r][c][:3] = pixel

io.imsave(out_img, rf_labels)

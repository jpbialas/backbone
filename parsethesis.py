import backbone
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from skimage import io
import os
import numpy as np
import csv

# writes spectral values to csv

c = 0
with open('thesistrain.csv', 'w') as fout:
    write = csv.writer(fout)
    for file in os.listdir('thesistrainsegs'):
        c += 1
        rgb = io.imread('thesistrainsegs/' + file)
        red = rgb[:,:,0]
        green = rgb[:,:,1]
        blue = rgb[:,:2]
        for i in range(rgb.shape[0]):
           for j in range(rgb.shape[1]):
               if not np.all(rgb[i, j, 0:3] == [255, 255, 255]):
                   row = rgb[i,j,0:3]
                   write.writerow(np.append(row, [c]))
c = 0
with open('thesistest.csv', 'w') as fout:
    write = csv.writer(fout)
    for file in os.listdir('thesistestsegs'):
        c += 1
        rgb = io.imread('thesistestsegs/' + file)
        red = rgb[:,:,0]
        green = rgb[:,:,1]
        blue = rgb[:,:2]
        for i in range(rgb.shape[0]):
           for j in range(rgb.shape[1]):
               if not np.all(rgb[i, j, 0:3] == [255, 255, 255]):
                   row = rgb[i,j,0:3]
                   write.writerow(np.append(row, [c]))

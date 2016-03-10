from skimage import io
import numpy as np
import csv

# writes pixel spectral values to a csv
# entries: r g b label x y
# keep x,y values to generate labelled image later

raw = io.imread('1-0003-0002.tif')
labelled = io.imread('1-0003-0002_3.tif')

# training set: left part of image
with open('newtrainloc.csv', 'w') as fout:
    write = csv.writer(fout)
    red = raw[:,:,0]
    green = raw[:,:,1]
    blue = raw[:,:2]
    for i in range(raw.shape[0]):
        for j in range(4434): # split along vertical line              
            rgb = labelled[i, j][:3]
            # if pixel is labelled
            if (np.all(rgb==[255, 0, 0]) or np.all(rgb==[0, 255, 0])):
                c = 1
                if np.all(rgb == [0, 255, 0]):
                    c = 0
                row = raw[i,j, :3]
                write.writerow(np.append(row, [c, i, j]))

# test set: right part of image
with open('newtestloc.csv', 'w') as fout:
    write = csv.writer(fout)
    red = raw[:,:,0]
    green = raw[:,:,1]
    blue = raw[:,:2]
    for i in range(raw.shape[0]):
        for j in range(4434, raw.shape[1]):
            rgb = labelled[i, j][:3]
            if (np.all(rgb==[255, 0, 0]) or np.all(rgb==[0, 255, 0])):
                c = 1
                if np.all(rgb == [0, 255, 0]):
                    c = 0
                row = raw[i,j, :3]
                write.writerow(np.append(row, [c, i, j]))

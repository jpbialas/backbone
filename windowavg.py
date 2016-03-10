import backbone
from skimage import io
import numpy as np
import csv
import sys
import os

# gets mean of spectral values of surrounding pixels in a window of specified radius

rad = int(sys.argv[1]) # radius, i.e. 2 corresponds to 5x5 window

def conv_avg(img, rad):
    '''
    Zero-padded average sliding window filter
    '''
    res = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not np.all(img[i, j, 0:3] == [255, 255, 255]):                
                val = np.zeros(4)
                for w in range(-rad, rad + 1):
                    for h in range(-rad, rad + 1):
                        r = i + w
                        c = j + h
                        if r>=0 and r<img.shape[0] and c>=0 and c<img.shape[1]:
                            val += img[r, c]
                        else:
                            val += np.array([255, 255, 255, 255]).astype('float')b
                val /= (2*rad + 1)**2
                res[i, j] = val
    return res

def main():
    raw = io.imread('1-0003-0002.tif')
    labelled = io.imread('1-0003-0002_3.tif')
    raw_conv = conv_avg(raw, rad)
    labelled_conv = conv_avg(labelled, rad)

    with open('newtrainw' + str(rad) + 'loc.csv', 'w') as fout:
        write = csv.writer(fout)
        for i in range(raw.shape[0]):
            for j in range(4434):
                rgb = labelled[i, j][:3]
                if (np.all(rgb==[255, 0, 0]) or np.all(rgb==[0, 255, 0])):
                    c = 1
                    if np.all(rgb == [0, 255, 0]):
                        c = 0
                    row = raw_conv[i,j, :3]
                    write.writerow(np.append(row, [c,i,j]))

    with open('newtestw' + str(rad) + 'loc.csv', 'w') as fout:
        write = csv.writer(fout)
        for i in range(raw.shape[0]):
            for j in range(4434, raw.shape[1]):
                rgb = labelled[i, j][:3]
                if (np.all(rgb==[255, 0, 0]) or np.all(rgb==[0, 255, 0])):
                    c = 1
                    if np.all(rgb == [0, 255, 0]):
                        c = 0
                    row = raw_conv[i,j, :3]
                    write.writerow(np.append(row, [c,i,j]))
   
if __name__ == '__main__':
    main()                   

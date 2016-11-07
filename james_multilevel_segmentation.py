import map_overlay
from map_overlay import MapOverlay
import numpy as np
import cv2
import sklearn
from sklearn.externals.joblib import Parallel, delayed
import datetime


def segment_conversion(small, big):
    small_n = np.max(small)
    big_n = np.max(big)
    convert = np.zeros(small_n + 1).astype('int')
    convert[small] = big
    return convert

def main(j):
    maps = [MapOverlay('datafromjoe/1-0003-0002.tif'),MapOverlay('datafromjoe/1-0003-0003.tif')]
    for i in range(10,1010, 10):
        time1 = datetime.datetime.now()
        maps[j].new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(j+2, j+2, i), i)
        time2 = datetime.datetime.now()
        print i, time2-time1

if __name__ == '__main__':
    Parallel(n_jobs=16)(delayed(main)(i) for i in range(2))
 
import map_overlay
from map_overlay import MapOverlay
import numpy as np
import cv2
import sklearn
from sklearn.externals.joblib import Parallel, delayed
import datetime


def segment_conversion(small, big):
    n = np.max(small)
    m = np.max(big)
    convert = np.zeros(small_n + 1).astype('int')
    convert[small] = big
    return convert

def main(j):
    print 'here'
    for i in range(100,110, 10):
        time1 = datetime.datetime.now()
        print time1
        maps[j].new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(j+2, j+2, i), i)
        time2 = datetime.datetime.now()
        print i, time2-time1

maps = [MapOverlay('datafromjoe/1-0003-0002.tif'),MapOverlay('datafromjoe/1-0003-0003.tif')]
if __name__ == '__main__':
    main(0)
    #Parallel(n_jobs=16)(delayed(main)(i) for i in range(2))
 
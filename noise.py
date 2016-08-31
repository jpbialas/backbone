from map_overlay import MapOverlay
import map_overlay
import seg_classify as sc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seg_features as sf

def main(my_map, start = 1, end = 10):
    img_num = my_map.name[-1]
    dims = my_map.rows, my_map.cols
    label_image = my_map.masks['damage'].reshape(dims).astype('uint8')
    segs = my_map.segmentations[50][1].ravel()
    plt.figure('Labelling')
    plt.imshow(label_image, cmap = 'gray')
    kernel = np.ones((3,3),np.uint8)
    for i in range(start, end+1):
        noise_level = i/10.0
        print 'currently on image: ',img_num, ' at noise level: ', noise_level
        noisy = cv2.erode(label_image, np.ones((3,3),np.uint8), iterations = int(noise_level*20)) #REPRESENTS MISSING SOME SEGMENTS
        noisy = cv2.dilate(noisy, kernel, iterations = int(noise_level*60))  #REPRESENTS BROAD LABELS
        labelling = sf.px2seg2(noisy.reshape(my_map.rows*my_map.cols,1), segs)
        indcs = np.where(labelling > .5)[0]
        np.savetxt('damagelabels50/Noise{}-3-{}.csv'.format(noise_level, img_num), indcs, fmt = '%d')



if __name__ == '__main__':
    map_train, map_test = map_overlay.basic_setup([100], 50, label_name = "Jared")
    main(map_train)
    main(map_test)
    plt.show()
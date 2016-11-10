import map_overlay
import seg_classify as sc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seg_features as sf
from convenience_tools import *

def main(my_map, start = 1, end = 2):
    img_num = my_map.name[-1]
    dims = my_map.rows, my_map.cols
    label_image = my_map.masks['damage'].reshape(dims).astype('uint8')
    segs = my_map.segmentations[50].ravel()
    plt.figure('Labelling')
    plt.imshow(label_image, cmap = 'gray')
    kernel = np.ones((3,3),np.uint8)
    for i in range(start, end+1):
        noise_level = i/10.0
        print 'currently on image: ',img_num, ' at noise level: ', noise_level
        #noisy = cv2.erode(label_image, np.ones((3,3),np.uint8), iterations = int(noise_level*20)) #REPRESENTS MISSING SOME SEGMENTS
        noisy = cv2.dilate(label_image, kernel, iterations = int(noise_level*10))  #REPRESENTS BROAD LABELS
        #labelling = sf.px2seg2(noisy.reshape(my_map.rows*my_map.cols,1), segs)
        #indcs = np.where(labelling > .01)[0]
        indcs = set(segs.ravel().astype('int')*noisy.ravel())
        np.savetxt('damagelabels50/Sim{}-3-{}.csv'.format(i, img_num), indcs, fmt = '%d')


def calc_percents():
    seg_percents = []
    px_percents = []
    #segs = np.load('processed_segments/shapefilewithfeatures003-003-50.npy')
    damage = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',')
    #damage_px = damage[segs]
    for i in range(50):
        noise = np.loadtxt('damagelabels50/Sim_{}-3-3.csv'.format(i), delimiter = ',')
        px_percents.append(1-damage.shape[0]/float(noise.shape[0]))
        seg_percents.append(np.setdiff1d(noise, damage).shape[0]/float(np.setdiff1d(noise, damage).shape[0] + damage.shape[0]))
    print seg_percents
    print px_percents
    return seg_percents

def main2(my_map):
    img_num = my_map.name[-1]
    dims = my_map.rows, my_map.cols
    label_image = my_map.masks['damage'].reshape(dims).astype('uint8')
    segs = my_map.segmentations[50].ravel()
    kernel = np.ones((3,3),np.uint8)

    for i in range(0,50):
        iters = i
        noisy = cv2.dilate(label_image, kernel, iterations = int(iters))
        result = np.bincount(segs.ravel().astype('int')*noisy.ravel())
        jared = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',')
        plt.figure(iters)
        plt.imshow(my_map.mask_segments_by_indx(np.concatenate((np.where(result>500)[0], jared)), 50))
        np.savetxt('damagelabels50/Sim_{}-3-3.csv'.format(i), np.concatenate((np.where(result>1000)[0], jared)), delimiter = ',', fmt = '%d')

    '''for i in range(1,5):
        iters = 10*i
        noisy = cv2.dilate(label_image, kernel, iterations = int(iters))
        result = np.bincount(segs.ravel().astype('int')*noisy.ravel())
        jared = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',')
        plt.figure("{} 2".format(iters))
        plt.imshow(my_map.mask_segments_by_indx(np.concatenate((np.where(result>1000)[0], jared)), 50))
        #np.savetxt('damagelabels50/Sim{}-3-3.csv'.format(i), np.concatenate((np.where(result>1000)[0], jared)), delimiter = ',', fmt = '%d')
    '''
    plt.show()


def visualize_noise(my_map):
    img_num = my_map.name[-1]
    for i in range(5):
        noise_level = 10*i
        img_name = 'damagelabels50/Sim_{}-3-{}.csv'.format(noise_level, img_num)
        #plt.figure(img_name)
        next_label = np.loadtxt(img_name, delimiter = ',')
        next_img = my_map.mask_segments_by_indx(next_label, 50, with_img = True)
        show_img(next_img)
    img_name = 'damagelabels50/Joe-3-{}.csv'.format(img_num)
    plt.figure(img_name)
    next_label = np.loadtxt(img_name, delimiter = ',')
    next_img = my_map.mask_segments_by_indx(next_label, 50, with_img = True)
    plt.imshow(next_img)

    img_name = 'damagelabels50/Jared-3-{}.csv'.format(img_num)
    plt.figure(img_name)
    next_label = np.loadtxt(img_name, delimiter = ',')
    next_img = my_map.mask_segments_by_indx(next_label, 50, with_img = True)
    plt.imshow(next_img)

    plt.show()


if __name__ == '__main__':
    map_train, map_test = map_overlay.basic_setup([100], 50, label_name = "Jared")
    #main(map_test)
    #main(map_train)
    #main2(map_test)
    #calc_percents()
    visualize_noise(map_test)
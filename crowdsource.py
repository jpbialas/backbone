import map_overlay
from map_overlay import MapOverlay
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv


def show_images(img_num):
    my_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
    shape = my_map.img.shape
    my_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-50.shp'.format(img_num, img_num), 50)
    labels = create_label_array()[img_num]
    prob_map = np.zeros(shape[:2])
    for row in labels:
        prob_map+=my_map.mask_segments_by_indx(row, 50, with_img = False).reshape(shape[:2])
    prob_map /= len(labels)
    masked_img = my_map.mask_helper(my_map.img, prob_map, opacity = .8)
    fig = plt.figure('Image 3-{}'.format(img_num))
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.imshow(masked_img)
    plt.title('Image 3-{}'.format(img_num)), plt.xticks([]), plt.yticks([])

def prob_labels(img_num):
    my_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
    shape = my_map.img.shape
    my_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-50.shp'.format(img_num, img_num), 50)
    labels = create_label_array()[img_num]
    segs = my_map.segmentations[50][1].ravel()
    nsegs = np.max(segs) + 1
    prob_labels = np.zeros(nsegs)
    for row in labels:
        prob_labels[row]+=1
    prob_labels /= len(labels)
    return prob_labels


def create_label_array(add_extra = True):
    all_labels = load_csv('labels.csv')
    labels = {2: [], 3: []}
    for value in all_labels.values():
        labels[value[0]].append(map(int, value[1][1:-1].split(',')))
    if add_extra:
        labels[2].append(np.loadtxt('damagelabels50/Joe-3-2.csv', delimiter = ',', dtype = 'int'))
        labels[3].append(np.loadtxt('damagelabels50/Joe-3-3.csv', delimiter = ',', dtype = 'int'))
        labels[2].append(np.loadtxt('damagelabels50/Luke-3-2.csv', delimiter = ',', dtype = 'int'))
        labels[3].append(np.loadtxt('damagelabels50/Luke-3-3.csv', delimiter = ',', dtype = 'int'))
    return labels

def load_csv(fn):
    with open(fn, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = dict((rows[0],(int(rows[1]),rows[2])) for rows in reader)
        return mydict

if __name__ == '__main__':
    print np.unique(prob_labels(2))
    print np.unique(prob_labels(3))
    plt.show()


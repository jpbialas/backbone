import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from convenience_tools import *
import os



def px2seg2(values, indcs):
    '''
        See px2seg
        This implementation is faster for larger numbers of segments. The original is
        better for smaller numbers of segments because each segment is computed by numpy
    '''
    pbar = custom_progress()
    n_segs = int(np.max(indcs))+1
    counts = np.bincount(indcs.ravel().astype('int'))
    counts = counts.reshape(counts.shape[0], 1)
    data = np.zeros((n_segs, values.shape[1]), dtype = 'float')
    for i in pbar(range(values.shape[0])):
        data[indcs[i]] += values[i]
    return data/counts

def px2seg(values, indcs):
    '''
        values: an array of length n, which is a raveled image of pixel values
        indcs: an array of length n which contains the index of which of the k segments that that 
                pixel belongs to
        RETURNS: An array of length k where each element holds the average of all values of pixels
                with the corresponding index

        Used for finding average color of each segment or fitting a pixel-based labelling to segents
    '''
    pbar = custom_progress()
    n_segs = int(np.max(indcs))+1
    data = np.zeros((n_segs, values.shape[1]), dtype = 'float')
    for i in pbar(range(n_segs)):
        indices = np.where(indcs == i)[0]
        data[i] = np.sum(values[indices], axis = 0)/indices.shape[0]
    return data


def color_edge(my_map, seg):
    '''
        my_map: map holding the image to extract features from
        seg: Segmentation level to find features for
        RETURNS:
            Average color and edge density for each segment
            Names of those features
    '''
    names = np.array(['red{}'.format(seg),'green{}'.format(seg), 'blue{}'.format(seg), 'ED{}'.format(seg)])
    p = os.path.join('features', "color_edge_{}.npy".format(my_map.segmentations[seg][0]))
    if os.path.exists(p):
        data = np.load(p)
    else:
        edges = cv2.Canny(my_map.img,50,100).reshape((my_map.rows,my_map.cols, 1))
        color_e = np.concatenate((my_map.img, edges), axis = 2).reshape((my_map.rows*my_map.cols, 4))
        segs = my_map.segmentations[seg][1].ravel()
        if seg <= 50:
            data = px2seg2(color_e, segs)
        else:
            data = px2seg(color_e, segs)
        np.save(p, data)
    return data, names


def ecognition_features(img_num, seg):
    '''
        img_num: image number of map to extract features from
        seg: Segmentation level to find features for
        RETURNS:
            All features extracted from ecognition
            Names for ecognition features
    '''
    json_file = open('segmentations/withfeatures{}/{}-{}-features.json'.format(img_num, img_num, seg))
    json_str = json_file.read()
    #print json_str

    json_data = json.loads(json_str.replace('inf', '0'))
    n_segs = len(json_data['features'])
    n_feats = len(json_data['features'][0]['properties'].values())

    data = np.zeros((n_segs, n_feats))
    json_features = json_data['features']
    names = json_features[0]['properties'].keys()
    for i in range(n_segs):
        data[i] = json_features[i]['properties'].values()
    return data, names

def show_shapes(my_map, rect, ellipse, n_segs, level):
    '''
        Not used:
        Draws rectangle and ellipse contours around each segment
    '''
    test = np.zeros(n_segs)
    test[i] = 1
    img = my_map.mask_segments(test, level, with_img = True)
    cv2.ellipse(img = img, box = ellipse, color = [255,0,0], thickness = 5)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)
    cv2.imshow("img",img)
    cv2.waitKey(1000)

def hog(my_map, seg):
    '''
        my_map: map holding the image to extract features from
        seg: Segmentation level to find features for
        RETURNS:
            HoG for each segment
            Names of those features
    '''
    bins = 16
    names = []
    for i in range(bins):
        names.append('hog{}'.format(i))
    p = os.path.join('features', "hog_seg_{}.npy".format(my_map.segmentations[seg][0]))
    if os.path.exists(p):
        data = np.load(p)
    else:
        pbar = custom_progress()
        bw_img = cv2.cvtColor(my_map.img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(bw_img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(bw_img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        angles = ang/2.0*np.pi*255
        segs = my_map.segmentations[seg][1].ravel()
        n_segs = int(np.max(segs))+1
        data = np.zeros((n_segs, 16), dtype = 'uint8')
        for i in pbar(range(n_segs)):
            indices = np.where(segs == i)[0]
            data[i] = np.histogram(angles.ravel()[indices], 16)[0]
        np.save(p, data)
    return data, names


def shapes(my_map, level):
    '''
        my_map: map holding the image to extract features from
        seg: Segmentation level to find features for
        RETURNS:
            Rectangle and Ellipse fits for each segment
            Names of those features
    '''
    names = np.array(['re{}'.format(level),'rf{}'.format(level),'ee{}'.format(level),'ef{}'.format(level)])
    p = os.path.join('features', "aspect_extent_{}.npy".format(my_map.segmentations[level][0]))
    if os.path.exists(p):
        data = np.load(p)
        data = np.clip(data, 0, 1)
        np.save(p,data)
    else:
        pbar = custom_progress()
        segs = my_map.segmentations[level][1]
        n_segs = int(np.max(segs))+1
        data = np.zeros((n_segs, 4), dtype = 'float')
        for i in pbar(range(n_segs)):
            next_shape = (segs == i)
            cnt, _ = cv2.findContours(next_shape.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = np.array(cnt)[0]
            rect = cv2.minAreaRect(cnt)

            if cnt.shape[0]>4:
                ellipse = cv2.fitEllipse(cnt)
            else:
                ellipse == rect

            h,w = rect[1]
            a,b = ellipse[1]
            
            leftBound = max(rect[0][0] - max(2*w, 2*a,2*b), 0)
            rightBound = min(rect[0][0] + max(2*w,2*a,2*b), my_map.cols)
            lower = max(rect[0][1] - max(2*w,2*a,2*b), 0)
            upper = max(rect[0][1] + max(2*w,2*a,2*b), my_map.rows)

            img = np.zeros_like(my_map.img)
            cv2.ellipse(img = img,box = ellipse, color = [1,0,0], thickness = -1)
            shape_cut = next_shape[lower:upper,leftBound:rightBound]
            ellipse_cut = img[lower:upper,leftBound:rightBound,0]
            area = np.sum(shape_cut)
            intersection = np.sum(shape_cut*ellipse_cut)
            union = area+np.sum(ellipse_cut)-intersection

            rect_elong =  0 if max(w,h) == 0 else min(w,h)/float(max(w,h))
            rect_fit = 0 if h*w ==0 else float(area)/(h*w)
            ellipse_elong = 0 if max(a,b) == 0 else min(a,b)/float(max(a,b))
            ellipse_fit = 0 if union == 0 else intersection/union

            data[i,0] = rect_elong
            data[i,1] = rect_fit
            data[i,2] = ellipse_elong
            data[i,3] = ellipse_fit
        np.save(p, data)
    return data, names
    

def multi_segs(my_map, base_seg, seg_levels, ecognition = True):
    '''
        my_map: map holding the image to extract features from
        base_seg: Segmentation level to find features for
        seg_levels: Additional segmentation levels to extract context features from
        ecognition: Boolean indicating whether to use ecognition features
        RETURNS:
            All features for each segment and context segments
            Names of those features
    '''
    img = my_map.img
    img_num = my_map.name[-1]
    h,w,_ = img.shape
    segs = my_map.segmentations[base_seg][1].ravel().astype('int')
    n_segs = int(np.max(segs))
    pbar = custom_progress()
    color_data, color_names = color_edge(my_map, base_seg)
    shape_data, shape_names = shapes(my_map, base_seg)
    hog_data, hog_names = hog(my_map, base_seg)
    if ecognition:
        james_data, james_names = ecognition_features(img_num, base_seg)
        data = np.concatenate((james_data,shape_data, hog_data, color_data), axis = 1)
        names = np.concatenate((james_names, shape_names, hog_names, color_names), axis = 0)
    else:
        data = np.concatenate((shape_data, hog_data, color_data), axis = 1)
        names = np.concatenate((shape_names, hog_names, color_names), axis = 0)    
    if len(seg_levels)>0:
        for seg in seg_levels:
            segmentation = my_map.segmentations[seg][1].ravel().astype('int')
            m_segs = int(np.max(segmentation))
            convert = np.zeros(n_segs+1).astype('int')
            convert[segs] = segmentation
            color_data, color_names = color_edge(my_map, seg)
            shape_data, shape_names = shapes(my_map, seg)
            hog_data, hog_names = hog(my_map, seg)
            next_data = np.concatenate((shape_data, color_data), axis = 1)[convert]
            next_names = np.concatenate((shape_names, color_names), axis = 0)
            data = np.concatenate((next_data, data), axis = 1)
            names = np.concatenate((next_names, names), axis = 0)
    return data, names


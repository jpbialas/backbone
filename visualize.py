import numpy as np
import cv2
import matplotlib.pyplot as plt
import px_features as features
from map_overlay import MapOverlay
import seg_classify as sc


def vis_seg():
    my_map, X, y, names = sc.setup_segs(2, 50, [400], .5, jared = False)
    segs = my_map.segmentations[50][1].ravel().astype('int')
    print segs.shape
    print names
    h,w = my_map.rows, my_map.cols
    print X.shape
    X = X[segs].reshape(h,w,X.shape[1])
    print X.shape


    colors = X[:,:,-4:-1]
    colors100 = X[:,:,-28:-25]
    rect_elong = X[:,:,24]
    rect_elong100 = X[:,:,0]
    rect_fit = X[:,:,25]
    rect_fit100 = X[:,:,1]
    ellipse_elong = X[:,:,26]
    ellipse_elong100 = X[:,:,2]
    ellipse_fit = X[:,:,27]
    ellipse_fit100 = X[:,:,3]
    edges = X[:,:,-1]
    edges100 = X[:,:,-25]

    print 'starting stuff'

    plt.figure()
    plt.imshow(colors.astype('uint8'))
    plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.imshow(colors100.astype('uint8'))
    plt.xticks([]), plt.yticks([])
    print ('done with color')

    plt.show()
    
    plt.figure()
    plt.imshow(edges, cmap = 'gray')
    plt.xticks([]), plt.yticks([])


    plt.figure()
    plt.imshow(edges100, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    print ('done with edges')
    plt.show()

    fig = plt.figure()
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.subplot(221)
    plt.imshow(rect_elong, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.imshow(rect_fit, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(ellipse_elong, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(ellipse_fit, cmap = 'gray')
    plt.xticks([]), plt.yticks([])


    fig = plt.figure()
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.subplot(221)
    plt.imshow(rect_elong100, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.imshow(rect_fit100, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(ellipse_elong100, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(ellipse_fit100, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    print ('done with shapes')
    plt.show()



def vis_px():
    fn = 'haiti/haiti_1002003.tif'
    img = cv2.imread(fn)
    bw_img = cv2.imread(fn, 0)
    myMap = MapOverlay(fn)
    h,w,c = img.shape

    '''
    norm = img
    blur = features.blurred(myMap.img, "haiti_1002003")[0]
    print blur.shape



    f1 = plt.figure('Px Colors', frameon = False)
    f1.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    ax = plt.Axes(f1, [0., 0., 1., 1.])
    ax.set_axis_off()
    f1.add_axes(ax)

    r = norm[:,:,0]
    plt.subplot(231)
    plt.imshow(r, 'Reds')
    plt.xticks([]), plt.yticks([])
    g = norm[:,:,1]
    plt.subplot(232)
    plt.imshow(g, 'Greens')
    plt.xticks([]), plt.yticks([])
    b = norm[:,:,2]
    plt.subplot(233)
    plt.imshow(b, 'Blues')
    plt.xticks([]), plt.yticks([])


    ave_r = blur[:,0].reshape(h,w)
    plt.subplot(234)
    plt.imshow(ave_r, 'Reds')
    plt.xticks([]), plt.yticks([])
    ave_g = blur[:,1].reshape(h,w)
    plt.subplot(235)
    plt.imshow(ave_g, 'Greens')
    plt.xticks([]), plt.yticks([])
    ave_b = blur[:,2].reshape(h,w)
    plt.subplot(236)
    plt.imshow(ave_b, 'Blues')
    plt.xticks([]), plt.yticks([])
    '''
    f2 = plt.figure('Px Edges', frameon = False)
    f2.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    ax = plt.Axes(f2, [0., 0., 1., 1.])
    ax.set_axis_off()
    f2.add_axes(ax)

    edge = features.edge_density(img, 100, img_name = 'haiti_1002003', amp = 1)[0].reshape(h,w)
    print np.max(edge)
    print edge/0.0096
    plt.subplot(111)
    plt.imshow(edge, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    
    '''
    f3 = plt.figure('Histogram of Gradients', frameon = False)
    f2.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    ax = plt.Axes(f3, [0., 0., 1., 1.])
    ax.set_axis_off()
    f3.add_axes(ax)

    hog = features.hog(bw_img, 50, img_name = 'haiti_1002003')[0]
    for i in range(hog.shape[1]):
        f3.add_subplot(4,4,i+1)
        plt.imshow(hog[:,i].reshape(h,w), 'gray')
        plt.xticks([]), plt.yticks([])'''

    plt.show()


vis_px()
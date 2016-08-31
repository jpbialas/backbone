from map_overlay import MapOverlay
import map_overlay
import seg_classify as sc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seg_features as sf

def main(img_num = 2):

    map_train, X_train, y_train, names = sc.setup_segs(img_num, 50, [100], .5, jared = True, new_feats = True)
    dims = map_train.rows, map_train.cols
    label_image = map_train.masks['damage'].reshape(dims).astype('uint8')

    plt.figure('Labelling')
    plt.imshow(label_image, cmap = 'gray')
    kernel = np.ones((3,3),np.uint8)
    noisy = cv2.erode(label_image, np.ones((3,3),np.uint8), iterations = 5) #REPRESENTS MISSING SOME SEGMENTS
    noisy = cv2.dilate(noisy, kernel, iterations = 20)  #REPRESENTS BROAD LABELS
    plt.figure('Bad Labelling')
    plt.imshow(noisy, cmap = 'gray')

    map_train.newPxMask(noisy, 'damage')
    '''print "HERE"
    data, names = sf.multi_segs(map_train, 50, [], use_james = True, joes_labels = False)
    print "here2"
    print data[:,-1]#plt.imshow(data[:, -1].reshape(dims), cmap = 'gray')
    print data.shape
    '''
    data = np.load('noisy_segs_data_3.npy')
    y = data[:,-1]>.01
    
    #y = np.load('noisy_segs_{}.npy'.format(img_num))
    plt.figure('segmented')
    #np.save('noisy_segs_data_{}.npy'.format(img_num), data)
    #np.save('noisy_segs_{}.npy'.format(img_num), y)

    plt.imshow(map_train.mask_segments(y, level = 50, with_img = False).reshape(dims), cmap = 'gray')

def main2():
    map_test, map_train = map_overlay.basic_setup()
    dims = map_train.rows, map_train.cols
    plt.figure("Joes")
    plt.imshow(map_train.masks['damage'].reshape(dims).astype('uint8'), cmap = 'gray')
    plt.show()



if __name__ == '__main__':
    #print 'done with imports'
    #main(2)
    main(3)
    map_train, X_train, y_train, names = sc.setup_segs(3, 50, [100], .5, jared = True, new_feats = True)
    dims = map_train.rows, map_train.cols
    #y = np.load('noisy_segs_3.npy')
    #print np.shape(y)
    #print y, reduce(lambda a,b:a or b, y)
 

    plt.figure('Joe Labels')
    map_train.newMask('datafromjoe/1-003-00{}-damage.shp'.format(3), 'damage2')
    plt.imshow(map_train.masks['damage2'].reshape(dims).astype('uint8'), cmap = 'gray')

    plt.show()
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from object_model import ObjectClassifier
from px_model import PxClassifier
from map_overlay import Geo_Map
import map_overlay
from mpl_toolkits.mplot3d import Axes3D
from tsne import tsne
from sklearn.decomposition import PCA
import seg_classify as sc
from convenience_tools import *

def pca(x, k):
    print("Starting PCA")
    pca = PCA(n_components=k)
    pca.fit(x)
    print("Variance Retained: " + str(pca.explained_variance_ratio_.sum()))
    print("Ending PCA")
    return pca.transform(x)


def assign_colors(labels, start=0):
    '''
    input: (n x 1) array of cluster assignments
    output: (n x 1) list of colors assignments
            List of colors used in order
            Number of points assigned to each ordered colo
    '''
    labels = labels.astype('int')
    colors = [-1]*len(labels)
    colorNames = [ "green", "red", "blue", "yellow", "orange", "grey", "cyan"]
    numbers = [0]*len(colorNames)
    for indx, i in enumerate(labels):
        if i < len(colorNames):
            colors[indx] = colorNames[i-start]
            numbers[i-start]+=1
        else:
            colors[indx] = "black"
    return colors, colorNames, numbers


def get_samples(img_num, n, label, load = True):
    if load:
        print('loading results for {}'.format(img_num))
        ans = np.loadtxt('temp/cached_{}.csv'.format(img_num), delimiter = ',')
        print('done loading')
    else:
        next_map = Geo_Map('datafromjoe/1-0003-000{}.tif'.format(img_num))
        next_map.newMask('datafromjoe/1-003-00{}-damage.shp'.format(img_num),'damage')
        all_damage = np.where(next_map.getLabels('damage') == 1)[0]

        sub_sample = all_damage[np.random.random_integers(0,all_damage.shape[0]-1, n)]
        print('generating for {}'.format(img_num))
        X, names = gen_features(next_map, 200, 200, 16)
        ans = X[sub_sample]
        print("done")
        np.savetxt('temp/cached_{}.csv'.format(img_num), ans, delimiter = ',')
    return ans, np.ones(n)*label

def plot(y, colors, k, name):
    if k == 2:
        xaxis = y[:,0]
        yaxis = y[:,1]
        fig = plt.figure(name)
        ax = fig.add_subplot(111)
        green_patch = mpatches.Patch(color='green', label='Joes Data')
        red_patch = mpatches.Patch(color='red', label='Jareds Data')
        #plt.legend(handles=[green_patch, red_patch])
        ax.scatter(xaxis,yaxis, s = 3, color = colors)#, alpha = 0.5)
    if k == 3:
        xaxis = np.array(y[:,0])
        yaxis = np.array(y[:,1])
        zaxis = np.array(y[:,2])
        fig = plt.figure(name)
        ax = fig.add_subplot(111, projection = '3d')
        #print colors
        print len(xaxis), len(yaxis)
        print len(colors)
        ax.scatter(xs=xaxis, ys=yaxis, zs = zaxis, zdir = 'z', s = 5, edgecolors = colors, c = colors)#, depthshade = True, alpha = .5)


def all_histos(NZ1, NZ2, Haiti, names, indices = None, prefix = ''):
    pbar = custom_progress()
    j = 0
    if indices == None:
        indices = np.arange(len(names))
        print indices
    for i in pbar(indices):
        hist1, bins1 = np.histogram(NZ1[:,i], bins=32)
        hist2, bins2 = np.histogram(NZ2[:,i], bins=32)
        hist3, bins3 = np.histogram(Haiti[:,i], bins=32)
        leftmost = min(bins1[0],bins2[0], bins3[0])
        rightmost = max(bins1[-1], bins2[-1], bins3[-1])

        fig = plt.figure('{}{}'.format(prefix,names[i]))
        plt.title(names[i])
        ax1 = plt.subplot(311)
        ax1.set_title('NZ1')
        width = 0.7 * (bins1[1] - bins1[0])
        center = (bins1[:-1] + bins1[1:]) / 2
        plt.bar(center, hist1, align='center', width=width)
        plt.xlim(leftmost, rightmost)
        plt.xticks([], [])
        
        ax2 = plt.subplot(312)
        ax2.set_title('NZ2')
        width = 0.7 * (bins2[1] - bins2[0])
        center = (bins2[:-1] + bins2[1:]) / 2
        plt.bar(center, hist2, align='center', width=width)
        plt.xlim(leftmost, rightmost)
        plt.xticks([], [])


        ax3 = plt.subplot(313)
        ax3.set_title('Haiti')
        width = 0.7 * (bins3[1] - bins3[0])
        center = (bins3[:-1] + bins3[1:]) / 2
        plt.bar(center, hist3, align='center', width=width)
        plt.xlim(leftmost, rightmost)

        #plt.close(fig)
        j+=1
    plt.show()


def look_at_features(names, indices):
    N_INDCS = -1

    map_test, map_train = map_overlay.basic_setup([100, 400], 50, label_name = "Jared")
    clf = ObjectClassifier()
    damage_labels = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    building_labels = np.loadtxt('damagelabels50/all_rooftops-3-3.csv', delimiter  =',').astype('int')
    segs = map_train.segmentations[50][1]
    n_segs = int(np.max(segs.ravel()))+1
    all_labels = np.zeros(n_segs) #Nothing = Green
    all_labels[damage_labels] = 1  #Damage = Red
    all_labels[building_labels] = 2  #Building = Blue

    all_data, y = clf._get_X_y(map_train, 'Jared')
    '''
    clf.fit(map_train, label_name = 'Jared') #<-this one changesthe training for the importance order
    names =  clf.feat_names
    print names

    importances = clf.model.feature_importances_
    indices = np.argsort(importances)[::-1]'''
    pbar = custom_progress()
    j = names.shape[0]
    for i in pbar(indices[::-1]):
        j-=1
        fig = plt.figure(names[i])
        plt.imshow(all_data[:,i][segs.astype('int')], cmap = 'gray')
        fig.savefig('/Users/jsfrank/Desktop/features/{}-{}'.format(j, names[i]), format='png')
        plt.close(fig)


def transition():
    N_INDCS = -1

    map_test, map_train = map_overlay.basic_setup([100], 50, label_name = "Jared")
    clf = ObjectClassifier()
    damage_labels = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    building_labels = np.loadtxt('damagelabels50/all_buildings-3-3.csv', delimiter  =',').astype('int')
    segs = map_train.segmentations[50][1].ravel()
    n_segs = int(np.max(segs))+1
    all_labels = np.zeros(n_segs) #Nothing = Green
    all_labels[damage_labels] = 1  #Damage = Red
    all_labels[building_labels] = 2  #Building = Blue
    all_data, y = clf._get_X_y(map_train, 'Jared')
    clf.fit(map_train, label_name = 'Jared')
    names =  clf.feat_names

    for i in range(50):
        new_damage = np.concatenate((all_data[damage_labels], all_data[building_labels[:i*10]]), axis = 0)
        print new_damage.shape
        all_histos(new_damage, all_data[building_labels], all_data[np.where(all_labels == 0)[0]], names, [17], prefix = "{} ".format(i*10))
    plt.show()


def comparepx():
    N_INDCS = -1
    NZ1, NZ2 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    Haiti = Geo_Map('haiti/haiti_1002003.tif')
    clf = PxClassifier(85,-1)

    haiti_data, names = clf.gen_features(Haiti)
    print np.mean(haiti_data, axis = 0), np.std(haiti_data, axis = 0)
    nz1_data, names = clf.gen_features(NZ1)
    print np.mean(nz1_data, axis = 0), np.std(nz1_data, axis = 0)
    nz2_data, names = clf.gen_features(NZ2)
    print np.mean(nz2_data, axis = 0), np.std(nz2_data, axis = 0)


    haiti_sub = haiti_data[np.random.choice(haiti_data.shape[0], 100000, replace = False)]
    nz1_sub = nz1_data[np.random.choice(nz1_data.shape[0], 100000, replace = False)]
    nz2_sub = nz2_data[np.random.choice(nz2_data.shape[0], 100000, replace = False)]
    print 'done subbing'
    print haiti_sub.shape
    print names
    all_histos(nz1_sub, nz2_sub, haiti_sub, names, indices = None, prefix = '')
    plt.show()

comparepx()
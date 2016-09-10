import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from object_model import ObjectClassifier
from px_model import PxClassifier
from map_overlay import MapOverlay
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
        next_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
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


def all_histos(damage, buildings, other, names, indices, prefix = ''):
    pbar = custom_progress()
    j = 0
    for i in pbar(indices):

        
        hist1, bins1 = np.histogram(damage[:,i], bins=32)
        hist2, bins2 = np.histogram(buildings[:,i], bins=32)
        hist3, bins3 = np.histogram(other[:,i], bins=32)
        leftmost = min(bins1[0],bins2[0], bins3[0])
        rightmost = max(bins1[-1], bins2[-1], bins3[-1])

        fig = plt.figure('{}{}'.format(prefix,names[i]))
        plt.title(names[i])
        ax1 = plt.subplot(311)
        ax1.set_title('Damage')
        width = 0.7 * (bins1[1] - bins1[0])
        center = (bins1[:-1] + bins1[1:]) / 2
        plt.bar(center, hist1, align='center', width=width)
        plt.xlim(leftmost, rightmost)
        plt.xticks([], [])
        
        ax2 = plt.subplot(312)
        ax2.set_title('Buildings')
        width = 0.7 * (bins2[1] - bins2[0])
        center = (bins2[:-1] + bins2[1:]) / 2
        plt.bar(center, hist2, align='center', width=width)
        plt.xlim(leftmost, rightmost)
        plt.xticks([], [])


        ax3 = plt.subplot(313)
        ax3.set_title('Other')
        width = 0.7 * (bins3[1] - bins3[0])
        center = (bins3[:-1] + bins3[1:]) / 2
        plt.bar(center, hist3, align='center', width=width)
        plt.xlim(leftmost, rightmost)


        fig.savefig('/Users/jsfrank/Desktop/px_histos/{}-{}-histo'.format(j, names[i]), format='png')
        plt.close(fig)
        j+=1


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


def comparepx(k):
    N_INDCS = -1
    map_test, map_train = map_overlay.basic_setup([100], 50, label_name = "all_buildings")
    segs = map_train.segmentations[50][1].ravel().astype('int')
    clf = PxClassifier(85,-1)
    damage_labels = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    building_labels = np.loadtxt('damagelabels50/all_buildings-3-3.csv', delimiter  =',').astype('int')
    
    n_segs = int(np.max(segs))+1
    all_labels = np.zeros(n_segs) #Nothing = Green
    all_labels[damage_labels] = 1  #Damage = Red
    all_labels[building_labels] = 2  #Building = Blue

    all_labels = all_labels[segs].ravel()
    #plt.imshow(all_labels.reshape(map_train.rows, map_train.cols), cmap = 'gray')
    

    indcs = np.random.choice(np.arange(map_train.rows*map_train.cols), 1000000, replace = False).astype('int')
    #print np.where(all_labels == 0)[0][indcs].shape
    #print np.where(all_labels == 1)[0][indcs].shape
    #print np.where(all_labels == 2)[0][indcs].shape
    all_data, names = clf.gen_features(map_train)
    #print all_data[:,:3].shape, map_train.rows, map_train.cols
    #plt.imshow(all_data[:,:3].reshape(map_train.rows, map_train.cols, 3))
    #plt.show()
    #print all_data[np.where(all_labels == 1)[0]][indcs], all_data[np.where(all_labels[indcs] == 1)[0]].shape

    '''clf.fit(map_train)
    clf.feature_importance()
    plt.show()
    names =  clf.feat_names
    importances = clf.model.feature_importances_
    indices = np.argsort(importances)[::-1]'''
    #print names[indices][:N_INDCS]
    all_histos(all_data[indcs][np.where(all_labels[indcs] == 1)[0]], all_data[indcs][np.where(all_labels[indcs] == 2)[0]], all_data[indcs][np.where(all_labels[indcs] == 0)[0]], names, np.arange(names.shape[0]), prefix = 'px ')
    #plt.show()


def compare(k):
    N_INDCS = -1

    map_test, map_train = map_overlay.basic_setup([100, 400], 50, label_name = "Jared")
    clf = ObjectClassifier()
    damage_labels = np.loadtxt('damagelabels50/Jared-3-3.csv', delimiter = ',').astype('int')
    building_labels = np.loadtxt('damagelabels50/all_rooftops-3-3.csv', delimiter  =',').astype('int')
    segs = map_train.segmentations[50][1].ravel()
    n_segs = int(np.max(segs))+1
    all_labels = np.zeros(n_segs) #Nothing = Green
    all_labels[damage_labels] = 1  #Damage = Red
    all_labels[building_labels] = 2  #Building = Blue

    all_data, y = clf._get_X_y(map_train, 'Jared')
    clf.fit(map_train, label_name = 'Jared') #<-this one changesthe training for the importance order
    names =  clf.feat_names
    print names

    importances = clf.model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print names[indices][:N_INDCS]
    plt.figure()
    std = np.std([tree.feature_importances_ for tree in clf.model.estimators_],
                 axis=0)
    plt.title("Feature importances")
    plt.bar(range(names.shape[0]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(names.shape[0]), names[indices])
    plt.xlim([-1, names.shape[0]])

    all_histos(all_data[damage_labels], all_data[building_labels], all_data[np.where(all_labels == 0)[0]], names, indices[:N_INDCS])
    #plt.show()
    return names, indices
    '''

    colors, color_names, _ = assign_colors(all_labels)
    print color_names

    print ("starting pca")
    y_pca = pca(all_data[:,:], 50)
    #indcs = np.random.choice(np.arange(n_segs), 5000, replace = False).astype('int')
    #print indcs
    y_pca = y_pca
    print('done pca')
    print y_pca.shape
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0, verbose = 2)
    np.set_printoptions(suppress=True)
    print ("starting tsne")
    y_tsne = model.fit_transform(y_pca)
    print ('done tsne')

    #plot(y_tsne, colors, k, 'TSNE')
    plot(y_tsne, np.array(colors), k, 'PCA')

    plt.show()

    '''
    
comparepx(2)
#look_at_features(names, indices)

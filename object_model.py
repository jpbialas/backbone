from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results
import crowdsource
import sklearn
from convenience_tools import *
from sklearn.externals.joblib import Parallel, delayed



class ObjectClassifier():

    def __init__(self, verbose = 0):
        self.verbose = verbose
        self.params = {
            "n_trees" : 85, 
            "base_seg" : 20, 
            "segs" : [50], 
            "thresh" : .5,
            "new_feats" : True,
            "EVEN" : 2
        }

    def sample(self, y, EVEN, n_samples = -1):
        if EVEN>0:
            zeros = np.where(y==0)[0]
            ones = np.where(y==1)[0]
            n0,n1 = zeros.shape[0], ones.shape[0]
            if EVEN == 2:
                replace = True
                n = max(n0, n1)
            else:
                replace = False
                n = min(n0, n1)
            if n_samples == -1:
                zero_samples = np.random.choice(zeros, n, replace = replace)
                one_samples = np.random.choice(ones, n, replace = replace)
            else:
                zero_samples = np.random.choice(zeros, n_samples//2, replace = replace)
                one_samples = np.random.choice(ones, n_samples//2, replace = replace)
            return np.concatenate((zero_samples, one_samples))
        else:
            return np.arange(y.shape[0])

    def _get_X_y(self, next_map, label_name, custom_labels = None, custom_data = None, custom_fn = None):
        img_num = next_map.name[-1]
        if custom_data is None:
            X, feat_names = sf.multi_segs(next_map, self.params['base_seg'], self.params['segs'], self.params['new_feats'])
            self.feat_names = feat_names
        else:
            X = custom_data
            print 'X ', X.shape
        if custom_labels is None:
            fn = 'damagelabels50/{}-3-{}.csv'.format(label_name, img_num) if custom_fn == None else custom_fn
            damage_indices = np.loadtxt(fn, delimiter = ',', dtype = 'int')
            y = np.zeros(X.shape[0])
            y[damage_indices] = 1
        else:
            y = custom_labels
            print 'Y ',y.shape
        return X, y

    def reset_model(self):
        self.model= RandomForestClassifier(n_estimators=self.params['n_trees'], n_jobs = -1, verbose = self.verbose)

    def testing_suite(self, map_test, prediction_prob, save = True):
        v_print('generating visuals', self.verbose)
        heat_fig = analyze_results.probability_heat_map(map_test, prediction_prob, self.test_name, save = save)
        analyze_results.ROC(map_test, map_test.getLabels('damage'), prediction_prob, self.test_name, save = save)
        map_test.newPxMask(prediction_prob.ravel()>.4, 'damage_pred')
        sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred', self.test_name, save)
        v_print('done generating visuals', self.verbose)

    def fit(self, map_train, label_name = "Jared", custom_labels = None, custom_data = None):
        v_print('starting fit', self.verbose)
        X, y = self._get_X_y(map_train, label_name, custom_labels, custom_data)
        samples = self.sample(y, self.params['EVEN']) 
        self.model= RandomForestClassifier(n_estimators=self.params['n_trees'], n_jobs = -1, verbose = self.verbose)#, class_weight = "balanced")
        self.model.fit(X[samples], y[samples])
        v_print('ending fit', self.verbose)

    def predict_proba(self, map_test, label_name = 'Jared', custom_labels = None, custom_data = None):
        segment_probs = self.predict_proba_segs(map_test, label_name, custom_labels, custom_data)
        px_probs = segment_probs[map_test.segmentations[self.params['base_seg']][1].astype('int')]
        v_print('ending predict', self.verbose)
        return px_probs

    def predict_proba_segs(self, map_test, label_name = 'Jared', custom_labels = None, custom_data = None):
        v_print('starting predict', self.verbose)
        X, y = self._get_X_y(map_test, label_name, custom_labels, custom_data)
        img_num = map_test.name[-1]
        self.test_name = analyze_results.gen_model_name("Segs", label_name, self.params['EVEN'], img_num, self.params['new_feats'])
        segment_probs = self.model.predict_proba(X)[:,1]
        v_print('ending predict', self.verbose)
        return segment_probs
    
    def predict(self, map_test, label_name = "Jared", thresh = .5):
        return self.predict_proba(map_test, label_name)>thresh

    def fit_and_predict(self, map_train, map_test, label_name = 'Jared'):
        self.fit(map_train, label_name)
        return self.predict_proba(map_test, label_name)

    def feature_importance(self):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1].astype('int')

        # Print the feature ranking
        print("Feature ranking:")


        for f in range(len(self.feat_names)):
            print("{}. feature {}: ({})".format(f + 1, self.feat_names[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.feat_names)), importances[indices], color="r", yerr=std[indices], align="center")
        #print(range(len(self.feat_names)))
        #print(indices.dtype)
        plt.xticks(range(len(self.feat_names)), np.array(self.feat_names)[indices])
        plt.xlim([-1, len(self.feat_names)])
        #plt.show()


def main(labels_2, labels_3, threshs_2, threshs_3, thresh):
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    segs_3 = map_3.segmentations[50][1]
    truth_3 = labels_3[segs_3.astype('int')]
    segs_2 = map_2.segmentations[50][1]
    truth_2 = labels_2[segs_2.astype('int')]

    print 'thresh {}'.format(thresh)
    model = ObjectClassifier()
    model.fit(map_3, custom_labels = labels_3>thresh)
    pred = model.predict_proba(map_2, "Jared")
    #np.savetxt('one_pred.csv', pred, delimiter = ',', fmt = '%1.3f')
    print sklearn.metrics.roc_auc_score(map_2.getLabels('damage'), pred.ravel())
    for i in threshs_2:
        print sklearn.metrics.roc_auc_score(truth_2.ravel()>i, pred.ravel())

def main_haiti():
    p_thresh = .06
    haiti_map = map_overlay.haiti_setup()
    model = ObjectClassifier()
    X, y = model._get_X_y(haiti_map, 'damage', custom_fn = 'damagelabels20/Jared.csv')
    print X.shape, y.shape
    segs = haiti_map.segmentations[20][1].astype('int')
    left = np.unique(segs.reshape(haiti_map.shape2d)[:,:2048]).astype(int)
    right = np.unique(segs.reshape(haiti_map.shape2d)[:,2048:]).astype(int)
    top = np.unique(segs.reshape(haiti_map.shape2d)[:2048,:]).astype(int)
    bottom = np.unique(segs.reshape(haiti_map.shape2d)[2048:,:]).astype(int)


    model.fit(haiti_map, custom_data = X[left], custom_labels=y[left])
    #model.feature_importance()
    proba = model.predict_proba(haiti_map, custom_data = X, custom_labels=y)
    g_truth = haiti_map.masks['damage'].reshape(4096,4096)
    analyze_results.ROC(haiti_map, g_truth[:,2048:].ravel(), proba[:,2048:].ravel(), 'Haiti Right')
    print analyze_results.FPR_from_FNR(g_truth[:,2048:].ravel(), proba[:,2048:].ravel())
    #analyze_results.probability_heat_map(haiti_map, proba, 'Haiti Right', save = False)
    fig = plt.figure()
    plt.imshow(haiti_map.mask_helper(haiti_map.img, proba.reshape(4096,4096)>p_thresh)[:, 2048:,:])
    plt.contour(g_truth.reshape(haiti_map.rows, haiti_map.cols)[:, 2048:], colors = 'green')
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.xticks([]), plt.yticks([])
    '''
    model.fit(haiti_map, custom_data = X[right], custom_labels=y[right])
    model.feature_importance()
    proba = model.predict_proba(haiti_map, custom_data = X, custom_labels=y)
    g_truth = haiti_map.masks['damage'].reshape(4096,4096)
    print analyze_results.ROC(haiti_map, g_truth[:,:2048].ravel(), proba[:,:2048].ravel(), 'Haiti Left')[1:3]
    #analyze_results.probability_heat_map(haiti_map, proba, 'Haiti Left', save = False)
    fig = plt.figure()
    plt.imshow(haiti_map.mask_helper(haiti_map.img, proba.reshape(4096,4096)>p_thresh)[:,:2048,:])
    plt.contour(g_truth.reshape(haiti_map.rows, haiti_map.cols)[:,:2048], colors = 'green')
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.xticks([]), plt.yticks([])

    model.fit(haiti_map, custom_data = X[top], custom_labels=y[top])
    #model.feature_importance()
    proba = model.predict_proba(haiti_map, custom_data = X, custom_labels=y)
    g_truth = haiti_map.masks['damage'].reshape(4096,4096)
    print analyze_results.ROC(haiti_map, g_truth[2048:,:].ravel(), proba[2048:,:].ravel(), 'Haiti Bottom')[1:3]
    #analyze_results.probability_heat_map(haiti_map, proba, 'Haiti Bottom', save = False)
    fig = plt.figure()
    plt.imshow(haiti_map.mask_helper(haiti_map.img, proba.reshape(4096,4096)>p_thresh)[2048:,:,:])
    plt.contour(g_truth.reshape(haiti_map.rows, haiti_map.cols)[2048:,:], colors = 'green')
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.xticks([]), plt.yticks([])


    model.fit(haiti_map, custom_data = X[bottom], custom_labels=y[bottom])
    #model.feature_importance()
    proba = model.predict_proba(haiti_map, custom_data = X, custom_labels=y)
    g_truth = haiti_map.masks['damage'].reshape(4096,4096)
    print analyze_results.ROC(haiti_map, g_truth[:2048,:].ravel(), proba[:2048,:].ravel(), 'Haiti Top')[1:3]
    #analyze_results.probability_heat_map(haiti_map, proba, 'Haiti Top', save = False)
    fig = plt.figure()
    plt.imshow(haiti_map.mask_helper(haiti_map.img, proba.reshape(4096,4096)>p_thresh)[:2048,:,:])
    plt.contour(g_truth.reshape(haiti_map.rows, haiti_map.cols)[:2048,:], colors = 'green')
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.xticks([]), plt.yticks([])
    '''

    plt.show()

if __name__ == '__main__':
    main_haiti()
    '''labels_2 = crowdsource.prob_labels(2)
    labels_3 = crowdsource.prob_labels(3)
    threshs_2 = np.unique(labels_2)[:-1]
    threshs_3 = np.unique(labels_3)[:-1]
    print 'trianing threshs', threshs_3
    print 'testing threshs', threshs_2

    Parallel(n_jobs=-1)(delayed(main)(labels_2,labels_3,threshs_2,threshs_3,thresh) for thresh in threshs_3)'''



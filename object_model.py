from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results
from labeler import Labelers
import sklearn
from convenience_tools import *
from sklearn.externals.joblib import Parallel, delayed



class ObjectClassifier():

    def __init__(self, verbose = 0):
        self.verbose   = verbose
        self.n_trees   = 85 
        self.base_seg  = 20 
        self.segs      = [50] 
        self.thresh    = .5
        self.new_feats = True
        self.even      = 2

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

    def get_X(self, next_map, custom_data = None):
        img_num = next_map.name[-1]
        if custom_data is None:
            X, feat_names = sf.multi_segs(next_map, self.base_seg, self.segs, self.new_feats)
            self.feat_names = feat_names
        else:
            X = custom_data
            self.feat_names = None
        return X

    def reset_model(self):
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)

    def fit(self, map_train, labels, custom_data = None):
        v_print('starting fit', self.verbose)
        X = self.get_X(map_train, custom_data)
        samples = self.sample(labels, self.even) 
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)#, class_weight = "balanced")
        self.model.fit(X[samples], labels[samples])
        v_print('ending fit', self.verbose)

    def predict_proba(self, map_test, custom_data = None):
        segment_probs = self.predict_proba_segs(map_test, custom_data)
        px_probs = segment_probs[map_test.segmentations[self.base_seg][1].astype('int')]
        v_print('ending predict', self.verbose)
        return px_probs

    def predict_proba_segs(self, map_test, custom_data = None):
        v_print('starting predict', self.verbose)
        X = self.get_X(map_test, custom_data)
        img_num = map_test.name[-1]
        segment_probs = self.model.predict_proba(X)[:,1]
        v_print('ending predict', self.verbose)
        return segment_probs
    
    def predict(self, map_test, thresh = .5):
        return self.predict_proba(map_test)>thresh

    def fit_and_predict(self, map_train, map_test, labels):
        self.fit(map_train, labels)
        return self.predict_proba(map_test)

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
        print(range(len(self.feat_names)))
        print(indices.dtype)
        plt.xticks(range(len(self.feat_names)), np.array(self.feat_names)[indices])
        plt.xlim([-1, len(self.feat_names)])
        



def main_haiti():
    y = Labelers().majority_vote()
    y2 = Labelers().majority_vote(labeler_indices = np.array([0,2]))
    p_thresh = .06
    haiti_map = map_overlay.haiti_setup()
    model = ObjectClassifier()
    X = model.get_X(haiti_map)
    train = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    test = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
    segs = haiti_map.segmentations[20][1].astype('int')
    train_segs = np.unique(segs[train]).astype(int)
    test_segs  = np.unique(segs[test]).astype(int)
    print test_segs.shape
    print train_segs.shape
    model.fit(haiti_map, y2[train_segs], X[train_segs])
    probs = model.predict_proba(haiti_map)
    g_truth = y[segs][test].ravel()
    proba = probs.reshape(4096,4096)[test].ravel()
    print analyze_results.FPR_from_FNR(y[segs][test].ravel(), probs.reshape(4096,4096)[test].ravel(), TPR = .95)
    analyze_results.ROC(haiti_map, g_truth.ravel(), proba.ravel(), 'Haiti Test')[:2]
    plt.show()

if __name__ == '__main__':
    main_haiti()



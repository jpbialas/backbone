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
        self.haiti_constants()


    def NZ_constants(self):
        self.n_trees   = 85 
        self.base_seg  = 50
        self.segs      = [100]
        self.even      = 2
        self.features  = sf.NZ_features 

    def haiti_constants(self):
        self.n_trees   = 200
        self.base_seg  = 20 
        self.segs      = [50]
        self.even      = 2
        self.features  = sf.haiti_features 


    def sample(self, y, EVEN, n_samples = -1):
        if EVEN>0:
            zeros = np.where(y==0)[0]
            ones = np.where(y==1)[0]
            n0,n1 = zeros.shape[0], ones.shape[0]
            replace = True if EVEN == 2 else False
            n = max(n0, n1) if EVEN == 2 else min(n0, n1)
            if n_samples == -1:
                zero_samples = np.random.choice(zeros, n, replace = replace)
                one_samples = np.random.choice(ones, n, replace = replace)
            else:
                zero_samples = np.random.choice(zeros, n_samples//2, replace = replace)
                one_samples = np.random.choice(ones, n_samples//2, replace = replace)
            return np.concatenate((zero_samples, one_samples))
        else:
            return np.arange(y.shape[0])

    def get_X(self, next_map):
        if next_map.X is None:
            X, feat_names = self.features(next_map, self.base_seg, self.segs)
            X = X[next_map.unique_segs(self.base_seg)]
            next_map.X = X
            self.feat_names = feat_names
        else:
            X = next_map.X
            self.feat_names = None
        return X

    def reset_model(self):
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)

    def fit(self, map_train, labels, indcs = None):
        if indcs is None:
            indcs = np.arange(labels.shape[0])
        v_print('starting fit', self.verbose)
        X = self.get_X(map_train)[indcs]
        y = labels[indcs]
        samples = self.sample(y, self.even) 
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)#, class_weight = "balanced")
        self.model.fit(X[samples], y[samples])
        v_print('ending fit', self.verbose)
    

    def predict_proba(self, map_test):
        seg_probs = self.predict_proba_segs(map_test, None)
        px_probs = map_test.seg_convert(self.base_seg, seg_probs)
        v_print('ending predict', self.verbose)
        return px_probs

    def predict_proba_segs(self, map_test, indcs = None):
        if indcs is None:
            indcs = np.arange(map_test.unique_segs(self.base_seg).shape[0])
        v_print('starting predict', self.verbose)
        X = self.get_X(map_test)[indcs]
        segment_probs = self.model.predict_proba(X)[:,1]
        v_print('ending predict', self.verbose)
        return segment_probs
    
    def predict(self, map_test, thresh = .5):
        return self.predict_proba(map_test)>thresh

    def fit_and_predict(self, map_train, map_test, labels, indcs = None):
        self.fit(map_train, labels, indcs)
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
    model      = ObjectClassifier(1)
    y          = Labelers().majority_vote()
    test       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    train      = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
    haiti_map  = map_overlay.haiti_setup()
    train_map  = haiti_map.sub_map(train)
    test_map   = haiti_map.sub_map(test)
    probs = model.fit_and_predict(train_map, test_map, y[train_map.unique_segs(20)])
    g_truth = y[test_map.segmentations[20]]
    print analyze_results.FPR_from_FNR(g_truth.ravel(), probs.ravel(), TPR = .95)
    analyze_results.probability_heat_map(test_map, probs.ravel(), '')
    analyze_results.ROC(g_truth.ravel(), probs.ravel(), 'Haiti Test')[:2]
    plt.figure('mask')
    plt.imshow(test_map.mask_helper(test_map.img, probs))
    plt.figure('mask 6%')
    plt.imshow(test_map.mask_helper(test_map.img, probs>0.06))
    plt.figure('mask 50%')
    plt.imshow(test_map.mask_helper(test_map.img, probs>0.5))
    plt.show()

def label_test():
    model      = ObjectClassifier(1)
    labelers   = Labelers()
    test       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    train      = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
    haiti_map  = map_overlay.haiti_setup()
    train_map  = haiti_map.sub_map(train)
    test_map   = haiti_map.sub_map(test)
    for email in labelers.emails:
        print email
        fig = plt.figure(email)
        fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
        y = (labelers.labeler(email) == labelers.majority_vote())
        probs = model.fit_and_predict(train_map, test_map, y[train_map.unique_segs(20)])
        plt.imshow(probs, cmap = 'seismic',  norm = plt.Normalize(0,1))
        plt.title(email), plt.xticks([]), plt.yticks([])    
    plt.figure('img')
    plt.imshow(test_map.img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()




if __name__ == '__main__':
    #main_haiti()
    label_test()



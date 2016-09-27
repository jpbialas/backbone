from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
#import matplotlib.pyplot as plt
import analyze_results
from sklearn.metrics import roc_curve
from convenience_tools import *



class ObjectClassifier():

    def __init__(self, verbose = 0):
        self.verbose = verbose
        self.params = {
            "n_trees" : 85, 
            "base_seg" : 50, 
            "segs" : [100], 
            "thresh" : .5,
            "new_feats" : True,
            "EVEN" : 1
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

    def _get_X_y(self, next_map, label_name, custom_labels = None, custom_data = None):
        img_num = next_map.name[-1]
        if custom_data is None:
            X, feat_names = sf.multi_segs(next_map, self.params['base_seg'], self.params['segs'], self.params['new_feats'])
            self.feat_names = feat_names
        else:
            X = custom_data
        if custom_labels is None:
            damage_indices = np.loadtxt('damagelabels50/{}-3-{}.csv'.format(label_name, img_num), delimiter = ',', dtype = 'int')
        else:
            damage_indices = custom_labels
        
        y = np.zeros(X.shape[0])
        y[damage_indices] = 1
        return X, y

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

    def predict_proba(self, map_test, label_name = 'Jared', custom_labels = None):
        v_print('starting predict', self.verbose)
        X, y = self._get_X_y(map_test, label_name, custom_labels)
        img_num = map_test.name[-1]
        self.test_name = analyze_results.gen_model_name("Segs", label_name, self.params['EVEN'], img_num, self.params['new_feats'])
        segment_probs = self.model.predict_proba(X)[:,1]
        px_probs = segment_probs[map_test.segmentations[self.params['base_seg']][1].astype('int')]
        v_print('ending predict', self.verbose)
        return px_probs
    
    def predict(self, map_test, label_name = "Jared", thresh = .5):
        return self.predict_proba(map_test, label_name)>thresh

    def fit_and_predict(self, map_train, map_test, label_name = 'Jared'):
        self.fit(map_train, label_name)
        return self.predict_proba(map_test, label_name)

if __name__ == '__main__':
    jared_test, jared_train = map_overlay.basic_setup([100], 50, label_name = "Jared")
    

    ob_clf1 = ObjectClassifier()
    pred_jared = ob_clf1.fit_and_predict(jared_train, jared_test, "Jared")
    #ob_clf1.testing_suite(jared_test, pred_jared)
    print(roc_curve(jared_test.getLabels('damage'), pred_jared.ravel()))
    #print analyze_results.ROC(jared_test, jared_test.getLabels('damage'), pred_jared, ob_clf1.test_name, save = False)




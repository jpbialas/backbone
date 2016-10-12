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
            "base_seg" : 50, 
            "segs" : [100], 
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

    def _get_X_y(self, next_map, label_name, custom_labels = None, custom_data = None):
        img_num = next_map.name[-1]
        if custom_data is None:
            X, feat_names = sf.multi_segs(next_map, self.params['base_seg'], self.params['segs'], self.params['new_feats'])
            self.feat_names = feat_names
        else:
            X = custom_data
        if custom_labels is None:
            damage_indices = np.loadtxt('damagelabels50/{}-3-{}.csv'.format(label_name, img_num), delimiter = ',', dtype = 'int')
            y = np.zeros(X.shape[0])
            y[damage_indices] = 1
        else:
            y = custom_labels
        
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

    def predict_proba(self, map_test, label_name = 'Jared', custom_labels = None):
        segment_probs = self.predict_proba_segs(map_test, label_name, custom_labels)
        px_probs = segment_probs[map_test.segmentations[self.params['base_seg']][1].astype('int')]
        v_print('ending predict', self.verbose)
        return px_probs

    def predict_proba_segs(self, map_test, label_name = 'Jared', custom_labels = None):
        v_print('starting predict', self.verbose)
        X, y = self._get_X_y(map_test, label_name, custom_labels)
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

def main(labels_2, labels_3, threshs_2, threshs_3, thresh):
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    segs_3 = map_3.segmentations[50][1]
    truth_3 = labels_3[segs_3.astype('int')]

    print 'thresh'
    model = ObjectClassifier()
    model.fit(map_2, custom_labels = labels_2>thresh)
    pred = model.predict_proba(map_3, "Jared")
    #np.savetxt('one_pred.csv', pred, delimiter = ',', fmt = '%1.3f')
    for i in threshs_3:
        print '\t testing at thresh i'
        print sklearn.metrics.roc_auc_score(truth_3.ravel()>i, pred.ravel())

if __name__ == '__main__':
    labels_2 = crowdsource.prob_labels(2)
    labels_3 = crowdsource.prob_labels(3)
    threshs_2 = np.unique(labels_2)[:-1]
    threshs_3 = np.unique(labels_3)[:-1]

    Parallel(n_jobs=-1)(delayed(main)(labels_2,labels_3,threshs_2,threshs_3,thresh) for thresh in threshs_2)



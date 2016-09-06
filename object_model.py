from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results
from convenience_tools import *



class ObjectClassifier():

    def __init__(self, verbose = 1):
        self.verbose = verbose
        self.params = {
            "n_trees" : 1000, 
            "base_seg" : 50, 
            "segs" : [100], 
            "thresh" : .5,
            "new_feats" : True,
            "EVEN" : True
        }

    def sample(self, y, EVEN, n_samples = -1):
        if EVEN:
            zeros = np.where(y==0)[0]
            ones = np.where(y==1)[0]
            n0,n1 = zeros.shape[0], ones.shape[0]
            if n_samples == -1:
                zero_samples = np.random.choice(zeros, min(n0, n1))
                one_samples = np.random.choice(ones, min(n0, n1))
            else:
                zero_samples = np.random.choice(zeros, n_samples//2)
                one_samples = np.random.choice(ones, n_samples//2)
            return np.concatenate((zero_samples, one_samples))
        else:
            return np.arange(y.shape[0])

    def _get_X_y(self, next_map, label_name):
        img_num = next_map.name[-1]
        X, feat_names = sf.multi_segs(next_map, self.params['base_seg'], self.params['segs'], self.params['new_feats'])
        damage_indices = np.loadtxt('damagelabels50/{}-3-{}.csv'.format(label_name, img_num), delimiter = ',', dtype = 'int')
        self.feat_names = feat_names
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

    def fit(self, map_train, label_name = "Jared"):
        v_print('starting fit', self.verbose)
        X, y = self._get_X_y(map_train, label_name)
        samples = self.sample(y, self.params['EVEN']) 
        self.model= RandomForestClassifier(n_estimators=self.params['n_trees'], n_jobs = -1, verbose = self.verbose)#, class_weight = "balanced")
        self.model.fit(X[samples], y[samples])
        v_print('ending fit', self.verbose)

    def predict_proba(self, map_test, label_name = 'Jared'):
        v_print('starting predict', self.verbose)
        X, y = self._get_X_y(map_test, label_name)
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
    joe_test, joe_train = map_overlay.basic_setup([100], 50, label_name = "jared_with_buildings")
    

    ob_clf2 = ObjectClassifier()
    pred_joe = ob_clf2.fit_and_predict(joe_train, joe_test, "jared_with_buildings")
    #analyze_results.ROC(joe_test, joe_test.getLabels('damage'), pred_joe, ob_clf2.test_name, save = False)
    print analyze_results.average_class_prob(joe_test, joe_test.getLabels('damage'), pred_joe, ob_clf2.test_name)
    #ob_clf2.testing_suite(joe_test, pred_joe)

    ob_clf1 = ObjectClassifier()
    pred_jared = ob_clf1.fit_and_predict(jared_train, jared_test, "Jared")
    #ob_clf1.testing_suite(jared_test, pred_jared)
    #analyze_results.ROC(jared_test, joe_test.getLabels('damage'), pred_jared, ob_clf1.test_name, save = False)
    print analyze_results.average_class_prob(jared_test, joe_test.getLabels('damage'), pred_jared, ob_clf1.test_name)

    analyze_results.compare_heatmaps(pred_jared, pred_joe)
    plt.show()



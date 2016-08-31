from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results



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
        data, feat_names = sf.multi_segs(next_map, self.params['base_seg'], self.params['segs'], self.params['new_feats'], True)
        damage_indices = np.loadtxt('damagelabels50/{}-3-{}.csv'.format(label_name, img_num), delimiter = ',', dtype = 'int')
        self.feat_names = feat_names
        X = data[:,:-1]
        y = np.zeros(X.shape[0])
        y[damage_indices] = 1
        return X, y


    def fit(self, map_train, label_name = "Jared"):
        X, y = self._get_X_y(map_train, label_name)
        samples = self.sample(y, self.params['EVEN']) 
        self.model= RandomForestClassifier(n_estimators=self.params['n_trees'], n_jobs = -1, verbose = self.verbose)#, class_weight = "balanced")
        self.model.fit(X[samples], y[samples])

    def predict_proba(self, map_test, label_name = 'Jared'):
        X, y = self._get_X_y(map_test, label_name)
        img_num = map_test.name[-1]
        model_name = analyze_results.gen_model_name("Segs", label_name, self.params['EVEN'], img_num, True)
        segment_probs = self.model.predict_proba(X)[:,1]
        px_probs = segment_probs[map_test.segmentations[self.params['base_seg']][1].astype('int')]
        return px_probs


if __name__ == '__main__':
    jared_test, jared_train = map_overlay.basic_setup([100], 50, label_name = "Jared")
    ob_clf1 = ObjectClassifier()
    ob_clf1.fit(jared_train, "Jared")
    pred_jared = ob_clf1.predict_proba(jared_test, "Jared")


    analyze_results.probability_heat_map(jared_test, pred_jared, 'Jared Heatmap', False)
    analyze_results.ROC(jared_test, jared_test.getLabels('damage'), pred_jared, 'Jared ROC', save = False)
    jared_test.newPxMask(pred_jared.ravel()>.7, 'damage_pred')
    sbs_fig = analyze_results.side_by_side(jared_test, 'damage', 'damage_pred', "Jared SBS", save = False)


    joe_test, joe_train = map_overlay.basic_setup([100], 50, label_name = "Joe")
    ob_clf2 = ObjectClassifier()
    ob_clf2.fit(joe_train, "Joe")
    pred_joe = ob_clf2.predict_proba(joe_test, "Joe")


    analyze_results.probability_heat_map(joe_test, pred_joe, 'Joe Heatmap', False)
    analyze_results.ROC(joe_test, joe_test.getLabels('damage'), pred_joe, 'Joe ROC', save = False)
    joe_test.newPxMask(pred_joe.ravel()>.7, 'damage_pred')
    sbs_fig = analyze_results.side_by_side(joe_test, 'damage', 'damage_pred', "Joe SBS", save = False)
    analyze_results.probability_heat_map(joe_test, pred_joe, 'Joe Heatmap', False)

    analyze_results.compare_heatmaps(pred_jared, pred_joe)
    plt.show()



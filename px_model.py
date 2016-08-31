from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import px_features
import matplotlib.pyplot as plt
import analyze_results
import os
import progressbar
from convenience_tools import *

class PxClassifier():

    def __init__(self, n_estimators, n_jobs, verbose = 1):
        self.verbose = verbose
        #self.model = RandomForestClassifier(n_estimators, n_jobs, verbose = verbose)
        self.params = {
            'frac_train': 0.01,
            'frac_test' : 1,
            'mask_train' : 'damage',
            'mask_test' : 'damage',
            'edge_k' : 100,
            'hog_k' : 50,
            'nbins' : 16,
            'EVEN' : True
        }

    def sample(self, labels, nsamples, EVEN):
        '''
        INPUT: 
            - labels:   (nx1 ndarray) Array containing binary labels
            - nsamples: (int) Value representingtotal number of indices to be sampled
                            NOTE: (If odd, produces list of length (nsamples-1))
            - EVEN:    (boolean) True if indices should be evenly sampled, false otherwise
        OUTPUT:
            - Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
                nsamples/2 indices have value 0
        '''
        if EVEN:
            n = labels.shape[0]
            zeros = np.where(labels == 0)[0]
            n_zeros = np.shape(zeros)[0]
            ones = np.where(labels == 1)[0]
            n_ones = np.shape(ones)[0]
            zero_samples = np.random.choice(zeros, nsamples/2)
            one_samples = np.random.choice(ones, nsamples/2)
            final_set = np.concatenate((zero_samples, one_samples))
        else:
            final_set = np.random.random_integers(0,y_train.shape[0]-1, int(n_train*params['frac_train']))
        return final_set


    def gen_features(self, new_map, params):
        '''
        input:
            - new_map: MapObject
        output:
            - feature representation of map
        '''
        #entropy, entropy_name = px_features.entr(new_map.bw_img, img_name = new_map.name)

        #glcm, glcm_name = px_features.GLCM(new_map.bw_img, 50, img_name = new_map.name)
        rgb, rgb_name = px_features.normalized(new_map.getMapData(), img_name = new_map.name)
        ave_rgb, ave_rgb_name = px_features.blurred(new_map.img, img_name = new_map.name)
        edges, edges_name = px_features.edge_density(new_map.bw_img, params['edge_k'], img_name = new_map.name, amp = 1)
        hog, hog_name = px_features.hog(new_map.bw_img, params['hog_k'], img_name = new_map.name, bins = params['nbins'])
        v_print('Concatenating', False)
        data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
        names = np.concatenate((rgb_name, ave_rgb_name, edges_name, hog_name))
        v_print('Done Concatenating', False)
        return data, names



    def testing_suite(self, map_test, prediction_prob):
        heat_fig = analyze_results.probability_heat_map(map_test, prediction_prob, self.test_name, save = True)
        analyze_results.ROC(map_test, map_test.getLabels('damage'), prediction_prob, self.test_name, save = True)
        map_test.newPxMask(prediction_prob.ravel()>.4, 'damage_pred')
        sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred', self.test_name, True)


    def fit(self, map_train):
        v_print('Generating Features', self.verbose)
        X_train, feat_names = self.gen_features(map_train, self.params)
        v_print('Done generating Features', self.verbose)
        self.feat_names = feat_names
        y_train =  map_train.getLabels(self.params['mask_train'])
        n_train = y_train.shape[0]
        train = self.sample(y_train, int(n_train*self.params['frac_train']), self.params['EVEN'])
        v_print("Start Modelling", self.verbose)
        self.model = RandomForestClassifier(n_estimators=85, n_jobs = -1, verbose = self.verbose)
        self.model.fit(X_train[train], y_train[train])
        v_print("Done Modelling", self.verbose)


    def predict_proba(self, map_test, label_name = "Jared"):
        v_print('Starting test gen', self.verbose)
        X_test, feat_names = self.gen_features(map_test, self.params)
        v_print('Done test gen', self.verbose)
        y_test = map_test.getLabels(self.params['mask_test'])
        n_test = y_test.shape[0]
        test = np.random.random_integers(0,y_test.shape[0]-1, int(n_test*self.params['frac_test']))
        img_num = map_test.name[-1]
        self.test_name = analyze_results.gen_model_name("Px", label_name, self.params['EVEN'], img_num, False)
        ground_truth = y_test[test]
        v_print('Starting Predction', self.verbose)
        prediction_prob = self.model.predict_proba(X_test[test])[:,1]
        v_print("Done with Prediction", self.verbose)
        '''if self.params['frac_test'] == 1:
            np.save('PXpredictions/'+self.test_name+'_probs.npy', prediction_prob)'''
        return prediction_prob

    def predict(self, map_test, label_name = "Jared", thresh = .5):
        return self.predict_proba(map_test, label_name)>thresh

    def fit_and_predict(self, map_train, map_test):
        self.fit(map_train)
        return self.prediction_prob(map_test)

    def feature_importance(self):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(len(self.feat_names)):
            print("{}. feature {}: ({})".format(f + 1, self.feat_names[indices[f]], importances[indices[f]]))
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.feat_names)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(self.feat_names)), self.feat_names[indices])
        plt.xlim([-1, len(self.feat_names)])


if __name__ == "__main__":
    print 'setting up'
    map_train, map_test = map_overlay.basic_setup([], label_name = "Jared")
    print 'done setting up'
    model = PxClassifier(85,-1)
    model.fit_and_predict(map_train, map_test)
    model.testing_suite()
    plt.show()


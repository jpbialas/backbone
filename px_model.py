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

    def __init__(self, n_estimators = 85, n_train = 265824, verbose = 0):
        self.verbose = verbose
        self.n_train = n_train
        self.params = {
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
        print nsamples
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
            final_set = np.random.random_integers(0,y_train.shape[0]-1, n_samples)
        return final_set


    def gen_features(self, new_map, params = None):
        '''
        input:
            - new_map: MapObject
        output:
            - feature representation of map
        '''
        #entropy, entropy_name = px_features.entr(new_map.bw_img, img_name = new_map.name)

        #glcm, glcm_name = px_features.GLCM(new_map.bw_img, 50, img_name = new_map.name)
        if new_map.X is None:
            if params == None:
                params = self.params
            rgb, rgb_name = px_features.normalized(new_map.getMapData(), img_name = new_map.name)
            ave_rgb, ave_rgb_name = px_features.blurred(new_map.img, img_name = new_map.name)
            edges, edges_name = px_features.edge_density(new_map.bw_img, params['edge_k'], img_name = new_map.name, amp = 1)
            hog, hog_name = px_features.hog(new_map.bw_img, params['hog_k'], img_name = new_map.name, bins = params['nbins'])
            max_d, max_d_names = px_features.bright_max_diff(new_map.img, params['edge_k'], img_name = new_map.name)
            v_print('Concatenating', self.verbose)
            data = np.concatenate((rgb, ave_rgb, edges, max_d, hog), axis = 1)
            names = np.concatenate((rgb_name, ave_rgb_name, edges_name, max_d_names, hog_name))
            v_print('Done Concatenating', self.verbose)
            new_map.X = data
            self.feat_names = names
        else:
            data = new_map.X
        return data



    def testing_suite(self, map_test, prediction_prob):
        v_print('generating visuals', self.verbose)
        heat_fig = analyze_results.probability_heat_map(map_test, prediction_prob, self.test_name, save = True)
        analyze_results.ROC(map_test.getLabels('damage'), prediction_prob, self.test_name, save = True)
        map_test.newPxMask(prediction_prob.ravel()>.4, 'damage_pred')
        sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred', self.test_name, True)
        v_print('done generating visuals', self.verbose)


    def fit(self, map_train, labels):
        v_print('Generating Features', self.verbose)
        X_train = self.gen_features(map_train, self.params)
        v_print('Done generating Features', self.verbose)
        train = self.sample(labels, int(self.n_train), self.params['EVEN'])
        v_print("Start Modelling", self.verbose)
        self.model = RandomForestClassifier(n_estimators = n_estimators, n_jobs = -1, verbose = self.verbose)
        self.model.fit(X_train[train], labels[train])
        v_print("Done Modelling", self.verbose)


    def predict_proba(self, map_test):
        v_print('Starting test gen', self.verbose)
        X_test = self.gen_features(map_test, self.params)
        v_print('Done test gen', self.verbose)
        prediction_prob = self.model.predict_proba(X_test)[:,1]
        v_print("Done with Prediction", self.verbose)
        return prediction_prob

    def predict(self, map_test, thresh = .5):
        return self.predict_proba(map_test)>thresh

    def fit_and_predict(self, map_train, map_test, labels):
        self.fit(map_train, labels)
        probs = self.predict_proba(map_test)
        return probs

    def feature_importance(self):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        v_print("Feature ranking:", self.verbose)
        for f in range(len(self.feat_names)):
            v_print("{}. feature {}: ({})".format(f + 1, self.feat_names[indices[f]], importances[indices[f]]), self.verbose)
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.feat_names)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(self.feat_names)), self.feat_names[indices])
        plt.xlim([-1, len(self.feat_names)])


if __name__ == "__main__":
    pass


    


import matplotlib
matplotlib.use('Agg')
from object_model import ObjectClassifier
import analyze_results
import map_overlay
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.externals.joblib import Parallel, delayed
from convenience_tools import *
import cv2

class al:    
    def __init__(self, postfix = '', random = False, show = False):
        self.start_n = 50
        self.batch_size = 50
        self.updates = 200
        self.verbose = 1
        self.TPR = .95
        self.path = 'al/'
        self.show = show
        self.postfix = postfix + '_rf' if not random else postfix+'_random'
        self.train = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
        self.test = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
        self.uncertainty = self.rf_uncertainty
        self.update_labels = self.simple_update
        self.haiti_map = map_overlay.haiti_setup()
        segs = self.haiti_map.segmentations[20][1].ravel().astype('int')
        segs = segs.reshape(self.haiti_map.shape2d)
        self.train_segs = np.unique(segs[self.train]).astype(int)
        self.test_segs = np.unique(segs[self.test]).astype(int)
        print self.test_segs, self.test_segs.shape
        model = ObjectClassifier(verbose = 0)
        X, y = model._get_X_y(self.haiti_map, 'damage', \
                              custom_fn = 'damagelabels20/Jared.csv')
        self.X_train, self.X_test = X[self.train_segs], X[self.test_segs]
        self.y_train, self.y_test = y[self.train_segs], y[self.test_segs]
        self.fprs = []
        self.training_labels = self._gen_training_labels()
        self.test_progress()

    def _gen_training_labels(self):
        training_labels = np.ones_like(self.y_train)*-1
        np.random.seed()
        for i in range(2):
            training_labels[np.random.choice(np.where(self.y_train==i)[0], self.start_n//2, replace = False)] = i
        return training_labels


    def _uncertain_order(self, importance, valid_indices):
        '''
        Returns valid indices sorted by each index's value in importance.
        '''
        order = valid_indices[np.argsort(importance[valid_indices])]
        order = order[::-1]
        return order

    def partial_segs_to_full(self, proba_segs, indcs):
        segs = self.haiti_map.segmentations[20][1].ravel().astype('int')
        all_segs = np.zeros(np.max(segs)+1)
        all_segs[indcs] = proba_segs
        return all_segs

    def show_selected(self):
        lab_train_indcs = np.where(self.training_labels != -1)[0]
        lab_indcs = self.train_segs[lab_train_indcs]
        img = self.haiti_map.mask_segments_by_indx(lab_indcs,\
                                     20, opacity = 1, with_img = True)
        img = cv2.cvtColor(img[self.train], cv2.COLOR_BGR2RGB)
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        cv2.imwrite('{}selected_{}{}.png'.format(self.path, n_labeled,\
                                                 self.postfix), img)


    def rf_uncertainty(self):
        thresh = .06
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.haiti_map, custom_data = self.X_train[training_sample],\
                  custom_labels=self.y_train[training_sample])
        proba_segs = model.predict_proba_segs(self.haiti_map, \
                                            custom_data = self.X_train,\
                                            custom_labels=self.y_train)
        if self.show:
            self.show_selected()
            fig = plt.figure()
            segs_train = self.haiti_map.segmentations[20][1].astype('int')[self.train]
            n_labeled = np.where(self.training_labels != -1)[0].shape[0]
            plt.imshow(self.partial_segs_to_full(proba_segs, self.train_segs)[segs_train], cmap = 'seismic', norm = plt.Normalize(0,1))
            fig.savefig('{}test_{}{}.png'.format(self.path, n_labeled, self.postfix), format='png')
            plt.close(fig)
        unknown_indcs = np.where(self.training_labels == -1)[0]
        uncertainties = 1-np.abs(proba_segs-thresh)
        return self._uncertain_order(uncertainties.ravel(), unknown_indcs)


    def random_uncertainty(self):
        self.show_selected()
        return np.random.permutation(np.where(self.training_labels == -1)[0])


    def simple_update(self, new_training):
        self.training_labels[new_training] = self.y_train[new_training]


    def test_progress(self):
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.haiti_map, custom_data = self.X_train[training_sample],\
                  custom_labels=self.y_train[training_sample])
        seg_map = self.haiti_map.segmentations[20][1].astype('int')[self.test]
        proba_segs = model.predict_proba_segs(self.haiti_map, \
                                              custom_data = self.X_test,\
                                              custom_labels=self.y_test)
        proba = self.partial_segs_to_full(proba_segs, self.test_segs)[seg_map]
        g_truth = self.haiti_map.masks['damage'].reshape(4096,4096)[self.test]
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        if self.show:
            fig, AUC = analyze_results.ROC(self.haiti_map, g_truth.ravel(), proba.ravel(), 'Haiti Test')[:2]
            fig.savefig('{}ROC_{}{}.png'.format(self.path, n_labeled, self.postfix), format='png')
            plt.close(fig)
        FPR = analyze_results.FPR_from_FNR(g_truth.ravel(), proba.ravel(), TPR = self.TPR)
        self.fprs.append(FPR)
        np.savetxt('{}fprs{}.csv'.format(self.path, self.postfix), self.fprs, delimiter = ',', fmt = '%1.4f')


    def update(self):
        new_training = self.uncertainty()[:self.batch_size]
        self.update_labels(new_training)
        self.test_progress()


    def run(self):
        for i in range(self.updates):
            self.update()

def run_al(i, n_runs):
    next_al = al(postfix = '_{}'.format(i%(n_runs/2)), random = i<n_runs/2)
    next_al.run()

if __name__ == '__main__':
    n_runs = 16
    Parallel(n_jobs=n_runs)(delayed(run_al)(i,n_runs) for i in range(n_runs))
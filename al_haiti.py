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
from labeler import Labelers
import cv2

class al:    
    def __init__(self, postfix = '', random = False, update = 'donmez', unique_email = None, show = False):
        self.set_params()
        self.show            = show
        self.unique_email    = unique_email
        self.update_type     = update
        self.postfix         = postfix #+ '_rf' if not random else postfix + '_random'
        self.uncertainty     = self.rf_uncertainty if not random else self.random_uncertainty
        self.setup_map_split()
        self.labelers        = Labelers()
        self.training_labels = self._gen_training_labels(self.labelers.majority_vote()[self.train_segs])
        self.test_progress()

    def set_params(self):
        self.start_n    = 50
        self.batch_size = 50
        self.updates    = 1800
        self.verbose    = 1
        self.TPR        = .95
        self.path       = 'al/'
        self.fprs       = []
        self.UIs        = []

    def setup_map_split(self):
        self.train      = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
        self.test       = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
        self.haiti_map  = map_overlay.haiti_setup()
        self.segs       = self.haiti_map.segmentations[20][1].astype('int')
        self.train_segs = np.unique(self.segs[self.train]).astype(int)
        self.test_segs  = np.unique(self.segs[self.test]).astype(int)
        model           = ObjectClassifier(verbose = 0)
        X               = model.get_X(self.haiti_map)
        self.X_train    = X[self.train_segs]
        self.X_test     = X[self.test_segs]


    def _gen_training_labels(self, y_train):
        training_labels = np.ones_like(self.train_segs)*-1
        np.random.seed()
        for i in range(2):
            sub_samp = np.where(y_train==i)[0]
            indices = np.random.choice(sub_samp, self.start_n//2, replace = False)
            self.labelers.donmez_vote(indices, .85, True)
            training_labels[indices] = i
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
        img = self.haiti_map.mask_segments_by_indx(lab_indcs, 20, 1, True)
        img = cv2.cvtColor(img[self.train], cv2.COLOR_BGR2RGB)
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        cv2.imwrite('{}selected_{}{}.png'.format(self.path, n_labeled,\
                                                 self.postfix), img)


    def rf_uncertainty(self):
        thresh = .06
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.haiti_map, self.training_labels[training_sample], self.X_train[training_sample])
        proba_segs = model.predict_proba_segs(self.haiti_map, self.X_train)
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
        if self.show:
            self.show_selected()
        return np.random.permutation(np.where(self.training_labels == -1)[0])


    def update_labels(self, new_training):
        if self.update_type == "donmez":
            new_labs = self.labelers.donmez_vote(self.train_segs[new_training], 0.85, True)
            self.UIs.append(self.labelers.UI())
            np.save('{}UIs{}.npy'.format(self.path, self.postfix), np.array(self.UIs))
        elif self.update_type == "majority":
            new_labs = self.labelers.majority_vote(self.train_segs[new_training])
        else:
            new_labs = self.labelers.labeler(self.unique_email)[self.train_segs[new_training]]
        self.training_labels[new_training] = new_labs


    def test_progress(self):
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.haiti_map, self.training_labels[training_sample], self.X_train[training_sample])
        proba_segs = model.predict_proba_segs(self.haiti_map, self.X_test)
        proba = self.partial_segs_to_full(proba_segs, self.test_segs)[self.segs][self.test]
        g_truth = self.labelers.majority_vote()[self.segs][self.test]
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        if self.show:
            fig, AUC = analyze_results.ROC(self.haiti_map, g_truth.ravel(), proba.ravel(), 'Haiti Test')[:2]
            fig.savefig('{}ROC_{}{}.png'.format(self.path, n_labeled, self.postfix), format='png')
            plt.close(fig)
        FPR = analyze_results.FPR_from_FNR(g_truth.ravel(), proba.ravel(), TPR = self.TPR)
        self.fprs.append(FPR)
        np.save('{}fprs{}.npy'.format(self.path, self.postfix), self.fprs)


    def update(self):
        new_training = self.uncertainty()[:self.batch_size]
        self.update_labels(new_training)
        self.test_progress()


    def run(self):
        for i in range(self.updates):
            self.update()



def run_al(i, n_runs):
    next_al = al(postfix = '_donmez_rf_{}'.format(i))
    next_al.run()

if __name__ == '__main__':
    n_runs = 16
    Parallel(n_jobs=n_runs)(delayed(run_al)(i,n_runs) for i in range(n_runs))
    #a = al()
    #a.run()
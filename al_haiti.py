import matplotlib
matplotlib.use('Agg')
from object_model import ObjectClassifier
import analyze_results
import map_overlay
import matplotlib.pyplot as plt
import sys
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
        self.postfix         = postfix
        self.uncertainty     = self.rf_uncertainty if not random else self.random_uncertainty
        self.setup_map_split()
        self.labelers        = Labelers()
        self.training_labels = self._gen_training_labels(self.labelers.majority_vote()[self.train_map.unique_segs(self.seg)])
        self.test_progress()

    def set_params(self):
        self.start_n    = 200
        self.batch_size = 50
        self.updates    = 700
        self.verbose    = 1
        self.TPR        = .95
        self.seg        = 20
        self.path       = 'al_7_200/'
        self.fprs       = []
        self.UIs        = []

    def setup_map_split(self):
        self.train      = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
        self.test       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2, 4096))
        self.haiti_map  = map_overlay.haiti_setup()
        self.train_map  = self.haiti_map.sub_map(self.train)
        self.test_map   = self.haiti_map.sub_map(self.test)

    def _gen_training_labels(self, y_train):
        training_labels = np.ones_like(y_train)*-1
        np.random.seed()
        for i in range(2):
            sub_samp = np.where(y_train==i)[0]
            train_indices = np.random.choice(sub_samp, self.start_n//2, replace = False)
            seg_indices = self.train_map.unique_segs(20)[train_indices]
            self.labelers.donmez_vote(seg_indices, .85, True)
            self.labelers.model_start(self.train_map, train_indices)
            training_labels[train_indices] = i
        return training_labels


    def _uncertain_order(self, importance, valid_indices):
        '''
        Returns valid indices sorted by each index's value in importance.
        '''
        order = valid_indices[np.argsort(importance[valid_indices])]
        order = order[::-1]
        return order

    def partial_segs_to_full(self, proba_segs, indcs):
        segs = self.haiti_map.segmentations[20].ravel().astype('int')
        all_segs = np.zeros(np.max(segs)+1)
        all_segs[indcs] = proba_segs
        return all_segs

    def show_selected(self):
        lab_train_indcs = np.where(self.training_labels != -1)[0]
        lab_indcs = self.train_map.unique_segs(self.seg)[lab_train_indcs]
        img = self.train_map.mask_segments_by_indx(lab_indcs, 20, 1, True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        cv2.imwrite('{}selected_{}{}.png'.format(self.path, n_labeled,\
                                                 self.postfix), img)


    def rf_uncertainty(self):
        thresh = .06
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.train_map, self.training_labels, training_sample)
        proba_segs = model.predict_proba_segs(self.train_map)
        if self.show:
            self.show_selected()
            fig = plt.figure()
            n_labeled = np.where(self.training_labels != -1)[0].shape[0]
            img = self.train_map.seg_convert(self.seg, proba_segs)
            plt.imshow(img, cmap = 'seismic', norm = plt.Normalize(0,1))
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
        train_segs = self.train_map.unique_segs(self.seg)
        if "donmez" in self.update_type:
            new_labs = self.labelers.donmez_vote(train_segs[new_training], 1, True)
            self.UIs.append(self.labelers.UI())
            np.save('{}UIs{}.npy'.format(self.path, self.postfix), np.array(self.UIs))
        elif self.update_type == "majority":
            new_labs = self.labelers.majority_vote(train_segs[new_training])
        elif self.update_type == "random":
            labelers = np.random.randint(0, len(self.labelers.labels), len(new_training))
            new_labs = self.labelers.labels[labelers, train_segs[new_training]]
        elif self.update_type == "email":
            new_labs = self.labelers.labeler(self.unique_email)[train_segs[new_training]]
        elif self.update_type == "model":
            new_labs = self.labelers.model_vote(self.train_map, new_training)
        elif self.update_type == "model_2":
            new_labs = self.labelers.model_vote(self.train_map, new_training, all = True)

        self.training_labels[new_training] = new_labs


    def test_progress(self):
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(self.training_labels, EVEN = 2)
        print self.labelers.rewards
        print np.unique(training_sample), self.training_labels[np.unique(training_sample)], training_sample.shape, np.unique(training_sample).shape
        print self.labelers.labeler('masexaue@mtu.edu')[self.train_map.unique_segs(self.seg)[np.unique(training_sample)]]
        model.fit(self.train_map, self.training_labels, training_sample)
        proba = model.predict_proba(self.test_map)
        g_truth = self.labelers.majority_vote()[self.test_map.segmentations[self.seg]]
        n_labeled = np.where(self.training_labels > -1)[0].shape[0]
        if self.show:
            fig, AUC = analyze_results.ROC(g_truth.ravel(), proba.ravel(), 'Haiti Test')[:2]
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



def run_al(i, update, random):
    assert(random == 'random' or random == 'rf')
    next_al = al(postfix = '_{}_{}_{}'.format(update, random, i), random = random == "random", update = update)
    next_al.run()

if __name__ == '__main__':
    #options = [('majority', 'random'), ('random', 'random'), ('majority', 'rf'), ('model', 'rf'), ('donmez', 'rf'), ('random', 'rf')]
    options = [('model', 'rf'), ('donmez_1', 'rf'), ('random', 'rf'), ('random', 'random')]
    option = options[int(sys.argv[2])]
    run_al(sys.argv[1], option[0], option[1])
    
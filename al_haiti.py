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
from Xie import EM

class al(object):
    """
    Responsible for setting up and running active learning experiments

    Parameters
    ----------
    postfix : String
        String added to the end of filenames when saving results. Important for being able to run multiple experiments at once without overwriting
    [random] : boolean
        New training data selected to be sampled is done so randomly if True, and based on the random forest probability if False
    [update_method] : String
        Describes how to assign labels to new training data based on a number of methods
        Examples include: donmez, donmez1, majority, random, email, yan, xie
    [unique_email] : String
        If update type is 'email', labels will be assigned according to unique_email's labels
    [show] : boolean
        If True, figures are generated at various stages to help illustrate progress

    Fields
    ------
    All Parameters are also set as fields with identical purposes

    start_n : int
        Number of initial training data to label
    batch_size : int
        Number of new training data to label for each iteration
    updates : int
        Number of iterations to run through
    verbose : int
        Prints various messages indicating progress if 1. Prints nothing if 0
    TPR : int
        TPR value to evaluate FPR at
    seg : int
        Segmentation level associated with data
    thresh : float
        Threshold around which to select most uncertain data for random forest method
    path : String
        Folder path to save all files in
    fprs : float list
        Stores all FPRs from evaluation as active learning run moves forward
    UIs  : float list
        Stores all UI confidences generated from Donmez run if that is the labeling method
    unceratinty : void->ndarray
        Function used to select new training data to label
    train : ndarray
        index list indicating portion of map to set as training (see Px_Map.submap for details)
    test : ndarray
        index list indicating portion of map to set as testing (see Px_Map.submap for details)
    haiti_map : Px_Map
        Haiti map with all data for active learning run
    train_map : Px_Map
        Sub map of Haiti Map used for training as assigned by train
    test_map : Px_Map
        Sub map of Haiti Map used for testing as assigned by test
    labelers : Labelers
        Stores all crowdsourcing labelers who have labeled the iamge
    training_labels : ndarray
        Array with length equal to the number of segments in the training data
        Indices that have not yet been assigned a label have value -1
        Indices that have been labeled as non damage have value 0
        Indices that have been labeled as damage have value 1
    """

    def __init__(self, postfix = '', random = False, update_method = 'donmez', unique_email = None, show = False):
        self.set_params()
        self.show            = show
        self.unique_email    = unique_email
        self.update_method   = update_method
        self.postfix         = postfix
        self.uncertainty     = self.rf_uncertainty if not random else self.random_uncertainty
        self.setup_map_split()
        self.labelers        = Labelers()
        self.training_labels = self._gen_training_labels(self.labelers.majority_vote()[self.train_map.unique_segs(self.seg)])
        self.test_progress()


    def set_params(self):
        """Sets basic parameters"""
        self.start_n    = 50
        self.batch_size = 50
        self.updates    = 700
        self.verbose    = 1
        self.TPR        = .95
        self.seg        = 20
        self.thresh     = .06
        self.path       = 'al_9/'
        self.fprs       = []
        self.UIs        = []

    def setup_map_split(self):
        """Splits haiti map into training portion and testing portion"""
        self.train      = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
        self.test       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2, 4096))
        self.haiti_map  = map_overlay.haiti_setup()
        self.train_map  = self.haiti_map.sub_map(self.train)
        self.test_map   = self.haiti_map.sub_map(self.test)

    def _gen_training_labels(self, y_train):
        """
        Initializes training_labels with start_n labels
        
        Labels are 50% damage and 50% non damage to give a good starting distribution
        """
        #initialize training labels
        training_labels = np.ones_like(y_train)*-1
        np.random.seed()
        #For each class value, choose start_n/2 random training examples to label using majority vote
        for i in range(2):
            sub_samp = np.where(y_train==i)[0]
            train_indices = np.random.choice(sub_samp, self.start_n//2, replace = False)
            seg_indices = self.train_map.unique_segs(self.seg)[train_indices]
            self.labelers.donmez_vote(seg_indices, .85, True)
            training_labels[train_indices] = i
        #If using Yan labeling model, also train models for each labeler to learn performance
        if self.update_method == 'yan':
            indcs = np.where(training_labels>-1)[0]
            self.labelers.model_start(self.train_map, indcs)
        return training_labels


    def _uncertain_order(self, importance, valid_indices):
        """Returns valid_indices sorted by each index's value in importance"""
        order = valid_indices[np.argsort(importance[valid_indices])]
        order = order[::-1]
        return order


    def show_selected(self):
        """Generates and saves image masking all segments that have already been labeled"""
        lab_train_indcs = np.where(self.training_labels != -1)[0]
        lab_indcs = self.train_map.unique_segs(self.seg)[lab_train_indcs]
        img = self.train_map.mask_segments_by_indx(lab_indcs, self.seg, 1, True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n_labeled = np.where(self.training_labels != -1)[0].shape[0]
        cv2.imwrite('{}selected_{}{}.png'.format(self.path, n_labeled,\
                                                 self.postfix), img)


    def rf_uncertainty(self):
        """
        Selcts [self.batch_size] new segments to label based on their distance from [self.thresh]
        First trains classifier on all labeled data, then predicts probability of all other segments 
        being damage and chooses segments closest to [self.thresh]

        Returns
        -------
        ndarray
            All indices of unlabeled data sorted in decreasing uncertainty
        """
        model = ObjectClassifier(NZ = 0, verbose = 0)
        #train and predict segs of classifier
        training_sample = model.sample(self.training_labels, EVEN = 2)
        model.fit(self.train_map, self.training_labels, training_sample)
        proba_segs = model.predict_proba_segs(self.train_map)
        #If show, save figures of heatmap from prediction
        if self.show:
            self.show_selected()
            fig = plt.figure()
            n_labeled = np.where(self.training_labels != -1)[0].shape[0]
            img = self.train_map.seg_convert(self.seg, proba_segs)
            plt.imshow(img, cmap = 'seismic', norm = plt.Normalize(0,1))
            fig.savefig('{}test_{}{}.png'.format(self.path, n_labeled, self.postfix), format='png')
            plt.close(fig)
        #choose indices whose predictions minus thresh were closest to zero
        unknown_indcs = np.where(self.training_labels == -1)[0]
        uncertainties = 1-np.abs(proba_segs-self.thresh)
        return self._uncertain_order(uncertainties.ravel(), unknown_indcs)


    def random_uncertainty(self):
        """Randomly selects [self.batch_size] new segments to be labeled"""
        if self.show:
            self.show_selected()
        return np.random.permutation(np.where(self.training_labels == -1)[0])


    def update_labels(self, new_training):
        """
        Assigns labels to the segment indices listed in new_training based on the update model
    
        Parameters
        ----------
        new_training : ndarray
            Array listing indices of segments that need labels. Note that these indices are 
            in terms of the the training indices. So if the smallest training segment index 
            is 33000, then index 0 in new_training will mean segment index 33000. In order
            to get indices in terms of all indices, need to use:
            self.train_map.unique_segs(self.seg)[new_training]
        """
        train_segs = self.train_map.unique_segs(self.seg)
        if self.update_method == "donmez":
            #Based on the donmez algorithm
            new_labs = self.labelers.donmez_vote(train_segs[new_training], .85, True)
            self.UIs.append(self.labelers.UI())
            np.save('{}UIs{}.npy'.format(self.path, self.postfix), np.array(self.UIs))
        elif self.update_method == "donmez_1":
            #Variant of the donmez algorithm in which the algorithm may only sample one labeler each time
            new_labs = self.labelers.donmez_pick_1(train_segs[new_training])
            self.UIs.append(self.labelers.UI())
            np.save('{}UIs{}.npy'.format(self.path, self.postfix), np.array(self.UIs))
        elif self.update_method == "majority":
            #Uses majority vote
            new_labs = self.labelers.majority_vote(train_segs[new_training])
        elif self.update_method == "random":
            #Randomly selects a labeler for each data point
            labelers = np.random.randint(0, len(self.labelers.labels), len(new_training))
            new_labs = self.labelers.labels[labelers, train_segs[new_training]]
        elif self.update_method == "email":
            #Uses labels that unique_email chose
            new_labs = self.labelers.labeler(self.unique_email)[train_segs[new_training]]
        elif self.update_method == "yan":
            #Uses the yan algorithm to pick new labels
            new_labs = self.labelers.model_vote(new_training)
        elif self.update_method == 'xie':
            #Uses the xie algorithm to pick new labels
            indcs = np.concatenate((np.where(self.training_labels>-1)[0], new_training))
            em = EM(self.train_map, self.labelers, train_segs[indcs])
            em.run()
            new_labs = (em.G[:,1]>0.5)[-self.batch_size:]
            print zip(new_labs, self.labelers.majority_vote(train_segs[new_training]))
        self.training_labels[new_training] = new_labs


    def test_progress(self):
        """Evaluates progress of active learning run by looking at results tested on testing map"""
        model = ObjectClassifier(NZ = 0, verbose = 0)
        #Pulls all training data thats been labeled and samples evenly between the classes
        training_sample = model.sample(self.training_labels, EVEN = 2)
        #Trains on training data and tests on test map
        model.fit(self.train_map, self.training_labels, training_sample)
        proba = model.predict_proba(self.test_map)
        #Uses majority vote as ground truth
        g_truth = self.labelers.majority_vote()[self.test_map.segmentations[self.seg]]
        n_labeled = np.where(self.training_labels > -1)[0].shape[0]
        #If show is true, saves the ROC curve
        if self.show:
            fig, AUC = analyze_results.ROC(g_truth.ravel(), proba.ravel(), 'Haiti Test')[:2]
            fig.savefig('{}ROC_{}{}.png'.format(self.path, n_labeled, self.postfix), format='png')
            plt.close(fig)
        #Evaluates progress by finding FPR at self.FNR and adding it to the fprs list
        FPR, thresh = analyze_results.FPR_from_FNR(g_truth.ravel(), proba.ravel(), TPR = self.TPR)
        self.fprs.append(FPR)
        #saves all fprs to document every iteration so results are not loss and progress can be seen mid-run
        np.save('{}fprs{}.npy'.format(self.path, self.postfix), self.fprs)


    def update(self):
        """Performs one active learning run by choosing uncertain data, assigning labels, and testing progress"""
        new_training = self.uncertainty()[:self.batch_size]
        self.update_labels(new_training)
        self.test_progress()


    def run(self):
        """Performs full active learning run by calling update repeatedly"""
        for i in range(self.updates):
            print 'Iteration {}'.format(i)
            self.update()



def run_al(i, update, random):
    assert(random == 'random' or random == 'rf')
    next_al = al(postfix = '_{}_{}_{}'.format(update, random, i), random = random == "random", update_method = update)
    next_al.run()

if __name__ == '__main__':
    #options = [('majority', 'random'), ('random', 'random'), ('majority', 'rf'), ('model', 'rf'), ('donmez', 'rf'), ('random', 'rf')]
    options = [('yan', 'rf'), ('donmez', 'rf'), ('majority', 'rf'), ('random', 'rf'), ('xie', 'random'), ('donmez', 'rf'), ('majority', 'rf'), ('majority', 'random')]
    option = options[int(sys.argv[2])]
    run_al(sys.argv[1], option[0], option[1])
    
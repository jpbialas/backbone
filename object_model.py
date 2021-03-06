import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results
import sklearn
from convenience_tools import *
from sklearn.externals.joblib import Parallel, delayed
import time



class ObjectClassifier(object):
    """
    Random Forest based classifier for Px_Map Objects

    Parameters
    ----------
    NZ : boolean
        True if imagery to be processed is from new zeland imagery. False if from Haiti
        This sets constants associated with these two datasets
    verbose : int
        1 causes code to print updates as the run progresses

    Fields
    ------
    verbose : int
        1 causes code to print updates as the run progresses
    n_trees : int
        Number of trees to use in the random forest classifier
    base_seg : int
        Px_Map segmentation level used for classification
    segs : int
        Px_Map segmentation levels used for contextual features
    even : int
        0 : training data will not be sampled with balanced classes
        1 : training data will undersample the majority class to balance the classes
        2 : training data will oversample the minority class to balance the classes
    features : Px_Map -> int -> int lit -> ndarray
        Takes Px_Map object, base_seg, and contextual segs and produces ndarray containing features
    feat_names : String list
        List containing names of all features used for classification
    model : sklearn.ensemble.forest.RandomForestClassifier
        RF model trained on training map
    """

    def __init__(self, NZ = True, verbose = 1):
        self.verbose   = verbose
        self.NZ_constants() if NZ else self.haiti_constants()


    def NZ_constants(self):
        """Sets field constants for NZ imagery"""
        self.n_trees   = 85 
        self.base_seg  = 50
        self.segs      = [100]
        self.even      = 1
        self.features  = sf.NZ_features 

    def haiti_constants(self):
        """Sets field constants for Haiti imagery"""
        self.n_trees   = 200
        self.base_seg  = 20 
        self.segs      = [50]
        self.even      = 2
        self.features  = sf.haiti_features 


    def sample(self, y, EVEN, n_samples = -1):
        """


        """
        if EVEN>0:
            zeros = np.where(y==0)[0]
            ones = np.where(y==1)[0]
            n0,n1 = zeros.shape[0], ones.shape[0]
            if n0 == 0 or n1 == 0:
                return np.arange(y.shape[0])
            replace = True if EVEN == 2 else False
            n = max(n0, n1) if EVEN == 2 else min(n0, n1)
            if n_samples == -1:
                zero_samples = np.random.choice(zeros, n, replace = replace)
                one_samples = np.random.choice(ones, n, replace = replace)
            else:
                zero_samples = np.random.choice(zeros, n_samples//2, replace = replace)
                one_samples = np.random.choice(ones, n_samples//2, replace = replace)
            return np.concatenate((zero_samples, one_samples))
        else:
            return np.arange(y.shape[0])

    def get_X(self, next_map):
        if next_map.X is None:
            X, feat_names = self.features(next_map, self.base_seg, self.segs)
            X = X[next_map.unique_segs(self.base_seg)]
            next_map.X = X
            self.feat_names = feat_names
        else:
            X = next_map.X
            self.feat_names = None
        return X

    def reset_model(self):
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)

    def fit(self, map_train, labels, indcs = None):
        if indcs is None:
            indcs = np.arange(labels.shape[0])
        v_print('starting fit', self.verbose)
        X = self.get_X(map_train)[indcs]
        y = labels[indcs]
        samples = self.sample(y, self.even)
        self.model= RandomForestClassifier(n_estimators=self.n_trees, n_jobs = -1, verbose = self.verbose)
        self.model.fit(X[samples], y[samples])
        v_print('ending fit', self.verbose)
    

    def predict_proba(self, map_test):
        seg_probs = self.predict_proba_segs(map_test, None)
        px_probs = map_test.seg_convert(self.base_seg, seg_probs)
        return px_probs

    def predict_proba_segs(self, map_test, indcs = None):
        if indcs is None:
            indcs = np.arange(map_test.unique_segs(self.base_seg).shape[0])
        v_print('starting predict', self.verbose)
        X = self.get_X(map_test)[indcs]
        try:
            segment_probs = self.model.predict_proba(X)[:,1]
        except IndexError:
            segment_probs = self.model.predict_proba(X).ravel()
        v_print('ending predict', self.verbose)
        return segment_probs
    
    def predict(self, map_test, thresh = .5):
        return self.predict_proba(map_test)>thresh

    def fit_and_predict(self, map_train, map_test, labels, indcs = None):
        self.fit(map_train, labels, indcs)
        return self.predict_proba(map_test)

    def feature_importance(self):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1].astype('int')
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(len(self.feat_names)):
            #print("{}. feature {}: ({})".format(f + 1, self.feat_names[indices[f]], importances[indices[f]]))
            g = f + 26
            h = g + 26
            print("{} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(f + 1, self.feat_names[indices[f]], importances[indices[f]], g + 1, self.feat_names[indices[g]], importances[indices[g]], h + 1, self.feat_names[indices[h]], importances[indices[h]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.feat_names)), importances[indices], color="r", yerr=std[indices], align="center")
        print(range(len(self.feat_names)))
        print(indices.dtype)
        plt.xticks(range(len(self.feat_names)), np.array(self.feat_names)[indices])
        plt.xlim([-1, len(self.feat_names)])
        

def main_NZ():
    model = ObjectClassifier()
    map_2, map_3 = map_overlay.basic_setup()
    labels = np.zeros(map_2.unique_segs(50).shape[0])
    labels[np.loadtxt('damagelabels50/Jared-3-2.csv', delimiter = ',').astype('int')] = 1
    probs = model.fit_and_predict(map_2, map_3, labels)
    np.save('object_probs.npy', probs)

def main_haiti():
    from Xie import EM
    from labeler import Labelers
    model      = ObjectClassifier(0,1)
    labelers   = Labelers()
    y          = labelers.majority_vote()
    train       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    test      = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
    haiti_map  = map_overlay.haiti_setup()
    train_map  = haiti_map.sub_map(train)
    test_map   = haiti_map.sub_map(test)
    #em = EM(train_map, labelers)
    #em.run()
    #y2 = em.G[:,1]>0.5
    g_truth    = y[test_map.segmentations[20]]
    FPRs = []
    TPRs = []
    for email in labelers.emails:
        print email
        a = time.time()
        labels = labelers.labeler(email)[test_map.segmentations[20]]
        b = time.time()
        FPR, TPR = analyze_results.confusion_analytics(g_truth.ravel(), labels.ravel())
        c = time.time()
        FPRs.append(FPR)
        TPRs.append(TPR)

    probs = model.fit_and_predict(train_map, test_map, y[train_map.unique_segs(20)])
    print analyze_results.FPR_from_FNR(g_truth.ravel(), probs.ravel(), TPR = .95)
    analyze_results.probability_heat_map(test_map, probs.ravel(), '')
    fig, _, _, _, _, _ = analyze_results.ROC(g_truth.ravel(), probs.ravel(), 'Classifier')
    plt.scatter(FPRs, TPRs)
    names = labelers.emails
    for i in range(len(FPRs)):
        plt.annotate(names[i], (FPRs[i], TPRs[i]))

    fig.savefig('All_ROCs/{}_ROC.png'.format('Classifier'), format='png')
    plt.show()

def label_test():
    from labeler import Labelers
    model      = ObjectClassifier(NZ = False)
    labelers   = Labelers()
    test       = np.ix_(np.arange(4096/3, 4096), np.arange(4096/2))
    train      = np.ix_( np.arange(4096/3, 4096), np.arange(4096/2, 4096))
    haiti_map  = map_overlay.haiti_setup()
    train_map  = haiti_map.sub_map(train)
    test_map   = haiti_map.sub_map(test)
    predictions = np.zeros((labelers.labels.shape[0], test_map.unique_segs(20).shape[0]))
    agreement = (labelers.labels == labelers.majority_vote())[:,train_map.unique_segs(20)]
    for i in range(labelers.labels.shape[0]):
        print labelers.emails[i]
        new_model = ObjectClassifier(NZ = False)
        new_model.fit(train_map, agreement[i])
        probs = new_model.predict_proba_segs(test_map)
        predictions[i] = probs
        print predictions
    best_labelers = np.argmax(predictions, axis = 0)
    print best_labelers
    np.save('predictions.npy', predictions)
    np.save('best.npy',best_labelers)
    assert(best_labelers.shape[0] == test_map.unique_segs(20).shape[0])
    model_labels = labelers.labels[best_labelers,test_map.unique_segs(20)]
    np.save('vote.npy', model_labels)





if __name__ == '__main__':
    main_haiti()
    #label_test()



from object_model import ObjectClassifier
from px_model import PxClassifier
import analyze_results
import map_overlay
from map_overlay import MapOverlay
import matplotlib.pyplot as plt
import timeit
import numpy as np
import sklearn
from convenience_tools import *


def bootstrap(L, k):
    ''' Returns k random samples of L of size |L| with replacement
    INPUT:
        L: (ndarray) 
        k: (int) number of backbone iterations to run
    '''
    return np.random.choice(L, (k, L.shape[0]), replace = True)


def test_progress(map_train, map_test, X_train, training_labels, test_truth, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = np.where(training_labels != -1)
    model.fit(map_train, custom_labels = training_labels[training_sample], custom_data = X_train[training_sample])
    prediction = model.predict_proba(map_test)
    test_segs = map_test.segmentations[50][1].ravel().astype('int')
    if show:
        fig = plt.figure()
        plt.imshow(prediction, cmap = 'seismic', norm = plt.Normalize(0,1))
        fig.savefig('al/pred_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
        fig, AUC, _, _, _, _ = analyze_results.ROC(map_test,test_truth[test_segs], prediction.ravel(), "", save = False)
        fig.savefig('al/ROC_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
    else:
        return sklearn.metrics.roc_auc_score(test_truth[test_segs], prediction.ravel())
    return AUC   


def uncertainty(data, y, k, m, frac_test = 1.0, verbose = True):
    '''Computes uncertainty for all unlabelled points as described in Mozafari 3.2 
    and returnst the m indices of the most uncertain
    NOTE: Smaller values are more certain b/c they have smaller variance
    Input:
        X: (ndarray) Contains all data
        y: (ndarray) Contains labels, 1 for damage, 0 for not damaged, -1 for unlabelled
        k: (int) number of backbone iterations to run
        m: (int) number of new data to select for labelling
    '''
    U = np.where(y < 0)[0]
    U = np.random.choice(U, frac_test*U.shape[0], replace = False)
    data_U =  np.take(data, U, axis = 0)
    L = np.where(y >= 0)[0]
    samples = bootstrap(L, k)
    X_U = np.zeros(U.shape[0])
    v_print("Staring model loop", verbose)
    pbar = custom_progress()
    for row in pbar(samples):
        model = ObjectClassifier(verbose = 0)
        training_sample = model.sample(training_labels, EVEN = 2)
        test = training_labels[training_sample]
        model.fit(next_map, custom_labels = training_labels[training_sample], custom_data = X[training_sample])
        prediction = model.predict_proba_segs(next_map)
        X_U += next_prediction
    X_U/=k
    uncertainties=X_U*(1-X_U)
    return U[np.argsort(uncertainties)[-m:]]
    

def rf_uncertainty(next_map, X, training_labels, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = np.where(training_labels != -1)
    test = training_labels[training_sample]
    model.fit(next_map, custom_labels = training_labels[training_sample], custom_data = X[training_sample])
    prediction = model.predict_proba_segs(next_map)
    if show:
        fig = plt.figure()
        img = next_map.mask_segments_by_indx(np.where(training_labels != -1)[0], 50, with_img = True)
        plt.imshow(img)
        fig.savefig('al/selected_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(prediction[next_map.segmentations[50][1].astype('int')], cmap = 'seismic', norm = plt.Normalize(0,1))
        fig.savefig('al/test_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
    unknown_indcs = np.where(training_labels == -1)[0]
    uncertainties = np.abs(prediction-.5)
    return unknown_indcs[np.argsort(uncertainties[unknown_indcs])]


def indcs2bools(indcs, segs):
    nsegs = np.max(segs)+1
    seg_mask = np.zeros(nsegs)
    seg_mask[indcs] = 1
    return seg_mask[segs]

def main(start_n=100, step_n=100, n_updates = 100, verbose = 1, show = False):
    '''
    Runs active learning on train, and tests on the other map. Starts with start_n labels, and increments by step_n size batches.
    If method is UNCERT, picks new indices with bootstrap Uncertainty, with a bootstrap size of boot_n.
    '''
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    print ('done setting up')
    segs_2 = map_2.segmentations[50][1].ravel().astype('int')
    segs_3 = map_3.segmentations[50][1].ravel().astype('int')
    model = ObjectClassifier(verbose = 0)
    X_2, train_truth = model._get_X_y(map_2, "Jared")
    X_3, test_truth = model._get_X_y(map_3, "Jared")
    training_labels = np.ones_like(train_truth)*-1
    rocs = []
    #set initial values
    print ('setting values')
    for i in range(2):
        training_labels[np.random.choice(np.where(train_truth==i)[0], start_n//2, replace = False)] = i
    #Test initial performance
    print('about to test progress for first time')
    next_roc = test_progress(map_2, map_3, X_2, training_labels, test_truth, show)
    rocs.append(next_roc)
    pbar = custom_progress()
    for i in pbar(range(n_updates)):
        most_uncertain = rf_uncertainty(map_2, X_2, training_labels, show)
        new_training = most_uncertain[:step_n]
        #The following step simulates the expert giving the new labels
        training_labels[new_training] = train_truth[new_training]
        #Test predictive performance on other map
        next_roc = test_progress(map_2, map_3, X_2, training_labels, test_truth, show)
        rocs.append(next_roc)
        np.savetxt('al/rocs.csv', rocs, delimiter = ',')




if __name__ == '__main__':
    main()
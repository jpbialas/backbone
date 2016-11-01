import matplotlib
matplotlib.use('Agg')
from object_model import ObjectClassifier
from px_model import PxClassifier
import analyze_results
import map_overlay
from map_overlay import MapOverlay
import matplotlib.pyplot as plt
import timeit
import numpy as np
import sklearn
from sklearn.externals.joblib import Parallel, delayed
from convenience_tools import *
import cv2


def bootstrap(L, k):
    ''' Returns k random samples of L of size |L| with replacement
    INPUT:
        L: (ndarray) 
        k: (int) number of backbone iterations to run
    '''
    return np.random.choice(L, (k, L.shape[0]), replace = True)


def test_progress(map_train, map_test, X_train, training_labels, test_truth, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = model.sample(training_labels, EVEN = 2)
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

def test_haiti_progress(haiti_map, X, y, training_labels, train_segs, test_segs, thresh, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = model.sample(training_labels, EVEN = 2)
    print ('\n')
    print ("Training size is: {}".format(training_sample.shape[0]))
    model.fit(haiti_map, custom_data = X[train_segs][training_sample], custom_labels=y[train_segs][training_sample])
    proba = model.predict_proba(haiti_map, custom_data = X, custom_labels=y)
    print haiti_map.masks['damage'].shape
    g_truth = haiti_map.masks['damage'].reshape(4096,4096)
    if show:
        fig, AUC, _,_,_,_ = analyze_results.ROC(haiti_map, g_truth[:,2048:].ravel(), proba[:,2048:].ravel(), 'Haiti Left')
        fig.savefig('al/ROC_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
    Prec, FPR = analyze_results.FPR_from_FNR(g_truth[:,2048:].ravel(), proba[:,2048:].ravel(), TPR = thresh)
    #AUC = sklearn.metrics.roc_auc_score(g_truth[:,2048:].ravel(), proba[:,2048:].ravel())
    return Prec, FPR


def uncertain_order(importance, valid_indices, decreasing = True):
    '''
    Returns valid indices sorted by each index's value in importance.
    '''
    order = valid_indices[np.argsort(importance[valid_indices])]
    if decreasing:
        order = order[::-1]
    return order

def LCB_helper(boot_sample, training_labels, X, i):
    obj_model = ObjectClassifier(verbose = 0)
    obj_model.reset_model()
    model = obj_model.model
    smpl = boot_sample[i]
    model.fit(X[smpl], training_labels[smpl])
    return model.predict_proba(X)[:,1].ravel()

def LCB(next_map, X, training_labels, k, show):
    '''
    Implementation of algorithm found in: http://www.jmlr.org/proceedings/papers/v16/chen11a/chen11a.pdf
    '''
    training_sample = np.where(training_labels != -1)[0]
    boot_sample = bootstrap(training_sample, k)
    #all_predictions = np.zeros_like(boot_sample)
    
    all_predictions = np.array(Parallel(n_jobs=-1)(delayed(LCB_helper)(boot_sample, training_labels, X, i) for i in range(k)))
    CPP = np.sum(all_predictions, axis = 0)/all_predictions.shape[0]
    pp = float(np.where(training_labels == 1)[0].shape[0])/training_sample.shape[0]
    if pp > 0.5:
        pp = 1-pp

    if show:
        fig = plt.figure()
        img = next_map.mask_segments_by_indx(np.where(training_labels != -1)[0], 50, with_img = True)
        plt.imshow(img)
        fig.savefig('al/selected_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
        fig = plt.figure()
        plt.imshow(CPP[next_map.segmentations[50][1].astype('int')], cmap = 'seismic', norm = plt.Normalize(0,1))
        fig.savefig('al/test_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)

    Pmax = np.mean([0.5, pp])
    Q = np.zeros_like(CPP)
    Q[CPP<Pmax] = CPP[CPP<Pmax]/Pmax
    Q[CPP>=Pmax] = (1 - CPP[CPP>=Pmax])/float(1-Pmax)
    unknown_indcs = np.where(training_labels == -1)[0]
    return uncertain_order(Q, unknown_indcs, decreasing=True)
    

def rf_uncertainty(next_map, X, training_labels, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = model.sample(training_labels, EVEN = 2)
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
    uncertainties = 1-np.abs(prediction-.5)
    # This returns only the indices whose values are -1 with most uncertain first
    return uncertain_order(uncertainties, unknown_indcs, decreasing=True)


def rf_uncertainty_haiti(haiti_map, X, y, training_labels, train_segs, show):
    model = ObjectClassifier(verbose = 0)
    training_sample = model.sample(training_labels, EVEN = 2)
    print ('\n')
    print ("Training size is: {}".format(training_sample.shape[0]))
    model.fit(haiti_map, custom_data = X[train_segs][training_sample], custom_labels=y[train_segs][training_sample])
    prediction = model.predict_proba_segs(haiti_map, custom_data = X, custom_labels=y)
    if show:
        lab_train_indcs = np.where(training_labels != -1)[0]
        print training_labels.shape, lab_train_indcs.shape, np.max(lab_train_indcs)
        print train_segs.shape, np.max(train_segs)
        lab_indcs = train_segs[lab_train_indcs]
        print lab_indcs.shape, np.max(lab_indcs)
        img = cv2.cvtColor(haiti_map.mask_segments_by_indx(lab_indcs, 20, opacity = 1, with_img = True)[:,:2048,:], cv2.COLOR_BGR2RGB)
        cv2.imwrite('al/selected_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), img)
        fig = plt.figure()
        plt.imshow(prediction[haiti_map.segmentations[20][1].astype('int')[:,:2048]], cmap = 'seismic', norm = plt.Normalize(0,1))
        fig.savefig('al/test_{}.png'.format(np.where(training_labels != -1)[0].shape[0]), format='png')
        plt.close(fig)
    unknown_indcs = np.where(training_labels == -1)[0]
    uncertainties = 1-np.abs(prediction[train_segs]-.5)
    # This returns only the indices whose values are -1 with most uncertain first
    return uncertain_order(uncertainties.ravel(), unknown_indcs, decreasing=True)



def random_uncertainty(training_labels, show):
    return np.random.permutation(np.where(training_labels == -1)[0])

def indcs2bools(indcs, segs):
    nsegs = np.max(segs)+1
    seg_mask = np.zeros(nsegs)
    seg_mask[indcs] = 1
    return seg_mask[segs]

def main_haiti(run_num, start_n = 50, step_n=50, n_updates = 2000, verbose = 1, show = False):
    if run_num >= 8:
        u_name = 'rf'
    else:
        u_name = 'random'

    print ("Setting up {}".format(run_num))
    haiti_map = map_overlay.haiti_setup()
    segs = haiti_map.segmentations[20][1].ravel().astype('int')
    left = np.unique(segs.reshape(haiti_map.shape2d)[:,:2048]).astype(int)
    right = np.unique(segs.reshape(haiti_map.shape2d)[:,2048:]).astype(int)
    #top = np.unique(segs.reshape(haiti_map.shape2d)[:2048,:]).astype(int)
    #bottom = np.unique(segs.reshape(haiti_map.shape2d)[2048:,:]).astype(int)
    train_segs, test_segs = left, right
    
    print ('done setting up {}'.format(run_num))
    model = ObjectClassifier(verbose = 0)
    X, y = model._get_X_y(haiti_map, 'damage', custom_fn = 'damagelabels20/Jared.csv')
    X_train, y_train = X[train_segs], y[train_segs]
    X_test, y_test = X[test_segs], y[test_segs]

    training_labels = np.ones_like(y_train)*-1
    fprs90 = []
    precs90 = []
    fprs95 = []
    precs95 = []
    #set initial values
    print ('setting values')
    np.random.seed()
    for i in range(2):
        training_labels[np.random.choice(np.where(y_train==i)[0], start_n//2, replace = False)] = i
    #Test initial performance
    print np.where(training_labels==1)[0], np.where(training_labels==0)[0]
    print('about to test progress for first time')
    next_prec90, next_fpr90 = test_haiti_progress(haiti_map, X, y, training_labels, train_segs, test_segs, .90, show)
    next_prec95, next_fpr95 = test_haiti_progress(haiti_map, X, y, training_labels, train_segs, test_segs, .95, show)
    precs90.append(next_prec90)
    fprs90.append(next_fpr90)
    precs95.append(next_prec95)
    fprs95.append(next_fpr95)
    pbar = custom_progress()
    for i in range(n_updates):
        if u_name == 'rf':
            most_uncertain = rf_uncertainty_haiti(haiti_map, X, y, training_labels, train_segs, show)
        else:
            most_uncertain = random_uncertainty(training_labels, show)
        new_training = most_uncertain[:step_n]
        #The following step simulates the expert giving the new labels
        training_labels[new_training] = y_train[new_training]
        #Test predictive performance on other map
        next_prec90, next_fpr90 = test_haiti_progress(haiti_map, X, y, training_labels, train_segs, test_segs, .90, show)
        next_prec95, next_fpr95 = test_haiti_progress(haiti_map, X, y, training_labels, train_segs, test_segs, .95, show)
        precs90.append(next_prec90)
        fprs90.append(next_fpr90)
        precs95.append(next_prec95)
        fprs95.append(next_fpr95)
        np.savetxt('al/rocs_{}_0.90_prec_{}.csv'.format(u_name, run_num%8), precs90, delimiter = ',')
        np.savetxt('al/rocs_{}_0.90_fpr_{}.csv'.format(u_name, run_num%8), fprs90, delimiter = ',')
        np.savetxt('al/rocs_{}_0.95_prec_{}.csv'.format(u_name, run_num%8), precs95, delimiter = ',')
        np.savetxt('al/rocs_{}_0.95_fpr_{}.csv'.format(u_name, run_num%8), fprs95, delimiter = ',')
    return np.array(rocs)


def main(run_num, start_n=50, step_n=50, n_updates = 200, verbose = 1, show = True):
    '''
    Runs active learning on train, and tests on the other map. Starts with start_n labels, and increments by step_n size batches.
    If method is UNCERT, picks new indices with bootstrap Uncertainty, with a bootstrap size of boot_n.
    '''
    print run_num
    map_2, map_3 = map_overlay.basic_setup([100], 50, label_name = "Jared")
    map_train = map_2
    map_test = map_3
    
    print ('done setting up')
    segs_train = map_train.segmentations[50][1].ravel().astype('int')
    segs_test = map_test.segmentations[50][1].ravel().astype('int')
    model = ObjectClassifier(verbose = 0)
    X_train, train_truth = model._get_X_y(map_train, "Jared")
    X_test, test_truth = model._get_X_y(map_test, "Jared")
    training_labels = np.ones_like(train_truth)*-1
    rocs = []
    #set initial values
    print ('setting values')
    for i in range(2):
        training_labels[np.random.choice(np.where(train_truth==i)[0], start_n//2, replace = False)] = i
    #Test initial performance
    print('about to test progress for first time')
    next_roc = test_progress(map_train, map_test, X_train, training_labels, test_truth, show)
    print next_roc
    rocs.append(next_roc)
    pbar = custom_progress()
    for i in pbar(range(n_updates)):
        #most_uncertain = LCB(map_train, X_train, training_labels, 10, show)
        #most_uncertain = rf_uncertainty(map_train, X_train, training_labels, show)
        most_uncertain = random_uncertainty(training_labels, show)
        new_training = most_uncertain[:step_n]
        #The following step simulates the expert giving the new labels
        training_labels[new_training] = train_truth[new_training]
        #Test predictive performance on other map
        next_roc = test_progress(map_train, map_test, X_train, training_labels, test_truth, show)
        print next_roc
        rocs.append(next_roc)
        np.savetxt('al_3-2_3-3/rocs_random_{}.csv'.format(run_num), rocs, delimiter = ',')
    return np.array(rocs)



if __name__ == '__main__':
    #main_haiti(0)
    #main(0)
    Parallel(n_jobs=16)(delayed(main_haiti)(i) for i in range(16))

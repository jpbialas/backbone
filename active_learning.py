import numpy as np
import matplotlib.pyplot as plt
import px_features
import analyze_results
import map_overlay
from convenience_tools import *
import cv2
import os
import px_classify
import seg_classify as sc
from sklearn.ensemble import RandomForestClassifier


def balance(labels, method = 0):
    '''
    Balances the given data by either undersampling the majority class or oversampling the minority class.
    The resampling method is specified in method
        - 0 = Undersample majority class
        - 1 = Oversample minority class

    '''
    split_labels = [np.where(labels==0)[0], np.where(labels==1)[0]]
    split_counts = [len(split_labels[0]), len(split_labels[1])]
    maj_class = np.argmax(split_counts)
    min_class = int(not maj_class)
    if method == 0:
        new_maj = np.random.choice(split_labels[maj_class], split_counts[min_class], replace = False)
        new_min = split_labels[min_class]
    else:
        new_maj = split_labels[maj_class]
        cls_diff = split_counts[maj_class] - split_counts[min_class]
        new_min_addition = np.random.choice(split_labels[min_class], cls_diff, replace = False)
        new_min = np.concatenate((split_labels[min_class], new_min_addition))
    return np.concatenate((new_min, new_maj))


def bootstrap(L, k):
    ''' Returns k random samples of L of size |L| with replacement
    INPUT:
        L: (ndarray) 
        k: (int) number of backbone iterations to run
    '''
    return np.random.choice(L, (k, L.shape[0]), replace = True)

def strawman_error(X, y, m):
    unlabelled = np.where(y < 0)[0]
    return np.random.choice(unlabelled, m, replace = False)


def uncertainty(data, y, k, m, frac_test = 1.0, verbose = True):
    ##NOTE: HAVE NOT YET IMPLEMENTED BALANCED HERE

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
        next_model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
        next_model.fit(data[row], y[row])
        next_prediction = next_model.predict(data_U)
        X_U += next_prediction
    X_U/=k
    uncertainties=X_U*(1-X_U)
    return U[np.argsort(uncertainties)[-m:]]

def trivial_error(X, y, m, frac_test = 1.0, verbose = True):
    '''Computes uncertainty from Random Forest's probability metric and returns the m indices
    of the most uncertain
    NOTE: This metric is the distance from 50% So larger values are more certain
    Input:
        X: (ndarray) Contains all data
        y: (ndarray) Contains labels, 1 for damage, 0 for not damaged, -1 for unlabelled
        m: (int) number of new data to select for labelling
    '''
    unlabelled = np.where(y < 0)[0]
    unlabelled = np.random.choice(unlabelled, frac_test*unlabelled.shape[0], replace = False)
    L = np.where(y >= 0)[0]
    model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
    model.fit(X[L], y[L])
    predictions = model.predict_proba(X[unlabelled])
    if predictions.shape[1]>1:
        uncertainties = predictions[:,1]
        uncertainties = np.abs(uncertainties-.5)/.5
        return unlabelled[np.argsort(uncertainties)[:m]]
    else:
        return unlabelled[:m]


def test_seg_progress(map_test, base_seg, X_train, X_test, y_train, indices, FPRs, FNRs, F1s, Accs, Confs, name, verbose = True):
    '''
    Trains X_train on y_train and tests on X_test. Adds resulting FPR, FNR, and Conf to FPRs, FNRs, and Confs
    '''
    model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
    model.fit(X_train[indices], y_train[indices])
    prediction = model.predict_proba(X_test)[:,1]>0.4
    full_predict = map_test.mask_segments(prediction, base_seg, False)
    map_test.newPxMask(full_predict.ravel(), 'damage_pred')


    cv2.imwrite('al/test_{}_{}.png'.format(name, indices.shape[0]),map_test.mask_segments(prediction.ravel(), 50, with_img = True))

    FPR, FNR, conf = analyze_results.confusion_analytics(map_test.getLabels('damage'), full_predict)
    _,_,Acc,F1 = analyze_results.prec_recall(map_test.getLabels('damage'), full_predict)
    FPRs.append(FPR)
    FNRs.append(FNR)
    Confs.append(conf)
    F1s.append(F1)
    Accs.append(Acc)

def run_active_learning_seg(start_n=100, step_n=100, boot_n = 100,  n_updates = 100, method = "UNCERT", train = 2, verbose = True):
    '''
    Runs active learning on train, and tests on the other map. Starts with start_n labels, and increments by step_n size batches.
    If method is UNCERT, picks new indices with bootstrap Uncertainty, with a bootstrap size of boot_n.
    '''
    base_seg = 50

    if method == "UNCERT":
        legend_name = 'Bootstrap Uncertainty'
    elif method == "Forest":
        legend_name = 'Random Forest Probability'

    if train == 2:
        map_train, X_train, y_train, names = sc.setup_segs(2, base_seg, [100],.5, jared = True)
        map_test, X_test, y_test, _ = sc.setup_segs(3,base_seg, [100],  .5, jared = True)
    else:
        map_train, X_train, y_train, names = sc.setup_segs(3, base_seg, [100],.5, jared = True)
        map_test, X_test, y_test, _ = sc.setup_segs(2,base_seg, [100],  .5, jared = True)

    
    X_axis = [start_n]

    # Initial sample of labelled data
    #sample = np.random.choice(np.arange(y_train.shape[0]), start_n, replace = False)
    sample = sc.sample(y_train, start_n)

    random_sample = sample.copy()

    y = np.ones_like(y_train)*-1
    y[sample] = y_train[sample]
    FPR, FNR, conf, f1, acc = [],[],[], [], []
    FPR_r, FNR_r, conf_r, f1_r, acc_r = [],[],[], [], []
    full_FPR, full_FNR, full_F1, full_Acc= [],[],[], []
    #Initial results from only initial samples
    test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, FPR, FNR, f1, acc, conf, name = method)
    test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, FPR_r, FNR_r, f1_r, acc_r, conf_r, name = 'random')
    #Calculates heuristics for full labelling(Convergence Value)
    test_seg_progress(map_test, base_seg, X_train, X_test, y_train, np.arange(y_train.shape[0]), full_FPR, full_FNR, full_F1, full_Acc, [], name = 'full')
    


    ###########################      MATPLOTLIB STUFF      ################################
    #Turns on interactive mode
    ##plt.ion()

    #Sets up FPR chart
    prec_comparisons = plt.figure('False Positive Rate')
    graph_FPR = plt.plot(X_axis, FPR, 'r-', label = legend_name)[0]
    graph_FPR_r = plt.plot(X_axis, FPR_r, 'r--', label = 'Random Selection')[0]
    plt.axhline(full_FPR[0], color = 'gray', label = 'Full Labelling')
    plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
    plt.legend()
    plt.title('False Positive Rate AL Comparison')
    plt.xlabel('Number of Labelled Samples')
    plt.ylabel('FPR')

    #Sets up FNR chart
    rec_comparisons = plt.figure('False Negative Rate')
    graph_FNR = plt.plot(X_axis, FNR, 'g-',label = legend_name)[0]
    graph_FNR_r = plt.plot(X_axis, FNR_r, 'g--', label = 'Random Selection')[0]
    plt.axhline(full_FNR[0], color = 'gray', label = 'Full Labelling')
    plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
    plt.legend()
    plt.title('False Negative Rate AL Comparison')
    plt.xlabel('Number of Labelled Samples')
    plt.ylabel('FNR')

    f1_comparisons = plt.figure('F1 Rate')
    graph_F1 = plt.plot(X_axis, f1, 'b-',label = legend_name)[0]
    graph_F1_r = plt.plot(X_axis, f1_r, 'b--', label = 'Random Selection')[0]
    plt.axhline(full_F1[0], color = 'gray', label = 'Full Labelling')
    plt.axis([start_n, start_n+step_n*n_updates,0,1])
    plt.legend()
    plt.title('F1 AL Comparison')
    plt.xlabel('Number of Labelled Samples')
    plt.ylabel('F1')

    acc_comparisons = plt.figure('Accuracy Rate')
    graph_Acc = plt.plot(X_axis, acc, 'b-',label = legend_name)[0]
    graph_Acc_r = plt.plot(X_axis, acc_r, 'b--', label = 'Random Selection')[0]
    plt.axhline(full_Acc[0], color = 'gray', label = 'Full Labelling')
    plt.axis([start_n, start_n+step_n*n_updates,0,1])
    plt.legend()
    plt.title('Accuracy AL Comparison')
    plt.xlabel('Number of Labelled Samples')
    plt.ylabel('Accuracy')

    # Draws plots
    ##plt.draw()
    ##plt.pause(0.05)

    ################################### END MATPLOTLIB STUFF #############################



    cv2.imwrite('al/sample_{}.png'.format(start_n),map_train.mask_segments_by_indx(sample, 50, with_img = True))
    cv2.imwrite('al/sample_random_{}.png'.format(start_n),map_train.mask_segments_by_indx(random_sample, 50, with_img = True))
    v_print('starting uncertainty', verbose)
    for i in range(1, n_updates):
        #plt.figure()
        #print ("WHAT IS GOING ON??")
        #plt.imshow(map_train.mask_segments_by_indx(sample, 50, with_img = True))
        #plt.show()

        X_axis.append(start_n+i*step_n)

        print "there are", np.where(y >= 0)[0].shape[0], "labelled points"
        #Uses AL method to find next sample indices
        next_indices = uncertainty(X_train, y, boot_n, step_n)
        #next_indices = trivial_error(X_train, y, step_n)
        sample = np.concatenate((sample, next_indices))
        y[sample] = y_train[sample]
        random_sample = np.concatenate((random_sample, strawman_error(X_train, y, boot_n)))

        print ("percent damage: ", np.sum(y_train[sample])/float(y_train[sample].shape[0]))

        cv2.imwrite('al/sample_{}.png'.format(start_n+i*step_n),map_train.mask_segments_by_indx(sample, 50, with_img = True))
        cv2.imwrite('al/sample_random_{}.png'.format(start_n+i*step_n),map_train.mask_segments_by_indx(random_sample, 50, with_img = True))
        #Update charts
        test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, FPR, FNR, f1, acc, conf, name = method)
        test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, FPR_r, FNR_r, f1_r, acc_r, conf_r, name = 'random')

        #####################    UPDATING MATPLOTLIB CHARTS    ############################
        graph_FPR.set_ydata(FPR)
        graph_FPR.set_xdata(X_axis)
        graph_F1.set_ydata(f1)
        graph_F1.set_xdata(X_axis)
        graph_Acc.set_ydata(acc)
        graph_Acc.set_xdata(X_axis)
        graph_FNR.set_ydata(FNR)
        graph_FNR.set_xdata(X_axis)
        graph_FPR_r.set_ydata(FPR_r)
        graph_FPR_r.set_xdata(X_axis)
        graph_FNR_r.set_ydata(FNR_r)
        graph_FNR_r.set_xdata(X_axis)
        graph_F1_r.set_ydata(f1_r)
        graph_F1_r.set_xdata(X_axis)
        graph_Acc_r.set_ydata(acc_r)
        graph_Acc_r.set_xdata(X_axis)

        ##plt.draw()
        ##plt.pause(0.05)
        ############################ END CHART UPDATEDS  ###########################

        #Save results for each iteration
        prec_comparisons.savefig('temp/FPR_jared {} {}.png'.format(legend_name, train))
        rec_comparisons.savefig('temp/FNR_jared {} {}.png'.format(legend_name, train))
        ##f1_comparisons.savefig('temp/F1_jared {} {}.png'.format(legend_name, train))
        ##acc_comparisons.savefig('temp/Acc_jared {} {}.png'.format(legend_name, train))
        np.save('temp/conf_jared {} {}.npy'.format(legend_name, train), np.array(conf))
        np.save('temp/conr_jared random {}.npy'.format(train),np.array(conf_r))

    ##plt.waitforbuttonpress()

if __name__ == '__main__':
    #map_train, map_test = map_overlay.basic_setup()
    print 'starting active learning'
    #run_active_learning(map_train, map_test, 100000, 100000)
    run_active_learning_seg(train = 3)









import numpy as np
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
import sklearn
import cv2
import matplotlib.pyplot as plt
from convenience_tools import *
import time


def confusion_matrix(y_true, y_pred):
    '''
        Generates confusion_matrix faster than sklearn
    '''
    TPTN = y_true == y_pred
    TP = np.sum(np.logical_and(TPTN, y_true))
    TN = np.sum(TPTN) - TP
    FPFN = 1 - TPTN
    FP = np.sum(np.logical_and(FPFN, y_pred))
    FN = np.sum(FPFN) - FP
    return np.array([[TN, FP],[FN, TP]])


def draw_segment_analysis(my_map, labels):
    pbar = custom_progress()
    data = np.zeros((my_map.rows*my_map.cols), dtype = 'uint8')
    segs = my_map.segmentation.ravel()
    n_segs = int(np.max(segs))
    for i in pbar(range(n_segs)):
        data[np.where(segs == i)] = labels[i]
    img = data.reshape(my_map.rows, my_map.cols)
    plt.imshow(img, cmap = 'gray')
    plt.show()

    return img


def confusion_analytics(y_true, y_pred):
    TPTN = y_true == y_pred
    TP = np.sum(np.logical_and(TPTN, y_true))
    TN = np.sum(TPTN) - TP
    FPFN = 1 - TPTN
    FP = np.sum(np.logical_and(FPFN, y_pred))
    FN = np.sum(FPFN) - FP
    FPR = float(FP)/(FP+TN)
    TPR = float(TP)/(TP+FN)
    return FPR, TPR



def prec_recall(y_true, y_pred):
    '''
    INPUT:
        - confusion_matrix: (2x2 ndarray) Confusion matrix of shape: (actual values) x (predicted values)
    OUTPUT:
        - (tuple) A Tuple containing the precision and recall of the confusion matrix

    '''
    conf = confusion_matrix(y_true, y_pred)
    TP = conf[1,1]
    FP = conf[0,1]
    TN = conf[0,0]
    FN = conf[1,0]
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    FPR = FP/(FP+TN)
    FNR = 1-recall
    return round(precision, 5), round(recall, 5), round(accuracy, 5), round(f1, 5)

def compare_heatmaps(pred1, pred2):
    fig = plt.figure('Difference')

    fig.subplots_adjust(bottom=0.05, left = 0.02, right = 0.98, top = 1, wspace = 0.02, hspace = 0)
    plt.subplot2grid((2,2),(0,0)),plt.imshow(pred2, cmap = 'seismic', norm = plt.Normalize(0,1))
    plt.title('Bad Labelling'), plt.xticks([]), plt.yticks([])

    plt.subplot2grid((2,2),(0,1)),plt.imshow(pred1, cmap = 'seismic', norm = plt.Normalize(0,1))
    plt.title('Good Labelling'), plt.xticks([]), plt.yticks([])

    diff = pred1-pred2
    plt.subplot2grid((2,2),(1,0), colspan = 2),plt.imshow(diff, cmap = 'seismic', norm = plt.Normalize(-1,1))
    plt.title('Difference'), plt.xticks([]), plt.yticks([])
    

def side_by_side(myMap, mask_true, mask_predict, name, save = False):
    fig = plt.figure('SBS_{}'.format(name))
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.subplot(121),plt.imshow(myMap.maskImg(mask_true))
    plt.title('Labelled Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(myMap.maskImg(mask_predict))
    plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
    if save:
        fig.savefig('Compare Methods/{}_sbs.png'.format(name), format='png', dpi = 2400)
    return fig

def probability_heat_map(map_test, full_predict, name, save = False):
    fig = plt.figure('HeatMap_{}'.format(name))
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    ground_truth = map_test.getLabels('damage')
    plt.contour(ground_truth.reshape(map_test.rows, map_test.cols), colors = 'green')
    plt.imshow(full_predict.reshape(map_test.rows, map_test.cols), cmap = 'seismic')
    plt.title('Damage Prediction'), plt.xticks([]), plt.yticks([])
    if save:
        fig.savefig('Compare Methods/{}_heatmap.png'.format(name), format='png', dpi = 2400)
    return fig

def contour_map(img, full_predict, title = 'Contour Map'):
    fig = plt.figure(title)
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.contour(full_predict, colors = 'green')
    plt.imshow(img)
    plt.title(title), plt.xticks([]), plt.yticks([])


def FPR_from_FNR(ground_truth, full_predict, TPR = .95, prec = False):
    FPRs, TPRs, threshs = roc_curve(ground_truth, full_predict.ravel())
    min_i = 0
    max_i = TPRs.shape[0]
    while max_i-min_i >= 1:
        test_i = np.floor((max_i+min_i)/2)
        test = TPRs[test_i]
        if test == TPR:
            min_i = test_i
            max_i = test_i+1
        elif test_i+1<len(TPRs) and TPRs[test_i+1]<TPR:
            min_i = test_i + 1
        else:
            max_i = test_i
    indx = min_i
    slope = (threshs[indx+1]-threshs[indx])/(TPRs[indx+1]-TPRs[indx])
    b = threshs[indx]-slope*TPRs[indx]
    thresh = slope*TPR + b
    print thresh, FPRs[min_i]
    if prec:
        return metrics.precision_score(ground_truth.ravel(), full_predict.ravel()>thresh)
    else:
        return FPRs[min_i], thresh



def ROC(ground_truth, full_predict, name, save = False):
    FPRs, TPRs, threshs = roc_curve(ground_truth.ravel(), full_predict.ravel())
    opt_thresh = threshs[np.argmin(FPRs**2 + (1-TPRs)**2)]

    fig = plt.figure('{} ROC'.format(name))
    AUC = sklearn.metrics.roc_auc_score(ground_truth.ravel(), full_predict.ravel())
    plt.plot(FPRs, TPRs)
    plt.title('ROC Curve (AUC = {})'.format(round(AUC, 5)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    indx = np.argmin(FPRs**2 + (1-TPRs)**2)
    if save:
        fig.savefig('Compare Methods/{}_ROC.png'.format(name), format='png', dpi = 2400)
        np.save('Compare Methods/'+name+'.npy', (FPRs, TPRs, threshs))
    return fig, AUC, opt_thresh, FPRs, TPRs, threshs


def feature_importance(model, labels, X):
    '''
    INPUT:
        - model:  (sklearn model) Trained sklearn model

    '''
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("{}. feature {}: ({})".format(f + 1, labels[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), labels[indices])
    plt.xlim([-1, X.shape[1]])
    #plt.show()
import numpy as np
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
import sklearn
import cv2
import matplotlib.pyplot as plt
from convenience_tools import *
import time


def confusion_matrix(y_true, y_pred):
    """
    Generates confusion_matrix just like sklearn but faster
    
    Parameters
    ----------
    y_true : ndarray
        1D array holding true labels 
    y_pred : ndarray
        1D array holding predicted labels
        1D array holding predicted labels

    """
    TPTN = y_true == y_pred
    TP = np.sum(np.logical_and(TPTN, y_true))
    TN = np.sum(TPTN) - TP
    FPFN = 1 - TPTN
    FP = np.sum(np.logical_and(FPFN, y_pred))
    FN = np.sum(FPFN) - FP
    return np.array([[TN, FP],[FN, TP]])


def confusion_analytics(y_true, y_pred):
    """
    Returns FPR and TPR
    
    Parameters
    ----------
    y_true : ndarray
        1D array holding true labels 
    y_pred : ndarray
        1D array holding predicted labels
        1D array holding predicted labels

    """
    TPTN = y_true == y_pred
    TP = np.sum(np.logical_and(TPTN, y_true))
    TN = np.sum(TPTN) - TP
    FPFN = 1 - TPTN
    FP = np.sum(np.logical_and(FPFN, y_pred))
    FN = np.sum(FPFN) - FP
    FPR = float(FP)/(FP+TN)
    TPR = float(TP)/(TP+FN)
    return FPR, TPR

def side_by_side(myMap, mask_true, mask_predict, name, save = False):
    """Generates figure that displays side by side images myMap masked by mask_true and amsk_predict"""
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
    """Produces heat map based on full_predict with true labels (stored as damage in map_test) 
    labeled with green contours"""
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
    """Produces figure of image wth full_predict shown as green contours over the image"""
    fig = plt.figure(title)
    fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    plt.contour(full_predict, colors = 'green')
    plt.imshow(img)
    plt.title(title), plt.xticks([]), plt.yticks([])


def FPR_from_FNR(ground_truth, full_predict, TPR = .95, prec = False):
    """
    Produces FPR at when TPR is [TPR]. If there is no threshold that produces this exact result,
    use linear interpolation between the two closest thresholds. 
    
    Parameters
    ----------
    ground_truth : ndarray
        1D array holding true labels 
    full_predict : ndarray
        1D array holding probabilities that each index is a 1
    TPR : float
        Recall to find FPR at
    prec : boolean
        If True, returns precision at that TPR instead of FPR

    Returns
    -------
    float
        FPR (or precision if prec == True) at set TRP
    float
        threshold that produces said FPR

    """
    FPRs, TPRs, threshs = roc_curve(ground_truth, full_predict.ravel())
    min_i = 0
    max_i = TPRs.shape[0]
    #Binary search for threshold corresponding with TPR
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
    #Use linear interpolation to find threshold and FPR at what would be a .95 Recall
    indx = min_i
    slope = (threshs[indx+1]-threshs[indx])/(TPRs[indx+1]-TPRs[indx])
    b = threshs[indx]-slope*TPRs[indx]
    thresh = slope*TPR + b
    slope2 = (FPRs[indx+1]-FPRs[indx])/(TPRs[indx+1]-TPRs[indx])
    b2 = FPRs[indx]-slope2*TPRs[indx]
    FPR = slope2*TPR + b2
    print thresh, FPRs[min_i], FPR
    if prec:
        return metrics.precision_score(ground_truth.ravel(), full_predict.ravel()>thresh)
    else:
        return FPR, thresh



def ROC(ground_truth, full_predict, name, save = False):
    """
    Produces FPR at when TPR is [TPR]. If there is no threshold that produces this exact result,
    use linear interpolation between the two closest thresholds. 
    
    Parameters
    ----------
    ground_truth : ndarray
        1D array holding true labels 
    full_predict : ndarray
        1D array holding probabilities that each index is a 1
    name : String
        name associated with ROC curve for unique saving
    save : boolean
        saves roc curve figure if save is True
    
    Returns
    matplolib.figure
        Figure holding ROC curve
    float
        Area under ROC curve
    float
        threshold leading to a point closest to (0,1)
    float list
        list of FPRs at each threshold
    float list
        list of TPRs at each threshold
    float list
        list of all thresholds
    """
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


import mapOverlay as MapOverlay
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt


def prec_recall_map(myMap, mask_true, mask_predict):
	'''
	INPUT:
		- confusion_matrix: (2x2 ndarray) Confusion matrix of shape: (actual values) x (predicted values)
	OUTPUT:
		- (tuple) A Tuple containing the precision and recall of the confusion matrix

	'''
	label = myMap.getLabels(mask_true)
	prediction = myMap.getLabels(mask_predict)
	conf = confusion_matrix(label, prediction)
	TP = conf[1,1]
	FP = conf[0,1]
	TN = conf[0,0]
	FN = conf[1,0]

	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)
	accuracy = float(TP + TN)/(TP+FN+TN+FP)

	return precision, recall, accuracy

def prec_recall(label, prediction):
	'''
	INPUT:
		- confusion_matrix: (2x2 ndarray) Confusion matrix of shape: (actual values) x (predicted values)
	OUTPUT:
		- (tuple) A Tuple containing the precision and recall of the confusion matrix

	'''
	conf = confusion_matrix(label, prediction)
	TP = conf[1,1]
	FP = conf[0,1]
	TN = conf[0,0]
	FN = conf[1,0]

	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)
	accuracy = float(TP + TN)/(TP+FN+TN+FP)

	return precision, recall, accuracy

def side_by_side(myMap, mask_true, mask_predict):
	fig = plt.figure()
	fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
	plt.subplot(121),plt.imshow(myMap.maskImg(mask_true))
	plt.title('Labelled Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(myMap.maskImg(mask_predict))
	plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
	fig.savefig('comparison.png', format='png', dpi=1200)
	plt.show()

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
from map_overlay import MapOverlay
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import cv2
import matplotlib.pyplot as plt
from convenience_tools import *


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
	f1 = float(2*precision*recall)/(precision+recall)


	return precision, recall, accuracy, f1

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
	conf = confusion_matrix(y_true, y_pred)
	TP = conf[1,1]
	FP = conf[0,1]
	TN = conf[0,0]
	FN = conf[1,0]
	recall = metrics.recall_score(y_true, y_pred)
	FPR = float(FP)/(FP+TN)
	FNR = 1-recall
	return round(FPR,5), round(FNR, 5), conf

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


def test_thresholds():
	print("loading")
	my_map = MapOverlay('datafromjoe/1-0003-0002.tif')
	my_map.newMask('datafromjoe/1-003-002-damage.shp', 'damage')
	predict_2 = np.loadtxt('temp/predictions.csv', delimiter = ',')
	labels = my_map.getLabels('damage')
	base_p, base_r, base_a, base_f =  prec_recall(labels, predict_2.ravel())
	predictions = np.loadtxt('temp/segments_100.csv', delimiter = ',')
	print("done loading")
	results = [[],[],[],[]]
	rate = 0.05
	for i in range(int(1/rate)):
		p,r,a,f = prec_recall(labels, predictions.ravel(), i*rate)
		results[0].append(p)
		results[1].append(r)
		results[2].append(a)
		results[3].append(f)
		print(i*rate, p, r, a ,f)
	fig = plt.figure("Threshold's Effect on F1")
	fig.suptitle("Threshold's Effect on F1", fontsize=14)
	ax = fig.add_subplot(111)

	t = np.arange(0,1,rate)
	ax.set_xlabel('Threshold (%)')
	ax.set_ylabel('Percentage (%)')
	precision = ax.plot(t, results[0], 'r-', label = 'Precision')
	recall = ax.plot(t, results[1], 'b-', label = 'Recall')
	#acc = ax.plot(t, results[2], 'g--', label = 'Accuracy')
	f1 =  ax.plot(t, results[3], 'y-', label = 'F1')
	base_p_graph = ax.plot(t, np.ones_like(t)*base_p, 'r--', label = 'Precision Compare')
	base_p_graph = ax.plot(t, np.ones_like(t)*base_r, 'b--', label = 'Recall Compare')
	base_p_graph = ax.plot(t, np.ones_like(t)*base_f, 'y--', label = 'F1 Compare')

	
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=6, mode="expand", borderaxespad=0.)
	plt.show()

#test_thresholds()

def side_by_side(myMap, mask_true, mask_predict):
	fig = plt.figure()
	fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
	plt.subplot(121),plt.imshow(myMap.maskImg(mask_true))
	plt.title('Labelled Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(myMap.maskImg(mask_predict))
	plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
	fig.savefig('temp/comparison.png', format='png', dpi=1200)

def probability_heat_map(map_test, full_predict):
	fig = plt.figure()
	fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
	ground_truth = map_test.getLabels('damage')
	plt.contour(ground_truth.reshape(map_test.rows, map_test.cols), colors = 'green')
	plt.imshow(full_predict.reshape(map_test.rows, map_test.cols), cmap = 'seismic')
	plt.title('Damage Prediction'), plt.xticks([]), plt.yticks([])


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
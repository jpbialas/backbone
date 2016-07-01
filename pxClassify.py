from mapOverlay import MapOverlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import features
import matplotlib.pyplot as plt
import analyzeResults


def v_print(myStr, vocal):
	if vocal:
		print(myStr)

def sample(labels, nsamples):
	'''
	INPUT: 
		- labels:   (nx1 ndarray) Array containing binary labels
		- nsamples: (int) Value representingtotal number of indices to be sampled
						NOTE: (If odd, produces list of length (nsamples-1))
		- [used]:   (sx1 ndarray) Array containing indices not to include in sampling
	OUTPUT:
		- Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
			nsamples/2 indices have value 0
	'''
	n = labels.shape[0]
	zeros = np.where(labels == 0)[0]
	n_zeros = np.shape(zeros)[0]

	ones = np.where(labels == 1)[0]
	n_ones = np.shape(ones)[0]

	zero_samples = zeros[np.random.random_integers(0,n_zeros-1, nsamples/2)]
	one_samples = ones[np.random.random_integers(0, n_ones-1, nsamples/2)]
	final_set = np.concatenate((zero_samples, one_samples))

	return final_set

def sample_split(labels, nsamples_train, nsamples_test):
	'''
	INPUT: 
		- labels:   (nx1 ndarray) Array containing binary labels
		- nsamples: (int) Value representingtotal number of indices to be sampled
						NOTE: (If odd, produces list of length (nsamples-1))
		- [used]:   (sx1 ndarray) Array containing indices not to include in sampling
	OUTPUT:
		- Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
			nsamples/2 indices have value 0
	'''
	n = labels.shape[0]
	train = labels[:n/2]
	train_set = sample(train, nsamples_train)
	test_set = np.random.random_integers(n/2, n-1, nsamples_test)
	return train_set,test_set


def gen_features(myMap, edge_k, hog_k, hog_bins):
	'''
	input:
		- myMap: MapObject
	output:
		- feature representation of map
	'''
	#entropy, entropy_name = features.entr(myMap.bw_img)
	rgb, rgb_name = features.normalized(myMap.getMapData())
	ave_rgb, ave_rgb_name = features.blurred(myMap.img)
	edges, edges_name = features.edge_density(myMap.bw_img, edge_k, amp = 1)
	hog, hog_name = features.hog(myMap.bw_img, hog_k)
	data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
	names = np.concatenate((rgb_name, ave_rgb_name, hog_name))
	return data, names



def predict_all(model, myMap, mask_name = 'damage', load = False, erode = False, vocal = True):
	'''
	INPUT:
		- model:  (sklearn model) Trained sklearn model
		- X: 	  (n x m array) containing the data that model is trained on
		- fn: 	  (string) filename of base map image used for data generation
		- [load]: (boolean) Optional paramater indicating whether or not to load a 
						saved file named "predictions.csv" containing the model predictions
		- [erode]: (boolean) Optional parameter indicating whether or not to reduce noise by eroding 
						positive predictions not near other positive predictions

	RESULT:
		- Saves learned predictions to 'predictions.csv'
		- Displays img with regions highlighted to represent learned data
	'''
	img = myMap.img
	h,w,c = img.shape
	X = gen_features(myMap, 100, 50, 16)[0]
	#X = myMap.get_features()
	y = myMap.getLabels(mask_name)

	if not load:
		v_print("Starting full prediction", vocal)
		full_predict = model.predict(X).reshape(h,w)
		v_print("Ending full prediction", vocal)
		np.savetxt('temp/predictions.csv', full_predict, delimiter = ',', fmt = "%d")
	else:
		v_print("Loading prediction", vocal)
		full_predict = np.loadtxt('temp/predictions.csv', delimiter = ',')
		v_print("Done loading prediction", vocal)
	if erode:
		kernel = np.ones((5,5),np.uint8)
		full_predict = cv2.erode(full_predict, kernel, iterations = 2)

	myMap.newPxMask(full_predict.ravel(), 'damage_pred')

	precision, recall, accuracy, f1 = analyzeResults.prec_recall_map(myMap, mask_name, 'damage_pred')
	print("precision = {}".format(precision))
	print("recall = {}".format(recall))
	print("accuracy = {}".format(accuracy))
	print("f1 = {}".format(f1))

	analyzeResults.side_by_side(myMap, mask_name, 'damage_pred')


def train_and_test(map_train, map_test, mask_train = 'damage', mask_test = 'damage', frac_train = 0.1, frac_test = 0.1, edge_k = 100, hog_k = 50, nbins = 16, n_trees = 85, vocal = True):
	v_print('Starting train load', vocal)
	X_train, labels = gen_features(map_train, edge_k, hog_k, nbins)
	v_print('Done load', vocal)

	v_print('Starting test load', vocal)
	X_test, labels = gen_features(map_test, edge_k, hog_k, nbins)
	v_print('Done test load', vocal)

	y_train =  map_train.getLabels(mask_train)
	n_train = y_train.shape[0]
	train = sample(y_train, int(n_train*frac_train))

	y_test = map_test.getLabels(mask_test)
	n_test = y_test.shape[0]
	test = np.random.random_integers(0,y_test.shape[0], int(n_test*frac_test))

	v_print("Starting Modelling", vocal)
	model= RandomForestClassifier(n_estimators=n_trees)
	model.fit(X_train[train], y_train[train])
	v_print("Done Modelling", vocal)


	v_print("Starting Testing", vocal)
	prediction = model.predict(X_test[test])
	ground_truth = y_test[test]
	v_print("Done Testing", vocal)
	precision, recall, accuracy, f1 = analyzeResults.prec_recall(ground_truth, prediction)
	return precision, recall,  accuracy, f1,  model, X_test, y_test, labels

def train_model(myMap, mask_name = 'damage', frac_train = 0.1, frac_test = 0.1, edge_k = 100, hog_k = 50, nbins = 16, n_trees = 85, vocal = True):
	'''
	INPUT:
		-fn: 		   (string) Filename of .tiff file containing map
		-mask_names:   (string list) List of mask names associated with positive labelling
		-[frac_train]: (float) Optional Fraction of total image to train data on
		-[frac_test]:  (float) Optional Fraction of total image to test data on
		-[edge_k]:     (int) Size of window used for edge density calculation
		-[hog_k]       (int) Size of window used for HOG calculations
		-[nbins]       (int) Number of bins used for HoG
		-[n_tress]     (int) Number of trees used in random forest
	OUTPUT:
		Tuple Containing:
			- (float) precision
			- (float) recall
			- (sklearn model) Trained sklearn model
			- (ndarray) Data collected from .tiff file
	'''
	v_print('Starting gen', vocal)
	X, labels = gen_features(myMap, edge_k, hog_k, nbins)
	v_print('Ending gen', vocal)

	y = myMap.getLabels(mask_name)
	n = y.shape[0]
	train, test = sample_split(y, int(n*frac_train), int(n*frac_test))

	v_print("Starting Modelling", vocal)
	model= RandomForestClassifier(n_estimators=n_trees)
	model.fit(X[train], y[train])
	v_print("Done Modelling", vocal)

	y_pred = model.predict(X[test])
	y_true = y[test]

	precision, recall, accuracy, f1 = analyzeResults.prec_recall(y_true, y_pred)
	return precision, recall,  accuracy, f1,  model, X, y, labels


if __name__ == "__main__":
	
	fn2 = 'datafromjoe/1-0003-0002.tif'
	mask_fn2 = 'datafromjoe/1-003-002-damage.shp'
	fn1 = 'datafromjoe/1-0003-0003.tif'
	mask_fn1 = 'datafromjoe/1-003-003-damage.shp'

	myMap1 = MapOverlay(fn1)
	myMap1.newMask(mask_fn1, 'damage')
	myMap2 = MapOverlay(fn2)
	myMap2.newMask(mask_fn2, 'damage')


	'''precision, recall, accuracy, f1,  model, X, y, names = train_model(myMap1)
	print("precision = {}".format(precision))
	print("recall = {}".format(recall))
	print("accuracy = {}".format(accuracy))
	print("f1 = {}".format(f1))'''
	precision, recall,  accuracy, f1,  model2, X, y, names = train_and_test(myMap1, myMap2, frac_test = 0.001)
	print("precision = {}".format(precision))
	print("recall = {}".format(recall))
	print("accuracy = {}".format(accuracy))
	print("f1 = {}".format(f1))
	

	#predict_all(model, myMap2, 'damage')
	predict_all(model2, myMap2, 'damage')


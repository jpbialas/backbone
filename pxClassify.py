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

def random_forest(X, y, vocal):
	'''
	Input: 
		- X: (n x m ndarray) n  m-dimensional vectors representing data to be learned
		- y: (n x 1 ndarray) contains labels for X
	Output:
		- sklearn model fit to input data
	'''

	v_print("Starting Modelling", vocal)
	model= RandomForestClassifier(n_estimators=85)
	model.fit(X, y)
	v_print("Done Modelling", vocal)
	return model


def sample(labels, nsamples_train, nsamples_test):
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
	train_labels = labels[:n/2]
	test_labels = labels[n/2:]

	train_zeros = np.where(train_labels == 0)[0]
	n_trainzeros = np.shape(train_zeros)[0]
	test_zeros = np.where(test_labels == 0)[0]
	n_testzeros = np.shape(test_zeros)[0]

	train_ones = np.where(train_labels == 1)[0]
	n_trainones = np.shape(train_ones)[0]
	test_ones = np.where(test_labels == 1)[0]
	n_testones = np.shape(test_ones)[0]

	zero_train_samples = train_zeros[np.random.random_integers(0,n_trainzeros-1, nsamples_train/2)]
	one_train_samples = train_ones[np.random.random_integers(0, n_trainones-1, nsamples_train/2)]
	train_set = np.concatenate((zero_train_samples, one_train_samples))

	zero_test_samples = test_zeros[np.random.random_integers(0,n_testzeros-1, nsamples_test/2)]
	one_test_samples = test_ones[np.random.random_integers(0, n_testones-1, nsamples_test/2)]
	test_set = np.concatenate((zero_test_samples, one_test_samples))
	return train_set,test_set

def sample2(labels, nsamples, used,  even = True):
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
	if even:
		zeros = np.where(labels == 0)[0]
		if used is not None:
			zeros = np.setdiff1d(zeros, used)
		nzeros = np.shape(zeros)[0]

		ones = np.where(labels == 1)[0]
		if used is not None:
			ones = np.setdiff1d(ones, used)
		nones = np.shape(ones)[0]

		zero_samples = zeros[np.random.random_integers(0,nzeros-1, nsamples/2)]
		one_samples = ones[np.random.random_integers(0, nones-1, nsamples/2)]
		return np.concatenate((zero_samples, one_samples))
	else:
		uniques = np.setdiff1d(np.arange(labels.shape[0]), used)
		return np.random.random_integers(0, uniques.shape[0], nsamples)



def gen_features(myMap):
	'''
	input:
		- myMap: MapObject
	output:
		- feature representation of map
	'''
	rgb, rgb_name = features.normalized(myMap.getMapData())
	ave_rgb, ave_rgb_name = features.blurred(myMap.img)
	edges, edges_name = features.edge_density(myMap.bw_img, 100, amp = 1)
	hog, hog_name = features.hog(myMap.bw_img, 50)
	data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
	names = np.concatenate((rgb_name, ave_rgb_name, edges_name, hog_name))
	return data, names



def predict_all(model, myMap, mask_name, load = False, erode = False, vocal = True):
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
	X = gen_features(myMap)[0]
	y = myMap.getLabels(mask_name)

	if not load:
		v_print("starting full prediction", vocal)
		full_predict = model.predict(X).reshape(h,w)
		v_print("ending full prediction", vocal)
		np.savetxt('predictions.csv', full_predict, delimiter = ',', fmt = "%d")
	else:
		v_print("loading prediction", vocal)
		full_predict = np.loadtxt('predictions.csv', delimiter = ',')
		v_print("done loading prediction", vocal)
	if erode:
		kernel = np.ones((5,5),np.uint8)
		full_predict = cv2.erode(full_predict, kernel, iterations = 2)

	myMap.newPxMask(full_predict.ravel(), 'damage_pred')

	precision, recall, accuracy = analyzeResults.prec_recall_map(myMap, mask_name, 'damage_pred')
	print("precision = ", precision)
	print("recall = ", recall)
	print("accuracy = ", accuracy)

	analyzeResults.side_by_side(myMap, mask_name, 'damage_pred')





def main(myMap, mask_names, frac_train, frac_test, vocal = True):
	'''
	INPUT:
		-fn: 		 (string) Filename of .tiff file containing map
		-mask_fn: 	 (string) Filename of .shp file outlining labelled data
		-name: 		 (string) Name associated with labeled data
		-frac_train: (float) Fraction of total image to train data on
		-frac_test:  (float) Fraction of total image to test data on
	OUTPUT:
		Tuple Containing:
			- (float) precision
			- (float) recall
			- (sklearn model) Trained sklearn model
			- (ndarray) Data collected from .tiff file
	'''
	y =  myMap.getLabels(mask_names[0])
	for i in range(1,len(mask_names)):
		y = y+myMap.getLabels(mask_names[i])
	y = y>0
	
	X, labels = gen_features(myMap)
	n = y.shape[0]
	
	train_size = int(n*frac_train)
	test_size = int(n*frac_test)

	
	train, test = sample(y, train_size, test_size)
	model = random_forest(X[train],y[train], vocal)

	y_pred = model.predict(X[test])
	y_true = y[test]

	precision, recall, accuracy = analyzeResults.prec_recall(y_true, y_pred)
	return precision, recall, accuracy, model, X, y, labels


def main2(folder, fn):
	'''
	INPUT:
		-fn: 		 (string) Filename of .tiff file containing map
	OUTPUT:
		Tuple Containing:
			- (float) precision
			- (float) recall
			- (sklearn model) Trained sklearn model
			- (ndarray) Data collected from .tiff file
	'''
	myMap = MapOverlay('{}/{}'.format(folder,fn))
	X, names = gen_features(myMap)
	

	myMap.newMask('{}/training-damage.shp'.format(folder), 'damage-train')
	myMap.newMask('{}/training-building.shp'.format(folder), 'building-train')
	myMap.newMask('{}/training-car.shp'.format(folder), 'car-train')
	myMap.newMask('{}/training-pavement.shp'.format(folder), 'pavement-train')
	myMap.newMask('{}/training-vegentation.shp'.format(folder), 'vege-train')

	
	damage = myMap.getLabels('damage-train')
	not_damage = myMap.getLabels('building-train') + myMap.getLabels('car-train') + myMap.getLabels('pavement-train') + myMap.getLabels('vege-train')
	not_damage = not_damage>0

	train = np.concatenate((np.where(damage == 1)[0], np.where(not_damage == 1)[0]))
	y_train = damage


	myMap.newMask('{}/validation-damage.shp'.format(folder), 'damage-valid')
	myMap.newMask('{}/validation-all.shp'.format(folder), 'all-validation')

	damage_valid = myMap.getLabels('damage-valid')
	all_validation = myMap.getLabels('all-validation')

	test = np.where(all_validation == 1)[0]
	y_test = damage_valid


	print("starting modelling")
	model = random_forest(X[train],y_train[train])
	print("done modelling")


	y_pred = model.predict(X[test])
	y_true = y_test[test]
	return model, X, ((y_test + y_train) > 0)


def tune_hyperparameters():
	pass	


if __name__ == "__main__":
	
	#fn1 = 'datafromjoe/1-0003-0003.tif'
	#mask_fn1 = 'datafromjoe/1-003-003-damage.shp'
	fn1 = 'jpl-data/clipped-image.tif'
	mask_fn1 = 'jpl-data/training-damage.shp'
	mask_fn12 = 'jpl-data/validation-damage.shp'
	myMap1 = MapOverlay(fn1)
	myMap1.newMask(mask_fn1, 'damage')
	myMap1.newMask(mask_fn12, 'damage2')
	precision, recall, accuracy, model, X, y, names = main(myMap1, ['damage', 'damage2'], 0.01, 0.01, vocal = True)
	analyzeResults.feature_importance(model, names, X)
	print("precision = ", precision)
	print("recall = ", recall)
	print("accuracy = ", accuracy)
	
	#fn2 = 'datafromjoe/1-0003-0002.tif'
	#mask_fn2 = 'datafromjoe/1-003-002-damage.shp'

	fn2 = 'jpl-data/clipped-image.tif'
	mask_fn2 = 'jpl-data/training-damage.shp'
	mask_fn22 = 'jpl-data/validation-damage.shp'

	myMap2 = MapOverlay(fn2)
	myMap2.newMask(mask_fn2, 'damage')
	myMap2.newMask(mask_fn22, 'damage2')
	myMap2.combine_masks('damage','damage2','damage_full')
	predict_all(model, myMap2, 'damage_full', load = False, erode = False)


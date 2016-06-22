from mapOverlay import MapOverlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import features
import matplotlib.pyplot as plt


'''For each of the following Models:
Input: 
	- X: (n x m ndarray) n  m-dimensional vectors representing data to be learned
	- y: (n x 1 ndarray) contains labels for X
Output:
	- sklearn model fit to input data
'''
def random_forest(X, y):
	model= RandomForestClassifier(n_estimators=85)
	model.fit(X, y)
	return model

def SVM(X,y):
	model = SVC()
	model.fit(X,y)
	return model

def NB(X, y):
	model = GaussianNB()
	model.fit(X,y)
	return model

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
def sample(labels, nsamples, used,  even = True):
	if even:
		zeros = np.where(labels == 0)[0]
		if used is not None:
			zeros = np.setdiff1d(zeros, used)
		nzeros = np.shape(zeros)[0]

		ones = np.where(labels == 1)[0]
		if used is not None:
			ones = np.setdiff1d(ones, used)
		nones = np.shape(ones)[0]

		print(nones, nsamples)
		zero_samples = zeros[np.random.random_integers(0,nzeros-1, nsamples/2)]
		one_samples = ones[np.random.random_integers(0, nones-1, nsamples/2)]
		return np.concatenate((zero_samples, one_samples))
	else:
		uniques = np.setdiff1d(np.arange(labels.shape[0]), used)
		return np.random.random_integers(0, uniques.shape[0], nsamples)


'''
INPUT:
	- confusion_matrix: (2x2 ndarray) Confusion matrix of shape: (actual values) x (predicted values)
OUTPUT:
	- (tuple) A Tuple containing the precision and recall of the confusion matrix

'''
def prec_recall(confusion_matrix):
	TP = confusion_matrix[1,1]
	FP = confusion_matrix[0,1]
	TN = confusion_matrix[0,0]
	FN = confusion_matrix[1,0]

	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)
	accuracy = float(TP + TN)/(TP+FN+TN+FP)

	return precision, recall, accuracy

def gen_features(img, myMap):
	return np.concatenate((features.normalized(myMap.getMapData()),features.blurred(myMap.getMap()), features.edge_density(img, 100, amp = 1), features.hog(img, 50)), axis = 1)


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
def predict_all(model, X, y, fn, load = False, erode = False):

	img = cv2.imread(fn)
	h,w,c = img.shape

	if not load:
		print("starting full prediction")
		full_predict = model.predict(X).reshape(h,w)
		print("ending full prediction")
		np.savetxt('predictions.csv', full_predict, delimiter = ',', fmt = "%d")
		np.savetxt('labels.csv', y, delimiter = ',', fmt = '%d')
	else:
		print("loading prediction")
		full_predict = np.loadtxt('predictions.csv', delimiter = ',')
		print("done loading prediction")
	highlights = full_predict
	if erode:
		kernel = np.ones((5,5),np.uint8)
		highlights = cv2.erode(highlights, kernel, iterations = 2)

	conf = confusion_matrix(y, full_predict.ravel())
	precision, recall, accuracy = prec_recall(conf)
	print("precision = ", precision)
	print("recall = ", recall)
	print("accuracy = ", accuracy)

	conf = confusion_matrix(y, ((highlights-.25)/.75).ravel())
	precision, recall, accuracy = prec_recall(conf)
	print("precision = ", precision)
	print("recall = ", recall)
	print("accuracy = ", accuracy)

	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	final = img/255.0*np.logical_not(highlights).reshape(h,w,1)
	plt.imshow(final)
	plt.show()



def predict_all2(model, X, fn, load = False, erode = False):

	img = cv2.imread(fn)
	h,w,c = img.shape

	if not load:
		print("starting full prediction")
		full_predict = model.predict(X).reshape(h,w)
		print("ending full prediction")
		np.savetxt('predictions.csv', full_predict, delimiter = ',', fmt = "%d")
	else:
		print("loading prediction")
		full_predict = np.loadtxt('predictions.csv', delimiter = ',')
		print("done loading prediction")
	if erode:
		kernel = np.ones((5,5),np.uint8)
		full_predict = cv2.erode(full_predict, kernel, iterations = 2)


	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	final = img/255.0*np.logical_not(full_predict.reshape(h,w,1))
	plt.imshow(final)
	plt.show()

'''
INPUT:
	- model:  (sklearn model) Trained sklearn model

'''
def feature_importance(model):
	labels = np.array(['red', 'green', 'blue', 'ave_red', 'ave_green', 'ave_blue', 'edge_density', 'hog 1', 'hog 2', 'hog 3', 'hog 4', 'hog 5', 'hog 6', 'hog 7', 'hog 8', 'hog 9', 'hog 10', 'hog 11', 'hog 12', 'hog 13', 'hog 14', 'hog 15', 'hog 16'])
	importances = model.feature_importances_
	std = np.std([tree.feature_importances_ for tree in model.estimators_],
	             axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
	    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	print(indices)
	print(labels[indices])
	plt.xticks(range(X.shape[1]), labels[indices])
	plt.xlim([-1, X.shape[1]])
	plt.show()


'''
INPUT:
	-fn: 		 (string) Filename of .tiff file containing map
	-mask_fn: 	 (string) Filename of .shp file outlining labelled data
	-name: 		 (string) Name associated with labeled data
	-model: 	 ((n x m ndarray) -> (n x 1 ndarray) -> sklearn model) Function that takes data and 
						labels as input and produces trained model
	-frac_train: (float) Fraction of total image to train data on
	-frac_test:  (float) Fraction of total image to test data on
OUTPUT:
	Tuple Containing:
		- (float) precision
		- (float) recall
		- (sklearn model) Trained sklearn model
		- (ndarray) Data collected from .tiff file
'''
def main(fn, mask_fn, mask_fn2, name, model, frac_train, frac_test):
	myMap = MapOverlay(fn)
	myMap.newMask(mask_fn, name)
	myMap.newMask(mask_fn2, "{}2".format(name))


	img = cv2.imread(fn, 0)
	y = (myMap.getLabels(name) + myMap.getLabels("{}2".format(name)))>0
	
	X = gen_features(img, myMap)
	n = y.shape[0]
	
	train_size = int(n*frac_train)
	test_size = int(n*frac_test)

	print("starting modelling")
	train = sample(y, train_size, np.array([]), even = True)
	model = model(X[train],y[train])
	print("done modelling")

	test = sample(y, test_size, train, even = True)
	y_pred = model.predict(X[test])
	y_true = y[test]

	conf = confusion_matrix(y_true, y_pred)
	print(conf)
	
	precision, recall, accuracy = prec_recall(conf)

	return precision, recall, accuracy, model, X, y

'''
INPUT:
	-fn: 		 (string) Filename of .tiff file containing map
	-mask_fn: 	 (string) Filename of .shp file outlining labelled data
	-name: 		 (string) Name associated with labeled data
	-model: 	 ((n x m ndarray) -> (n x 1 ndarray) -> sklearn model) Function that takes data and 
						labels as input and produces trained model
OUTPUT:
	Tuple Containing:
		- (float) precision
		- (float) recall
		- (sklearn model) Trained sklearn model
		- (ndarray) Data collected from .tiff file
'''
def main2(folder, fn, model_fun):
	myMap = MapOverlay('{}/{}'.format(folder,fn))
	img = cv2.imread('{}/{}'.format(folder,fn), 0)
	X = np.concatenate((features.normalized(myMap.getMapData()),features.blurred(myMap.getMap()), features.edge_density(img, 100, amp = 1), features.hog(img, 100)), axis = 1)
	

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
	model = model_fun(X[train],y_train[train])
	print("done modelling")


	y_pred = model.predict(X[test])
	y_true = y_test[test]

	conf = confusion_matrix(y_true, y_pred)
	print(conf)
	
	precision, recall, accuracy = prec_recall(conf)
	print(precision, recall, accuracy)

	predict_all(model, X, ((y_test + y_train) > 0), '{}/{}'.format(folder,fn))
	return precision, recall, accuracy, model, X, ((y_test + y_train) > 0)


def tune_hyperparameters():
	pass	


if __name__ == "__main__":
	
	mask_fn = 'datafromjoe/1-003-003-damage.shp'

	fn2 = 'jpl-data/clipped-image.tif'
	mask_fn2 = 'jpl-data/training-damage.shp'
	mask_fn3 = 'jpl-data/validation-damage.shp'

	precision, recall, accuracy, model, X, y = main(fn2, mask_fn2, mask_fn3, 'damage', random_forest, 0.01, 0.1)
	
	#precision, recall, accuracy, model, X, y = main2('jpl-data', 'clipped-image.tif', random_forest)
	feature_importance(model)
	print("precision = ", precision)
	print("recall = ", recall)
	print("accuracy = ", accuracy)
	#predict_all(model, X, y, fn2, False, True)

	fn = 'jpl-data/1-0003-0003.tif'
	img = cv2.imread(fn, 0)
	myMap = MapOverlay(fn)
	X2 = gen_features(img, myMap)
	predict_all2(model, X2, fn, False, False)

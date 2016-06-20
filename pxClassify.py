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
def sample(labels, nsamples, used = None):
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
	return precision, recall

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
def predict_all(model, X, fn, load = False, erode = False):

	img = cv2.imread(fn, 0)
	h,w = img.shape

	if not load:
		print("starting full prediction")
		full_predict = model.predict(X).reshape(h,w)
		print("ending full prediction")
		np.savetxt('predictions.csv', full_predict, delimiter = ',', fmt = "%d")
	else:
		print("loading prediction")
		full_predict = np.loadtxt('predictions.csv', delimiter = ',')
		print("done loading prediction")
	highlights = (full_predict*0.75)+0.25
	if erode:
		kernel = np.ones((5,5),np.uint8)
		highlights = cv2.erode(highlights, kernel, iterations = 4)

	plt.imshow(img*highlights, cmap = 'gray')
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
def main(fn, mask_fn, name, model, frac_train, frac_test):
	myMap = MapOverlay(fn)
	myMap.newMask(mask_fn, name)
	img = cv2.imread(fn, 0)
	h, w = img.shape
	y = myMap.getLabels(name)
	X = np.concatenate((features.normalized(myMap.getMapData()),features.blurred(myMap.getMap()), features.edge_density(img, 100, amp = 1)), axis = 1)
	n = y.shape[0]
	
	train_size = int(n*frac_train)
	test_size = int(n*frac_test)

	print("starting modelling")
	train = sample(y, train_size)
	model = model(X[train],y[train])
	print("done modelling")

	test = sample(y, test_size, used = train)
	y_pred = model.predict(X[test])
	y_true = y[test]

	conf = confusion_matrix(y_true, y_pred)
	print(conf)
	
	precision, recall = prec_recall(conf)

	return precision, recall, model, X


if __name__ == "__main__":
	fn = 'jpl-data/clipped-image.tif'
	mask_fn = 'jpl-data/training-damage.shp'
	precision, recall, model, X = main(fn, mask_fn, 'damage', random_forest, 0.0001, 0.01)
	print("precision = ", precision)
	print("recall = ", recall)
	predict_all(model, X, fn, True, True)


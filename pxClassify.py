from mapOverlay import MapOverlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import features
import matplotlib.pyplot as plt

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

def prec_recall(confusion_matrix):
	TP = confusion_matrix[1,1]
	FP = confusion_matrix[0,1]
	TN = confusion_matrix[0,0]
	FN = confusion_matrix[1,0]

	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)
	return precision, recall

def predict_all(model, X, fn, erode = False):
	img = cv2.imread(fn, 0)
	h,w = img.shape
	print("starting full prediction")
	full_predict = model.predict(X).reshape(h,w)
	print("ending full prediction")
	highlights = (full_predict*0.75)+0.25

	if erode:
		kernel = np.ones((5,5),np.uint8)
		highlights = cv2.erode(highlights, kernel, iterations = 2)

	np.savetxt('predictions.csv', full_predict, delimiter = ',', fmt = "%d")

	plt.imshow(img*highlights, cmap = 'gray')
	plt.show()


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
	precision, recall, model, X = main(fn, mask_fn, 'damage', random_forest, 0.01, 0.01)
	print("precision = ", precision)
	print("recall = ", recall)
	predict_all(model, X, fn, True)


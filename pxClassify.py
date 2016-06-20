from mapOverlay import MapOverlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import features

def random_forest(X, y):
	model= RandomForestClassifier(n_estimators=80)
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


def sample(labels, nsamples):
	zeros = np.where(labels == 0)[0]
	nzeros = np.shape(zeros)[0]

	ones = np.where(labels == 1)[0]
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


def main(fn, mask_fn, name, model, frac_train, frac_test):
	myMap = MapOverlay(fn)
	myMap.newMask(mask_fn, name)
	img = cv2.imread(fn, 0)
	y = myMap.getLabels(name)
	X = np.concatenate((features.normalized(myMap.getMapData()),features.blurred(myMap.getMap()), features.edge_density(img, 100, amp = 1)), axis = 1)
	n = y.shape[0]
	
	train_size = int(n*frac_train)
	test_size = int(n*frac_test)

	print("starting prediction")
	train = sample(y, train_size)
	model = model(X[train],y[train])
	print("done predicting")

	test = sample(y, test_size)
	y_pred = model.predict(X[test])
	y_true = y[test]

	conf = confusion_matrix(y_true, y_pred)
	return prec_recall(conf)


if __name__ == "__main__":
	fn = 'jpl-data/clipped-image.tif'
	mask_fn = 'jpl-data/training-damage.shp'
	precision, recall = main(fn, mask_fn, 'damage', random_forest, 0.01, 0.01)
	print("precision = ", precision)
	print("recall = ", recall)


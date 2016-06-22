import cv2
import matplotlib.pyplot as plt
import numpy as np
from pxClassify import prec_recall

from sklearn.metrics import confusion_matrix

def load_data():
	return np.loadtxt('predictions.csv', delimiter = ','), np.loadtxt('labels.csv', delimiter = ',')

def generate_graph(ksize, data, labels, min_v = 0, max_v = 5):
	kernel = np.ones((ksize, ksize))
	prec = []
	rec = []
	for i in range(min_v, max_v):
		print(i)
		precision, recall, _ = test(i, kernel, data, labels)
		prec.append(precision)
		rec.append(recall)
	plt.plot(np.arange(min_v, max_v, 1), prec, 'rs', np.arange(min_v, max_v, 1), rec, 'bs')
	plt.show()

def run(fn = 'jpl-data/1-0003-0003.tif', data = None):
	if data is None:
		data = np.loadtxt('predictions.csv', delimiter = ',')
	img = cv2.imread(fn)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h,w,c = img.shape
	final = img/255.0*np.logical_not(data.reshape(h,w,1))
	plt.imshow(final)
	plt.show()


def test(indx, kernel, data, labels):
	eroded = cv2.erode(data, kernel, iterations = indx)
	return prec_recall(confusion_matrix(labels, eroded.ravel()))




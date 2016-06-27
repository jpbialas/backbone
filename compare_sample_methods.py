import numpy as np
import cv2
import matplotlib.pyplot as plt
import features
import pxClassify
from pxClassify import gen_features
from mapOverlay import MapOverlay
from mpl_toolkits.mplot3d import Axes3D
from tsne import tsne
from sklearn.decomposition import PCA

def pca(x, k):
    print("Starting PCA")
    pca = PCA(n_components=k)
    pca.fit(x)
    print("Variance Retained: " + str(pca.explained_variance_ratio_.sum()))
    print("Ending PCA")
    return pca.transform(x)


def assign_colors(labels, start=0):
	'''
	input: (n x 1) array of cluster assignments
	output: (n x 1) list of colors assignments
	        List of colors used in order
	        Number of points assigned to each ordered colo
	'''
	labels = labels.astype('int')
	colors = [-1]*len(labels)
	colorNames = [ "green", "red", "pink", "yellow", "orange", "grey", "cyan", "red", "gray"]
	numbers = [0]*len(colorNames)
	for indx, i in enumerate(labels):
		if i < len(colorNames):
			colors[indx] = colorNames[i-start]
			numbers[i-start]+=1
		else:
			colors[indx] = "black"
	return colors, colorNames, numbers

def method_old(myMap):

	not_damage = myMap.getLabels('building-train') + myMap.getLabels('car-train') + myMap.getLabels('pavement-train') + myMap.getLabels('vege-train')
	not_damage = not_damage>0

	indices = np.arange(not_damage.shape[0])
	return np.random.random_integers(0, indices.shape[0], 100000)

def method_new(myMap):
	labels = (myMap.getLabels('damage') + myMap.getLabels("damage2"))>0
	
	zeros = np.where(labels == 0)[0]
	nzeros = np.shape(zeros)[0]

	ones = np.where(labels == 1)[0]
	nones = np.shape(ones)[0]

	zero_samples = zeros[np.random.random_integers(0,nzeros-1, 100000)]
	one_samples = ones[np.random.random_integers(0, nones-1, 100000)]
	return zero_samples, one_samples



def compare(k):
	myMap = MapOverlay('jpl-data/clipped-image.tif')
	img = cv2.imread('jpl-data/clipped-image.tif', 0)
	myMap.newMask('jpl-data/training-damage.shp', 'damage')
	myMap.newMask('jpl-data/validation-damage.shp', 'damage2')
	myMap.newMask('jpl-data/training-building.shp', 'building-train')
	myMap.newMask('jpl-data/training-car.shp', 'car-train')
	myMap.newMask('jpl-data/training-pavement.shp', 'pavement-train')
	myMap.newMask('jpl-data/training-vegentation.shp', 'vege-train')

	X = gen_features(myMap)

	samples_old = X[method_old(myMap)]
	res_undamaged, res_damaged = method_new(myMap)
	samples_new = X[res_undamaged]
	sample_new_dam = X[res_damaged]

	label_old = np.ones(100000)
	label_new = np.zeros(100000)
	label_new_damage = np.ones(100000)*2

	all_samples = np.concatenate((samples_old, samples_new, sample_new_dam))
	all_labels = np.concatenate((label_old, label_new, label_new_damage))

	colors, color_names, _ = assign_colors(all_labels)
	print(color_names)

	y = pca(all_samples, k)
	if k == 2:
		xaxis = y[:,0]
		yaxis = y[:,1]
		fig = plt.figure('test')
		ax = fig.add_subplot(111)
		ax.scatter(xaxis,yaxis, s = 1, color = colors, alpha = 0.5)
	if k == 3:
		xaxis = np.array(y[:,0])
		yaxis = np.array(y[:,1])
		zaxis = np.array(y[:,2])
		fig = plt.figure('test')
		ax = fig.add_subplot(111, projection = '3d')
		ax.scatter(xs=xaxis, ys=yaxis, zs = zaxis, zdir = 'z', s = 5, edgecolors = colors, c = colors, depthshade = True, alpha = .5)
	plt.show()
compare(2)

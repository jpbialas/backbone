import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def get_samples(img_num, n, label, load = True):
	if load:
		print('loading results for {}'.format(img_num))
		ans = np.loadtxt('temp/cached_{}.csv'.format(img_num), delimiter = ',')
		print('done loading')
	else:
		next_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
		next_map.newMask('datafromjoe/1-003-00{}-damage.shp'.format(img_num),'damage')
		all_damage = np.where(next_map.getLabels('damage') == 1)[0]

		sub_sample = all_damage[np.random.random_integers(0,all_damage.shape[0]-1, n)]
		print('generating for {}'.format(img_num))
		X, names = gen_features(next_map, 200, 200, 16)
		ans = X[sub_sample]
		print("done")
		np.savetxt('temp/cached_{}.csv'.format(img_num), ans, delimiter = ',')
	return ans, np.ones(n)*label

def plot(y, colors, k, name):
	if k == 2:
		xaxis = y[:,0]
		yaxis = y[:,1]
		fig = plt.figure(name)
		ax = fig.add_subplot(111)
		green_patch = mpatches.Patch(color='green', label='Image 002')
		red_patch = mpatches.Patch(color='red', label='Image 003')
		plt.legend(handles=[green_patch, red_patch])
		ax.scatter(xaxis,yaxis, s = 1, color = colors, alpha = 0.5)
	if k == 3:
		xaxis = np.array(y[:,0])
		yaxis = np.array(y[:,1])
		zaxis = np.array(y[:,2])
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.scatter(xs=xaxis, ys=yaxis, zs = zaxis, zdir = 'z', s = 5, edgecolors = colors, c = colors, depthshade = True, alpha = .5)


def all_histos(X1, X2):
	names = ['red', 'green', 'blue', 'ave_red', 'ave_green', 'ave_blue', 'edges', 'hog 0', 'hog 1', 'hog 2', 'hog 3', 'hog 4', 'hog 5', 'hog 6', 'hog 7', 'hog 8', 'hog 9', 'hog 10', 'hog 11', 'hog 12', 'hog 13', 'hog 14', 'hog 15']
	for i in range(X1.shape[1]):


		hist1, bins1 = np.histogram(X1[:,i], bins=32)
		hist2, bins2 = np.histogram(X2[:,i], bins=32)
		leftmost = min(bins1[0],bins2[0])
		rightmost = max(bins1[-1], bins2[-1])

		plt.figure('{}'.format(names[i]))
		plt.title(names[i])
		ax1 = plt.subplot(211)
		ax1.set_title('002')
		width = 0.7 * (bins1[1] - bins1[0])
		center = (bins1[:-1] + bins1[1:]) / 2
		plt.bar(center, hist1, align='center', width=width)
		plt.xlim(leftmost, rightmost)
		
		ax2 = plt.subplot(212)
		ax2.set_title('003')

		width = 0.7 * (bins2[1] - bins2[0])
		center = (bins2[:-1] + bins2[1:]) / 2
		plt.bar(center, hist2, align='center', width=width)
		plt.xlim(leftmost, rightmost)
		



def compare(k):
	samples_002, labels_002 = get_samples(2, 100000, 0, load = False)
	samples_003, labels_003 = get_samples(3, 100000, 1, load = False)

	all_samples = np.concatenate((samples_002, samples_003))
	all_labels = np.concatenate((labels_002, labels_003))
	print(all_samples.shape)

	colors, color_names, _ = assign_colors(all_labels)
	print(color_names)

	'''y2 = tsne(all_samples.T)
	plot(y2, colors, k)'''

	all_histos(samples_002,samples_003)


	y = pca(all_samples, k)
	plot(y, colors, k, 'PCA')
	plt.show()
	
compare(2)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import features
from map_overlay import MapOverlay

def vis():
	fn = 'jpl-data/clipped-image.tif'
	img = cv2.imread(fn)
	bw_img = cv2.imread(fn, 0)
	myMap = MapOverlay(fn)
	h,w,c = img.shape

	
	norm = features.normalized(myMap.getMapData())
	blur = features.blurred(myMap.img)


	f1 = plt.figure('Colors', frameon = False)
	ax = plt.Axes(f1, [0., 0., 1., 1.])
	ax.set_axis_off()
	f1.add_axes(ax)

	r = norm[:,0].reshape(h,w)
	plt.subplot(231)
	plt.imshow(r, 'Reds')
	g = norm[:,1].reshape(h,w)
	plt.subplot(232)
	plt.imshow(g, 'Greens')
	b = norm[:,2].reshape(h,w)
	plt.subplot(233)
	plt.imshow(b, 'Blues')


	ave_r = blur[:,0].reshape(h,w)
	plt.subplot(234)
	plt.imshow(ave_r, 'Reds')
	ave_g = blur[:,1].reshape(h,w)
	plt.subplot(235)
	plt.imshow(ave_g, 'Greens')
	ave_b = blur[:,2].reshape(h,w)
	plt.subplot(236)
	plt.imshow(ave_b, 'Blues')

	f2 = plt.figure('Edges', frameon = False)
	ax = plt.Axes(f2, [0., 0., 1., 1.])
	ax.set_axis_off()
	f2.add_axes(ax)

	edge = features.edge_density(img, 100, amp = 1).reshape(h,w)
	plt.subplot(111)
	plt.imshow(edge, 'gray')
	

	f3 = plt.figure('Histogram of Gradients', frameon = False)
	ax = plt.Axes(f3, [0., 0., 1., 1.])
	ax.set_axis_off()
	f3.add_axes(ax)


	hog = features.hog(bw_img, 50)
	for i in range(hog.shape[1]):
		f3.add_subplot(4,4,i+1)
		plt.imshow(hog[:,i].reshape(h,w), 'gray')

	plt.show()


vis()
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.feature as feat
from skimage.filters.rank import windowed_histogram
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os

def edge_density(img, ksize, img_name = "", amp = 1):
	'''
	INPUT:
		- img:   (nxm ndarray) Gray-scale image
		- ksize: (int) Kernel size for window to consider
		- [amp]: (int) Optional value that amplifies the effects of more dense regions
	OUTPUT:
		- (n*m x 1 ndarray) Array containing labels related to texture of each pixel's surrounidng area
	'''
	p = os.path.join('features', "edge_{}_{}_{}.npy".format(ksize,amp,img_name))
	if os.path.exists(p):
		return np.load(p)
	else:
		new_img = img.copy()
		edges = cv2.Canny(img,50,100)
		density = cv2.blur(edges, ksize = (ksize,ksize), borderType = cv2.BORDER_REFLECT)/float(ksize)
		n = density.ravel().shape[0]
		res = density.ravel().reshape((n,1))**amp
		np.save(p, res)
		return res, np.array(['edges'])


def normalized(data, img_name = ""):
	'''
	INPUT: 
		- data: (ndarray) Data to be normalized
	OUTPUT:
		- (n*m ndarray) Normalized data with 0 mean and standard normal variance
	'''
	p = os.path.join('features', "norm_{}.npy".format(img_name))
	if os.path.exists(p):
		return np.load(p)
	else:
		res = (data - np.mean(data))/np.std(data)
		np.save(p, res)
		return res, np.array(['red', 'green', 'blue'])
	

def entr(bw_img, img_name = "", disk_size = 5):
	p = os.path.join('features', "entropy_{}_{}.npy".format(disk_size,img_name))
	if os.path.exists(p):
		return np.load(p)
	else:
		h,w = bw_img.shape
		print "starting entropy"
		entr_img = entropy(bw_img, disk(disk_size))
		print("done with entropy")
		plt.imshow(entr_img, cmap = 'gray')
		plt.show()
		res = entr_img.reshape((h*w,1))
		np.save(p,res)
		return res, np.array(['entropy'])



def mirror_border(img, border_width, border_height):
	return cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width, cv2.BORDER_REFLECT)




def GLCM(img, k, img_name = ""):
	mirror_img = mirror_border(img, k/2, k/2)
	print("done extending mirror")
	new_img = np.copy(img)
	for i in range(k/2,img.shape[0]):
		print(i)
		for j in range(k/2, img.shape[1]):
			new_window = mirror_img[i-k/2: i+k/2+1, j-k/2: j+k/2+1]
			glcm = greycomatrix(new_window, [1],[0], symmetric = True, normed = True)
			contrast = greycoprops(glcm, 'contrast')
			new_img[i-k/2,j] = contrast
	return new_img,np.array('GLCM')



def blurred(img, img_name = "", ksize = 101):
	'''
	INPUT:
		- img: (n x m x c ndarray) Image to be blurred
		- [ksize]: (int) Optional odd value representing kernel size.
					NOTE: It is intentionally initialized to a large value to better encompass
						  general color of surrounding area.
	OUTPUT:
		- (n*m x c ndarray) Ravelled image with gaussian blur applied
	'''
	p = os.path.join('features', "blurred_{}_{}.npy".format(ksize,img_name))
	if os.path.exists(p):
		return np.load(p)
	else:
		h,w,c = img.shape
		res = cv2.GaussianBlur(img,(ksize,ksize),0).reshape(h*w,c)
		np.save(p, res)
		return res, np.array(['ave_red','ave_green', 'ave_blue'])


def hog(bw_img, ksize, img_name = "", bins = 16):
	p = os.path.join('features', "hog_{}_{}_{}.npy".format(ksize,bins,img_name))
	if os.path.exists(p):
		return np.load(p)
	else:
		h,w = bw_img.shape
		selem = np.ones((ksize, ksize)).astype('float')
		gx = cv2.Sobel(bw_img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(bw_img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		angles = ang/2.0*np.pi*255

		hist = windowed_histogram(angles.astype('uint8'), selem, n_bins = bins)

		names = []
		for i in range(bins):
			names.append('hog {}'.format(i))

		res = hist.reshape(h*w, bins)
		np.save(p, res)
		return res, np.array(names)
		#print(hist.shape)
		#return np.std(np.sort(hist, axis = 2), axis = 2).reshape(h*w,1)

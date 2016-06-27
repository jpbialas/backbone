import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.feature as feat
from skimage.filters.rank import windowed_histogram
from skimage.feature import greycomatrix, greycoprops


def edge_density(img, ksize, amp = 1):
	'''
	INPUT:
		- img:   (nxm ndarray) Gray-scale image
		- ksize: (int) Kernel size for window to consider
		- [amp]: (int) Optional value that amplifies the effects of more dense regions
	OUTPUT:
		- (n*m x 1 ndarray) Array containing labels related to texture of each pixel's surrounidng area
	'''
	new_img = img.copy()

	edges = cv2.Canny(img,50,100)
	density = cv2.blur(edges, ksize = (ksize,ksize), borderType = cv2.BORDER_REFLECT)/float(ksize)
	n = density.ravel().shape[0]
	return density.ravel().reshape((n,1))**amp, np.array(['edges'])


def normalized(data):
	'''
	INPUT: 
		- data: (ndarray) Data to be normalized
	OUTPUT:
		- (n*m ndarray) Normalized data with 0 mean and standard normal variance
	'''
	return (data - np.mean(data))/np.std(data), np.array(['red', 'green', 'blue'])
	

def mirror_border(img, border_width, border_height):
	return cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width, cv2.BORDER_REFLECT)

def GLCM(img, k):
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
	return new_img



def blurred(img, ksize = 101):
	'''
	INPUT:
		- img: (n x m x c ndarray) Image to be blurred
		- [ksize]: (int) Optional odd value representing kernel size.
					NOTE: It is intentionally initialized to a large value to better encompass
						  general color of surrounding area.
	OUTPUT:
		- (n*m x c ndarray) Ravelled image with gaussian blur applied
	'''
	h,w,c = img.shape
	return cv2.GaussianBlur(img,(ksize,ksize),0).reshape(h*w,c), np.array(['ave_red','ave_green', 'ave_blue'])


def hog(bw_img, ksize, bins = 16):
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

	return hist.reshape(h*w, bins), np.array(names)
	#print(hist.shape)
	#return np.std(hist, axis = 2).reshape(h*w,1)

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.feature as feat
from skimage.filters.rank import windowed_histogram

'''
INPUT:
	- img:   (nxm ndarray) Gray-scale image
	- ksize: (int) Kernel size for window to consider
	- [amp]: (int) Optional value that amplifies the effects of more dense regions
OUTPUT:
	- (n*m x 1 ndarray) Array containing labels related to texture of each pixel's surrounidng area
'''
def edge_density(img, ksize, amp = 1):
	new_img = img.copy()

	edges = cv2.Canny(img,50,100)
	density = cv2.blur(edges, ksize = (ksize,ksize), borderType = cv2.BORDER_REFLECT)/float(ksize)
	n = density.ravel().shape[0]
	return density.ravel().reshape((n,1))**amp

'''
INPUT: 
	- data: (ndarray) Data to be normalized
OUTPUT:
	- (n*m ndarray) Normalized data with 0 mean and standard normal variance
'''
def normalized(data):
	return (data - np.mean(data))/np.std(data)


'''
INPUT:
	- img: (n x m x c ndarray) Image to be blurred
	- [ksize]: (int) Optional odd value representing kernel size.
				NOTE: It is intentionally initialized to a large value to better encompass
					  general color of surrounding area.
OUTPUT:
	- (n*m x c ndarray) Ravelled image with gaussian blur applied
'''
def blurred(img, ksize = 101):
	h,w,c = img.shape
	return cv2.GaussianBlur(img,(ksize,ksize),0).reshape(h*w,c)


def hog(bw_img, ksize):
	h,w = bw_img.shape
	selem = np.ones((ksize, ksize)).astype('float')
	gx = cv2.Sobel(bw_img, cv2.CV_32F, 1, 0)
	gy = cv2.Sobel(bw_img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(gx, gy)
	angles = (ang).astype('float')/(2.0*np.pi)
	hist = windowed_histogram(angles, selem, n_bins = 16)
	return hist.reshape(h*w, 16)
	#print(hist.shape)
	#return np.std(hist, axis = 2).reshape(h*w,1)

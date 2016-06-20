import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.feature as feat

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
	return (data - np.mean(img))/np.std(data)


'''
INPUT:
	- img: (n x m x c ndarray) Image to be blurred
	- [ksize]: (int) Optional odd value representing kernel size.
				NOTE: It is intentionally initialized to a large value to better encompass
					  general color of surrounding area.
OUTPUT:
	- (n*m x c ndarray) Ravelled image with gaussian blur applied
'''
def blurred(img, ksize = 21):
	h,w,c = img.shape
	return cv2.GaussianBlur(img,(21,21),0).reshape(h*w,c)


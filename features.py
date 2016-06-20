import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.feature as feat

def edge_density(img, ksize, downsample = 0, amp = 1):
	new_img = img.copy()
	for i in range(downsample):
		new_img = cv2.pyrDown(new_img)

	edges = cv2.Canny(img,50,100)
	density = cv2.blur(edges, ksize = (ksize,ksize), borderType = cv2.BORDER_REFLECT)/float(ksize)
	n = density.ravel().shape[0]
	return density.ravel().reshape((n,1))**amp

def normalized(img):
	return (img - np.mean(img))/np.std(img)

def blurred(img):
	h,w,c = img.shape
	return cv2.GaussianBlur(img,(21,21),0).reshape(h*w,c)


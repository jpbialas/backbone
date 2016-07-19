import pathos.multiprocessing as mp
import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import matplotlib.pyplot as plt

class multiT():
	def test(self, i):
		print("starting, ", i)
		for j in range(self.k/2, self.new_img.shape[1]):
			new_window = self.mirror_img[i-self.k/2: i+self.k/2+1, j-self.k/2: j+self.k/2+1]
			glcm = greycomatrix(new_window, [1],[0], symmetric = True, normed = True)
			contrast = greycoprops(glcm, 'contrast')
			self.new_img[i-self.k/2,j] = contrast
		print("ending, ", i)

	def GLCM(self):
		p = mp.ProcessingPool(4)
		p.map(self.test, range(self.k/2,self.img.shape[0]))

	def mirror_border(self, img, border_width, border_height):
		return cv2.copyMakeBorder(img, border_height, border_height, border_width, border_width, cv2.BORDER_REFLECT)

	def show_and_save(self):
		np.savetxt('results.csv', self.new_img)
		plt.imshow(self.new_img)
		plt.show()

	def __init__(self, fn, k):
		self.k = k
		self.img = cv2.imread(fn, 0)
		self.mirror_img = self.mirror_border(self.img, self.k/2, self.k/2)
		print("done extending mirror")
		self.new_img = np.copy(self.img)

if __name__ == '__main__':
	fn = 'jpl-data/clipped-image.tif'
	Test = multiT(fn,11)
	Test.GLCM()
	Test.show_and_save()

import analyzeResults
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mapOverlay import MapOverlay

def compare(fn, mask_fn, data_fn):
	myMap = MapOverlay(fn)
	myMap.newMask(mask_fn, 'damage')
	full_predict = np.loadtxt(data_fn, delimiter = ',')
	myMap.newPxMask(full_predict.ravel(), 'damage_pred')
	analyzeResults.side_by_side(myMap, 'damage', 'damage_pred')

compare('datafromjoe/1-0003-0002.tif','datafromjoe/1-003-002-damage.shp', 'predictions.csv')
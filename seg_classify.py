from map_overlay import MapOverlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import seg_features as sf
import matplotlib.pyplot as plt
import analyze_results


def visualize(img_num, seg_level):
	new_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
	new_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(img_num, img_num, seg_level), seg_level)
	data, names = sf.shapes(new_map, seg_level)

	fig = plt.figure('Features {}'.format(seg_level))
	fig.subplots_adjust(bottom=0, left = 0, right = 1, top = 1, wspace = 0, hspace = 0)
	for i in range(data.shape[1]):
		sf.visualize_segments(new_map, data[:,i], seg_level, names[i], 220+i+1)


def setup_segs(img_num, base_seg, segs, thresh):
	new_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
	new_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(img_num, img_num, base_seg), base_seg)
	for seg in segs:
		new_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(img_num, img_num, seg), seg)
	new_map.newMask('datafromjoe/1-003-00{}-damage.shp'.format(img_num), 'damage')
	data, names = sf.multi_segs(new_map, base_seg, segs)#sf.color_edge(new_map, base_seg)
	X = data[:,:-1]
	y = data[:,-1]>255*thresh	
	return new_map, X, y, names


def sample(y):
	zeros = np.where(y==0)[0]
	ones = np.where(y==1)[0]
	n0,n1 = zeros.shape[0], ones.shape[0]
	zero_samples = zeros[np.random.random_integers(0,n0-1, min(n0,n1))]
	one_samples = ones[np.random.random_integers(0, n1-1, min(n0,n1))]
	return np.concatenate((zero_samples, one_samples))


def full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees, verbose = True):
	CUSTOM = True
	EVEN = False

	model= RandomForestClassifier(n_estimators=n_trees, n_jobs = -1, verbose = verbose)
	samples = sample(y_train)

	if EVEN:
		model.fit(X_train[samples], y_train[samples])
	else:
		model.fit(X_train, y_train)

	ground_truth = y_test

	if CUSTOM:
		prediction = model.predict_proba(X_test)[:,1]
		full_predict = prediction[map_test.segmentations[base_seg][1].astype('int')]
		analyzeResults.probability_heat_map(map_test, full_predict)
		prediction = prediction>.4
		full_predict = (full_predict>.4).ravel()
		'''for i in range(10):
			next_predict = full_predict.ravel()>(i/10.0)
			print i/10.0
			print analyzeResults.prec_recall(ground_truth, next_predict)'''
	else:
		prediction = model.predict(X_test)
		full_predict = map_test.mask_segments(prediction, base_seg, False)
		map_test.newPxMask(full_predict.ravel(), 'damage_pred')
		analyzeResults.side_by_side(map_test, 'damage', 'damage_pred')

	analyzeResults.feature_importance(model, names, X_train)
	print full_predict.shape, ground_truth.shape
	print "pred",analyzeResults.prec_recall(map_test.getLabels('damage'), full_predict)
	print "truth",analyzeResults.prec_recall(map_test.getLabels('damage'), map_test.mask_segments(y_test, base_seg, False))
	

def test2(n_trees = 500, base_seg = 50, segs = [100, 400], thresh = .5):
	map_train, X_train, y_train, names = setup_segs(2, base_seg, segs, thresh)
	map_test, X_test, y_test, _ = setup_segs(3, base_seg, segs,  thresh)
	
	print("2,3")
	full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees)
	print("3,2")
	full_run(map_test, X_test, y_test, map_train, X_train, y_train, names, base_seg, n_trees)
	

if __name__ == '__main__':
	#cache_all()
	#for i in range(10):
	#	print(i/10.0)
	test2()
	plt.show()
	'''for i in [2,3]:
		for j in [50,100,200,400,1000]:
			visualize(i, j)
			plt.show(block = False)'''
	#load_all()
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


def setup_segs(img_num, base_seg, segs, thresh, jared = False, new_feats = True):
	new_map = MapOverlay('datafromjoe/1-0003-000{}.tif'.format(img_num))
	new_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(img_num, img_num, base_seg), base_seg)
	for seg in segs:
		new_map.new_segmentation('segmentations/withfeatures{}/shapefilewithfeatures003-00{}-{}.shp'.format(img_num, img_num, seg), seg)
	
	#data, names = sf.multi_segs(new_map, base_seg, segs, new_feats)#sf.color_edge(new_map, base_seg)
	
	if jared:
		jared_load = np.loadtxt('jaredlabels/{}.csv'.format(img_num), delimiter = ',', dtype = 'int')
		new_map.new_seg_mask(jared_load, base_seg, 'damage')
		data, names = sf.multi_segs(new_map, base_seg, segs, new_feats, True)
		X = data[:,:-1]
		y = np.zeros(X.shape[0])
		y[jared_load] = 1
	else:
		new_map.newMask('datafromjoe/1-003-00{}-damage.shp'.format(img_num), 'damage')
		data, names = sf.multi_segs(new_map, base_seg, segs, new_feats, True)
		X = data[:,:-1]
		y = data[:,-1]>255*thresh	

	return new_map, X, y, names


def sample(y, n_samples = -1):
	zeros = np.where(y==0)[0]
	ones = np.where(y==1)[0]
	n0,n1 = zeros.shape[0], ones.shape[0]
	if n_samples == -1:
		zero_samples = np.random.choice(zeros, min(n0, n1))#zeros[np.random.random_integers(0,n0-1, min(n0,n1))]
		one_samples = np.random.choice(ones, min(n0, n1))#ones[np.random.random_integers(0, n1-1, min(n0,n1))]
	else:
		zero_samples = np.random.choice(zeros, n_samples//2)#zeros[np.random.random_integers(0,n0-1, n_samples//2)]
		one_samples = np.random.choice(ones, n_samples//2)#ones[np.random.random_integers(0, n1-1, n_samples//2)]

	return np.concatenate((zero_samples, one_samples))


def full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees, jared, new_feats, EVEN, verbose = True):
	model= RandomForestClassifier(n_estimators=n_trees, n_jobs = -1, verbose = verbose)#, class_weight = "balanced")
	samples = sample(y_train)
	img_num = map_test.name[-1]

	if EVEN:
		model.fit(X_train[samples], y_train[samples])
		thresh = .7
	else:
		thresh = .4
		model.fit(X_train, y_train)

	label_name = "Jared" if jared else "Joe"
	model_name = analyze_results.gen_model_name("Segs", label_name, EVEN, img_num, new_feats)

	prediction_prob = model.predict_proba(X_test)[:,1]
	full_predict = prediction_prob[map_test.segmentations[base_seg][1].astype('int')]
	heat_fig = analyze_results.probability_heat_map(map_test, full_predict, model_name, save = True)
	analyze_results.ROC(map_test, map_test.getLabels('damage'), full_predict, model_name, save = True)


	prediction = prediction_prob>thresh
	full_predict = (full_predict>thresh).ravel()

	map_test.newPxMask(full_predict.ravel(), 'damage_pred')
	sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred', model_name, save = True)
	

	#analyze_results.feature_importance(model, names, X_train)
	##print full_predict.shape, y_test.shape
	##print "pred",analyze_results.prec_recall(map_test.getLabels('damage'), full_predict)
	##print analyze_results.confusion_analytics(map_test.getLabels('damage'), full_predict)
	return full_predict
	

def test(n_trees = 1000, base_seg = 50, segs = [100], thresh = .01, jared = True, new_feats=True, EVEN = True):
	map_train, X_train, y_train, names = setup_segs(3, base_seg, segs, thresh, jared, new_feats)
	map_test, X_test, y_test, _ = setup_segs(2, base_seg, segs,  thresh, jared, new_feats)

	full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees, jared, new_feats, EVEN)
	full_run(map_test, X_test, y_test, map_train, X_train, y_train, names, base_seg, n_trees, jared, new_feats, EVEN)

def compare(n_trees = 1000, base_seg = 50, segs = [100], thresh = .5, jared = True, new_feats=True, EVEN = True):
	jared = True
	map_train, X_train, y_train, names = setup_segs(3, base_seg, segs, thresh, jared)
	map_test, X_test, y_test, _ = setup_segs(2, base_seg, segs,  thresh, jared)
	
	print("jared")
	pred1 = full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees, jared, new_feats, EVEN)

	jared = False
	map_train, X_train, y_train, names = setup_segs(3, base_seg, segs, thresh, jared)
	map_test, X_test, y_test, _ = setup_segs(2, base_seg, segs,  thresh, jared)

	print('joe')
	pred2 = full_run(map_train, X_train, y_train, map_test, X_test, y_test, names, base_seg, n_trees, jared, new_feats, EVEN)

	#pred2 = full_run(map_test, X_test, y_test, map_train, X_train, y_train, names, base_seg, n_trees, jared, new_feats, EVEN)

	plt.figure('Difference')
	diff = pred1-pred2
	plt.imshow(diff, cmap = 'seismic', norm = plt.Normalize(-1,1))
	plt.show()


if __name__ == '__main__':
	test(jared = True, EVEN = True, new_feats = False)
	test(jared = True, EVEN = False, new_feats = False)
	test(jared = False, EVEN = True, new_feats = False)
	test(jared = False, EVEN = False, new_feats = False)
	
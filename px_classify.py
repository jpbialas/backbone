from map_overlay import MapOverlay
import map_overlay
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import px_features
import matplotlib.pyplot as plt
import analyze_results
import os
import progressbar
from convenience_tools import *

def sample(labels, nsamples):
	'''
	NOW UNUSED
	INPUT: 
		- labels:   (nx1 ndarray) Array containing binary labels
		- nsamples: (int) Value representingtotal number of indices to be sampled
						NOTE: (If odd, produces list of length (nsamples-1))
		- [used]:   (sx1 ndarray) Array containing indices not to include in sampling
	OUTPUT:
		- Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
			nsamples/2 indices have value 0
	'''
	n = labels.shape[0]
	zeros = np.where(labels == 0)[0]
	n_zeros = np.shape(zeros)[0]

	ones = np.where(labels == 1)[0]
	n_ones = np.shape(ones)[0]

	zero_samples = zeros[np.random.random_integers(0,n_zeros-1, nsamples/2)]
	one_samples = ones[np.random.random_integers(0, n_ones-1, nsamples/2)]
	final_set = np.concatenate((zero_samples, one_samples))

	return final_set

def sample_split(labels, nsamples_train, nsamples_test):
	'''
	NOW UNUSED
	INPUT: 
		- labels:   (nx1 ndarray) Array containing binary labels
		- nsamples: (int) Value representingtotal number of indices to be sampled
						NOTE: (If odd, produces list of length (nsamples-1))
		- [used]:   (sx1 ndarray) Array containing indices not to include in sampling
	OUTPUT:
		- Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
			nsamples/2 indices have value 0
	'''
	n = labels.shape[0]
	train = labels[:n/2]
	train_set = sample(train, nsamples_train)
	test_set = np.random.random_integers(n/2, n-1, nsamples_test)
	return train_set,test_set


def gen_features(myMap, edge_k, hog_k, hog_bins):
	'''
	input:
		- myMap: MapObject
	output:
		- feature representation of map
	'''
	#entropy, entropy_name = px_features.entr(myMap.bw_img, img_name = myMap.name)

	#glcm, glcm_name = px_features.GLCM(myMap.bw_img, 50, img_name = myMap.name)
	rgb, rgb_name = px_features.normalized(myMap.getMapData(), img_name = myMap.name)
	ave_rgb, ave_rgb_name = px_features.blurred(myMap.img, img_name = myMap.name)
	edges, edges_name = px_features.edge_density(myMap.bw_img, edge_k, img_name = myMap.name, amp = 1)
	hog, hog_name = px_features.hog(myMap.bw_img, hog_k, img_name = myMap.name)
	v_print('Concatenating', False)
	data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
	names = np.concatenate((rgb_name, ave_rgb_name, edges_name, hog_name))
	v_print('Done Concatenating', False)
	return data, names



def train_and_test(map_train, map_test, mask_train = 'damage', mask_test = 'damage', frac_train = 0.01, frac_test = 0.01, edge_k = 100, hog_k = 50, nbins = 16, n_trees = 85, EVEN = True, verbose = True):
	model, labels, _ = train(map_train, mask_train,frac_train,  edge_k, hog_k, nbins, n_trees, EVEN, verbose)
	return test(map_test, model, labels, mask_test, frac_test, edge_k, hog_k, nbins, n_trees, EVEN, verbose)

def test(map_test, model, labels, mask_test = 'damage', frac_test = 0.01, edge_k = 100, hog_k = 50, nbins = 16, n_trees = 85, EVEN = True, verbose = True):
	v_print('Starting test gen', verbose)
	X_test, labels = gen_features(map_test, edge_k, hog_k, nbins)
	v_print('Done test gen', verbose)
	y_test = map_test.getLabels(mask_test)
	n_test = y_test.shape[0]
	if frac_test<1:
		test = np.random.random_integers(0,y_test.shape[0]-1, int(n_test*frac_test))
	else:
		test = np.arange(n_test)

	ground_truth = y_test[test]
	v_print('Starting Predction', verbose)
	prediction_prob = model.predict_proba(X_test[test])[:,1]
	v_print("Done with Prediction", verbose)
	np.save(os.path.join('temp', "prediction_{}.npy".format(map_test.name)), prediction_prob)
	if frac_test == 1:
		heat_fig = analyze_results.probability_heat_map(map_test, prediction_prob)

		heat_fig.savefig('../../Compare Methods/px_heatmap_{}.png'.format(map_test.name), format='png', dpi = 2400)

		roc_name = 'Px Even' if EVEN else 'Px'
		analyze_results.ROC(map_test, map_test.getLabels('damage'), prediction_prob, roc_name)


		map_test.newPxMask(prediction_prob.ravel()>.4, 'damage_pred')
		sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred')
		sbs_fig.savefig('../../Compare Methods/px_sbs_{}.png'.format(map_test.name), format='png', dpi = 2400)
	else:
		v_print("Done Testing", verbose)
		prediction = prediction_prob>.4
		print analyze_results.prec_recall(ground_truth, prediction)

	return model, X_test, y_test, labels	

def train(map_train, mask_train = 'damage',frac_train = 0.01,  edge_k = 100, hog_k = 50, nbins = 16, n_trees = 85, EVEN = True, verbose = True):
	v_print('Starting train gen', verbose)
	X_train, labels = gen_features(map_train, edge_k, hog_k, nbins)
	v_print('Done train gen', verbose)

	y_train =  map_train.getLabels(mask_train)
	n_train = y_train.shape[0]

	if EVEN:
		train = sample(y_train, int(n_train*frac_train))
	else:
		train = np.random.random_integers(0,y_train.shape[0]-1, int(n_train*frac_train))

	v_print("Starting Modelling", verbose)
	model= RandomForestClassifier(n_estimators=n_trees, n_jobs = -1, verbose = verbose)
	model.fit(X_train[train], y_train[train])
	v_print("Done Modelling", verbose)

	return model, labels, X_train


def fit_to_segs(map_test, segs = 50, probabilities = None):
	segs = map_test.segmentations[segs][1].ravel()
	ground_truth = map_test.getLabels('damage')
	if probabilities is None:
		probabilities = np.load(os.path.join('temp', "prediction_{}.npy".format(map_test.name)))
	pbar = custom_progress()
	n_segs = int(np.max(segs))+1
	data = np.zeros((n_segs), dtype = 'float')
	for i in pbar(range(n_segs)):
		indices = np.where(segs == i)[0]
		data.itemset(i, np.sum(probabilities[indices], axis = 0)/indices.shape[0])
	probs = data[segs.reshape(map_test.rows, map_test.cols).astype('int')]
	analyze_results.probabilty_heat_map(map_test, probs)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.1)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.15)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.2)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.25)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.3)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.35)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.4)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.45)
	print analyze_results.prec_recall(ground_truth, probs.ravel()>.5)
	

def main_test():
	map_train, map_test = map_overlay.basic_setup([50], 50, jared = False)
	#fit_to_segs(map_test)
	#fit_to_segs(map_train)
	#plt.show()

	model, X, y1, names = train_and_test(map_train, map_test, frac_test = 1, EVEN = False, verbose = True)
	analyze_results.feature_importance(model, names, X)


	model, X, y2, names = train_and_test(map_test, map_train, frac_test = 1, EVEN = False, verbose = True)
	analyze_results.feature_importance(model, names, X)

	#plt.show()

	plt.show()

if __name__ == "__main__":
	main_test()


	


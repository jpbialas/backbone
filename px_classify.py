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

def sample(labels, nsamples, EVEN):
	'''
	INPUT: 
		- labels:   (nx1 ndarray) Array containing binary labels
		- nsamples: (int) Value representingtotal number of indices to be sampled
						NOTE: (If odd, produces list of length (nsamples-1))
		- EVEN:    (boolean) True if indices should be evenly sampled, false otherwise
	OUTPUT:
		- Returns random list of indices from labels such that nsamples/2 of the indices have value 1 and 
			nsamples/2 indices have value 0
	'''
	if EVEN:
		n = labels.shape[0]
		zeros = np.where(labels == 0)[0]
		n_zeros = np.shape(zeros)[0]
		ones = np.where(labels == 1)[0]
		n_ones = np.shape(ones)[0]
		zero_samples = np.random.choice(zeros, nsamples/2)
		one_samples = np.random.choice(ones, nsamples/2)
		final_set = np.concatenate((zero_samples, one_samples))
	else:
		final_set = np.random.random_integers(0,y_train.shape[0]-1, int(n_train*params['frac_train']))
	return final_set


def gen_features(myMap, params):
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
	edges, edges_name = px_features.edge_density(myMap.bw_img, params['edge_k'], img_name = myMap.name, amp = 1)
	hog, hog_name = px_features.hog(myMap.bw_img, params['hog_k'], img_name = myMap.name, bins = params['nbins'])
	v_print('Concatenating', False)
	data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
	names = np.concatenate((rgb_name, ave_rgb_name, edges_name, hog_name))
	v_print('Done Concatenating', False)
	return data, names


def testing_suite(map_test, prediction_prob, model_name):
	heat_fig = analyze_results.probability_heat_map(map_test, prediction_prob, model_name, save = True)
	analyze_results.ROC(map_test, map_test.getLabels('damage'), prediction_prob, model_name, save = True)
	map_test.newPxMask(prediction_prob.ravel()>.4, 'damage_pred')
	sbs_fig = analyze_results.side_by_side(map_test, 'damage', 'damage_pred', model_name, True)


def train(map_train, params, verbose = True):
	v_print('Starting train gen', verbose)
	X_train, feat_names = gen_features(map_train, params)
	v_print('Done train gen', verbose)
	y_train =  map_train.getLabels(params['mask_train'])
	n_train = y_train.shape[0]
	train = sample(y_train, int(n_train*params['frac_train']), params['EVEN'])
	v_print("Starting Modelling", verbose)
	model= RandomForestClassifier(n_estimators=params['n_trees'], n_jobs = -1, verbose = verbose)
	model.fit(X_train[train], y_train[train])
	v_print("Done Modelling", verbose)
	return model, X_train



def test(map_test, model, params, jared = True,  verbose = True):
	v_print('Starting test gen', verbose)
	X_test, feat_names = gen_features(map_test, params)
	v_print('Done test gen', verbose)
	y_test = map_test.getLabels(params['mask_test'])
	n_test = y_test.shape[0]
	if params['frac_test']<1:
		test = np.random.random_integers(0,y_test.shape[0]-1, int(n_test*params['frac_test']))
	else:
		test = np.arange(n_test)
	img_num = map_test.name[-1]
	label_name = "Jared" if jared else "Joe"
	model_name = analyze_results.gen_model_name("Px", label_name, params['EVEN'], img_num, False)
	ground_truth = y_test[test]
	v_print('Starting Predction', verbose)
	prediction_prob = model.predict_proba(X_test[test])[:,1]
	v_print("Done with Prediction", verbose)
	if params['frac_test'] == 1:
		np.save('PXpredictions/'+model_name+'_probs.npy', prediction_prob)
		testing_suite(map_test, prediction_prob, model_name)
	else:
		v_print("Done Testing", verbose)
		prediction = prediction_prob>.4
		v_print (analyze_results.prec_recall(ground_truth, prediction), verbose)
	return model, X_test, prediction_prob

def train_and_test(map_train, map_test, params, jared = True, verbose = True):

	label_name = "Jared" if jared else "Joe"
	img_num = map_test.name[-1]
	model_name = analyze_results.gen_model_name("Px", label_name, params['EVEN'], img_num, False)
	p = 'PXpredictions/'+model_name+'_probs.npy'
	if os.path.exists(p) and False:
		v_print ("loading results", verbose)
		prediction = np.load(p)
	else:
		v_print ("Running Tests", verbose)
		model, X_train = train(map_train, params, verbose)
		model, X_test, prediction = test(map_test, model, params, jared, verbose)
	return prediction


def main_test_new(params, EVEN, jared):
	map_test, map_train = map_overlay.basic_setup([100], 50, jared)
	
	params['EVEN'] = EVEN

	pred1 = train_and_test(map_train, map_test, params, jared = True, verbose = True)
	pred2 = train_and_test(map_train, map_test, params, jared = False, verbose = True)

	plt.figure('Difference')
	diff = (pred1-pred2).reshape((map_test.rows, map_test.cols))
	plt.imshow(diff, cmap = 'seismic', norm = plt.Normalize(-1,1))
	plt.show()
	

def main_test(params, EVEN, jared):
	map_train, map_test = map_overlay.basic_setup([100], 50, jared)
	params['EVEN'] = EVEN
	train_and_test(map_train, map_test, params, jared = jared, verbose = True)
	train_and_test(map_test, map_train, params, jared = jared, verbose = True)
	plt.show()


if __name__ == "__main__":

	params = {
		'frac_train': 0.01,
		'frac_test' : 1,
		'mask_train' : 'damage',
		'mask_test' : 'damage',
		'edge_k' : 100,
		'hog_k' : 50,
		'nbins' : 16,
		'n_trees' : 85,
		'EVEN' : True
	}
	print('1')
	main_test(params, EVEN  = True, jared = True)
	plt.show()
	'''print('2')
	main_test(params, EVEN = True, jared = False)
	print('3')
	main_test(params, EVEN = False, jared = True)
	print('4')
	main_test(params, EVEN = False, jared = False)'''


	


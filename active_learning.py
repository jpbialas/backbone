import numpy as np
import px_features
import analyze_results
import map_overlay
from convenience_tools import *
import os
import px_classify
from sklearn.ensemble import RandomForestClassifier


def bootstrap(L, k):
	''' Returns k random samples of L of size |L| with replacement
	INPUT:
		L: (ndarray) 
		k: (int) number of backbone iterations to run
	'''
	n = L.shape[0]
	return np.random.choice(L, (k, n), replace = True)


def uncertainty(X, y, k, m, frac_test = 1.0, verbose = True):
	'''Computes uncertainty for all unlabelled points as described in Mozafari 3.2 
	and returnst the m indices of the most uncertain
	NOTE: Smaller values are more certain b/c they have smaller variance
	Input:
		X: (ndarray) Contains all data
		y: (ndarray) Contains labels, 1 for damage, 0 for not damaged, -1 for unlabelled
		k: (int) number of backbone iterations to run
		m: (int) number of new data to select for labelling
	'''
	unlabelled = np.where(y < 0)[0]
	unlabelled = np.random.choice(unlabelled, frac_test*unlabelled.shape[0], replace = False)
	L = np.where(y > 0)[0]
	samples = bootstrap(L, k)
	uncertainties = np.zeros(unlabelled.shape[0])
	v_print("Staring model loop", verbose)
	pbar = custom_progress()
	for row in samples:
		next_model = RandomForestClassifier(n_estimators = 85, n_jobs = -1, verbose = verbose)
		next_model.fit(X[row], y[row])
		print uncertainties.shape
		next_prediction = next_model.predict(X[unlabelled])
		print next_prediction.shape
		uncertainties += next_prediction
	uncertainties/=k
	print uncertainties
	print np.sum(uncertainties)
	return np.argsort(uncertainties)[-m:]

def trivial_error(X, y, m):
	'''Computes uncertainty from Random Forest's probability metric and returns the m indices
	of the most uncertain
	NOTE: This metric is the distance from 50% So larger values are more certain
	Input:
		X: (ndarray) Contains all data
		y: (ndarray) Contains labels, 1 for damage, 0 for not damaged, -1 for unlabelled
		m: (int) number of new data to select for labelling
	'''
	unlabelled = np.where(y < 0)[0]
	unlabelled = np.random.choice(unlabelled, frac_test*unlabelled.shape[0], replace = False)
	L = np.where(y > 0)[0]
	model = RandomForestClassifier(n_estimators = 85, n_jobs = -1, verbose = verbose)
	model.fit(X[L], y[L])
	uncertainties = model.predict_proba(X[unlabelled])[:,1]
	uncertainties = np.abs(uncertainties-.5)/.5
	return np.argsort(uncertainties)[:m]

def test_progress(X_train, X_test, y_train, y_test, indices, verbose = True):
	v_print("testing progress", verbose)
	model = RandomForestClassifier(n_estimators = 85, n_jobs = -1, verbose = verbose)
	model.fit(X_train[indices], y_train[indices])
	test = np.random.random_integers(0,y_test.shape[0]-1, int(y_test.shape[0]*.01))
	results = model.predict(X_test[test])
	print analyze_results.prec_recall(map_test.getLabels('damage')[test], results[test])


def run_active_learning(map_train, map_test, start_n, step_n, verbose = True):
	v_print('generating features', verbose)
	X_train, _ = px_classify.gen_features(map_train, 100, 50, 16)
	#np.save(os.path.join('temp', "train.npy"), X_train)
	X_test, _ = px_classify.gen_features(map_test, 100, 50, 16)
	#np.save(os.path.join("temp","test.npy"), X_test)

	v_print('done generating features', verbose)
	y_train = map_train.getLabels('damage')
	sample = np.random.random_integers(0,y_train.shape[0]-1, start_n)
	y = np.ones_like(y_train)*-1
	y[sample] = y_train[sample]
	test_progress(X_train, X_test, y_train, y_test, sample)
	v_print('starting uncertainty', verbose)
	for i in range(10):
		next_indices = uncertainty(X_train, y, 10, step_n, frac_test = .1)
		sample = sample.concatenate(sample, next_indices)
		test_progress(X_train, X_test, y_train, y_test, sample)
		y = np.ones_like(y_train)*-1
		y[sample] = y_train[sample]


if __name__ == '__main__':
	map_train, map_test = map_overlay.basic_setup()
	print 'starting active learning'
	run_active_learning(map_train, map_test, 10000, 10000)










import numpy as np
import matplotlib.pyplot as plt
import px_features
import analyze_results
import map_overlay
from convenience_tools import *
import os
import px_classify
import seg_classify as sc
from sklearn.ensemble import RandomForestClassifier


def bootstrap(L, k):
	''' Returns k random samples of L of size |L| with replacement
	INPUT:
		L: (ndarray) 
		k: (int) number of backbone iterations to run
	'''
	return np.random.choice(L, (k, L.shape[0]), replace = True)

def strawman_error(X, y, m):
	unlabelled = np.where(y < 0)[0]
	return np.random.choice(unlabelled, m, replace = False)


def uncertainty(data, y, k, m, frac_test = 1.0, verbose = True):
	'''Computes uncertainty for all unlabelled points as described in Mozafari 3.2 
	and returnst the m indices of the most uncertain
	NOTE: Smaller values are more certain b/c they have smaller variance
	Input:
		X: (ndarray) Contains all data
		y: (ndarray) Contains labels, 1 for damage, 0 for not damaged, -1 for unlabelled
		k: (int) number of backbone iterations to run
		m: (int) number of new data to select for labelling
	'''
	U = np.where(y < 0)[0]
	U = np.random.choice(U, frac_test*U.shape[0], replace = False)
	data_U =  np.take(data, U, axis = 0)
	L = np.where(y > 0)[0]
	samples = bootstrap(L, k)
	X_U = np.zeros(U.shape[0])
	v_print("Staring model loop", verbose)
	pbar = custom_progress()
	for row in pbar(samples):
		next_model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
		next_model.fit(data[row], y[row])
		next_prediction = next_model.predict(data_U)
		X_U += next_prediction
	X_U/=k
	uncertainties=X_U*(1-X_U)
	return U[np.argsort(uncertainties)[-m:]]

def trivial_error(X, y, m, frac_test = 1.0):
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
	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = verbose)
	model.fit(X[L], y[L])
	uncertainties = model.predict_proba(X[unlabelled])[:,1]
	uncertainties = np.abs(uncertainties-.5)/.5
	return unlabelled[np.argsort(uncertainties)[:m]]


def test_seg_progress(map_test, base_seg, X_train, X_test, y_train, indices, precision, recall, f1, verbose = True):
	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
	model.fit(X_train[indices], y_train[indices])
	prediction = model.predict(X_test)
	full_predict = map_test.mask_segments(prediction, base_seg, False)
	map_test.newPxMask(full_predict.ravel(), 'damage_pred')
	prec, rec, _, _f1 = analyze_results.prec_recall(map_test.getLabels('damage'), full_predict)
	precision.append(prec)
	recall.append(rec)
	f1.append(_f1)

def run_active_learning_seg(start_n=1000, step_n=100, boot_n = 100,  n_updates = 100, verbose = True):
	base_seg = 50
	X_axis = [0]
	map_train, X_train, y_train, names = sc.setup_segs(2, base_seg, [100, 400],.5)
	map_test, X_test, y_test, _ = sc.setup_segs(3,base_seg, [100,400],  .5)


	sample = np.random.random_integers(0,y_train.shape[0]-1, start_n)
	random_sample = sample.copy()

	y = np.ones_like(y_train)*-1
	y[sample] = y_train[sample]
	precision, recall, f1 = [],[],[]
	prec_r, rec_r, f1_r = [],[],[]
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, precision, recall, f1)
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, prec_r, rec_r, f1_r)
	
	plt.ion()
	graph_prec = plt.plot(X_axis, precision, label = 'Precision')[0]
	graph_rec = plt.plot(X_axis, recall, label = 'Recall')[0]
	graph_f1 = plt.plot(X_axis, f1, label = 'F1')[0]
	graph_p_r = plt.plot(X_axis, prec_r, label = 'Prec Comparison')[0]
	graph_r_r = plt.plot(X_axis, rec_r, label = 'Rec Comparison')[0]
	graph_f_r = plt.plot(X_axis, f1_r, label = 'F1 Comparison')[0]
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#   ncol=4, mode="expand", borderaxespad=0.)
	plt.axis([0, n_updates, 0, 1])
	plt.legend()
	plt.draw()
	plt.pause(0.01)

	v_print('starting uncertainty', verbose)
	for i in range(1, n_updates):
		X_axis.append(i)

		print "there are", np.where(y < 0)[0].shape, "unlabelled"
		#next_indices = uncertainty(X_train, y, boot_n, step_n)
		next_indices = trivial_error(X_train, y, step_n)
		sample = np.concatenate((sample, next_indices))
		random_sample = np.concatenate((random_sample, strawman_error(X_train, y, boot_n)))
		test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, precision, recall, f1)
		test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, prec_r, rec_r, f1_r)
		graph_prec.set_ydata(precision)
		graph_rec.set_ydata(recall)
		graph_f1.set_ydata(f1)
		graph_p_r.set_ydata(prec_r)
		graph_r_r.set_ydata(rec_r)
		graph_f_r.set_ydata(f1_r)
		graph_prec.set_xdata(X_axis)
		graph_rec.set_xdata(X_axis)
		graph_f1.set_xdata(X_axis)
		graph_p_r.set_xdata(X_axis)
		graph_r_r.set_xdata(X_axis)
		graph_f_r.set_xdata(X_axis)
		plt.draw()
		plt.pause(0.01)
		y = np.ones_like(y_train)*-1
		y[sample] = y_train[sample]

	plt.waitforbuttonpress()

if __name__ == '__main__':
	#map_train, map_test = map_overlay.basic_setup()
	print 'starting active learning'
	#run_active_learning(map_train, map_test, 100000, 100000)
	run_active_learning_seg()









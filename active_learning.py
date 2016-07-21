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
	L = np.where(y >= 0)[0]
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

def trivial_error(X, y, m, frac_test = 1.0, verbose = True):
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
	L = np.where(y >= 0)[0]
	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
	model.fit(X[L], y[L])
	predictions = model.predict_proba(X[unlabelled])
	if predictions.shape[1]>1:
		uncertainties = predictions[:,1]
		uncertainties = np.abs(uncertainties-.5)/.5
		return unlabelled[np.argsort(uncertainties)[:m]]
	else:
		return unlabelled[:m]


def test_seg_progress(map_test, base_seg, X_train, X_test, y_train, indices, precision, recall, f1, verbose = True):
	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
	model.fit(X_train[indices], y_train[indices])
	prediction = model.predict_proba(X_test)[:,1]>0.4
	full_predict = map_test.mask_segments(prediction, base_seg, False)
	map_test.newPxMask(full_predict.ravel(), 'damage_pred')
	prec, rec, _, _f1 = analyze_results.prec_recall(map_test.getLabels('damage'), full_predict)
	precision.append(prec)
	recall.append(rec)
	f1.append(_f1)

def run_active_learning_seg(start_n=100, step_n=100, boot_n = 100,  n_updates = 100, method = "UNCERT", train = 3, verbose = True):
	base_seg = 50

	if method == "UNCERT":
		legend_name = 'Bootstrap Uncertainty'
	elif method == "Forest":
		legend_name = 'Random Forest Probability'

	if train == 2:
		map_train, X_train, y_train, names = sc.setup_segs(2, base_seg, [100, 400],.5)
		map_test, X_test, y_test, _ = sc.setup_segs(3,base_seg, [100,400],  .5)
	else:
		map_train, X_train, y_train, names = sc.setup_segs(3, base_seg, [100, 400],.5)
		map_test, X_test, y_test, _ = sc.setup_segs(2,base_seg, [100,400],  .5)

	
	X_axis = [start_n]
	map_train, X_train, y_train, names = sc.setup_segs(3, base_seg, [100, 400],.5)
	map_test, X_test, y_test, _ = sc.setup_segs(2,base_seg, [100,400],  .5)

	sample = np.random.choice(np.arange(y_train.shape[0]), start_n, replace = False)
	random_sample = sample.copy()

	y = np.ones_like(y_train)*-1
	y[sample] = y_train[sample]
	precision, recall, f1 = [],[],[]
	prec_r, rec_r, f1_r = [],[],[]
	full_prec, full_rec, full_f = [],[],[]
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, precision, recall, f1)
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, prec_r, rec_r, f1_r)
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, np.arange(y_train.shape[0]), full_prec, full_rec, full_f)
	
	plt.ion()
	prec_comparisons = plt.figure('Precision')
	graph_prec = plt.plot(X_axis, precision, 'r-', label = legend_name)[0]
	graph_p_r = plt.plot(X_axis, prec_r, 'r--', label = 'Random Selection')[0]
	plt.axhline(full_prec[0], color = 'gray', label = 'Full Labelling')
	plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
	plt.legend()
	plt.title('Precision AL Comparison')
	plt.xlabel('Number of Labelled Samples')
	plt.ylabel('Precision (%)')

	rec_comparisons = plt.figure('Recall')
	graph_rec = plt.plot(X_axis, recall, 'g-',label = legend_name)[0]
	graph_r_r = plt.plot(X_axis, rec_r, 'g--', label = 'Random Selection')[0]
	plt.axhline(full_rec[0], color = 'gray', label = 'Full Labelling')
	plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
	plt.legend()
	plt.title('Recall AL Comparison')
	plt.xlabel('Number of Labelled Samples')
	plt.ylabel('Recall (%)')

	F1_comparisons = plt.figure('F1')
	graph_f1 = plt.plot(X_axis, f1, 'b-', label = legend_name)[0]
	graph_f_r = plt.plot(X_axis, f1_r, 'b--', label = 'Random Selection')[0]
	plt.axhline(full_f[0], color = 'gray', label = 'Full Labelling')
	plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
	plt.legend()
	plt.title('F1 AL Comparison')
	plt.xlabel('Number of Labelled Samples')
	plt.ylabel('F1 (%)')

	plt.draw()
	plt.pause(0.05)

	v_print('starting uncertainty', verbose)
	for i in range(1, n_updates):
		X_axis.append(start_n+i*step_n)

		print "there are", np.where(y >= 0)[0].shape[0], "labelled points"
		next_indices = uncertainty(X_train, y, boot_n, step_n)
		#next_indices = trivial_error(X_train, y, step_n)
		sample = np.concatenate((sample, next_indices))
		y[sample] = y_train[sample]
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
		plt.pause(0.05)
		prec_comparisons.savefig('temp/Precision {} {}.png'.format(legend_name, train))
		rec_comparisons.savefig('temp/Recall {} {}.png'.format(legend_name, train))
		F1_comparisons.savefig('temp/F1 {} {}.png'.format(legend_name, train))

	plt.waitforbuttonpress()

if __name__ == '__main__':
	#map_train, map_test = map_overlay.basic_setup()
	print 'starting active learning'
	#run_active_learning(map_train, map_test, 100000, 100000)
	run_active_learning_seg()









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


def test_seg_progress(map_test, base_seg, X_train, X_test, y_train, indices, FPRs, FNRs, Confs, verbose = True):
	'''
	Trains X_train on y_train and tests on X_test. Adds resulting FPR, FNR, and Conf to FPRs, FNRs, and Confs
	'''
	model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, verbose = 0)
	model.fit(X_train[indices], y_train[indices])
	prediction = model.predict_proba(X_test)[:,1]>0.4
	full_predict = map_test.mask_segments(prediction, base_seg, False)
	map_test.newPxMask(full_predict.ravel(), 'damage_pred')
	FPR, FNR, conf = analyze_results.confusion_analytics(map_test.getLabels('damage'), full_predict)
	FPRs.append(FPR)
	FNRs.append(FNR)
	Confs.append(conf)

def run_active_learning_seg(start_n=100, step_n=100, boot_n = 100,  n_updates = 100, method = "UNCERT", train = 3, verbose = True):
	'''
	Runs active learning on train, and tests on the other map. Starts with start_n labels, and increments by step_n size batches.
	If method is UNCERT, picks new indices with bootstrap Uncertainty, with a bootstrap size of boot_n.
	'''
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

	# Initial sample of labelled data
	sample = np.random.choice(np.arange(y_train.shape[0]), start_n, replace = False)
	random_sample = sample.copy()

	y = np.ones_like(y_train)*-1
	y[sample] = y_train[sample]
	FPR, FNR, conf = [],[],[]
	FPR_r, FNR_r, conf_r = [],[],[]
	full_FPR, full_FNR= [],[]
	#Initial results from only initial samples
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, FPR, FNR, conf)
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, FPR_r, FNR_r, conf_r)
	#Calculates heuristics for full labelling(Convergence Value)
	test_seg_progress(map_test, base_seg, X_train, X_test, y_train, np.arange(y_train.shape[0]), full_FPR, full_FNR, [])
	
	plt.ion()
	#Sets up FPR chart
	prec_comparisons = plt.figure('False Positive Rate')
	graph_FPR = plt.plot(X_axis, FPR, 'r-', label = legend_name)[0]
	graph_FPR_r = plt.plot(X_axis, FPR_r, 'r--', label = 'Random Selection')[0]
	plt.axhline(full_FPR[0], color = 'gray', label = 'Full Labelling')
	plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
	plt.legend()
	plt.title('False Positive Rate AL Comparison')
	plt.xlabel('Number of Labelled Samples')
	plt.ylabel('FPR (%)')

	#Sets up FNR chart
	rec_comparisons = plt.figure('False Negative Rate')
	graph_FNR = plt.plot(X_axis, FNR, 'g-',label = legend_name)[0]
	graph_FNR_r = plt.plot(X_axis, FNR_r, 'g--', label = 'Random Selection')[0]
	plt.axhline(full_FNR[0], color = 'gray', label = 'Full Labelling')
	plt.axis([start_n, start_n+step_n*n_updates, 0, 1])
	plt.legend()
	plt.title('False Negative Rate AL Comparison')
	plt.xlabel('Number of Labelled Samples')
	plt.ylabel('FNR (%)')


	plt.draw()
	plt.pause(0.05)

	v_print('starting uncertainty', verbose)
	for i in range(1, n_updates):
		X_axis.append(start_n+i*step_n)

		print "there are", np.where(y >= 0)[0].shape[0], "labelled points"
		#Uses AL method to find next sample indices
		next_indices = uncertainty(X_train, y, boot_n, step_n)
		#next_indices = trivial_error(X_train, y, step_n)
		sample = np.concatenate((sample, next_indices))
		y[sample] = y_train[sample]
		#Update charts
		random_sample = np.concatenate((random_sample, strawman_error(X_train, y, boot_n)))
		test_seg_progress(map_test, base_seg, X_train, X_test, y_train, sample, FPR, FNR, conf)
		test_seg_progress(map_test, base_seg, X_train, X_test, y_train, random_sample, FPR_r, FNR_r, conf_r)
		graph_FPR.set_ydata(FPR)
		graph_FPR.set_xdata(X_axis)
		graph_FNR.set_ydata(FNR)
		graph_FNR.set_xdata(X_axis)
		graph_FPR_r.set_ydata(FPR)
		graph_FPR_r.set_xdata(X_axis)
		graph_FNR_r.set_ydata(FNR)
		graph_FNR_r.set_xdata(X_axis)
		plt.draw()
		plt.pause(0.05)
		#Save results for each iteration
		prec_comparisons.savefig('temp/FPR {} {}.png'.format(legend_name, train))
		rec_comparisons.savefig('temp/FNR {} {}.png'.format(legend_name, train))
		np.save('temp/conf {} {}.npy'.format(legend_name, train), np.array(conf))
		np.save('temp/conr random {}.npy'.format(train),np.array(conf_r))

	plt.waitforbuttonpress()

if __name__ == '__main__':
	#map_train, map_test = map_overlay.basic_setup()
	print 'starting active learning'
	#run_active_learning(map_train, map_test, 100000, 100000)
	run_active_learning_seg()









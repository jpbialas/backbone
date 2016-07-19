import numpy as np
from sklearn.ensemble import RandomForestClassifier


def bootstrap(L, k):
	''' Returns k random samples of L of size |L| with replacement
	INPUT:
		L: (ndarray) 
		k: (int) number of backbone iterations to run
	'''
	n = L.shape[0]
	return np.random.choice(L, (k, n), replace = True)


def uncertainty(X, y, k, m):
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
	L = np.where(y > 0)[0]
	samples = bootstrap(L, k)
	uncertainties = np.zeros(samples.shape[1])
	for row in samples:
		next_model = RandomForestClassifier(n_estimators = 85, n_jobs = -1)
		next_model.fit(X[row], y[row])
		uncertainties += model.predict(X[unlabelled])
	uncertainties/=k
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
	L = np.where(y > 0)[0]
	model = RandomForestClassifier(n_estimators = 85, n_jobs = -1)
	model.fit(X[L], y[L])
	uncertainties = model.predict_proba(X[unlabelled])[:,1]

	uncertainties = np.abs(uncertainties-.5)/.5
	return np.argsort(uncertainties)[:m]



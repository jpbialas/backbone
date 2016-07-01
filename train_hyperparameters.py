import numpy as np
import matplotlib.pyplot as plt
import cv2
from mapOverlay import MapOverlay
import analyzeResults
import pxClassify as clas
import pathos.multiprocessing as mp





def main(param):
	print(param)
	global map_train
	global map_test
	global results
	global step
	global minimum

	precision, recall, accuracy, f1, model, X, y, names = \
			clas.train_and_test(map_train, map_test, 'damage', 'damage', edge_k = param, frac_train = .01, frac_test = 0.01,  vocal = True)
	
	print("precision = {}".format(precision))
	print("recall = {}".format(recall))
	print("accuracy = {}".format(accuracy))
	print("f1 = {}".format(f1))

	indx = (param-minimum)/step
	results[0, indx] = precision
	results[1, indx] = recall
	results[2, indx] = accuracy
	results[3, indx] = f1
	return(precision,recall,accuracy,f1)

if __name__ == '__main__':
	global map_train
	global map_test
	global results 
	global minimum
	global maximum
	global step

	minimum = 50
	maximum = 50
	step = 1

	map_train = MapOverlay('datafromjoe/1-0003-0003.tif')
	map_train.newMask('datafromjoe/1-003-003-damage.shp', 'damage')

	map_test = MapOverlay('datafromjoe/1-0003-0002.tif')
	map_test.newMask('datafromjoe/1-003-002-damage.shp', 'damage')


	main(100)
	'''results = np.zeros((minimum, (maximum-minimum)/step)) #[[],[],[],[]]

	t = np.arange(minimum, maximum + step, step)
	p = mp.ProcessingPool(3)
	p.map(main, range(minimum, maximum+step, step))
	'''
	for i in range(10, 205, 5):
		print(i)
		p,r,a,f = main(i)
		results[0].append(p)
		results[1].append(r)
		results[2].append(a)
		results[3].append(f)
	'''
	fig = plt.figure('Edge Density Window Size')
	fig.suptitle('Edge Density Window Size', fontsize=14)
	ax = fig.add_subplot(111)

	ax.set_xlabel('Window Width/Height')
	ax.set_ylabel('Percentage')
	precision = ax.plot(t, results[0], 'r--', label = 'Precision')
	recall = ax.plot(t, results[1], 'b--', label = 'Recall')
	acc = ax.plot(t, results[2], 'g--', label = 'Accuracy')
	f1 =  ax.plot(t, results[3], 'y--', label = 'F1')
	
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=4, mode="expand", borderaxespad=0.)
	plt.show()
	np.savetxt('hyper_results.csv',np.array(results), delimiter = ',')'''
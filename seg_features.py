import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from convenience_tools import *
import os

def px2seg_labels():
	data = np.zeros((n_segs, 5))
	counts = np.bincount(segs.ravel().astype('int'))
	print counts
	print np.where(color_e[:,-1]>0)[0].shape
	for i in pbar(range(segs.shape[0])):
		data[segs[i]] += color_e[i]
	data=data/np.expand_dims(counts,1).astype('float')

def color_edge(my_map, seg, joes_labels):
	print joes_labels
	names = np.array(['red{}'.format(seg),'green{}'.format(seg), 'blue{}'.format(seg), 'ED{}'.format(seg)])
	p = os.path.join('features', "color_edge_{}.npy".format(my_map.segmentations[seg][0]))
	if os.path.exists(p) and joes_labels:
		data = np.load(p)
	else:
		pbar = custom_progress()
		edges = cv2.Canny(my_map.img,50,100).reshape((my_map.rows,my_map.cols, 1))
		labels = my_map.getLabels('damage').reshape(my_map.rows, my_map.cols, 1)*255
		color_e = np.concatenate((my_map.img, edges, labels), axis = 2).reshape((my_map.rows*my_map.cols, 5))
		segs = my_map.segmentations[seg][1].ravel()
		n_segs = int(np.max(segs))+1
		data = np.zeros((n_segs, 5), dtype = 'uint8')
		for i in pbar(range(n_segs)):
			indices = np.where(segs == i)[0]
			data[i] = np.sum(color_e[indices], axis = 0)/indices.shape[0]
		if joes_labels:
			np.save(p, data)
	return data, names


def features_from_james(img_num, seg):
	json_file = open('segmentations/withfeatures{}/{}-{}-features.json'.format(img_num, img_num, seg))
	json_str = json_file.read()
	#print json_str

	json_data = json.loads(json_str.replace('inf', '0'))
	n_segs = len(json_data['features'])
	n_feats = len(json_data['features'][0]['properties'].values())

	data = np.zeros((n_segs, n_feats))
	json_features = json_data['features']
	names = json_features[0]['properties'].keys()
	for i in range(n_segs):
		data[i] = json_features[i]['properties'].values()
	return data, names

def show_shapes(my_map, rect, ellipse, n_segs, level):
	test = np.zeros(n_segs)
	test[i] = 1
	img = my_map.mask_segments(test, level, with_img = True)
	cv2.ellipse(img = img, box = ellipse, color = [255,0,0], thickness = 5)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(img,[box],0,(0,0,255),2)
	cv2.imshow("img",img)
	cv2.waitKey(1000)

def hog(my_map, seg):
	bins = 16
	names = []
	for i in range(bins):
		names.append('hog{}'.format(i))
	p = os.path.join('features', "hog_seg_{}.npy".format(my_map.segmentations[seg][0]))
	if os.path.exists(p):
		data = np.load(p)
	else:
		pbar = custom_progress()
		bw_img = cv2.cvtColor(my_map.img, cv2.COLOR_RGB2GRAY)
		gx = cv2.Sobel(bw_img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(bw_img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		angles = ang/2.0*np.pi*255
		segs = my_map.segmentations[seg][1].ravel()
		n_segs = int(np.max(segs))+1
		data = np.zeros((n_segs, 16), dtype = 'uint8')
		for i in pbar(range(n_segs)):
			indices = np.where(segs == i)[0]
			data[i] = np.histogram(angles.ravel()[indices], 16)[0]
		np.save(p, data)
	return data, names


def shapes(my_map, level):
	names = np.array(['re{}'.format(level),'rf{}'.format(level),'ee{}'.format(level),'ef{}'.format(level)])
	p = os.path.join('features', "aspect_extent_{}.npy".format(my_map.segmentations[level][0]))
	if os.path.exists(p):
		data = np.load(p)
		data = np.clip(data, 0, 1)
		np.save(p,data)
	else:
		pbar = custom_progress()
		segs = my_map.segmentations[level][1]
		n_segs = int(np.max(segs))+1
		data = np.zeros((n_segs, 4), dtype = 'float')
		for i in pbar(range(n_segs)):
			next_shape = (segs == i)
			cnt, _ = cv2.findContours(next_shape.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			cnt = np.array(cnt)[0]
			rect = cv2.minAreaRect(cnt)

			if cnt.shape[0]>4:
				ellipse = cv2.fitEllipse(cnt)
			else:
				ellipse == rect

			h,w = rect[1]
			a,b = ellipse[1]
			
			leftBound = max(rect[0][0] - max(2*w, 2*a,2*b), 0)
			rightBound = min(rect[0][0] + max(2*w,2*a,2*b), my_map.cols)
			lower = max(rect[0][1] - max(2*w,2*a,2*b), 0)
			upper = max(rect[0][1] + max(2*w,2*a,2*b), my_map.rows)

			img = np.zeros_like(my_map.img)
			cv2.ellipse(img = img,box = ellipse, color = [1,0,0], thickness = -1)
			shape_cut = next_shape[lower:upper,leftBound:rightBound]
			ellipse_cut = img[lower:upper,leftBound:rightBound,0]
			area = np.sum(shape_cut)
			intersection = np.sum(shape_cut*ellipse_cut)
			union = area+np.sum(ellipse_cut)-intersection

			rect_elong =  0 if max(w,h) == 0 else min(w,h)/float(max(w,h))
			rect_fit = 0 if h*w ==0 else float(area)/(h*w)
			ellipse_elong = 0 if max(a,b) == 0 else min(a,b)/float(max(a,b))
			ellipse_fit = 0 if union == 0 else intersection/union

			data[i,0] = rect_elong
			data[i,1] = rect_fit
			data[i,2] = ellipse_elong
			data[i,3] = ellipse_fit
		np.save(p, data)
	return data, names
	

def multi_segs(my_map, base_seg, seg_levels, use_james = True, joes_labels = True):
	img = my_map.img
	img_num = my_map.name[-1]
	h,w,_ = img.shape
	segs = my_map.segmentations[base_seg][1].ravel().astype('int')
	n_segs = int(np.max(segs))
	pbar = custom_progress()
	color_data, color_names = color_edge(my_map, base_seg, joes_labels)
	shape_data, shape_names = shapes(my_map, base_seg)
	hog_data, hog_names = hog(my_map, base_seg)
	if use_james:
		james_data, james_names = features_from_james(img_num, base_seg)
		data = np.concatenate((james_data,shape_data, hog_data, color_data), axis = 1)
		names = np.concatenate((james_names, shape_names, hog_names, color_names), axis = 0)
	else:
		data = np.concatenate((shape_data, hog_data, color_data), axis = 1)
		names = np.concatenate((shape_names, hog_names, color_names), axis = 0)

	
	if len(seg_levels)>0:
		for seg in seg_levels:
			segmentation = my_map.segmentations[seg][1].ravel().astype('int')
			m_segs = int(np.max(segmentation))
			convert = np.zeros(n_segs+1).astype('int')
			convert[segs] = segmentation
			color_data, color_names = color_edge(my_map, seg, joes_labels)
			color_data = color_data[:,:-1]
			shape_data, shape_names = shapes(my_map, seg)
			hog_data, hog_names = hog(my_map, seg)

			#james_data, james_names = features_from_james(img_num, seg)
			#james_names = [s + str(seg) for s in james_names]

			next_data = np.concatenate((shape_data, color_data), axis = 1)[convert]
			next_names = np.concatenate((shape_names, color_names), axis = 0)
			#next_data = color_data[convert]
			#next_names = color_names
			data = np.concatenate((next_data, data), axis = 1)
			names = np.concatenate((next_names, names), axis = 0)
	return data, names


def visualize_segments(my_map, data, seg_level, name, position):
	pbar = custom_progress()
	segs = my_map.segmentations[seg_level][1]
	full_image = np.clip(data[segs.astype('int')],0,1)
	#print max(data), min(data), max(full_image.ravel()), min(full_image.ravel())
	plt.subplot(position)
	plt.title(name), plt.xticks([]), plt.yticks([])
	plt.imshow(full_image, cmap = 'gray')
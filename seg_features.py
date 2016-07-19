import numpy as np
import matplotlib.pyplot as plt
import cv2
import progressbar
import os

def custom_progress():
	return progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',])


def color_edge(my_map, segs):
	names = np.array(['ave_red_{}'.format(segs),'ave_green_{}'.format(segs), 'ave_blue_{}'.format(segs), 'edge_density_{}'.format(segs)])
	p = os.path.join('features', "color_edge_{}.npy".format(my_map.segmentations[segs][0]))
	if os.path.exists(p):
		data = np.load(p)
	else:
		pbar = custom_progress()
		edges = cv2.Canny(my_map.img,50,100).reshape((my_map.rows,my_map.cols, 1))
		labels = my_map.getLabels('damage').reshape(my_map.rows, my_map.cols, 1)*255
		color_e = np.concatenate((my_map.img, edges, labels), axis = 2).reshape((my_map.rows*my_map.cols, 5))
		segs = my_map.segmentations[segs][1].ravel()
		n_segs = int(np.max(segs))+1
		data = np.zeros((n_segs, 5), dtype = 'uint8')
		for i in pbar(range(n_segs)):
			indices = np.where(segs == i)[0]
			data[i] = np.sum(color_e[indices], axis = 0)/indices.shape[0]
		np.save(p, data)
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


def shapes(my_map, level):
	names = np.array(['rect_elong_{}'.format(level),'rect_fit_{}'.format(level),'ellipse_elong_{}'.format(level),'ellipse_fit_{}'.format(level)])
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
	

def multi_segs(my_map, base_seg, seg_levels):
	img = my_map.img
	h,w,_ = img.shape
	segs = my_map.segmentations[base_seg][1].ravel().astype('int')
	n_segs = int(np.max(segs))
	pbar = custom_progress()
	color_data, color_names = color_edge(my_map, base_seg)
	shape_data, shape_names = shapes(my_map, base_seg)
	data = np.concatenate((shape_data, color_data), axis = 1)
	names = np.concatenate((shape_names, color_names), axis = 0)
	if len(seg_levels)>0:
		for seg in seg_levels:
			segmentation = my_map.segmentations[seg][1].ravel().astype('int')
			m_segs = int(np.max(segmentation))
			convert = np.zeros(n_segs+1).astype('int')
			convert[segs] = segmentation
			color_data, color_names = color_edge(my_map, seg)
			color_data = color_data[:,:-1]
			shape_data, shape_names = shapes(my_map, seg)

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

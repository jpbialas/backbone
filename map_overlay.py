import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
from osgeo import gdal, osr, ogr, gdalconst #For shapefile...raster
import os
from convenience_tools import *


''''
from map_overlay import MapOverlay
map_test = MapOverlay('datafromjoe/1-0003-0002.tif')
map_test.new_segmentation('segmentations/withfeatures2/2-1000-features.json', 1000)


'''
def basic_setup(segs = [], base_seg = 50, jared = False):

	map_train = MapOverlay('datafromjoe/1-0003-0002.tif')
	map_test = MapOverlay('datafromjoe/1-0003-0003.tif')


	
	map_train.new_segmentation('segmentations/withfeatures2/shapefilewithfeatures003-002-{}.shp'.format(base_seg), base_seg)
	map_test.new_segmentation('segmentations/withfeatures3/shapefilewithfeatures003-003-{}.shp'.format(base_seg), base_seg)


	for seg in segs:
		map_train.new_segmentation('segmentations/withfeatures2/shapefilewithfeatures003-002-{}.shp'.format(seg), seg)
		map_test.new_segmentation('segmentations/withfeatures3/shapefilewithfeatures003-003-{}.shp'.format(seg), seg)

	if not jared:
		map_train.newMask('datafromjoe/1-003-002-damage.shp', 'damage')
		map_test.newMask('datafromjoe/1-003-003-damage.shp', 'damage')

	if jared:
		map_train.new_seg_mask(np.loadtxt('jaredlabels/2.csv', delimiter = ','), base_seg, 'damage')
		map_test.new_seg_mask(np.loadtxt('jaredlabels/3.csv', delimiter = ','), base_seg, 'damage')


	return map_train, map_test


class MapOverlay:

	'''
	INPUT: 
		- map_fn: (string) Filename of map to be used as the base geographic raster for the object
	RESULT: 
		- Instantiates object with stored constants applicable to input map
	'''
	def __init__(self, map_fn):
		self.masks = {}
		self.segmentations = {}
		self.name = map_fn[map_fn.rfind('/')+1: map_fn.find('.tif')]
		driver = gdal.GetDriverByName('HFA')
		driver.Register()
		ds = gdal.Open(map_fn, 0)
		if ds is None:
			print('Could not open ' + map_fn)
			sys.exit(1)

		self.map_ds = ds
		geotransform = ds.GetGeoTransform()

		self.map_fn = map_fn
		self.img = cv2.cvtColor(cv2.imread(map_fn), cv2.COLOR_BGR2RGB)
		self.bw_img = cv2.imread(map_fn, 0)

		self.cols = ds.RasterXSize
		self.rows = ds.RasterYSize
		self.bands = ds.RasterCount

		self.originX = geotransform[0]
		self.originY = geotransform[3]
		self.pixelWidth = geotransform[1]
		self.pixelHeight = geotransform[5]

	
	def getMapData(self):
		'''
		OUTPUT: 
			- img: (n*m x 3 ndarray) Raveled image associated with map_fn
		'''
		h,w,c = self.img.shape
		return self.img.reshape(h*w, c)


	def get_features(self):
		return self.features, self.label_names
	
	def gen_features(self, edge_k, hog_k, hog_bins):
		'''
		input:
			- myMap: MapObject
		output:
			- feature representation of map
		'''
		rgb, rgb_name = features.normalized(self.getMapData())
		ave_rgb, ave_rgb_name = features.blurred(self.img)
		edges, edges_name = features.edge_density(self.bw_img, edge_k, amp = 1)
		hog, hog_name = features.hog(self.bw_img, hog_k)
		data = np.concatenate((rgb, ave_rgb, edges, hog), axis = 1)
		names = np.concatenate((rgb_name, ave_rgb_name, edges_name, hog_name))
		self.features =  data
		self.label_names =  names


	def getLabels(self, mask_name):
		'''
		INPUT: 
			- mask_name: (string) Name of mask to get labels for
		OUTPUT: 
			- (n x 1 ndarray) Vector of binary labels indicating which pixels lie within mask
		'''
		return self.masks[mask_name]


	
	def latlonToPx(self, x, y):
		'''
		INPUT: 
			- x: (float) Lattitude Coordinate
			- y: (float) Longitude Coordinates
		OUTPUT: 
			Tuple containing x and y coordinates of point in image associated with lat/lon
		'''
		xOffset = int((x - self.originX) / self.pixelWidth)
		yOffset = int((y - self.originY) / self.pixelHeight)
		return (xOffset, yOffset)

	
	def _projectShape(self, shape_fn, mask_name):
		'''
		INPUT: 
			-shape_fn:  (string) Shape filename to be reprojected
			-mask_name: (string) Custom name to associate with newly projected shape file
		OUTPUT: 
			- (DataSource object) New Shapefile with coordinates relative to base map raster

		NOTE: Second half of code heavily inspired by: https://pcjericks.github.io/py-gdalogr-cookbook/projection.html

		'''
		driver = ogr.GetDriverByName('ESRI Shapefile')
		in_ds = driver.Open(shape_fn, 0) #Second parameter makes it read only. Other option is 1
		if in_ds is None:
		  print('Could not open file')
		  sys.exit(1)
		in_lyr = in_ds.GetLayer()


		targetSR = osr.SpatialReference()
		targetSR.ImportFromWkt(self.map_ds.GetProjectionRef())

		coordTrans = osr.CoordinateTransformation(in_lyr.GetSpatialRef(), targetSR)

		outfile = 'cache\{}.shp'.format(mask_name)
		outDataSet = driver.CreateDataSource(outfile)

		dest_srs = osr.SpatialReference()
		dest_srs.ImportFromEPSG(4326)
		outLayer = outDataSet.CreateLayer(mask_name, targetSR, geom_type=ogr.wkbMultiPolygon)


		inLayerDefn = in_lyr.GetLayerDefn()
		for i in range(0, inLayerDefn.GetFieldCount()):
			fieldDefn = inLayerDefn.GetFieldDefn(i)
			outLayer.CreateField(fieldDefn)

		outLayerDefn = outLayer.GetLayerDefn()

		inFeature = in_lyr.GetNextFeature()
		while inFeature:
		    # get the input geometry
		    geom = inFeature.GetGeometryRef()
		    # reproject the geometry
		    geom.Transform(coordTrans)
		    # create a new feature
		    outFeature = ogr.Feature(outLayerDefn)
		    # set the geometry and attribute
		    outFeature.SetGeometry(geom)
		    for i in range(0, outLayerDefn.GetFieldCount()):
		        outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
		    # add the feature to the shapefile
		    outLayer.CreateFeature(outFeature)
		    # destroy the features and get the next input feature
		    outFeature.Destroy()
		    inFeature.Destroy()
		    inFeature = in_lyr.GetNextFeature()

		return outDataSet, targetSR


	def maskImg(self, mask_name):
		'''
		INPUT: 
			- (string) Name of mask to overlay
		RESULT: 
			- Displays base map overlayed with relevant mask shapes
		'''

		mask = self.masks[mask_name]
		h,w = self.rows, self.cols
		adj_mask = np.logical_not(mask).reshape(h,w,1)

		
		overlay = self.img*adj_mask #+mask.reshape(h,w,1)*np.array([0.,0.,1.])
		return overlay

	def mask_segments(self, i, level, with_img = True):
		'''
			masks_segments based on i, a boolean list indicating labelled segments
		'''
		mask = np.in1d(self.segmentations[level][1],np.where(i))
		if with_img:
			h,w = self.rows, self.cols
			adj_mask = np.logical_not(mask).reshape(h,w,1)
			overlay = self.img*adj_mask
			return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
		else:
			return mask

	def mask_segments_by_indx(self, indcs, level, with_img = True):
		'''
			masks_segments based on indcs, a list of indices
		'''
		mask = np.in1d(self.segmentations[level][1],indcs)
		if with_img:
			h,w = self.rows, self.cols
			adj_mask = np.logical_not(mask).reshape(h,w,1)
			overlay = self.img*adj_mask
			return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
		else:
			return mask


	def new_seg_mask(self, indcs, level, mask_name):
		'''
			generates new mask labelling given only segment index numbers at level @level
		'''
		mask = self.mask_segments_by_indx(indcs, level, False)
		self.masks[mask_name] = mask.ravel()


	def newPxMask(self, mask, mask_name):
		h,w,c = self.img.shape
		self.masks[mask_name] = mask.ravel()


	def combine_masks(self, old1, old2, new):
		old_mask1 = self.masks[old1]
		old_mask2 = self.masks[old2]
		del self.masks[old1]
		del self.masks[old2]
		self.newPxMask((old_mask1+old_mask2 > 0), new)
	


	def newMask(self, shape_fn, mask_name):
		''' 
		INPUT: 
			-shape_fn:  (string) Shape filename to be reprojected
			-mask_name: (string )Custom name to associate with newly projected shape file
		RESULT: Generates mask with the dimensions of the base map raster containing '1's in pixels
				 Where the shapes cover and '0' in all other locations.
				 Additionally adds the mask to the map dictionary
		'''
		dataSource, targetSR =  self._projectShape(shape_fn, mask_name)

		layer = dataSource.GetLayer()
		lat_max, lat_min, lon_min, lon_max = layer.GetExtent()

		maxx, miny = self.latlonToPx(lat_min,lon_max)
		minx, maxy = self.latlonToPx(lat_max,lon_min)
		nrows = min(maxy-miny, self.rows)
		ncols = min(maxx-minx, self.cols)
		minx = max(0, minx)
		miny = max(0, miny)
		maxx = min(maxx, self.cols)
		maxy = min(maxy, self.rows)

		xres=(lat_max-lat_min)/float(maxx-minx)
		yres=(lon_max-lon_min)/float(maxy-miny)

		geotransform=(lat_min,xres,0,lon_max,0, -yres)
		dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Byte)
		dst_ds.SetGeoTransform(geotransform)
		padding = ((miny, self.rows - maxy),(self.cols-maxx,minx)) # in order to fill whole image

		band = dst_ds.GetRasterBand(1) #Initialize with 1 band
		band.Fill(0) #initialise raster with zeros
		band.SetNoDataValue(0)
		band.FlushCache()

		gdal.RasterizeLayer(dst_ds, [1], layer)#, options = ["ATTRIBUTE=ID"])

		mask = dst_ds.GetRasterBand(1).ReadAsArray()
		mask = np.pad(mask, padding, mode = 'constant', constant_values = 0)
		
		mask = np.fliplr(mask)
		mask = mask>0
		self.masks[mask_name] = mask.ravel()

	def new_segmentation(self, shape_fn, level, mask_name = 'segments'):

		name = shape_fn[shape_fn.rfind('/')+1: shape_fn.find('.shp')]
		p = os.path.join('processed_segments', "{}.npy".format(name))
		#self.segment_name = name
		if os.path.exists(p):
			self.segmentations[level] =  (name,np.load(p))
		else:
			dataSource, targetSR =  self._projectShape(shape_fn, mask_name)

			layer = dataSource.GetLayer()
			lat_max, lat_min, lon_min, lon_max = layer.GetExtent()

			maxx, miny = self.latlonToPx(lat_min,lon_max)
			minx, maxy = self.latlonToPx(lat_max,lon_min)
			nrows = min(maxy-miny, self.rows)
			ncols = min(maxx-minx, self.cols)
			minx = max(0, minx)
			miny = max(0, miny)
			maxx = min(maxx, self.cols)
			maxy = min(maxy, self.rows)

			xres=(lat_max-lat_min)/float(maxx-minx)
			yres=(lon_max-lon_min)/float(maxy-miny)

			geotransform=(lat_min,xres,0,lon_max,0, -yres)
			dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Byte)
			dst_ds.SetGeoTransform(geotransform)
			padding = ((miny, self.rows - maxy),(self.cols-maxx,minx)) # in order to fill whole image

			nfeatures = layer.GetFeatureCount()
			print(nfeatures)
			print("starting map segmentation reading")
			full_mask = np.zeros((self.rows, self.cols))
			pbar = custom_progress()
			for i in pbar(range(nfeatures)):
				band = dst_ds.GetRasterBand(1) #Initialize with 1 band
				band.Fill(0) #initialise raster with zeros
				band.SetNoDataValue(0)
				band.FlushCache()
				new_layer = dataSource.CreateLayer("{}_{}".format(mask_name,i), targetSR, geom_type=ogr.wkbMultiPolygon)
				feature = layer.GetFeature(i)
				new_layer.CreateFeature(feature)
				layer.DeleteFeature(i)
				gdal.RasterizeLayer(dst_ds, [1], new_layer)
				mask = dst_ds.GetRasterBand(1).ReadAsArray()
				mask = np.pad(mask, padding, mode = 'constant', constant_values = 0)
				mask = np.fliplr(mask)
				mask = mask>0
				feature.Destroy()
				full_mask += mask*i
				dataSource.DeleteLayer("{}_{}".format(mask_name,i))
			print("finished map segmentation reading")
			dataSource.Destroy()
			self.segmentations[level] = (name,full_mask)
			np.save(p, full_mask)

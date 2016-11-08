import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
from osgeo import gdal, osr, ogr, gdalconst #For shapefile...raster
import os
from convenience_tools import *
import seg_features as sf

def convert_shapefile_to_indcs(shape_fn, map_fn, segs_fn, thresh = 0.5):
    '''
        Used to convert shapefile labeling scheme into segment index labels
    '''
    my_map = MapOverlay(map_fn)
    segs = np.load(segs_fn).astype('int')
    my_map.newMask(shape_fn, 'damage')
    mask = my_map.masks['damage'].reshape(segs.shape)
    fitted = sf.px2seg2(mask.reshape((my_map.cols*my_map.rows, 1)), segs.ravel())
    return np.where((fitted>thresh)>0)[0]


def basic_setup(segs = [100], base_seg = 50, label_name = "Jared"):
    '''
        Returns 1-003-002 and 1-003-002 as map objects with relevant segmentations and damage labeling
    ''' 
    #Generate Maps
    map_train = MapOverlay('datafromjoe/1-0003-0002.tif')
    map_test = MapOverlay('datafromjoe/1-0003-0003.tif')
    #Add Segmentations
    map_train.new_segmentation('segmentations/withfeatures2/shapefilewithfeatures003-002-{}.shp'.format(base_seg), base_seg)
    map_test.new_segmentation('segmentations/withfeatures3/shapefilewithfeatures003-003-{}.shp'.format(base_seg), base_seg)
    for seg in segs:
        map_train.new_segmentation('segmentations/withfeatures2/shapefilewithfeatures003-002-{}.shp'.format(seg), seg)
        map_test.new_segmentation('segmentations/withfeatures3/shapefilewithfeatures003-003-{}.shp'.format(seg), seg)
    #Add Damage Labels
    map_train.new_seg_mask(np.loadtxt('damagelabels50/{}-3-2.csv'.format(label_name), delimiter = ','), base_seg, 'damage')
    map_test.new_seg_mask(np.loadtxt('damagelabels50/{}-3-3.csv'.format(label_name), delimiter = ','), base_seg, 'damage')
    return map_train, map_test

def haiti_setup(segs = [50], base_seg = 20, label_name = "Jared"):
    '''
        Returns haiti map object with relevant segmentations and damage labeling
    ''' 
    haiti_map = MapOverlay('haiti/haiti_1002003.tif')
    haiti_map.new_segmentation('segmentations/haiti/segment-{}.shp'.format(base_seg), base_seg)
    for seg in segs:
        haiti_map.new_segmentation('segmentations/haiti/segment-{}.shp'.format(seg), seg)
    damage = np.loadtxt('damagelabels20/{}.csv'.format(label_name), delimiter = ',')
    haiti_map.new_seg_mask(np.loadtxt('damagelabels20/{}.csv'.format(label_name), delimiter = ','), base_seg, 'damage')
    return haiti_map


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
        self.shape = self.img.shape
        self.shape2d = self.img.shape[:2]
        self.bw_img = cv2.imread(map_fn, 0)
        self.cols, self.rows, self.bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
        self.originX, self.originY = geotransform[0], geotransform[3]
        self.pixelWidth, self.pixelHeight = geotransform[1], geotransform[5]

    
    def getMapData(self):
        '''
        OUTPUT: 
            - img: (n*m x 3 ndarray) Raveled image associated with map_fn
        '''
        h,w,c = self.img.shape
        return self.img.reshape(h*w, c)


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
        PRVIATE METHOD
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
          return None
        in_lyr = in_ds.GetLayer()
        targetSR = osr.SpatialReference()
        targetSR.ImportFromWkt(self.map_ds.GetProjectionRef())
        coordTrans = osr.CoordinateTransformation(in_lyr.GetSpatialRef(), targetSR)
        outfile = 'cache\{}.shp'.format(mask_name)
        outDataSet = driver.CreateDataSource(outfile)
        outLayer = outDataSet.CreateLayer(mask_name, targetSR, geom_type=ogr.wkbMultiPolygon)
        inLayerDefn = in_lyr.GetLayerDefn()
        fid_fd = ogr.FieldDefn('FID', ogr.OFTReal)
        print fid_fd.GetType()
        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            outLayer.CreateField(fieldDefn)
        outLayer.CreateField(fid_fd)
        outLayerDefn = outLayer.GetLayerDefn()
        inFeature = in_lyr.GetNextFeature()
        indx=0
        while inFeature:
            # get the input geometry
            geom = inFeature.GetGeometryRef()
            # reproject the geometry
            geom.Transform(coordTrans)
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()-1): #-1 to ignore fid
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
            outFeature.SetField('FID', indx)
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)
            # destroy the features and get the next input feature
            outFeature.Destroy()
            inFeature.Destroy()
            inFeature = in_lyr.GetNextFeature()
            indx+=1
        return outDataSet, targetSR

    def mask_helper(self, img, mask, opacity = 0.4):
        h,w,c = img.shape
        adj_mask = np.logical_not(mask).reshape(h,w,1)
        gap_image = img*adj_mask
        red_mask = ([255,0,0]*mask.reshape(h,w,1)).astype('uint8')
        masked_image = gap_image+red_mask
        output = img.copy()
        cv2.addWeighted(masked_image, opacity, output, 1.0-opacity, 0, output)
        return output


    def maskImg(self, mask_name, opacity = 0.4):
        '''
        INPUT: 
            - (string) Name of mask to overlay
        RESULT: 
            - Displays base map overlayed with relevant mask shapes
        '''

        mask = self.masks[mask_name]
        return self.mask_helper(self.img, mask, opacity)

    def mask_segments(self, i, level, with_img = True, opacity = .4):
        '''
            masks_segments based on i, a boolean list indicating labelled segments
        '''
        mask = np.in1d(self.segmentations[level][1],np.where(i))
        return self.mask_helper(self.img, mask, opacity) if with_img else mask

    def mask_segments_by_indx(self, indcs, level,  with_img = True, opacity = .4):
        '''
            masks_segments based on indcs, a list of indices
        '''
        mask = np.in1d(self.segmentations[level][1],indcs)
        return self.mask_helper(self.img, mask, opacity) if with_img else mask


    def new_seg_mask(self, indcs, level, mask_name):
        '''
            generates new mask labelling given only segment index numbers at level @level
        '''
        mask = self.mask_segments_by_indx(indcs, level, False)
        self.masks[mask_name] = mask.ravel()


    def newPxMask(self, mask, mask_name):
        '''
            adds boolean mask in shape of image to masks with associated name
        '''
        h,w,c = self.img.shape
        self.masks[mask_name] = mask.ravel()


    def combine_masks(self, old1, old2, new):
        old_mask1 = self.masks[old1]
        old_mask2 = self.masks[old2]
        del self.masks[old1]
        del self.masks[old2]
        self.newPxMask((old_mask1+old_mask2 > 0), new)
    

    def rasterize_shp(self, shape_fn, mask_name):
        '''
            Creates index raster in the shape of the base image from shapefile segmentation
        '''
        dataSource, targetSR =  self._projectShape(shape_fn, mask_name)
        layer = dataSource.GetLayer()
        lat_max, lat_min, lon_min, lon_max = layer.GetExtent()
        maxx, miny = self.latlonToPx(lat_min,lon_max)
        minx, maxy = self.latlonToPx(lat_max,lon_min)
        nrows, ncols = min(maxy-miny, self.rows), min(maxx-minx, self.cols)
        minx, miny, maxx, maxy = max(0, minx), max(0, miny), min(maxx, self.cols), min(maxy, self.rows)
        xres = (lat_max-lat_min)/float(maxx-minx)
        yres = (lon_max-lon_min)/float(maxy-miny)
        geotransform =(lat_min,xres,0,lon_max,0, -yres)
        dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Int32)
        dst_ds.SetGeoTransform(geotransform)
        padding = ((miny, self.rows - maxy),(self.cols-maxx,minx)) # in order to fill whole image
        gdal.RasterizeLayer(dst_ds, [1], layer, options = ['ATTRIBUTE=FID'])
        mask = np.pad(dst_ds.GetRasterBand(1).ReadAsArray(), padding, mode = 'edge')
        mask = np.fliplr(mask)
        dataSource.Destroy()
        return mask

    def newMask(self, shape_fn, mask_name):
        ''' 
        INPUT: 
            -shape_fn:  (string) Shape filename to be reprojected
            -mask_name: (string )Custom name to associate with newly projected shape file
        RESULT: Generates mask with the dimensions of the base map raster containing '1's in pixels
                 Where the shapes cover and '0' in all other locations.
                 Additionally adds the mask to the map dictionary
        '''
        mask = self.rasterize_shp(shape_fn, mask_name)>0
        self.masks[mask_name] = mask.ravel()

    def new_segmentation(self, shape_fn, level, mask_name = 'segments', save = True):

        name = shape_fn[shape_fn.rfind('/')+1: shape_fn.find('.shp')]
        p = os.path.join('processed_segments', "{}.npy".format(name))
        #self.segment_name = name
        if os.path.exists(p) and save:
            self.segmentations[level] =  (name,np.load(p))
        else:
            mask = self.rasterize_shp(shape_fn, mask_name)
            self.segmentations[level] = (name,mask)
            if save:
                np.save(p, mask)

if __name__ == '__main__':
    haiti_setup()
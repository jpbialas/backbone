import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
from osgeo import gdal, osr, ogr, gdalconst #For shapefile...raster
import os
from convenience_tools import *
import seg_features as sf


def basic_setup(segs = [100], base_seg = 50, label_name = "Jared"):
    """
    Generates 1-003-002 and 1-003-002 as map objects with relevant segmentations and damage labeling.

    Parameters
    ----------
    segs : int list
        List of integers representing segmentation level to add to Map Object
    base_seg: int
        Primary segmentation level to use for classification on Map Objects
    label_name : String
        Name of labeling to use as 'damage'. These are pulled from the damagelabels50 folder.

    Returns
    -------
    Geo_Map, Geo_Map
        Geo_Map objects containing 1-003-002 and 1-003-002 with relevant segmentations and damage label
    """ 

    #Generate Maps
    map_train = Geo_Map('datafromjoe/1-0003-0002.tif')
    map_test = Geo_Map('datafromjoe/1-0003-0003.tif')
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
    """
    Generates Haiti_Map as map object with relevant segmentations and damage labeling.

    Parameters
    ----------
    segs : int list
        List of integers representing segmentation level to add to Map Object
    base_seg: int
        Primary segmentation level to use for classification on Map Objects
    label_name : String
        Name of labeling to use as 'damage'. These are pulled from the damagelabels50 folder.

    Returns
    -------
    Geo_Map
        Geo_Map objects HaitiMap with relevant segmentations and damage label
    """ 

    #Generates Map
    haiti_map = Geo_Map('haiti/haiti_1002003.tif')
    #Adds Segmentations
    haiti_map.new_segmentation('segmentations/haiti/segment-{}.shp'.format(base_seg), base_seg)
    for seg in segs:
        haiti_map.new_segmentation('segmentations/haiti/segment-{}.shp'.format(seg), seg)
    #Adds Damage Labels
    damage = np.loadtxt('damagelabels20/{}.csv'.format(label_name), delimiter = ',')
    haiti_map.new_seg_mask(np.loadtxt('damagelabels20/{}.csv'.format(label_name), delimiter = ','), base_seg, 'damage')
    return haiti_map


class Px_Map(object):
    """
    Px_Map Class holds map image with associated features, masks, and segmentations

    Parameters
    ----------
    segs : img
        Image of Map Region

    Fields
    ------
    img : ndarray
        (h,w,c) image of map that all other fields will be in relation to
    masks : (String : boolean ndarray) dict
        Dictionary holding useful masks of map image. Masks are 2d boolean arrays of the same shape
        as img and have value 1 for pixels in which the mask's purpose is demarked as true, and zero otherwise.
        For example, a mask holding a damage labeling will be 1 for all pixels containing damage and zero otherwise.
    segmentations : (int : int ndarray) dict
        Dictionary holding segmentations of map image. segmentations are 2d int arrays of the same shape
        as img. Each entry holds the segment index of the pixel at that location in the map image.

        Example:
        If a 4x4 map image is segmented into 4 quadrants the segmentation might look like the following:

        0 0 0 1 1 1
        0 0 0 1 1 1
        2 2 2 3 3 3 
        2 2 2 3 3 3

        Segmentations are identified with the a numeric key. Usually this key is the 'level' corresponding 
        to the segmentation as executed in eCognition. Indices will most often start at 0, but if a Px_Map 
        object is a subimage of another, it may start at any value and not be sequential.
    rows : int
        Rows in map image
    cols : int
        Cols in map image
    bands : int
        Bands in map image
    shape : (int, int, int)
        Shape of img
    bw_img : ndarray
        Black and white version of img
    """ 

    def __init__(self, img):
        self.rows, self.cols, self.bands = img.shape
        self.X = None
        self.img = img
        self.masks = {}
        self.segmentations = {}
        self.shape = self.img.shape
        self.bw_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)


    def sub_map(self, ix):
        """
        Generates new Px_Map object whose map is a subset of the current instnce

        Parameters
        ----------
        ix : ndarray
            Array or Open Mesh representing subset of indices of Map Image to include in sub image.
            Use np.ix_ to generate 2D subset of image indices

        Returns
        -------
        Px_Map
            Sub Map of current instance

        """
        f = lambda x: x.reshape((self.rows, self.cols))[ix]
        sub_img = self.img[ix]
        sub_map = Px_Map(sub_img)
        sub_map.masks = {k: f(v).ravel() for k, v in self.masks.items()}
        sub_map.segmentations = {k: f(v) for k, v in self.segmentations.items()}
        return sub_map

    
    def unique_segs(self, level):
        """
        Returns all unique segment indices at given level.

        Useful for subimages which keep the same segment indexing as their parent.

        Parameters
        ----------
        level : int
            segmentation level 

        Returns
        -------
        int ndarray
            array of unique segment indices
        """
        return np.unique(self.segmentations[level]).astype(int)

    
    def seg_convert(self, seg, arr):
        """
        Converts array to be in terms of segment id starting from 0.

        This function is helps with complicated indexing. When a Px_Map instance is
        a sub_map of a different object, its indexing can start at any value and the segment indices may
        not be sequential. This makes indexing in terms of the segmentation complicated. Therefore, this function
        takes arr, an array holding values in an array of length unique_segs and places those values into a new
        array so that the positions of the values correspond to the segmenation indexes of the parent map object.

        Parameters
        ----------
        seg : int
            key for segmentation in self.segmentations        
        arr : 
            array of length self.unique_segs(segs) holding values

        Example:
        For a sub_map with the following segmentation stored in self.segmentations(seg):

        3 3 3 4 4 4
        3 3 3 4 4 4
        8 8 8 9 9 9 
        8 8 8 9 9 9

        and array arr of values: [18, 7, 23, 101]

        18 corresponds to segmentation index 3. 
        7 corresopnds to segmenation index 4.
        23 corresponds to segmentation index 8. etc

        seg_convert(seg, arr) will return the following array
        
        18 18 18  7   7   7
        18 18 18  7   7   7
        23 23 23 101 101 101
        23 23 23 101 101 101
        """
        uniques = self.unique_segs(seg)
        assert len(uniques)==arr.shape[0], \
            'Shape Error: arr must be the same length as uniques: {} != {}'.format(len(uniques), len(arr))
        assert seg in self.segmentations, \
            'Key Error: seg must be a key for a segmentation: {}'.format(seg)
        all_segs = np.zeros(uniques[-1]+1)
        all_segs[uniques] = arr
        return all_segs[self.segmentations[seg]]


    def getMapData(self):
        """Returns (h*w, c) image of map image color data"""
        h,w,c = self.img.shape
        return self.img.reshape(h*w, c)


    def getLabels(self, mask_name):
        """Returns pixel map associated with mask_name"""
        return self.masks[mask_name]

    
    def mask_helper(self, img, mask, opacity = 0.4):
        """
        Returns image with mask highlighted in red at given opacity

        Parameters
        ----------
        img : ndarray
            (h,w,3) Image to mask
        mask : ndarray
            (h,w) Mask of same shape as image containing either binary values or floats < 1
            to mask img with
        opacity : float
            Opacity for mask overlay

        Returns
        -------
            Image with mask overlayed according to pixel value in mask
        """
        h,w,c = img.shape
        adj_mask = np.logical_not(mask).reshape(h,w,1)
        gap_image = img*adj_mask
        red_mask = ([255,0,0]*mask.reshape(h,w,1)).astype('uint8')
        masked_image = gap_image+red_mask
        output = img.copy()
        cv2.addWeighted(masked_image, opacity, output, 1.0-opacity, 0, output)
        return output


    def maskImg(self, mask_name, opacity = 0.4):
        """Masks instance's map image with mask associated with mask_name at given. See mask_helper"""
        mask = self.masks[mask_name]
        return self.mask_helper(self.img, mask, opacity)


    def mask_segments(self, seg_bool_mask, level, with_img = True, opacity = .4):
        """
        Masks instance's map image with segment indices indicated by bool mask

        Useful for visualizing prediction output of Object-Based classifier

        Parameters
        ----------
        seg_bool_mask : ndarray
            Boolean array of shape (unique_segs(level), ) indicating which segments to mask in map image
        level : ing
            Segmentation level corresponding to boolean mask
        with_img: boolean
            Indicates whether or not to have image in background of mask
        opactiy : float
            Indicates opacity of mask over image if with_img is True

        Returns
        -------
        ndarray
            Image masked with segments indicated in seg_bool_mask
        """
        mask = np.in1d(self.segmentations[level],np.where(seg_bool_mask))
        return self.mask_helper(self.img, mask, opacity) if with_img else mask

    def mask_segments_by_indx(self, indcs, seg,  with_img = True, opacity = .4):
        """
        Masks instance's map image with segment indices indicated by segment index list

        Similar function as mask_segments, but indices to mask are indicated by index list instead
        of boolean mask.
        """
        assert seg in self.segmentations, \
            'Key Error: seg must be a key for a segmentation: {}'.format(seg)
        mask = np.in1d(self.segmentations[seg],indcs)
        return self.mask_helper(self.img, mask, opacity) if with_img else mask


    def new_seg_mask(self, indcs, level, mask_name):
        """
        Adds new mask labelling to self.masks absed on segment ids
        
        Parameters
        ----------
        indcs : ndarray
            segmentation ids to mark as 1 for new mask
        level : int
            segmentation key for relevant segmentation
        mask_name : String
            name to identify new mask by
        """
        mask = self.mask_segments_by_indx(indcs, level, False)
        self.masks[mask_name] = mask.ravel()


    def newPxMask(self, mask, mask_name):
        """Adds boolean mask in shape of image to masks with associated name"""
        assert mask.shape == (self.rows, self.cols), \
            "New Mask must be the same shape as the image: {} != {}".format(mask.shape, (self.rows, self.cols)) 
        self.masks[mask_name] = mask.ravel()


    def combine_masks(self, old1, old2, new_name):
        """
        Generates new mask from two pre-existing masks. Removes old masks

        Parameters
        ----------
        old1 : String
            Original mask name 1
        old2 : String
            Original mask name 2
        new_name : String
            New mask name
        """
        assert old1 in self.masks, \
            'Key Error: old1 must be a key for a mask: {}'.format(old1) 
        assert old2 in self.masks, \
            'Key Error: old2 must be a key for a mask: {}'.format(old2)
        old_mask1 = self.masks[old1]
        old_mask2 = self.masks[old2]
        del self.masks[old1]
        del self.masks[old2]
        self.newPxMask((old_mask1+old_mask2 > 0), new_name)


    def new_seg_raster(self, raster, seg):
        """Adds new segmenation given segmentation mask"""
        assert(raster.shape == (self.rows, self.cols))
        self.segmentations[seg] = raster


class Geo_Map(Px_Map):
    """
    Geo_Map Class is a subclass of Px_Map that includes lat/lon support. 
    It is instantiated with Tiff files and shapefiles instead of numpy based counterparts

    Note Geo_Map objects are not pickleable due to gdal elements. Px_Map objects are.

    Parameters
    ----------
    map_fn : String
        Path name to map file that will serve as base image. Should be a .tiff file

    Additional Fields
    ------
    name : String
        name of map_fn
    map_ds : GDAL Dataste
        Dataset corresponding to map file
    originX : float
        Latitude of map transform center
    OriginY : float
        Longitude of map transform center
    pixelWidth : float
        Width resolution of each pixel
    pixelHeight : float
        Height resolution of each pixel
    map_fn : String
        Path name to map file that will serve as base image. Should be a .tiff file
    """ 

    def __init__(self, map_fn):
        self.name = map_fn[map_fn.rfind('/')+1: map_fn.find('.tif')]
        driver = gdal.GetDriverByName('HFA')
        driver.Register()
        ds = gdal.Open(map_fn, 0)
        if ds is None:
            print('Could not open ' + map_fn)
            sys.exit(1)
        self.map_ds = ds
        geotransform = ds.GetGeoTransform()
        self.originX, self.originY = geotransform[0], geotransform[3]
        self.pixelWidth, self.pixelHeight = geotransform[1], geotransform[5]
        self.map_fn = map_fn
        Px_Map.__init__(self, cv2.cvtColor(cv2.imread(map_fn), cv2.COLOR_BGR2RGB))
        self.cols, self.rows, self.bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount


    def new_segmentation(self, shape_fn, level, save = True):
        """
        Creates new index map segmentation (see segmentations in Px_Map) from shapefile.

        If save is True, saves segmentation in procesed_segments folder

        Parameters
        ----------
        shape_fn : String
            Path to shapefile to import
        level : int
            Segmentation level as assigned in eCognition to save segmentation under
        save : boolean
            Saves rasterizationin processed_segments folder if True
        """
        name = shape_fn[shape_fn.rfind('/')+1: shape_fn.find('.shp')]
        p = os.path.join('processed_segments', "{}.npy".format(name))
        if os.path.exists(p) and save:
            mask = np.load(p)
        else:
            mask = self.rasterize_shp(shape_fn, 'segments')
            if save:
                np.save(p, mask)
        self.new_seg_raster(mask, level)
            
    

    def latlonToPx(self, x, y):
        """Returns Latitude and Longitude tuple corresponding to input pixel coordinates"""
        xOffset = int((x - self.originX) / self.pixelWidth)
        yOffset = int((y - self.originY) / self.pixelHeight)
        return (xOffset, yOffset)

    
    def _projectShape(self, shape_fn, mask_name):
        """
        Private method for repreojecting shapefile projection method into base_map's projection method.

        This ensures the two are properly aligned.

        Parameters
        ----------
        shape_fn : String
            Path to shapefile that needs to be reprojected
        mask_name : String
            Custom name to associate with newly projected shape file

        Returns
        -------
        osgeo.ogr.DataSource
            Datasource of new reprojected shapfile
        osgeo.osr.SpatialReference
            Spatial Reference that the shapefile was reprojected to
        """
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
        #Create new datasource and layer to hold reprojected shapefile
        outDataSet = driver.CreateDataSource(outfile)
        outLayer = outDataSet.CreateLayer(mask_name, targetSR, geom_type=ogr.wkbMultiPolygon)
        inLayerDefn = in_lyr.GetLayerDefn()
        fid_fd = ogr.FieldDefn('FID', ogr.OFTReal)
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
        print "IM HERE", type(outDataSet), type(targetSR)
        return outDataSet, targetSR

    def rasterize_shp(self, shape_fn, mask_name):
        '''
            Creates index raster in the shape of the base image from shapefile segmentation
        '''
        """
        Convert shapefile segmentation of image into a pixel-based map.

        This is necessary to take eCognition segmentations and convert them into a useful form
        
        Parameters
        ----------
        shape_fn : String
            Path to shapefile that needs to be reprojected

        Returns
        -------
        ndarray
            Rasterized pixel-map based segmentation
        """
        #First reprojects shapefile to be consistent with base-map's projection
        dataSource, targetSR =  self._projectShape(shape_fn, mask_name)
        layer = dataSource.GetLayer()
        lat_max, lat_min, lon_min, lon_max = layer.GetExtent()
        #Determine all pixel-based attributes of shapefile based on the base-map's pixel projection
        maxx, miny = self.latlonToPx(lat_min,lon_max)
        minx, maxy = self.latlonToPx(lat_max,lon_min)
        nrows, ncols = min(maxy-miny, self.rows), min(maxx-minx, self.cols)
        minx, miny, maxx, maxy = max(0, minx), max(0, miny), min(maxx, self.cols), min(maxy, self.rows)
        xres = (lat_max-lat_min)/float(maxx-minx)
        yres = (lon_max-lon_min)/float(maxy-miny)
        #generate geotransofrm and datasource based on pixel information
        geotransform =(lat_min,xres,0,lon_max,0, -yres)
        dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Int32)
        dst_ds.SetGeoTransform(geotransform)
        #Rasterize shapefil into dst_ds using pixel information
        gdal.RasterizeLayer(dst_ds, [1], layer, options = ['ATTRIBUTE=FID'])
        padding = ((miny, self.rows - maxy),(self.cols-maxx,minx)) 
        #Pad rasterization with reflection in case the shapefile's shape became slightly different than the map's during the projection
        mask = np.pad(dst_ds.GetRasterBand(1).ReadAsArray(), padding, mode = 'edge')
        mask = np.fliplr(mask)
        #Destroy datasource to free memory
        dataSource.Destroy()
        return mask


if __name__ == '__main__':
    haiti_map = haiti_setup()

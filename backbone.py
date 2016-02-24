#getsegs returns individual image segments as images. It works only with 
#tif files. Resulting tif files are georeferenced. 
#
#getsegs takes 3 input arguments:
#segments- a shape file containing the polygons delineating segment locations
#          the necessary support files should be in the same directory 
#raster- The image file we want to break down into individual segments
#output- the directory the resulting image segments are written to. They are
#        named $dirname/smallseg-N.tif where N is an arbitrary unique number
#
#
#layervalues retruns a dictionary of values representing average pixel value for a given layer, indexed by object number
#
#layervalues takes 3 input arguments:
#input-	a directory containing a collection of image segments created by getsegs()
#layer-	the layer we wish to compute the average intensity for
#alpha-	the layer containing the alpha channel for the object
#
#
#
#haralickvalues returns a dictionary of arrays representing the average of 4 directions for each of the 14 haralick textures
#
#haralickvalues takes one input argument:
#input- a directory containing a collection of image segments created by getsegs()
#
#
#
#labelsegs returns an array of segment numbers that match a given polygon training/validation file
#
#labelsegs takes 3 arguments:
#segments- a shape file containing the polygons delineating segment locations
#          the necessary support files should be in the same directory
#labels-  a shape file containing the polygons delineating label locations
#          the necessary support files should be in the same directory
#overlap- the amount of overlap in decimal percentage required to be considered a match
#	   computed by area

 
def layervalues(input, layer, alpha):
	import os
	import string
	from skimage import io
	import numpy
	
	results={}

	for filename in os.listdir(input):
                if filename[-3:] == 'tif':                        
                        fents=filename.split('-')
                        sents=fents[1].split('.')
                        object=sents[0]
                        actualfile=os.path.join(input,filename)
                        segment=io.imread(actualfile)
                        shape=numpy.shape(segment)
                        pixcount=0
                        pixtotal=0
                        i=0
                        while i<shape[0]:
                                j=0
                                while j<shape[1]:
                                        if segment[i,j,alpha]==255:
                                                pixcount+=1
                                                pixtotal+=segment[i,j,layer]
                                        j+=1
                                i+=1
                        if pixcount>0:
                                average=pixtotal/pixcount
                        else:
                                average=0
                        results[int(object)]=average
	return(results)

def haralickvalues(input):
	import os
	import string
	import numpy
	import mahotas
	
	results={}

	for filename in os.listdir(input):
		fents=filename.split('-')
		sents=fents[1].split('.')
		object=sents[0]
		actualfile=os.path.join(input,filename)
		segment=mahotas.imread(actualfile)
		grey=mahotas.colors.rgb2gray(segment[:,:,0:3],numpy.dtype(int))
		results[int(object)]=mahotas.features.haralick(grey,ignore_zeros=True,compute_14th_feature=True,return_mean=True)
	return(results)


def labelsegs(segments,labels,overlap):
	from osgeo import ogr,gdal
	import os
	import math
	labeledsegments=[]

	driver=ogr.GetDriverByName("ESRI Shapefile")

	#load the segmentation layer
	val_dataSource=driver.Open(segments,0)
	val_layer=val_dataSource.GetLayer()
        val_defn =val_layer.GetLayerDefn()
        for i in range(val_defn.GetFieldCount()):
                print val_defn.GetFieldDefn(i).GetName()

	count=1
	for feature in val_layer:
		#get geometry for segment
		val_geom=feature.GetGeometryRef()
		val_area=val_geom.GetArea()
		
		#load the label data
		lab_dataSource=driver.Open(labels,0)
		lab_layer=lab_dataSource.GetLayer()
                # lab_defn =lab_layer.GetLayerDefn()
                # for i in range(lab_defn.GetFieldCount()):
                #         print lab_defn.GetFieldDefn(i).GetName()
		for lab_feature in lab_layer:
                        # print lab_feature.GetField("Classvalue")
			#get geometry for label
			lab_geom=lab_feature.GetGeometryRef()

			#find how big the intersection is
			intersect=val_geom.Intersection(lab_geom)
			intersect_area=intersect.GetArea()

			if intersect_area>overlap*val_area:
				string= "segment %d" % (count)
				labeledsegments.append((count, lab_feature.GetField("Classvalue")))
		count=count+1
	return labeledsegments


def getsegs(segments, raster, output):
	from osgeo import ogr,gdal
	import os

	driver=ogr.GetDriverByName("ESRI Shapefile")

	#load the segmentation layer
	val_dataSource=driver.Open(segments,1)
	val_layer=val_dataSource.GetLayer()

	#create a new arbitrary field in the attributes
	field=ogr.FieldDefn("forsegs", ogr.OFTInteger)
	val_layer.CreateField(field)

	#assign a unique value to our arbitrary field
	count=1
	for feature in val_layer:
		feature.SetField("forsegs",count)
		val_layer.SetFeature(feature)
		count=count+1

	#write the file
	val_dataSource=None

	#use gdalwarp to cut the individual segment from the master raster
	i=0
	while i<count:
		#note: gdalwarp must be in the path, this may be OS specific
		command='gdalwarp -q -cutline %s -cwhere \"forsegs=%d\" -crop_to_cutline -of GTiff %s  %s\/smallseg-%d.tif' % (segments,i,raster,output,i)
		os.system(command)
		i=i+1

	#reopen our shape file
	val_dataSource=driver.Open(segments,1)
	val_layer=val_dataSource.GetLayer()
	layerDefinition = val_layer.GetLayerDefn()

	for i in range(layerDefinition.GetFieldCount()):
		#find the field we created
		if layerDefinition.GetFieldDefn(i).GetName()[:7]=="forsegs":
			#and delete it
			val_layer.DeleteField(i)
	#write out the shapefile
	val_dataSource=None

def make_cv_segments(data, k):
        import numpy as np
        res = []
        temp = data
        np.random.shuffle(data)
        size = int(len(data) / k)
        for i in range(k - 1):
                fold = data[i * size : (i+1) * size]
                res.append(fold[:, 0], fold[:, 1])
        last = data[(k - 1) * size : ]
        res.append(last[:, 0], last[:, 1])
        return res
        

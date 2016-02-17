from osgeo import ogr,gdal
import os

segments="/research/jpbialas/fullrastertry3.shp"

driver=ogr.GetDriverByName("ESRI Shapefile")

#load the segmentation layer
val_dataSource=driver.Open(segments,1)
val_layer=val_dataSource.GetLayer()

for feature in val_layer:
	val_geom=feature.GetGeometryRef()
	val_area=val_geom.GetArea()
	if val_area<0.0000000001:	
		#print(val_area)
		featId=feature.GetFID()
		val_layer.DeleteFeature(featId)

#write the file
#val_dataSource=None
val_dataSource.ExecuteSQL("REPACK fullrastertry3")

#val_layer.DeleteField(i)


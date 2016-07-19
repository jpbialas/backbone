import numpy as np
import matplotlib.pyplot as plt
from osgeo import ogr
import sys
import matplotlib.pyplot as plt
import analyzeResults
from mapOverlay import MapOverlay

i = 0

mask_fn1 = 'datafromjoe/1-003-002-damage.shp'
fn = 'datafromjoe/1-0003-0002.tif'
shape_fn = 'segmentations/withfeatures2/shapefilewithfeatures003-002-100.shp'
map2 = MapOverlay(fn)
map2.newMask('datafromjoe/1-003-002-damage.shp', 'damage')
print('making segmentation')
map2.new_segmentation(shape_fn)
analyzeResults.segmented_labels(map2)
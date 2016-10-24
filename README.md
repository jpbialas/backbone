# backbone
code for backbone project

Note: If using anaconda with gdal, one or both of the following environment variables might need to be set:

* `export LD_LIBRARY_PATH="/path/to/anaconda/lib"`
* `export GDAL_DATA="/path/to/anaconda/share/gdal"`

To convert input shapefile to output geojson:
* `ogr2ogr -f GeoJSON -preserve_fid -t_srs EPSG:4326 output.json input.shp`

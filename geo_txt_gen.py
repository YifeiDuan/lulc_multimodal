import rasterio
from rasterio.transform import xy

from pyproj import Transformer

def get_geo_coords(tif_path):
    # e.g. tif_path = "/content/drive/MyDrive/Courses/6.8300/Final Project/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"

    with rasterio.open(tif_path) as src:
        # Get the lat/lon of the center pixel
        center_x = src.width // 2
        center_y = src.height // 2
        easting, northing = xy(src.transform, center_y, center_x)
    
        # Transform the coordinate to standard lat/lon under "epsg:4326" system (WGS 84 coordinate system)
        transformer = Transformer.from_crs(src.crs, "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)

    return lon, lat


def gen_geo_txt(tif_path):

    # Get coordinates (lon, lat)
    lon, lat = get_geo_coords(tif_path=tif_path)

    # Generate text
    

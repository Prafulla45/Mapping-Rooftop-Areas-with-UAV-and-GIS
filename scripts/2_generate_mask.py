import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
import numpy as np

def generate_mask(image_path, rooftops_geojson, output_mask_path):
    """
    Generate a segmentation mask for rooftops.

    Args:
        image_path (str): Path to the satellite image.
        rooftops_geojson (str): Path to the GeoJSON file with rooftop polygons.
        output_mask_path (str): Path to save the generated mask.
    """
    # Check if the satellite image exists
    if not os.path.exists(image_path):
        print(f"Error: Satellite image file not found: {image_path}")
        return

    print("Reading satellite image...")
    with rasterio.open(image_path) as src:
        image_bounds = src.bounds  # Get the bounding box of the image
        transform = src.transform  # Get the transform (pixel-to-world mapping)
        out_shape = (src.height, src.width)  # Image dimensions

    print("Loading rooftop polygons...")
    rooftops = gpd.read_file(rooftops_geojson)
    rooftops = rooftops.to_crs(crs=src.crs)  # Match CRS with satellite image

    # Create a bounding box geometry from the image bounds
    bbox = box(*image_bounds)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=src.crs)

    # Filter rooftops that intersect with the satellite image bounds
    rooftops = rooftops[rooftops.geometry.intersects(bbox)]
    print(f"Filtered to {len(rooftops)} rooftops within the satellite image bounds.")

    print("Rasterizing rooftop polygons to create a mask...")
    # Rasterize polygons into a binary mask
    mask = rasterize(
        [(geom, 1) for geom in rooftops.geometry],  # Assign value 1 to rooftops
        out_shape=out_shape,  # Match the dimensions of the satellite image
        transform=transform,
        fill=0,  # Background value
        dtype="uint8"
    )

    print(f"Saving mask to {output_mask_path}...")
    with rasterio.open(
        output_mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="uint8",
        crs=src.crs,
        transform=transform,
    ) as dst:
        dst.write(mask, indexes=1)
    print("Mask saved successfully!")

if __name__ == "__main__":
    # Define absolute paths
    image_path = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/satellite_image.tif"
    rooftops_geojson = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/rooftops.geojson"
    output_mask_path = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/rooftop_mask.tif"

    generate_mask(image_path, rooftops_geojson, output_mask_path)

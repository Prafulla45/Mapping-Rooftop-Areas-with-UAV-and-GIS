import os
import osmnx as ox
import geopandas as gpd

def extract_osm_rooftops(bounding_box):
    """
    Extract building footprints (rooftops) from OpenStreetMap within a bounding box.

    Args:
        bounding_box (tuple): (north, south, east, west) coordinates.

    Returns:
        GeoDataFrame: Contains building polygons.
    """
    print("Fetching building footprints from OpenStreetMap...")

    # Fetch building geometries
    buildings = ox.geometries_from_bbox(*bounding_box, tags={"building": True})

    # Reset index for better handling and remove invalid geometries
    rooftops = buildings.reset_index()
    rooftops = rooftops[rooftops.geometry.notnull()]
    print(f"Extracted {len(rooftops)} building footprints.")
    return rooftops

if __name__ == "__main__":
    # Define the bounding box for the area of interest (north, south, east, west)
    # Example: Kathmandu, Nepal
    bounding_box = (27.7172, 27.7050, 85.3240, 85.3150)

    # Extract rooftops
    rooftops = extract_osm_rooftops(bounding_box)

    # Define output path
    output_dir = r"D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data"
    output_path = os.path.join(output_dir, "rooftops.geojson")

    # Check if the directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist.")
    else:
        print(f"Directory {output_dir} exists, proceeding to save...")

    # Save rooftops to a GeoJSON file
    try:
        print(f"Saving rooftops to {output_path}...")
        rooftops.to_file(output_path, driver="GeoJSON")
        print("Rooftops saved successfully!")
    except Exception as e:
        print(f"Error saving rooftops: {e}")

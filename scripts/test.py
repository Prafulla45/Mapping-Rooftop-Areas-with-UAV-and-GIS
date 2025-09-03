import rasterio
import matplotlib.pyplot as plt
import numpy as np

file_path = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/data/satellite_image_rgb.tif"

with rasterio.open(file_path) as src:
    image = src.read([1, 2, 3])  # Read RGB bands
    image = image.transpose((1, 2, 0))  # Rearrange for visualization

# Normalize the image to [0, 1]
image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

# Display the image
plt.imshow(image_normalized)
plt.title("Satellite Image (Normalized RGB)")
plt.axis("off")
plt.show()

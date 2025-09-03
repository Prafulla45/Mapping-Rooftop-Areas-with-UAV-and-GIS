import numpy as np
import tensorflow as tf
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from PIL import Image

def load_and_preprocess_image(img_path, target_size=(256, 256)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    return img_arr

def post_process(predictions, threshold=0.5):
    binary_images = (predictions > threshold).astype(np.uint8)
    return binary_images

def save_as_geotiff(pred_img, output_path, transform, crs="EPSG:4326"):
    if pred_img.shape[-1] == 4:
        pred_img = pred_img[..., :3]  

    if pred_img.ndim == 3 and pred_img.shape[2] == 3:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=pred_img.shape[0],
            width=pred_img.shape[1],
            count=3,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(pred_img[:, :, 0], 1)  
            dst.write(pred_img[:, :, 1], 2)  
            dst.write(pred_img[:, :, 2], 3)  
    else:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=pred_img.shape[0],
            width=pred_img.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(pred_img, 1)

def display_images(original_img, pred_mask_img, overlay_img, ax=None):
    if ax is None:
        ax = plt.gca()

    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pred_mask_img, cmap='gray')
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    ax[2].imshow(overlay_img)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

def main():
    model_path = "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/models/unet_final.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully.")

    image_paths = [
        "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/inputs/image.png",
        "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/inputs/image1.png",
        "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/inputs/image2.png",
        "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/inputs/image3.png"
    ]

    output_paths = {
        "original": [
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/original1.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/original2.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/original3.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/original4.tif"
        ],
        "mask": [
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/mask1.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/mask2.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/mask3.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/mask4.tif"
        ],
        "overlay": [
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/overlay1.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/overlay2.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/overlay3.tif",
            "D:/Rishikesh/BCT/VI/Minor/rooftop-detection/outputs/overlay4.tif"
        ]
    }

    min_x, max_x = 85.3100, 85.3120  
    min_y, max_y = 27.7000, 27.7020  

    transform = from_origin(min_x, max_y, (max_x - min_x) / 256, (max_y - min_y) / 256)

    original_images = []
    mask_images = []
    overlay_images = []

    for i, image_path in enumerate(image_paths):
        img_arr = load_and_preprocess_image(image_path)

        prediction = model.predict(img_arr)
        pred_mask = post_process(prediction, threshold=0.65)[0, :, :, 0]

        original_img = Image.open(image_path).resize((256, 256))
        original_img_arr = np.array(original_img, dtype=np.uint8)
        save_as_geotiff(original_img_arr, output_paths["original"][i], transform)

        save_as_geotiff(pred_mask, output_paths["mask"][i], transform)

        overlay_img = np.array(original_img) / 255.0  
        overlay_img[:, :, 0] = np.maximum(overlay_img[:, :, 0], pred_mask)  
        overlay_img_arr = (overlay_img * 255).astype(np.uint8)
        save_as_geotiff(overlay_img_arr, output_paths["overlay"][i], transform)

        print(f"Saved {output_paths['original'][i]}, {output_paths['mask'][i]}, and {output_paths['overlay'][i]}.")

        original_images.append(original_img)
        mask_images.append(pred_mask)
        overlay_images.append(overlay_img)

    fig, ax = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))

    for i in range(len(image_paths)):
        ax[i, 0].imshow(original_images[i])
        ax[i, 0].set_title(f"Original {i+1}")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(mask_images[i], cmap='gray')
        ax[i, 1].set_title(f"Mask {i+1}")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(overlay_images[i])
        ax[i, 2].set_title(f"Overlay {i+1}")
        ax[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

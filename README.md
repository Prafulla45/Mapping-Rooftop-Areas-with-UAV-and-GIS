# Mapping Rooftop Areas with UAV and GIS

A deep learning pipeline for detecting and mapping building rooftop areas from UAV (drone) and satellite imagery using semantic segmentation. The project combines OpenStreetMap data, GIS raster processing, and two segmentation model architectures (U-Net with TensorFlow/Keras and DeepLabV3+ with PyTorch) to produce georeferenced rooftop masks.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Pipeline](#pipeline)
  - [Step 1 - Extract OSM Building Footprints](#step-1---extract-osm-building-footprints)
  - [Step 2 - Generate Segmentation Masks](#step-2---generate-segmentation-masks)
  - [Step 3 - Fine-Tune the Model](#step-3---fine-tune-the-model)
  - [Step 4 - Run Inference](#step-4---run-inference)
- [Models](#models)
- [Inputs and Outputs](#inputs-and-outputs)
- [Notebooks](#notebooks)

---

## Overview

This project addresses the problem of automatically identifying and delineating rooftop areas from aerial and satellite imagery. The workflow is as follows:

1. Building footprint polygons are fetched from OpenStreetMap for a given geographic area.
2. The polygons are rasterized against a reference satellite image to create binary segmentation masks.
3. A segmentation model (U-Net or DeepLabV3+) is trained or fine-tuned on the image-mask pairs.
4. The trained model is applied to new images to produce predicted rooftop masks and visual overlays, which are saved as georeferenced GeoTIFF files.

The project was developed and tested using imagery of Kathmandu, Nepal (bounding box approximately 27.70 N, 85.31 E).

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # Raw satellite imagery
│   ├── preprocessed/         # Preprocessed imagery and masks
│   └── predictions/          # Model prediction outputs
├── inputs/                   # Input PNG images for quick inference
├── models/
│   ├── unet_final.h5         # Trained U-Net model (Keras/TensorFlow)
│   └── unet_final_2.h5       # Alternative U-Net checkpoint
├── notebooks/
│   └── data-modelling.ipynb  # Exploratory notebook for data modelling
├── outputs/                  # GeoTIFF outputs (original, mask, overlay)
├── scripts/
│   ├── 1_extract_osm.py      # Fetch building footprints from OSM
│   ├── 2_generate_mask.py    # Rasterize footprints to binary masks
│   ├── 3_fine_tune_model.py  # Fine-tune DeepLabV3+ (PyTorch)
│   └── 4_inference.py        # Run inference with the fine-tuned model
├── inference.py              # Inference script using U-Net (TensorFlow)
├── requirements.txt          # Python dependencies
└── __init__.py
```

---

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training; CPU inference is supported)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries used:

| Library | Purpose |
|---|---|
| TensorFlow / Keras | U-Net model loading and inference |
| PyTorch / torchvision | DeepLabV3+ training and inference |
| rasterio | Reading and writing GeoTIFF raster files |
| geopandas | Handling vector GIS data |
| osmnx | Querying OpenStreetMap building footprints |
| OpenCV | Image blending and visualization |
| Pillow | Image loading and resizing |
| NumPy | Array operations |

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Prafulla45/Mapping-Rooftop-Areas-with-UAV-and-GIS.git
cd Mapping-Rooftop-Areas-with-UAV-and-GIS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your satellite imagery (GeoTIFF format) in the `data/raw/` directory.

4. Update the file paths inside each script to point to your local data directories before running.

---

## Pipeline

### Step 1 - Extract OSM Building Footprints

**Script:** `scripts/1_extract_osm.py`

Fetches building polygon geometries from OpenStreetMap using a bounding box and saves them as a GeoJSON file.

```bash
python scripts/1_extract_osm.py
```

Edit the `bounding_box` variable (north, south, east, west) and `output_dir` path to match your area of interest.

---

### Step 2 - Generate Segmentation Masks

**Script:** `scripts/2_generate_mask.py`

Rasterizes the OSM building footprints against a reference satellite image to produce a binary mask GeoTIFF where rooftop pixels are 1 and background pixels are 0.

```bash
python scripts/2_generate_mask.py
```

Update `image_path`, `rooftops_geojson`, and `output_mask_path` to your local paths.

---

### Step 3 - Fine-Tune the Model

**Script:** `scripts/3_fine_tune_model.py`

Fine-tunes a pretrained DeepLabV3+ (ResNet-101 backbone) model on the generated image-mask pairs using PyTorch. BatchNorm layers are replaced with GroupNorm to support single-image batch sizes.

```bash
python scripts/3_fine_tune_model.py
```

Update `image_paths` and `mask_paths` to your local `.tif` file paths. The trained weights are saved to `models/fine_tuned_rooftop_model.pth`.

---

### Step 4 - Run Inference

**Script:** `scripts/4_inference.py`

Loads the fine-tuned DeepLabV3+ model and runs inference on a test image. The predicted rooftop mask is overlaid on the original image in green and saved to `outputs/inference_result.png`.

```bash
python scripts/4_inference.py
```

**Alternative - U-Net inference:**

**Script:** `inference.py`

Runs inference using the saved U-Net (`.h5`) model. Processes multiple input images and saves original, mask, and overlay as georeferenced GeoTIFF files to the `outputs/` directory.

```bash
python inference.py
```

---

## Models

| File | Architecture | Framework |
|---|---|---|
| `models/unet_final.h5` | U-Net | TensorFlow / Keras |
| `models/unet_final_2.h5` | U-Net (alternate checkpoint) | TensorFlow / Keras |
| `models/fine_tuned_rooftop_model.pth` | DeepLabV3+ ResNet-101 | PyTorch (generated by training) |

The U-Net model uses a threshold of 0.65 on sigmoid output to produce binary masks. The DeepLabV3+ model uses argmax over two output classes (background and rooftop).

---

## Inputs and Outputs

**Inputs (`inputs/`):**
- `image.png`, `image1.png`, `image2.png`, `image3.png` - Sample UAV/satellite image tiles used for U-Net inference.

**Outputs (`outputs/`):**
- `original{n}.tif` - Georeferenced original image tiles
- `mask{n}.tif` - Predicted binary rooftop masks
- `overlay{n}.tif` - Original images with predicted masks blended in

All output GeoTIFFs use EPSG:4326 (WGS 84) as the coordinate reference system

---

## Notebooks

`notebooks/data-modelling.ipynb` contains exploratory analysis and model development experiments used during the research phase of the project.

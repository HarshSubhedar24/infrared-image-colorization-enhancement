"""
==============================================================
BE FINAL YEAR PROJECT

Infrared Image Colorization and Enhancement 

STEP 01
Combine RGB Bands (Window-Based Version)

==============================================================
"""


import os
import glob
import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.windows import Window

from scene_selection import (
    display_existing_output_files,
    select_scene_directory,
)


# ==========================================================
# SELECT SCENE FOLDER
# ==========================================================

BASE_PATH = select_scene_directory()

# ==========================================================
# AUTOMATICALLY FIND REQUIRED BANDS
# ==========================================================

def find_band(folder, keyword):
    files = glob.glob(os.path.join(folder, f"*{keyword}.TIF"))

    if len(files) == 0:
        raise FileNotFoundError(f"{keyword} not found in:\n{folder}")

    return files[0]


print("=" * 60)
print("Scanning Scene Folder...")
print("=" * 60)

blue_band = find_band(str(BASE_PATH), "SR_B2")
green_band = find_band(str(BASE_PATH), "SR_B3")
red_band = find_band(str(BASE_PATH), "SR_B4")

print("✓ Blue Band  :", os.path.basename(blue_band))
print("✓ Green Band :", os.path.basename(green_band))
print("✓ Red Band   :", os.path.basename(red_band))

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

output_folder = os.path.join(str(BASE_PATH), "Output")
display_existing_output_files(Path(output_folder))
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "RGB_Image.png")
TILE_SIZE = 512

# ==========================================================
# READ LANDSAT BANDS
# ==========================================================

# ==========================================================
# PERCENTILE CONTRAST STRETCH
# ==========================================================

def get_percentiles(path):
    with rasterio.open(path) as src:
        sample_height = max(1, src.height // 8)
        sample_width = max(1, src.width // 8)
        sample = src.read(
            1,
            out_shape=(1, sample_height, sample_width),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float32)
    valid = sample[sample > 0]
    if valid.size == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, 2)), float(np.percentile(valid, 98))


def percentile_stretch(image, p2, p98):
    stretched = image.astype(np.float32, copy=True)
    np.clip(stretched, p2, p98, out=stretched)
    np.subtract(stretched, p2, out=stretched)
    np.divide(stretched, max(p98 - p2, 1e-6), out=stretched)
    np.multiply(stretched, 255.0, out=stretched)
    return np.rint(stretched).astype(np.uint8)


print("\nReading Landsat Bands...")
with rasterio.open(red_band) as red_src, rasterio.open(green_band) as green_src, rasterio.open(blue_band) as blue_src:
    if not (red_src.width == green_src.width == blue_src.width and red_src.height == green_src.height == blue_src.height):
        raise ValueError("Band dimensions do not match.")

    dimensions = (red_src.width, red_src.height)
    percentiles = [
        get_percentiles(path)
        for path in (red_band, green_band, blue_band)
    ]
    profile = {
        "driver": "PNG",
        "width": red_src.width,
        "height": red_src.height,
        "count": 3,
        "dtype": "uint8",
    }
    print("✓ Bands verified:", dimensions)
    print("\nSaving RGB Image in windows...")
    with rasterio.open(output_file, "w", **profile) as destination:
        for top in range(0, red_src.height, TILE_SIZE):
            for left in range(0, red_src.width, TILE_SIZE):
                height = min(TILE_SIZE, red_src.height - top)
                width = min(TILE_SIZE, red_src.width - left)
                window = Window(left, top, width, height)
                rgb_tile = np.empty((3, height, width), dtype=np.uint8)
                for channel_index, (source, (p2, p98)) in enumerate(zip(
                    (red_src, green_src, blue_src),
                    percentiles,
                )):
                    band = source.read(1, window=window)
                    rgb_tile[channel_index] = percentile_stretch(band, p2, p98)
                    del band
                destination.write(rgb_tile, window=window)
                del rgb_tile
    print("✓ RGB Image Saved Successfully")
    print("Location :", output_file)

# ==========================================================
# DISPLAY PREVIEW
# ==========================================================

print("\nGenerating Preview...")

with rasterio.open(output_file) as output:
    preview = output.read(
        out_shape=(3, 1200, 1200),
        resampling=rasterio.enums.Resampling.bilinear,
    ).transpose(1, 2, 0)

plt.figure(figsize=(10, 10))
plt.imshow(preview)
plt.title("Generated RGB Image")
plt.axis("off")
plt.show()

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
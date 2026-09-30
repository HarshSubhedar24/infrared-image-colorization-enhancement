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

# ==========================================================
# READ LANDSAT BANDS
# ==========================================================

print("\nReading Landsat Bands...")

with rasterio.open(red_band) as src:
    red = src.read(1)

with rasterio.open(green_band) as src:
    green = src.read(1)

with rasterio.open(blue_band) as src:
    blue = src.read(1)

print("✓ Bands Loaded Successfully")
print("Image Size :", red.shape)

# ==========================================================
# VERIFY IMAGE DIMENSIONS
# ==========================================================

assert red.shape == green.shape == blue.shape, \
    "Band dimensions do not match!"

print("✓ Band Dimensions Verified")

# ==========================================================
# PERCENTILE CONTRAST STRETCH
# ==========================================================

def percentile_stretch(image):

    image = image.astype(np.float32)

    valid = image[image > 0]

    if len(valid) == 0:
        return np.zeros_like(image)

    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)

    image = np.clip(image, p2, p98)

    image = (image - p2) / max((p98 - p2), 1e-6)

    return image


print("\nApplying Contrast Stretch...")

red = percentile_stretch(red)
green = percentile_stretch(green)
blue = percentile_stretch(blue)

print("✓ Contrast Stretch Completed")

# ==========================================================
# CONVERT TO UINT8
# ==========================================================

print("\nConverting to 8-bit...")

red = (red * 255).astype(np.uint8)
green = (green * 255).astype(np.uint8)
blue = (blue * 255).astype(np.uint8)

print("✓ Conversion Completed")

# ==========================================================
# STACK RGB CHANNELS
# ==========================================================

print("\nCombining RGB Bands...")

rgb = np.stack((red, green, blue), axis=-1)

print("✓ RGB Image Created")
print("RGB Shape :", rgb.shape)

# ==========================================================
# SAVE IMAGE
# ==========================================================

print("\nSaving RGB Image...")

success = cv2.imwrite(
    output_file,
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
)

if success:
    print("✓ RGB Image Saved Successfully")
    print("Location :", output_file)
else:
    print("Failed to save image.")

# ==========================================================
# DISPLAY PREVIEW
# ==========================================================

print("\nGenerating Preview...")

preview = cv2.resize(rgb, (1200, 1200))

plt.figure(figsize=(10, 10))
plt.imshow(preview)
plt.title("Generated RGB Image")
plt.axis("off")
plt.show()

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
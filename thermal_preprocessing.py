"""
==============================================================
BE FINAL YEAR PROJECT

Infrared Image Colorization and Enhancement using
Deep Learning

STEP 02
Thermal Image Preparation (Window-Based Version)

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import gc
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
# CONFIGURATION
# ==========================================================

# Parent folder and scene are selected interactively at runtime.
BASE_PATH = select_scene_directory()

# Tile size
# Larger value = Faster
# Smaller value = Less RAM

TILE_SIZE = 1024

# Contrast Stretch

LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98

# Temperature Conversion

TEMPERATURE_SCALE = 0.00341802
TEMPERATURE_OFFSET = 149.0

# Preview Image Size

PREVIEW_SIZE = 1000

# ==========================================================
# CONSOLE HEADER
# ==========================================================

print("\n")
print("=" * 60)
print("THERMAL IMAGE PREPARATION")
print("WINDOW BASED PROCESSING")
print("=" * 60)

print("\nSearching for Landsat Thermal Band...")

# ==========================================================
# SEARCH FOR ST_B10
# ==========================================================

thermal_path = None

for root, dirs, files in os.walk(BASE_PATH):

    for file in files:

        if file.upper().endswith("ST_B10.TIF"):

            thermal_path = Path(root) / file

            break

    if thermal_path is not None:
        break

if thermal_path is None:

    raise FileNotFoundError(
        "\nERROR : ST_B10.TIF not found."
    )

print("SUCCESS")
print("\nThermal Band Found")

print(thermal_path)

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

OUTPUT_FOLDER = BASE_PATH / "Output"
display_existing_output_files(OUTPUT_FOLDER)

PNG_FOLDER = OUTPUT_FOLDER / "Thermal_PNG"

TIFF_FOLDER = OUTPUT_FOLDER / "Thermal_TIFF"

PREVIEW_FOLDER = OUTPUT_FOLDER / "Preview"

PNG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

TIFF_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOutput folders created successfully.")

# ==========================================================
# OUTPUT FILES
# ==========================================================

PNG_OUTPUT = PNG_FOLDER / "Thermal_Image.png"

TIFF_OUTPUT = TIFF_FOLDER / "Temperature_Celsius.tif"

PREVIEW_OUTPUT = PREVIEW_FOLDER / "Thermal_Preview.png"

print("\nOutput Files")

print("------------------------------")

print("PNG      :", PNG_OUTPUT)

print("GeoTIFF  :", TIFF_OUTPUT)

print("Preview  :", PREVIEW_OUTPUT)

print("-" * 30)

# ==========================================================
# OPEN RASTER
# ==========================================================

print("\nOpening Thermal Raster...")

src = rasterio.open(thermal_path)

print("SUCCESS")

print("\nRaster Information")

print("------------------------------")

print("Width       :", src.width)

print("Height      :", src.height)

print("Bands       :", src.count)

print("Data Type   :", src.dtypes[0])

print("CRS         :", src.crs)

print("------------------------------")

# ==========================================================
# STORE METADATA
# ==========================================================

profile = src.profile.copy()

transform = src.transform

crs = src.crs

nodata = src.nodata

print("\nMetadata Loaded Successfully")

# ==========================================================
# WINDOW INFORMATION
# ==========================================================

rows = src.height

cols = src.width

print("\nWindow Configuration")

print("------------------------------")

print("Tile Size :", TILE_SIZE)

print("Rows      :", rows)

print("Columns   :", cols)

print("------------------------------")

print("\nPART 1A COMPLETED SUCCESSFULLY")

print("=" * 60)

# ==========================================================
# PART 1B
# UTILITY FUNCTIONS
# ==========================================================

print("\nLoading Utility Functions...")

# ==========================================================
# FUNCTION :
# Convert DN -> Kelvin
# ==========================================================

def dn_to_kelvin(dn):
    """
    Convert Landsat ST_B10 Digital Numbers (DN)
    into Surface Temperature (Kelvin).
    """

    return (
        dn.astype(np.float32)
        * TEMPERATURE_SCALE
    ) + TEMPERATURE_OFFSET


# ==========================================================
# FUNCTION :
# Kelvin -> Celsius
# ==========================================================

def kelvin_to_celsius(kelvin):
    """
    Convert Kelvin to Celsius.
    """

    return (
        kelvin - 273.15
    ).astype(np.float32)


# ==========================================================
# FUNCTION :
# Remove NoData Pixels
# ==========================================================

def remove_nodata(image, nodata_value):
    """
    Replace NoData pixels with NaN.
    """

    image = image.astype(np.float32)

    if nodata_value is not None:
        image[image == nodata_value] = np.nan
    else:
        image[image == 0] = np.nan

    return image


# ==========================================================
# FUNCTION :
# Print Temperature Statistics
# ==========================================================

def print_statistics(image):

    valid = image[np.isfinite(image)]

    print("\nTemperature Statistics")
    print("-" * 35)

    print(f"Minimum : {np.min(valid):.2f} °C")
    print(f"Maximum : {np.max(valid):.2f} °C")
    print(f"Mean    : {np.mean(valid):.2f} °C")
    print(f"Median  : {np.median(valid):.2f} °C")
    print(f"Std Dev : {np.std(valid):.2f}")

    print("-" * 35)


# ==========================================================
# FUNCTION :
# Memory Cleanup
# ==========================================================

def cleanup():
    """
    Release unused memory.
    """

    gc.collect()


# ==========================================================
# FUNCTION :
# Create Processing Windows
# ==========================================================

def generate_windows(dataset):

    """
    Generate processing windows for the raster.

    This allows us to process very large
    Landsat images using a constant amount
    of RAM.
    """

    for row in range(0, dataset.height, TILE_SIZE):

        for col in range(0, dataset.width, TILE_SIZE):

            width = min(
                TILE_SIZE,
                dataset.width - col
            )

            height = min(
                TILE_SIZE,
                dataset.height - row
            )

            yield Window(
                col_off=col,
                row_off=row,
                width=width,
                height=height
            )


print("Utility Functions Loaded Successfully")

print("\nPART 1B COMPLETED SUCCESSFULLY")

print("=" * 60)

# ==========================================================
# PART 1C
# PASS 1 : GLOBAL STATISTICS
# ==========================================================

print("\nStarting Pass 1 : Global Statistics")
print("=" * 60)

# Storage for global statistics
global_min = np.inf
global_max = -np.inf
global_sum = 0.0
global_count = 0

# Store a bounded sample from each tile for statistics. The complete raster
# remains windowed and is never concatenated in memory.
temperature_samples = []
STATISTICS_SAMPLES_PER_TILE = 4096

tile_number = 0

# ----------------------------------------------------------
# Process one tile at a time
# ----------------------------------------------------------

for window in generate_windows(src):

    tile_number += 1

    print(f"Processing Tile {tile_number}", end="\r")

    # Read only the current window
    thermal_dn = src.read(
        1,
        window=window
    )

    # Remove NoData
    thermal_dn = remove_nodata(
        thermal_dn,
        nodata
    )

    # Convert DN → Kelvin
    thermal_kelvin = dn_to_kelvin(
        thermal_dn
    )

    # Convert Kelvin → Celsius
    thermal_celsius = kelvin_to_celsius(
        thermal_kelvin
    )

    # Valid pixels only
    valid_pixels = thermal_celsius[
        np.isfinite(thermal_celsius)
    ]

    if valid_pixels.size == 0:

        continue

    # Update global statistics
    global_min = min(
        global_min,
        np.min(valid_pixels)
    )

    global_max = max(
        global_max,
        np.max(valid_pixels)
    )

    global_sum += np.sum(valid_pixels)

    global_count += valid_pixels.size

    # Keep only a bounded, evenly distributed sample from this tile.
    sample_count = min(STATISTICS_SAMPLES_PER_TILE, valid_pixels.size)
    sample_indices = np.linspace(
        0,
        valid_pixels.size - 1,
        num=sample_count,
        dtype=np.intp,
    )
    temperature_samples.append(valid_pixels[sample_indices].copy())

    # Free memory
    del thermal_dn
    del thermal_kelvin
    del thermal_celsius
    del valid_pixels

    cleanup()

print("\n")

# ----------------------------------------------------------
# Combine all valid pixels
# ----------------------------------------------------------

print("Combining Sampled Valid Pixels...")

temperature_values = np.concatenate(
    temperature_samples
)

print("Completed")

# ----------------------------------------------------------
# Calculate Global Mean
# ----------------------------------------------------------

global_mean = (
    global_sum / global_count
)

# ----------------------------------------------------------
# Calculate Global Standard Deviation
# ----------------------------------------------------------

global_std = np.std(temperature_values)

# ----------------------------------------------------------
# Print Statistics
# ----------------------------------------------------------

print("\nGlobal Temperature Statistics")

print("-" * 40)

print(f"Minimum : {global_min:.2f} °C")

print(f"Maximum : {global_max:.2f} °C")

print(f"Mean    : {global_mean:.2f} °C")

print(f"Std Dev : {global_std:.2f}")

print("-" * 40)

print("\nPASS 1 COMPLETED")

print("=" * 60)

# ==========================================================
# PART 1D
# GLOBAL PERCENTILE CALCULATION
# ==========================================================

print("\nCalculating Global Percentiles...")
print("=" * 60)

# ----------------------------------------------------------
# Calculate 2% and 98% percentiles
# ----------------------------------------------------------

p2 = np.percentile(
    temperature_values,
    LOW_PERCENTILE
)

p98 = np.percentile(
    temperature_values,
    HIGH_PERCENTILE
)

print("\nPercentiles Computed Successfully")

print("------------------------------------------")
print(f"{LOW_PERCENTILE}% Percentile  : {p2:.2f} °C")
print(f"{HIGH_PERCENTILE}% Percentile : {p98:.2f} °C")
print("------------------------------------------")

# ----------------------------------------------------------
# Safety Check
# ----------------------------------------------------------

if p98 <= p2:

    raise ValueError(
        "\nInvalid percentile values detected."
    )

print("\nPercentile values are valid.")

# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------

cleanup()

print("\nPART 1D COMPLETED SUCCESSFULLY")
print("=" * 60)

# ==========================================================
# PART 2A
# PREPARE OUTPUT FILES & START PASS 2
# ==========================================================

print("\n")
print("=" * 60)
print("PART 2A : GENERATING THERMAL OUTPUT")
print("=" * 60)



# ----------------------------------------------------------
# Update GeoTIFF Profile
# ----------------------------------------------------------

output_profile = profile.copy()

output_profile.update(
    dtype=rasterio.float32,
    count=1,
    compress="lzw"
)

# ----------------------------------------------------------
# Allocate PNG Buffer
# ----------------------------------------------------------

print("\nAllocating Output Image...")

thermal_png = np.zeros(
    (src.height, src.width),
    dtype=np.uint8
)

print("Output Buffer Created")

# ----------------------------------------------------------
# Open Output GeoTIFF
# ----------------------------------------------------------

dst = rasterio.open(
    str(TIFF_OUTPUT),
    "w",
    **output_profile
)
print("GeoTIFF Ready")

# ----------------------------------------------------------
# Start Pass 2
# ----------------------------------------------------------

print("\nStarting Pass 2...")
print("=" * 60)

tile_number = 0

for window in generate_windows(src):

    tile_number += 1

    print(
        f"Processing Tile {tile_number}",
        end="\r"
    )

    # ---------------------------------------------
    # Read Current Tile
    # ---------------------------------------------

    thermal_dn = src.read(
        1,
        window=window
    )

    # ---------------------------------------------
    # Remove NoData
    # ---------------------------------------------

    thermal_dn = remove_nodata(
        thermal_dn,
        nodata
    )

    # ---------------------------------------------
    # Convert DN → Kelvin
    # ---------------------------------------------

    thermal_kelvin = dn_to_kelvin(
        thermal_dn
    )

    # ---------------------------------------------
    # Convert Kelvin → Celsius
    # ---------------------------------------------

    thermal_celsius = kelvin_to_celsius(
        thermal_kelvin
    )

    # =====================================================
    # PART 2B STARTS HERE
    # =====================================================
    # ----------------------------------------------------------
    # Contrast Stretch using Global Percentiles
    # ----------------------------------------------------------

    thermal_celsius = np.clip(
        thermal_celsius,
        p2,
        p98
    )

    thermal_celsius = (
        thermal_celsius - p2
    ) / max((p98 - p2), 1e-6)

    # ----------------------------------------------------------
    # Convert to 8-bit
    # ----------------------------------------------------------

    thermal_8bit = (
        thermal_celsius * 255
    )

    thermal_8bit = np.nan_to_num(
        thermal_8bit,
        nan=0
    )

    thermal_8bit = thermal_8bit.astype(
        np.uint8
    )

    # ----------------------------------------------------------
    # Save Original Temperature to GeoTIFF
    # (Not the normalized image)
    # ----------------------------------------------------------

    original_temperature = kelvin_to_celsius(
        dn_to_kelvin(
            thermal_dn
        )
    )

    original_temperature = np.nan_to_num(
        original_temperature,
        nan=-9999
    )

    dst.write(
        original_temperature.astype(np.float32),
        1,
        window=window
    )

    # ----------------------------------------------------------
    # Copy Processed Tile into PNG Buffer
    # ----------------------------------------------------------

    row_start = window.row_off
    row_end = row_start + window.height

    col_start = window.col_off
    col_end = col_start + window.width

    thermal_png[
        row_start:row_end,
        col_start:col_end
    ] = thermal_8bit

    # ----------------------------------------------------------
    # Free Memory
    # ----------------------------------------------------------

    del thermal_dn
    del thermal_kelvin
    del thermal_celsius
    del thermal_8bit
    del original_temperature

    cleanup()
# ==========================================================
# PART 2C
# SAVE OUTPUT FILES
# ==========================================================

print("\n")
print("=" * 60)
print("PART 2C : SAVING OUTPUT FILES")
print("=" * 60)

# ----------------------------------------------------------
# Close GeoTIFF
# ----------------------------------------------------------

dst.close()

print("\n✓ Temperature GeoTIFF Saved Successfully")
print("Location :", TIFF_OUTPUT)

# ----------------------------------------------------------
# Save Thermal PNG
# ----------------------------------------------------------

print("\nSaving Thermal PNG...")

success = cv2.imwrite(
    str(PNG_OUTPUT),
    thermal_png
)

if success:

    print("✓ Thermal PNG Saved Successfully")
    print("Location :", PNG_OUTPUT)

else:

    print("✗ Failed to Save Thermal PNG")

# ----------------------------------------------------------
# Final Cleanup
# ----------------------------------------------------------

cleanup()

print("\nOutput Files Created Successfully")

print("=" * 60)
print("PART 2C COMPLETED SUCCESSFULLY")
print("=" * 60)

# ==========================================================
# PART 2D
# PREVIEW & HISTOGRAM
# ==========================================================

print("\n")
print("=" * 60)
print("PART 2D : GENERATING PREVIEW")
print("=" * 60)

# ----------------------------------------------------------
# Resize Preview Image
# ----------------------------------------------------------

print("\nGenerating Preview...")

preview = cv2.resize(
    thermal_png,
    (PREVIEW_SIZE, PREVIEW_SIZE),
    interpolation=cv2.INTER_NEAREST
)

# ----------------------------------------------------------
# Preview File Names
# ----------------------------------------------------------

GRAY_PREVIEW = PREVIEW_FOLDER / "Thermal_Gray_Preview.png"

INFERNO_PREVIEW = PREVIEW_FOLDER / "Thermal_Inferno_Preview.png"

HISTOGRAM_PREVIEW = PREVIEW_FOLDER / "Thermal_Histogram.png"

# ----------------------------------------------------------
# Preview 1 : Grayscale
# ----------------------------------------------------------

gray_figure = plt.figure(figsize=(8, 8), dpi=100)

plt.imshow(preview, cmap="gray")

plt.title("Thermal Image (Training Input)")

plt.axis("off")

plt.savefig(
    str(GRAY_PREVIEW),
    dpi=100,
    bbox_inches="tight"
)

plt.show()

plt.close(gray_figure)

print("✓ Saved :", GRAY_PREVIEW)

# ----------------------------------------------------------
# Preview 2 : Inferno
# ----------------------------------------------------------

inferno_figure = plt.figure(figsize=(8, 8), dpi=100)

plt.imshow(preview, cmap="inferno")

plt.title("Thermal Image (Visualization)")

plt.axis("off")

plt.savefig(
    str(INFERNO_PREVIEW),
    dpi=100,
    bbox_inches="tight"
)

plt.show()

plt.close(inferno_figure)

print("✓ Saved :", INFERNO_PREVIEW)

# ----------------------------------------------------------
# Histogram
# ----------------------------------------------------------


histogram_figure = plt.figure(figsize=(9, 5), dpi=100)

plt.hist(
    thermal_png.ravel(),
    bins=256,
    color="darkred",
    edgecolor="black"
)

plt.title("Thermal Image Histogram")

plt.xlabel("Pixel Value (0-255)")

plt.ylabel("Number of Pixels")

plt.grid(True)

# Save histogram
plt.savefig(
    str(HISTOGRAM_PREVIEW),
    dpi=100,
    bbox_inches="tight"
)

plt.show()

plt.close(histogram_figure)

print("✓ Histogram Saved :", HISTOGRAM_PREVIEW)
# ----------------------------------------------------------
# Image Information
# ----------------------------------------------------------

print("\nImage Information")
print("-" * 40)

print("Image Shape      :", thermal_png.shape)
print("Data Type        :", thermal_png.dtype)
print("Minimum Pixel    :", np.min(thermal_png))
print("Maximum Pixel    :", np.max(thermal_png))
pixel_mean, pixel_std = cv2.meanStdDev(thermal_png)
print("Mean Pixel       : {:.2f}".format(float(pixel_mean[0, 0])))
print("Standard Deviation : {:.2f}".format(float(pixel_std[0, 0])))

print("-" * 40)

print("\nPART 2D COMPLETED SUCCESSFULLY")

print("=" * 60)

# ==========================================================
# PART 2E
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 65)
print("THERMAL IMAGE PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nPROJECT INFORMATION")
print("-" * 40)

print("Project : Infrared Image Colorization and Enhancement")
print("Step    : Step 2 - Thermal Image Preparation")

print("\nINPUT FILE")
print("-" * 40)

print("Thermal Band :")
print(thermal_path)

print("\nOUTPUT FILES")
print("-" * 40)

print("Thermal PNG (ESRGAN Input)")
print(PNG_OUTPUT)

print()

print("Temperature GeoTIFF")
print(TIFF_OUTPUT)

print("\nIMAGE DETAILS")
print("-" * 40)

print(f"Image Size : {src.width} x {src.height}")

print(f"Tile Size  : {TILE_SIZE} x {TILE_SIZE}")

print(f"Data Type  : uint8 (PNG)")
print(f"GeoTIFF    : float32")

print("\nTEMPERATURE STATISTICS")
print("-" * 40)

print(f"Minimum Temperature : {global_min:.2f} °C")

print(f"Maximum Temperature : {global_max:.2f} °C")

print(f"Mean Temperature    : {global_mean:.2f} °C")

print(f"Standard Deviation  : {global_std:.2f}")

print()

print(f"{LOW_PERCENTILE}% Percentile  : {p2:.2f} °C")

print(f"{HIGH_PERCENTILE}% Percentile : {p98:.2f} °C")

print("\nNEXT STEP")
print("-" * 40)

print("✓ Thermal PNG is ready for ESRGAN training.")

print("✓ Temperature GeoTIFF is ready for scientific analysis.")

print("\nWORKFLOW STATUS")
print("-" * 40)

print("[✓] Dataset Downloaded")

print("[✓] Required Bands Selected")

print("[✓] RGB Image Generated")

print("[✓] Thermal Image Prepared")

print("[ ] ESRGAN Training")

print("[ ] Enhanced Thermal Generation")

print("[ ] Pix2Pix Training")

print("[ ] Model Evaluation")

print("[ ] GUI Development")

print("\nCleaning Temporary Memory...")

cleanup()

try:
    src.close()
except:
    pass

print("\n")
print("=" * 65)
print("\nAll Steps Completed Successfully")
print("=" * 65)

print("READY FOR STEP 3 : ESRGAN TRAINING")
print("=" * 65)

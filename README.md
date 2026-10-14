# Infrared Image Colorization and Enhancement

## Project title

**Infrared Image Colorization and Enhancement for Improved Object
Interpretation**

This project develops a remote-sensing pipeline for processing Landsat 8/9
imagery. It prepares visible RGB imagery and thermal imagery, enhances the
thermal image with a low-memory ESRGAN model, aligns the enhanced thermal
image to the RGB grid, and prepares the data for a future Pix2Pix
thermal-to-RGB colorization stage.

## Project objective

The objective is to improve the visual interpretation of thermal/infrared
imagery by:

1. Extracting RGB bands B2, B3, and B4.
2. Converting Landsat ST_B10 thermal data to Celsius.
3. Creating training data for thermal super-resolution.
4. Enhancing the thermal image with ESRGAN.
5. Aligning the enhanced thermal image to the original RGB pixel grid.
6. Translating the aligned thermal image to a colorized RGB image with Pix2Pix
   (planned).
7. Evaluating the generated result with image-quality metrics (planned).

## Implemented workflow

```text
Landsat 8/9 scene
        |
        +--> SR_B2, SR_B3, SR_B4
        |        |
        |        +--> RGB_Image.png
        |
        +--> ST_B10
                 |
                 +--> Temperature_Celsius.tif
                 +--> Thermal_Image.png
                              |
                              v
                    ESRGAN dataset preparation
                              |
                              v
                       ESRGAN training
                              |
                              v
                    Enhanced_Thermal_Image.tif
                              |
                              v
                  Pix2Pix grid alignment
                              |
                              v
                  Enhanced_Thermal_Pix2Pix.tif
                              |
                              v
             Pix2Pix colorization and evaluation
                         (planned)
```

## Current progress

| Stage | Status |
|---|---|
| Scene selection and reuse | Complete |
| RGB generation from B2/B3/B4 | Complete |
| Thermal preprocessing from ST_B10 | Complete |
| ESRGAN dataset preparation | Complete |
| ESRGAN training | Complete |
| ESRGAN inference | Complete |
| Thermal alignment to RGB grid | Complete |
| Pix2Pix dataset preparation | Complete |
| Pix2Pix training | Planned |
| Colorized RGB generation | Planned |
| Evaluation | Planned |
| Final GUI | Planned |

The former GIS-only `clean_enhanced_thermal.py` stage has been removed from
the project. The current alignment stage is
`align_enhanced_thermal.py`, whose purpose is to create an exact RGB-grid
thermal input for Pix2Pix.

## Repository files

| File | Purpose |
|---|---|
| `scene_selection.py` | Selects scenes interactively and supports scene reuse through `INFRARED_SCENE_DIRECTORY`. |
| `run_pipeline.py` | Selects one scene and runs all implemented stages sequentially. |
| `rgb_combine.py` | Reads SR_B4, SR_B3, and SR_B2 in windows, applies percentile stretching, and writes an RGB PNG. |
| `thermal_preprocessing.py` | Processes ST_B10 in memory-safe windows, converts values to Celsius, writes thermal TIFF/PNG outputs, and creates previews. |
| `esrgan_dataset_preparation.py` | Builds a legacy per-scene paired dataset, defines the ESRGAN model, trains checkpoints, runs tiled inference, and creates comparison previews. |
| `prepare_combined_esrgan_dataset.py` | Combines thermal patches from multiple preprocessed scenes using scene-level train/validation/test splits. |
| `train_esrgan.py` | Standalone 50-epoch shared ESRGAN training entry point. |
| `run_esrgan_inference.py` | Standalone ESRGAN inference entry point. |
| `align_enhanced_thermal.py` | Warps the enhanced thermal raster onto the original georeferenced thermal/RGB grid. |
| `running commands.txt` | Windows commands for the complete pipeline and individual stages. |

## Dataset layout

The default scene root is:

```text
D:\datasets
```

Each scene is expected to be in its own directory, for example:

```text
D:\datasets\scene_001\
```

Landsat input files should be inside the selected scene directory. The band
search logic identifies the required Landsat products by their band names:

```text
SR_B2  Blue
SR_B3  Green
SR_B4  Red
ST_B10 Thermal
```

Generated files are stored below:

```text
scene_001\
└── Output\
    ├── RGB_Image.png
    ├── Thermal_PNG\
    │   ├── Thermal_Image.png
    │   └── thermal preview files
    ├── Thermal_TIFF\
    │   └── Temperature_Celsius.tif
    ├── ESRGAN_Dataset\
    │   └── (optional legacy per-scene dataset)\
    └── Enhanced_Thermal\
        ├── Enhanced_Thermal_Image.tif
        ├── Enhanced_Thermal_Pix2Pix.tif
        └── Thermal_Before_After_Comparison.png
```

## Scene selection

Every standalone script can select a scene interactively. The scene selector:

- Uses `D:\datasets` by default.
- Sorts scene names naturally, so `scene_010` follows `scene_009`.
- Accepts either a scene number or a complete scene path.
- Reports existing output files before processing.

The complete pipeline selects the scene once and passes it to every child
stage using:

```text
INFRARED_SCENE_DIRECTORY
```

This prevents repeated scene selection during one pipeline run.

## Complete pipeline

From the project directory, run:

```powershell
C:\Python314\python.exe .\run_pipeline.py
```

The runner asks for the scene once and then runs every implemented stage
automatically in this order:

1. RGB preparation
2. Thermal preprocessing
3. ESRGAN inference using the shared checkpoint
4. Pix2Pix thermal alignment
5. Pix2Pix dataset preparation

The shared dataset preparation and training commands are run separately
before scene processing. The pipeline stops immediately if a processing
stage fails. Individual scripts can
still be run independently when a single stage needs to be rerun.

## Individual commands

```powershell
# RGB preparation
C:\Python314\python.exe .\rgb_combine.py

# Thermal preprocessing
C:\Python314\python.exe .\thermal_preprocessing.py

# ESRGAN dataset preparation
C:\Python314\python.exe .\esrgan_dataset_preparation.py

# ESRGAN training
C:\Python314\python.exe .\train_esrgan.py

# ESRGAN inference
C:\Python314\python.exe .\run_esrgan_inference.py

# Align enhanced thermal output to the RGB grid
C:\Python314\python.exe .\align_enhanced_thermal.py
```

## RGB preparation

`rgb_combine.py`:

- Locates SR_B2, SR_B3, and SR_B4.
- Verifies matching dimensions.
- Reads the bands in 512 x 512 windows.
- Estimates 2nd and 98th percentile values from downsampled samples.
- Applies per-band contrast stretching to uint8 values from 0 to 255.
- Writes `Output\RGB_Image.png`.
- Generates a preview image.

The RGB PNG is a display/training image. It does not independently carry
the original GeoTIFF CRS metadata. The georeferenced thermal TIFF is used as
the reference grid for later alignment.

## Thermal preprocessing

`thermal_preprocessing.py` processes the Landsat ST_B10 band without loading
the complete scene into memory.

The thermal conversion is:

```text
Kelvin   = DN * 0.00341802 + 149.0
Celsius  = Kelvin - 273.15
```

The script:

- Reads the thermal band in 1024 x 1024 windows.
- Handles NoData and invalid values.
- Uses bounded samples for statistics instead of concatenating every pixel.
- Writes the Celsius GeoTIFF:
  `Output\Thermal_TIFF\Temperature_Celsius.tif`
- Creates an 8-bit normalized thermal PNG:
  `Output\Thermal_PNG\Thermal_Image.png`
- Generates grayscale, inferno, and histogram previews.
- Uses low-DPI previews to avoid unnecessary memory use.

The 8-bit thermal PNG is intended for ESRGAN input. The Celsius TIFF retains
the georeferenced thermal data used by the alignment stage.

## ESRGAN

The ESRGAN implementation is designed for a system with limited GPU memory,
such as an NVIDIA RTX 3050 Laptop GPU with approximately 4 GB VRAM.

### Dataset preparation

The legacy per-scene ESRGAN dataset preparation stage:

- Finds thermal PNG files below the selected scene.
- Creates paired low-resolution and high-resolution thermal patches.
- Uses a scale factor of 4.
- Splits data into training, validation, and test sets.
- Writes data under `Output\ESRGAN_Dataset`.

For normal use, prepare one combined dataset from multiple scenes:

```powershell
C:\Python314\python.exe .\prepare_combined_esrgan_dataset.py
```

This command scans `D:\datasets`, uses scenes whose thermal preprocessing
output already exists, assigns complete scenes to train/validation/test
splits, and writes:

```text
D:\datasets\ESRGAN_Combined_Dataset\
├── train\
├── validation\
├── test\
├── metadata.json
└── checkpoints\
```

Scene-level splitting prevents patches from the same scene appearing in
multiple splits.

The default dataset ratios are:

```text
Training:   80%
Validation: 10%
Testing:    10%
```

### Training configuration

The shared-model training entry point uses:

```text
Epochs:              50
Batch size:          1
Feature channels:    8
RRDB blocks:         1
Discriminator:       8 channels
HR crop size:        64 x 64
Mixed precision:     Enabled where supported
Adversarial training: Disabled by default
```

Train this shared model once, not once per selected scene. Checkpoints are
stored under:

```text
D:\datasets\ESRGAN_Combined_Dataset\checkpoints\
```

The expected final checkpoint is:

```text
esrgan_epoch_0050.pt
```

### Inference configuration

Inference uses tiled processing to avoid GPU out-of-memory errors:

```text
Tile size: 160
Overlap:   24
Scale:     4
```

The inference stage:

- Uses overlapping tiles to reduce visible seams.
- Writes only the central tile regions.
- Processes the image with mixed precision when available.
- Reports progress during long scenes.
- Applies the configured output contrast processing.
- Writes:
  `Output\Enhanced_Thermal\Enhanced_Thermal_Image.tif`
- Saves:
  `Output\Enhanced_Thermal\Thermal_Before_After_Comparison.png`

The comparison preview includes image dimensions, pixel statistics, and file
sizes in MB. Similar-looking overview images are expected because large
images are downsampled for display; the enhanced raster has a higher pixel
count.

For scene processing, `run_pipeline.py` reuses the shared checkpoint. It
does not run ESRGAN dataset preparation or ESRGAN training.

## Pix2Pix alignment

The ESRGAN output is 4x larger than the original thermal/RGB grid and does
not independently contain reliable georeferencing because ESRGAN starts from
a PNG input. Before Pix2Pix dataset generation, run:

```powershell
C:\Python314\python.exe .\align_enhanced_thermal.py
```

This stage uses:

```text
Input:
  Output\Enhanced_Thermal\Enhanced_Thermal_Image.tif

Reference:
  Output\Thermal_TIFF\Temperature_Celsius.tif

Output:
  Output\Enhanced_Thermal\Enhanced_Thermal_Pix2Pix.tif
```

The output is written to the exact reference grid:

- Same width and height as the RGB image.
- Same CRS as the original thermal GeoTIFF.
- Same affine transform.
- Same north/south and east/west pixel orientation.
- Bilinear resampling for the enhanced thermal values.
- Windowed processing for large rasters.

For Pix2Pix, use:

```text
Input:  Enhanced_Thermal_Pix2Pix.tif
Target: RGB_Image.png
```

Do not use the removed `Enhanced_Thermal_Clean.tif` output. It is no longer
part of the project.

## Planned Pix2Pix stage

The following model files are planned but not yet implemented:

```text
train_pix2pix.py
run_pix2pix_inference.py
```

The planned Pix2Pix dataset will contain paired thermal and RGB patches:

```text
Pix2Pix_Dataset\
├── train\
│   ├── thermal\
│   └── rgb\
├── validation\
│   ├── thermal\
│   └── rgb\
└── test\
    ├── thermal\
    └── rgb\
```

The planned model is:

- One-channel enhanced thermal input.
- Three-channel RGB target.
- U-Net generator.
- PatchGAN discriminator.
- Adversarial loss plus L1 reconstruction loss.

Before training, paired patches must be visually checked to confirm that
roads, water, buildings, vegetation, and scene boundaries occupy the same
locations in both modalities.

The implemented `pix2pix_dataset_preparation.py` creates synchronized
thermal/RGB PNG patches, writes an `dataset_manifest.json` file, and saves
paired previews for each split. It uses a deterministic seed and rejects
patches with excessive thermal or RGB NoData.

## Planned evaluation

The final evaluation stage is planned to include:

- PSNR
- SSIM
- LPIPS
- FID
- Side-by-side visual comparisons
- Thermal, RGB, and colorized output previews

Evaluation should be performed on held-out test scenes or test patches and
should not reuse training patches.

## Memory and performance considerations

Large Landsat scenes can exceed available RAM or GPU memory. The project
therefore uses:

- Windowed raster reads and writes.
- Downsampled percentile/statistical samples.
- Bounded thermal statistics.
- Low-memory ESRGAN model settings.
- Batch size 1 for training.
- Mixed precision where available.
- Overlapping tiled inference.
- Periodic progress reporting.
- Compressed GeoTIFF output with BigTIFF enabled when needed.

On Windows, PyTorch CUDA DLL loading may fail with `WinError 1455` when the
system paging file is too small. Enable a system-managed or sufficiently
large Windows paging file and restart the system before retrying.

## Software requirements

The project is written for Python and currently uses the Python interpreter:

```text
C:\Python314\python.exe
```

The main dependencies are:

- NumPy
- Rasterio
- GDAL (through Rasterio)
- OpenCV
- Matplotlib
- Pillow
- PyTorch
- TorchVision, if required by the installed PyTorch workflow

CUDA-enabled PyTorch is recommended for ESRGAN training and inference.
CPU execution is possible but substantially slower for large scenes.

## Known limitations

- Pix2Pix dataset creation and Pix2Pix training are not implemented yet.
- Colorized RGB generation is not implemented yet.
- Quantitative evaluation is not implemented yet.
- RGB PNG output does not retain GeoTIFF CRS metadata by itself.
- ESRGAN output must be aligned with `align_enhanced_thermal.py` before
  thermal/RGB patch pairing.
- Model quality depends on scene coverage, patch quality, training duration,
  and available GPU memory.
- A low-capacity ESRGAN model may produce overly smooth results or a narrow
  output intensity range on some scenes.

## Recommended next steps

1. Implement `pix2pix_dataset_preparation.py`.
2. Generate paired thermal/RGB patches from
   `Enhanced_Thermal_Pix2Pix.tif` and `RGB_Image.png`.
3. Visually validate patch alignment.
4. Implement and train the Pix2Pix model.
5. Add colorized RGB inference.
6. Add PSNR, SSIM, LPIPS, and FID evaluation.
7. Add a final GUI after the command-line stages are stable.

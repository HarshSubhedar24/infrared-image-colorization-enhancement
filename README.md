# Infrared Image Colorization and Enhancement

## Project title

**Infrared Image Colorization and Enhancement for Improved Object
Interpretation**

This project processes Landsat 8/9 imagery to enhance thermal information and
produce a Pix2Pix colorized RGB representation. The workflow combines visible
bands, thermal preprocessing, shared ESRGAN enhancement, geospatial alignment,
and a four-channel Pix2Pix fusion model.

## Project objective

The project is designed to:

1. Extract visible RGB information from Landsat surface-reflectance bands.
2. Convert Landsat `ST_B10` data to temperature in Celsius.
3. Enhance the thermal image with ESRGAN.
4. Align the enhanced thermal image to the RGB grid.
5. Combine visible RGB guidance and enhanced thermal information.
6. Generate a colorized RGB output with Pix2Pix.
7. Preserve spatial metadata in the final GeoTIFF.
8. Provide a visual preview and saved comparison image.

Quantitative evaluation with PSNR, SSIM, LPIPS, or FID is not implemented yet.

## Current workflow

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
                     Shared ESRGAN inference
                              |
                              v
                 Enhanced_Thermal_Image.tif
                              |
                              v
                 Thermal/RGB grid alignment
                              |
                              v
               Enhanced_Thermal_Pix2Pix.tif
                              |
                              v
              Four-channel Pix2Pix inference
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
 Pix2Pix_Colorized_RGB.png              Pix2Pix_Colorized_RGB.tif
          |
          v
     Pix2Pix_Preview.png
```

## Current implementation status

| Component | Status |
|---|---|
| Landsat scene selection | Complete |
| RGB preparation from B2/B3/B4 | Complete |
| Thermal preprocessing from ST_B10 | Complete |
| Combined ESRGAN dataset preparation | Complete |
| Shared ESRGAN training | Complete |
| ESRGAN tiled inference | Complete |
| Enhanced thermal alignment | Complete |
| Combined Pix2Pix dataset preparation | Complete |
| Four-channel Pix2Pix model | Complete |
| Shared Pix2Pix training | Complete |
| Pix2Pix tiled inference | Complete |
| Tile-overlap seam reduction | Complete |
| Visible RGB/detail preservation | Complete |
| Automatic Pix2Pix stage in main pipeline | Complete |
| Thermal/RGB preview window | Complete |
| Saved Pix2Pix preview image | Complete |
| PSNR/SSIM/LPIPS/FID evaluation | Not implemented |
| Final project GUI | Not implemented |

## Repository files

| File | Purpose |
|---|---|
| `scene_selection.py` | Selects a scene with a Windows folder picker and supports the `INFRARED_SCENE_DIRECTORY` environment variable. |
| `run_pipeline.py` | Runs the complete per-scene processing workflow. |
| `rgb_combine.py` | Reads B2/B3/B4, applies percentile stretching, and creates `RGB_Image.png`. |
| `thermal_preprocessing.py` | Converts ST_B10 to Celsius and creates thermal TIFF/PNG outputs. |
| `prepare_combined_esrgan_dataset.py` | Builds a scene-level combined ESRGAN dataset. |
| `train_esrgan.py` | Trains the shared ESRGAN model. |
| `run_esrgan_inference.py` | Runs shared ESRGAN inference on one scene. |
| `esrgan_dataset_preparation.py` | Contains ESRGAN data preparation, model, training, inference, and legacy per-scene helpers. |
| `align_enhanced_thermal.py` | Aligns enhanced thermal output to the reference thermal/RGB grid. |
| `prepare_combined_pix2pix_dataset.py` | Builds the shared four-channel Pix2Pix dataset from multiple scenes. |
| `pix2pix_model.py` | Defines the four-channel U-Net generator, PatchGAN discriminator, dataloader, and training loop. |
| `train_pix2pix.py` | Trains the shared Pix2Pix model. |
| `run_pix2pix_inference.py` | Runs tiled Pix2Pix inference and creates the interactive and saved previews. |
| `pix2pix_dataset_preparation.py` | Creates optional per-scene paired Pix2Pix patches. |
| `running commands.txt` | Windows command reference for the workflow. |

## Scene input requirements

Each selected scene must contain files whose names identify these Landsat
bands:

```text
SR_B2  Blue
SR_B3  Green
SR_B4  Red
ST_B10 Thermal
```

Scenes should be stored in separate directories below a common parent folder,
for example:

```text
D:\datasets\scene_001\
D:\datasets\scene_002\
D:\datasets\scene_003\
```

The scene picker can browse any drive. Generated outputs are written inside
the selected scene's `Output` directory.

## Complete per-scene pipeline

Run this command from the project directory:

```powershell
C:\Python314\python.exe .\run_pipeline.py
```

The script selects one scene, validates the required bands, and passes the
selected path to all child scripts through `INFRARED_SCENE_DIRECTORY`.
Therefore, only one folder-selection dialog appears.

The stages run in this order:

1. **RGB preparation** using `rgb_combine.py`.
2. **Thermal preprocessing** using `thermal_preprocessing.py`.
3. **ESRGAN inference** using an existing shared checkpoint.
4. **Thermal/RGB alignment** using `align_enhanced_thermal.py`.
5. **Pix2Pix inference** using the existing four-channel checkpoint.

The pipeline does not prepare training datasets or retrain either model.
Training is performed separately.

## Per-scene output layout

```text
scene_001\
└── Output\
    ├── RGB_Image.png
    ├── Thermal_PNG\
    │   └── Thermal_Image.png
    ├── Thermal_TIFF\
    │   └── Temperature_Celsius.tif
    ├── Enhanced_Thermal\
    │   ├── Enhanced_Thermal_Image.tif
    │   ├── Enhanced_Thermal_Pix2Pix.tif
    │   └── Thermal_Before_After_Comparison.png
    ├── Pix2Pix_Colorized_RGB.png
    ├── Pix2Pix_Colorized_RGB.tif
    └── Pix2Pix_Preview.png
```

`RGB_Image.png` is an 8-bit display/training image and does not independently
retain GeoTIFF CRS metadata. The final Pix2Pix GeoTIFF uses the aligned
thermal raster as its geospatial reference.

## RGB preparation

`rgb_combine.py`:

- Reads `SR_B4` as red, `SR_B3` as green, and `SR_B2` as blue.
- Verifies matching source dimensions.
- Reads large rasters in windows.
- Estimates percentile ranges from bounded samples.
- Applies per-band contrast stretching to `[0, 255]`.
- Writes `Output\RGB_Image.png`.

## Thermal preprocessing

`thermal_preprocessing.py` processes `ST_B10` in memory-safe windows. The
thermal conversion is:

```text
Kelvin  = DN * 0.00341802 + 149.0
Celsius = Kelvin - 273.15
```

The script handles NoData values, writes the georeferenced Celsius TIFF, and
creates an 8-bit thermal PNG for ESRGAN processing.

## Shared ESRGAN workflow

### Prepare the combined dataset

After thermal preprocessing has been completed for multiple scenes, run:

```powershell
C:\Python314\python.exe .\prepare_combined_esrgan_dataset.py
```

The script scans a parent scene directory, finds scenes containing
`Output\Thermal_PNG\Thermal_Image.png`, extracts thermal patches, and assigns
complete scenes to training, validation, and test splits. Scene-level
splitting prevents patches from one scene leaking across multiple splits.

The combined dataset is written under:

```text
ESRGAN_Combined_Dataset\
├── train\
├── validation\
├── test\
├── metadata.json
└── checkpoints\
```

### Train ESRGAN

```powershell
C:\Python314\python.exe .\train_esrgan.py
```

The shared checkpoint is normally:

```text
ESRGAN_Combined_Dataset\checkpoints\esrgan_epoch_0050.pt
```

The model is trained once and reused for new scenes. A new scene does not
require ESRGAN retraining unless it should contribute to future model
training or the existing model performs poorly on its data.

### ESRGAN inference

`run_esrgan_inference.py` uses tiled processing to reduce GPU memory use. The
shared checkpoint is searched in the project dataset location and in the
location beside the scene folders. An explicit checkpoint can be supplied:

```powershell
$env:ESRGAN_CHECKPOINT_PATH = "D:\path\to\esrgan_epoch_0050.pt"
```

The enhanced output is:

```text
Output\Enhanced_Thermal\Enhanced_Thermal_Image.tif
```

## Thermal/RGB alignment

Run independently when required:

```powershell
C:\Python314\python.exe .\align_enhanced_thermal.py
```

The alignment stage uses:

```text
Input:
  Output\Enhanced_Thermal\Enhanced_Thermal_Image.tif

Reference:
  Output\Thermal_TIFF\Temperature_Celsius.tif

Output:
  Output\Enhanced_Thermal\Enhanced_Thermal_Pix2Pix.tif
```

The aligned raster matches the reference dimensions, CRS, affine transform,
orientation, and pixel grid. Do not use the removed
`Enhanced_Thermal_Clean.tif` output.

## Shared four-channel Pix2Pix workflow

### Pix2Pix input design

The current model is a four-channel fusion model:

```text
Channel 1: visible B2 / blue
Channel 2: visible B3 / green
Channel 3: visible B4 / red
Channel 4: aligned enhanced thermal
Target:    three-channel RGB image
```

The model uses a U-Net generator and conditional PatchGAN discriminator.
Training uses adversarial loss and L1 reconstruction loss. The previous
thermal-only checkpoint is incompatible with this four-channel architecture.

### Prepare the shared Pix2Pix dataset

At least 10 valid scenes are required by default. Each scene must contain:

```text
Output\Enhanced_Thermal\Enhanced_Thermal_Pix2Pix.tif
Output\RGB_Image.png
```

Run:

```powershell
C:\Python314\python.exe .\prepare_combined_pix2pix_dataset.py
```

The script creates:

```text
Pix2Pix_Combined_Dataset\
├── train\
│   ├── input_rgb\
│   ├── input_thermal\
│   └── rgb\
├── validation\
│   ├── input_rgb\
│   ├── input_thermal\
│   └── rgb\
├── test\
│   ├── input_rgb\
│   ├── input_thermal\
│   └── rgb\
├── dataset_manifest.json
└── checkpoints\
```

The default dataset location is:

```text
D:\datasets\Pix2Pix_Combined_Dataset
```

The output location can be overridden with:

```powershell
$env:PIX2PIX_DATASET_ROOT = "D:\path\to\Pix2Pix_Combined_Dataset"
```

The dataset builder recreates its output directory. Use it deliberately and
keep validated checkpoints backed up until a new training run is confirmed.

### Train Pix2Pix

```powershell
C:\Python314\python.exe .\train_pix2pix.py
```

The main checkpoints are:

```text
D:\datasets\Pix2Pix_Combined_Dataset\checkpoints\pix2pix_best.pt
D:\datasets\Pix2Pix_Combined_Dataset\checkpoints\pix2pix_latest.pt
D:\datasets\Pix2Pix_Combined_Dataset\checkpoints\training_history.json
```

The best checkpoint is selected using validation L1 loss. A checkpoint can be
specified for resuming:

```powershell
$env:PIX2PIX_RESUME_CHECKPOINT = "D:\path\to\pix2pix_latest.pt"
C:\Python314\python.exe .\train_pix2pix.py
```

After rebuilding the combined dataset, training from the beginning is the
safest option because scene-level split assignments may change.

## Pix2Pix inference and preview

The complete pipeline runs Pix2Pix automatically. To run only Pix2Pix
inference for an already prepared scene:

```powershell
C:\Python314\python.exe .\run_pix2pix_inference.py
```

The script uses:

```text
Output\RGB_Image.png
Output\Enhanced_Thermal\Enhanced_Thermal_Pix2Pix.tif
```

The checkpoint is searched in:

1. `PIX2PIX_CHECKPOINT_PATH`, when configured.
2. `D:\datasets\Pix2Pix_Combined_Dataset\checkpoints\pix2pix_best.pt`.
3. The project-local combined dataset checkpoint.

For example:

```powershell
$env:PIX2PIX_CHECKPOINT_PATH = "D:\path\to\pix2pix_best.pt"
```

Inference uses 256-pixel tiles with 64-pixel overlap and raised-cosine
feathering. This reduces visible tile-grid seams. The output preserves 70%
of the visible RGB input during compositing so clouds and fine visible
structures are less likely to disappear. If thermal NoData overlaps a valid
visible RGB pixel, the original visible pixel is retained. Genuine black
background pixels remain black.

The generated files are:

```text
Output\Pix2Pix_Colorized_RGB.png
Output\Pix2Pix_Colorized_RGB.tif
Output\Pix2Pix_Preview.png
```

The interactive preview is a fixed 1400×800 white window:

```text
Left:  Thermal Image (Training Input)
Right: Pix2Pix Colorized RGB
```

The image areas remain black so black NoData borders are visible. The same
two-panel layout is saved as `Output\Pix2Pix_Preview.png`. No original RGB
or comparison image is displayed in this preview.

## Adding new scenes

### Process a new scene only

An existing ESRGAN and Pix2Pix checkpoint can be reused for a new scene.
Run:

```powershell
C:\Python314\python.exe .\run_pipeline.py
```

No model retraining is required just because a new scene was added to the
storage folder.

### Add new scenes to ESRGAN training

After preprocessing the new scene and other source scenes:

```powershell
C:\Python314\python.exe .\prepare_combined_esrgan_dataset.py
C:\Python314\python.exe .\train_esrgan.py
```

### Add new scenes to Pix2Pix training

First ensure the new scene has aligned thermal and RGB outputs, then rebuild
and retrain:

```powershell
C:\Python314\python.exe .\prepare_combined_pix2pix_dataset.py
C:\Python314\python.exe .\train_pix2pix.py
```

Retraining is particularly recommended when new scenes add different
locations, seasons, cloud conditions, lighting, or land-cover types, or when
the existing model performs poorly on them.

## Individual commands

```powershell
# Process one scene through all production stages
C:\Python314\python.exe .\run_pipeline.py

# Prepare RGB only
C:\Python314\python.exe .\rgb_combine.py

# Prepare thermal data only
C:\Python314\python.exe .\thermal_preprocessing.py

# Run shared ESRGAN inference
C:\Python314\python.exe .\run_esrgan_inference.py

# Align enhanced thermal data
C:\Python314\python.exe .\align_enhanced_thermal.py

# Run Pix2Pix inference and open/save its preview
C:\Python314\python.exe .\run_pix2pix_inference.py

# Rebuild shared ESRGAN training data
C:\Python314\python.exe .\prepare_combined_esrgan_dataset.py

# Train shared ESRGAN
C:\Python314\python.exe .\train_esrgan.py

# Rebuild shared Pix2Pix training data
C:\Python314\python.exe .\prepare_combined_pix2pix_dataset.py

# Train shared Pix2Pix
C:\Python314\python.exe .\train_pix2pix.py
```

## Memory and performance

The workflow is designed for large Landsat scenes and limited GPU memory. It
uses:

- Windowed raster reads and writes.
- Bounded/downsampled statistics.
- Small ESRGAN model settings.
- Batch size 1 for training.
- CUDA mixed precision when available.
- Tiled ESRGAN and Pix2Pix inference.
- Overlap blending for Pix2Pix tiles.
- LZW-compressed GeoTIFF output with BigTIFF support.

CUDA-enabled PyTorch is recommended. CPU execution is supported but can be
substantially slower.

On Windows, PyTorch CUDA loading can fail with `WinError 1455` when the
system paging file is too small. Increase the Windows paging file and restart
the system before retrying.

## Software requirements

The current development interpreter is:

```text
C:\Python314\python.exe
```

Main dependencies:

- Python
- NumPy
- Rasterio
- GDAL through Rasterio
- OpenCV
- Pillow
- Matplotlib
- PyTorch
- Tkinter for the preview window

CUDA-enabled PyTorch is recommended for model training and inference.

## Known limitations

- PSNR, SSIM, LPIPS, and FID evaluation are not implemented.
- The RGB PNG does not itself contain CRS metadata.
- Pix2Pix training currently uses `RGB_Image.png` as the RGB target.
- The visible-RGB preservation blend improves detail retention but means the
  final image is not produced exclusively by the generator.
- Model quality depends on scene diversity, alignment accuracy, checkpoint
  quality, training duration, and available GPU memory.
- The final project GUI and automated evaluation report are not implemented.

## Recommended next steps

1. Implement PSNR and SSIM evaluation against held-out scenes.
2. Compare Pix2Pix output against an RGB baseline, not only against the RGB
   target.
3. Add cloud/detail preservation statistics to evaluation.
4. Test the complete pipeline on multiple unseen scenes.
5. Add a final project GUI after evaluation is stable.

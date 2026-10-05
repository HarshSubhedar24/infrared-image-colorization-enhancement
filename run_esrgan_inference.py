from pathlib import Path

from esrgan_dataset_preparation import (
    enhance_thermal_image,
    show_thermal_comparison,
)
from scene_selection import select_scene_directory


SCENE_DIRECTORY = select_scene_directory()
OUTPUT_DIRECTORY = SCENE_DIRECTORY / "Output"

THERMAL_IMAGE = OUTPUT_DIRECTORY / "Thermal_PNG" / "Thermal_Image.png"

CHECKPOINT = (
    OUTPUT_DIRECTORY
    / "ESRGAN_Dataset"
    / "checkpoints"
    / "esrgan_epoch_0050.pt"
)

OUTPUT_IMAGE = (
    OUTPUT_DIRECTORY
    / "Enhanced_Thermal"
    / "Enhanced_Thermal_Image.tif"
)

for required_path in (THERMAL_IMAGE, CHECKPOINT):
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required file was not found for {SCENE_DIRECTORY.name}: "
            f"{required_path}\n"
            "Prepare the thermal image and train an ESRGAN checkpoint for "
            "this scene first."
        )


enhanced_image = enhance_thermal_image(
    thermal_path=THERMAL_IMAGE,
    checkpoint_path=CHECKPOINT,
    output_path=OUTPUT_IMAGE,
    tile_size=160,
    overlap=24,
    device="cuda",
    mixed_precision=True,
)

print(f"Enhanced image saved at: {enhanced_image}")
show_thermal_comparison(
    THERMAL_IMAGE,
    enhanced_image,
    comparison_path=OUTPUT_IMAGE.parent / "Thermal_Before_After_Comparison.png",
)
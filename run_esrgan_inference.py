from pathlib import Path
import os

from esrgan_dataset_preparation import (
    enhance_thermal_image,
    show_thermal_comparison,
)
from scene_selection import select_scene_directory


SCENE_DIRECTORY = select_scene_directory()
OUTPUT_DIRECTORY = SCENE_DIRECTORY / "Output"

THERMAL_IMAGE = OUTPUT_DIRECTORY / "Thermal_PNG" / "Thermal_Image.png"

def find_shared_checkpoint(scene_directory: Path) -> Path:
    """Find the shared checkpoint without assuming a dataset drive."""
    candidates = []
    configured_path = os.environ.get("ESRGAN_CHECKPOINT_PATH")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.extend(
        (
            Path(__file__).resolve().parent
            / "ESRGAN_Combined_Dataset"
            / "checkpoints"
            / "esrgan_epoch_0050.pt",
            scene_directory.parent
            / "ESRGAN_Combined_Dataset"
            / "checkpoints"
            / "esrgan_epoch_0050.pt",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No shared ESRGAN checkpoint was found for {scene_directory.name}.\n"
        f"Searched:\n{searched}\n"
        "Run prepare_combined_esrgan_dataset.py and train_esrgan.py, or set "
        "ESRGAN_CHECKPOINT_PATH to an existing checkpoint."
    )


CHECKPOINT = find_shared_checkpoint(SCENE_DIRECTORY)

OUTPUT_IMAGE = (
    OUTPUT_DIRECTORY
    / "Enhanced_Thermal"
    / "Enhanced_Thermal_Image.tif"
)

if not THERMAL_IMAGE.is_file():
    raise FileNotFoundError(
        f"Thermal preprocessing output was not found for {SCENE_DIRECTORY.name}: "
        f"{THERMAL_IMAGE}\nRun thermal_preprocessing.py first."
    )

print(f"Using shared ESRGAN checkpoint: {CHECKPOINT}")


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
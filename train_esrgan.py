from pathlib import Path

from esrgan_dataset_preparation import train_esrgan
from scene_selection import select_scene_directory


SCENE_DIRECTORY = select_scene_directory()
DATASET_ROOT = SCENE_DIRECTORY / "Output" / "ESRGAN_Dataset"

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(
        f"ESRGAN dataset was not found for {SCENE_DIRECTORY.name}: "
        f"{DATASET_ROOT}\n"
        "Prepare the ESRGAN dataset for this scene before training."
    )


checkpoint_path = train_esrgan(
    dataset_root=DATASET_ROOT,
    epochs=50,
    batch_size=1,
    learning_rate=1e-4,
    device="cuda",
    feature_channels=8,
    rrdb_blocks=1,
    discriminator_channels=8,
    mixed_precision=True,
    train_hr_crop_size=64,
    optimizer_name="sgd",
    adversarial_training=False,
)

print(f"Final checkpoint: {checkpoint_path}")
from esrgan_dataset_preparation import train_esrgan
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent / "ESRGAN_Combined_Dataset"
CHECKPOINT_ROOT = DATASET_ROOT / "checkpoints"

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(
        f"Combined ESRGAN dataset was not found: "
        f"{DATASET_ROOT}\n"
        "Run prepare_combined_esrgan_dataset.py first."
    )


checkpoint_path = train_esrgan(
    dataset_root=DATASET_ROOT,
    checkpoint_directory=CHECKPOINT_ROOT,
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
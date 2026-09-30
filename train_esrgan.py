from pathlib import Path

from esrgan_dataset_preparation import train_esrgan


DATASET_ROOT = Path(
    r"C:\Users\subhe\OneDrive\Desktop\datasets\scene_017\Output\ESRGAN_Dataset"
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